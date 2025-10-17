package rag

import (
    "context"
    "fmt"
    "strings"
    "crypto/sha256"
    "encoding/hex"
    "time"

    "stats-agent/llmclient"

    "go.uber.org/zap"
)

// Tokenize request/response now shared via llmclient.

func (r *RAG) ensureEmbeddingTokenLimit(ctx context.Context, content string) string {
	trimmed := strings.TrimSpace(content)
	minCharThreshold := r.minTokenCheckCharThreshold
	if minCharThreshold > 0 && len(trimmed) <= minCharThreshold {
		return content
	}

	// Check if content is already under limit
	tokenCount, err := r.countTokensForEmbedding(ctx, trimmed)
	if err != nil {
		if r.logger != nil {
			r.logger.Debug("Token counting failed for embedding content", zap.Error(err))
		}
		return content
	}

    targetTokens := r.embeddingTokenTarget

	if tokenCount <= targetTokens {
		return content
	}

	// Content exceeds limit - use forward token accumulation
	// Split by words and build up until we hit the token limit
	words := strings.Fields(content)
	if len(words) == 0 {
		return content
	}

	var accumulated []string
	const checkInterval = 5 // Check tokens every 5 words to reduce API calls

	for i, word := range words {
		accumulated = append(accumulated, word)

		// Check token count every N words or at the end
		shouldCheck := (len(accumulated)%checkInterval == 0) || (i == len(words)-1)
		if !shouldCheck {
			continue
		}

		testText := strings.Join(accumulated, " ")
    tokens, err := r.countTokensForEmbedding(ctx, testText)
		if err != nil {
			// On error, fall back to character ratio truncation
			if r.logger != nil {
				r.logger.Warn("Token counting failed during truncation, using character ratio fallback", zap.Error(err))
			}
			runes := []rune(content)
			safeLen := len(runes) * targetTokens / tokenCount
			return string(runes[:safeLen])
		}

		if tokens > targetTokens {
			// Exceeded limit - backtrack by removing last batch of words
			wordsToRemove := min(checkInterval, len(accumulated))
			accumulated = accumulated[:len(accumulated)-wordsToRemove]

			// Fine-tune by adding back words one at a time
			for j := 0; j < wordsToRemove; j++ {
				if i-wordsToRemove+j+1 >= len(words) {
					break
				}
				testWords := append(accumulated, words[i-wordsToRemove+j+1])
				testText := strings.Join(testWords, " ")
				tokens, err := r.countTokensForEmbedding(ctx, testText)
				if err != nil || tokens > targetTokens {
					break
				}
				accumulated = testWords
			}
			break // Done truncating
		}
	}

	if len(accumulated) == 0 {
		// Couldn't fit even one word - use very conservative character truncation
		runes := []rune(content)
		safeLen := len(runes) * targetTokens / tokenCount / 2 // Extra conservative
		if safeLen < 1 {
			safeLen = 1
		}
		return string(runes[:safeLen])
	}

    result := strings.Join(accumulated, " ")
    if r.logger != nil {
        finalTokens, _ := r.countTokensForEmbedding(ctx, result)
        r.logger.Debug("Truncated embedding content using token accumulation",
            zap.Int("original_tokens", tokenCount),
            zap.Int("final_tokens", finalTokens),
            zap.Int("target_tokens", targetTokens),
            zap.Int("original_words", len(words)),
            zap.Int("final_words", len(accumulated)))
    }
    return result
}

// countTokensWithCache computes token counts with an LRU cache, heuristic estimation for long texts,
// and asynchronous validation for long inputs to keep the cache warm.
func (r *RAG) countTokensWithCache(ctx context.Context, text string) (int, error) {
    trimmed := strings.TrimSpace(text)
    if trimmed == "" {
        return 0, nil
    }

    // Key: 8-byte prefix of sha256 for speed and low collision risk
    sum := sha256.Sum256([]byte(trimmed))
    key := hex.EncodeToString(sum[:8])

    // Fast path: cache hit
    if r.tokenCache != nil {
        r.tokenCacheMu.RLock()
        if val, ok := r.tokenCache.Get(key); ok {
            if v, ok2 := val.(int); ok2 {
                r.tokenCacheMu.RUnlock()
                return v, nil
            }
        }
        r.tokenCacheMu.RUnlock()
    }

    // For short inputs, compute exact synchronously
    if len(trimmed) < 1000 {
        count, err := r.countTokensForEmbeddingExact(ctx, trimmed)
        if err == nil && r.tokenCache != nil {
            r.tokenCacheMu.Lock()
            r.tokenCache.Add(key, count)
            r.tokenCacheMu.Unlock()
        }
        return count, err
    }

    // For long inputs, return an estimate and validate asynchronously
    estimate := estimateTokens(trimmed)

    if r.tokenCache != nil {
        go func(txt string, cacheKey string) {
            // Background context with timeout to avoid hanging
            ctx2, cancel := context.WithTimeout(context.Background(), 5*time.Second)
            defer cancel()
            if count, err := r.countTokensForEmbeddingExact(ctx2, txt); err == nil {
                r.tokenCacheMu.Lock()
                r.tokenCache.Add(cacheKey, count)
                r.tokenCacheMu.Unlock()
            }
        }(trimmed, key)
    }

    return estimate, nil
}

// estimateTokens uses a conservative multiplier based on word count.
func estimateTokens(text string) int {
    // Roughly ~1.3 tokens per word for English (empirical)
    return int(float64(len(strings.Fields(text))) * 1.3)
}

func (r *RAG) countTokensForEmbedding(ctx context.Context, text string) (int, error) {
    // Route through cache-aware path
    return r.countTokensWithCache(ctx, text)
}

// countTokensForEmbeddingExact hits the tokenize endpoint directly (no caching/estimation).
func (r *RAG) countTokensForEmbeddingExact(ctx context.Context, text string) (int, error) {
    if r.cfg == nil || strings.TrimSpace(r.cfg.EmbeddingLLMHost) == "" {
        return 0, fmt.Errorf("embedding LLM host not configured")
    }
    client := llmclient.New(r.cfg, r.logger)
    return client.Tokenize(ctx, r.cfg.EmbeddingLLMHost, text)
}

// EmbeddingWindow represents a single window of text with its embedding.
type EmbeddingWindow struct {
	WindowIndex int
	WindowStart int
	WindowEnd   int
	WindowText  string
	Embedding   []float32
}

// createEmbeddingWindows splits text into multiple windows and generates an embedding for each.
// This ensures all content is searchable, even if it exceeds the embedding model's token limit.
func (r *RAG) createEmbeddingWindows(ctx context.Context, content string) ([]EmbeddingWindow, error) {
	trimmed := strings.TrimSpace(content)
	if trimmed == "" {
		return nil, nil
	}

    targetTokens := r.embeddingTokenTarget

	// Check total token count
    totalTokens, err := r.countTokensForEmbedding(ctx, trimmed)
	if err != nil {
		return nil, fmt.Errorf("failed to count tokens: %w", err)
	}

	// If content fits in one window, create single embedding
	if totalTokens <= targetTokens {
        embedding, err := r.embedder(ctx, trimmed)
		if err != nil {
			return nil, fmt.Errorf("failed to create embedding: %w", err)
		}
		return []EmbeddingWindow{{
			WindowIndex: 0,
			WindowStart: 0,
			WindowEnd:   len(trimmed),
			WindowText:  trimmed,
			Embedding:   embedding,
		}}, nil
	}

	// Split into multiple windows
	words := strings.Fields(trimmed)
	var windows []EmbeddingWindow
	windowIndex := 0
	currentPos := 0

    for i := 0; i < len(words); {
		accumulated := []string{}
		startPos := currentPos

		// Build up words until we hit the token limit
		const checkInterval = 10
		for j := i; j < len(words); j++ {
			accumulated = append(accumulated, words[j])

			// Check token count every N words or at the end
			if (len(accumulated)%checkInterval == 0) || (j == len(words)-1) {
				testText := strings.Join(accumulated, " ")
            tokens, err := r.countTokensForEmbedding(ctx, testText)
				if err != nil {
					return nil, fmt.Errorf("failed to count tokens for window: %w", err)
				}

				if tokens > targetTokens {
					// Backtrack
					wordsToRemove := min(checkInterval, len(accumulated))
					accumulated = accumulated[:len(accumulated)-wordsToRemove]

					// Fine-tune by adding back one word at a time
					for k := 0; k < wordsToRemove; k++ {
						testWithOne := append(accumulated, words[j-wordsToRemove+k+1])
						testText := strings.Join(testWithOne, " ")
                tokens, _ := r.countTokensForEmbedding(ctx, testText)
						if tokens <= targetTokens {
							accumulated = testWithOne
						} else {
							break
						}
					}
					break
				}
			}
		}

		if len(accumulated) == 0 {
			// Edge case: single word exceeds limit - include it anyway
			accumulated = []string{words[i]}
		}

		windowText := strings.Join(accumulated, " ")
        embedding, err := r.embedder(ctx, windowText)
		if err != nil {
			return nil, fmt.Errorf("failed to create embedding for window %d: %w", windowIndex, err)
		}

		endPos := currentPos + len(windowText)
		windows = append(windows, EmbeddingWindow{
			WindowIndex: windowIndex,
			WindowStart: startPos,
			WindowEnd:   endPos,
			WindowText:  windowText,
			Embedding:   embedding,
		})

		i += len(accumulated)
		currentPos = endPos + 1 // +1 for space between windows
		windowIndex++
	}

    return windows, nil
}

// createEmbeddingWindowsBatch splits each chunk into windows and generates embeddings in a single batch call.
// It returns a slice of windows per input chunk, preserving order.
func (r *RAG) createEmbeddingWindowsBatch(ctx context.Context, chunks []string) ([][]EmbeddingWindow, error) {
    if len(chunks) == 0 {
        return nil, nil
    }

    targetTokens := r.embeddingTokenTarget

    // First pass: compute window texts and positions per chunk
    type rawWindow struct {
        chunkIdx   int
        start      int
        end        int
        text       string
        windowIdx  int
    }
    var allWindows []rawWindow
    perChunkCounts := make([]int, len(chunks))

    for ci, content := range chunks {
        trimmed := strings.TrimSpace(content)
        if trimmed == "" {
            perChunkCounts[ci] = 0
            continue
        }

        // Count total tokens with cache
        totalTokens, err := r.countTokensForEmbedding(ctx, trimmed)
        if err != nil {
            return nil, fmt.Errorf("failed to count tokens: %w", err)
        }

        // One window fits
        if totalTokens <= targetTokens {
            allWindows = append(allWindows, rawWindow{
                chunkIdx:  ci,
                start:     0,
                end:       len(trimmed),
                text:      trimmed,
                windowIdx: 0,
            })
            perChunkCounts[ci] = 1
            continue
        }

        // Multi-window split mirroring createEmbeddingWindows logic
        words := strings.Fields(trimmed)
        windowIndex := 0
        currentPos := 0
        for i := 0; i < len(words); {
            accumulated := []string{}
            startPos := currentPos
            const checkInterval = 10

            for j := i; j < len(words); j++ {
                accumulated = append(accumulated, words[j])
                if (len(accumulated)%checkInterval == 0) || (j == len(words)-1) {
                    testText := strings.Join(accumulated, " ")
                    tokens, err := r.countTokensForEmbedding(ctx, testText)
                    if err != nil {
                        return nil, fmt.Errorf("failed to count tokens for window: %w", err)
                    }
                    if tokens > targetTokens {
                        // Backtrack
                        wordsToRemove := min(checkInterval, len(accumulated))
                        accumulated = accumulated[:len(accumulated)-wordsToRemove]
                        for k := 0; k < wordsToRemove; k++ {
                            testWithOne := append(accumulated, words[j-wordsToRemove+k+1])
                            testText := strings.Join(testWithOne, " ")
                            tokens, _ := r.countTokensForEmbedding(ctx, testText)
                            if tokens <= targetTokens {
                                accumulated = testWithOne
                            } else {
                                break
                            }
                        }
                        break
                    }
                }
            }

            if len(accumulated) == 0 {
                accumulated = []string{words[i]}
            }

            windowText := strings.Join(accumulated, " ")
            endPos := currentPos + len(windowText)

            allWindows = append(allWindows, rawWindow{
                chunkIdx:  ci,
                start:     startPos,
                end:       endPos,
                text:      windowText,
                windowIdx: windowIndex,
            })

            i += len(accumulated)
            currentPos = endPos + 1
            windowIndex++
        }
        perChunkCounts[ci] = windowIndex
    }

    // Flatten texts for a single embedding call
    flatTexts := make([]string, len(allWindows))
    for i, w := range allWindows {
        flatTexts[i] = w.text
    }

    embeddings, err := r.embedBatch(ctx, flatTexts)
    if err != nil {
        return nil, err
    }
    if len(embeddings) != len(allWindows) {
        return nil, fmt.Errorf("embedding batch size mismatch: got %d, want %d", len(embeddings), len(allWindows))
    }

    // Distribute embeddings back to per-chunk windows
    result := make([][]EmbeddingWindow, len(chunks))
    for i := range result {
        if perChunkCounts[i] > 0 {
            result[i] = make([]EmbeddingWindow, 0, perChunkCounts[i])
        }
    }

    for i, w := range allWindows {
        result[w.chunkIdx] = append(result[w.chunkIdx], EmbeddingWindow{
            WindowIndex: w.windowIdx,
            WindowStart: w.start,
            WindowEnd:   w.end,
            WindowText:  w.text,
            Embedding:   embeddings[i],
        })
    }

    return result, nil
}
