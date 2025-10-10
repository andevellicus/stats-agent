package rag

import (
    "context"
    "fmt"
    "strings"

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
	if targetTokens <= 0 {
		targetTokens = 480 // BGE default
	}

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

func (r *RAG) countTokensForEmbedding(ctx context.Context, text string) (int, error) {
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
	if targetTokens <= 0 {
		targetTokens = 480 // BGE default
	}

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
