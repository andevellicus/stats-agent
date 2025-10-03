package rag

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"stats-agent/web/format"

	"go.uber.org/zap"
)

type ragTokenizeRequest struct {
	Content string `json:"content"`
}

type ragTokenizeResponse struct {
	Tokens []int `json:"tokens"`
}

func (r *RAG) ensureEmbeddingTokenLimit(ctx context.Context, content string) string {
	trimmed := strings.TrimSpace(content)
	if len(trimmed) <= minTokenCheckCharThreshold {
		return content
	}

	tokenCount, err := r.countTokensForEmbedding(ctx, trimmed)
	if err != nil {
		if r.logger != nil {
			r.logger.Debug("Token counting failed for embedding content", zap.Error(err))
		}
		return content
	}

	if tokenCount <= embeddingTokenSoftLimit {
		return content
	}

	runes := []rune(content)
	safeLen := len(runes) * embeddingTokenTarget / tokenCount
	if safeLen <= 0 {
		safeLen = len(runes)
	}
	if safeLen >= len(runes) {
		return content
	}

	truncated := string(runes[:safeLen])
	balanced, _ := format.CloseUnbalancedTags(truncated)
	if r.logger != nil {
		r.logger.Debug("Truncated embedding content to respect token limit",
			zap.Int("original_tokens", tokenCount),
			zap.Int("target_tokens", embeddingTokenTarget),
			zap.Int("original_length", len(runes)),
			zap.Int("truncated_length", len([]rune(balanced))))
	}
	return balanced
}

func (r *RAG) countTokensForEmbedding(ctx context.Context, text string) (int, error) {
	if r.cfg == nil || strings.TrimSpace(r.cfg.MainLLMHost) == "" {
		return 0, fmt.Errorf("main LLM host not configured")
	}

	payload, err := json.Marshal(ragTokenizeRequest{Content: text})
	if err != nil {
		return 0, fmt.Errorf("marshal tokenize request: %w", err)
	}

	client := &http.Client{Timeout: r.cfg.LLMRequestTimeout}
	url := fmt.Sprintf("%s/tokenize", strings.TrimRight(r.cfg.MainLLMHost, "/"))

	for attempt := 0; attempt < 3; attempt++ {
		req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(payload))
		if err != nil {
			return 0, fmt.Errorf("create tokenize request: %w", err)
		}
		req.Header.Set("Content-Type", "application/json")

		resp, err := client.Do(req)
		if err != nil {
			if ctx.Err() != nil {
				return 0, ctx.Err()
			}
			if attempt == 2 {
				return 0, fmt.Errorf("tokenize request failed: %w", err)
			}
			time.Sleep(200 * time.Millisecond * time.Duration(attempt+1))
			continue
		}

		bodyCloser := resp.Body
		if resp.StatusCode == http.StatusServiceUnavailable {
			bodyCloser.Close()
			if attempt < 2 {
				time.Sleep(200 * time.Millisecond * time.Duration(attempt+1))
			}
			continue
		}

		if resp.StatusCode != http.StatusOK {
			defer bodyCloser.Close()
			responseBody, _ := io.ReadAll(bodyCloser)
			return 0, fmt.Errorf("tokenize status %s: %s", resp.Status, strings.TrimSpace(string(responseBody)))
		}

		var tokenizeResp ragTokenizeResponse
		decodeErr := json.NewDecoder(bodyCloser).Decode(&tokenizeResp)
		bodyCloser.Close()
		if decodeErr != nil {
			return 0, fmt.Errorf("decode tokenize response: %w", decodeErr)
		}

		return len(tokenizeResp.Tokens), nil
	}

	return 0, fmt.Errorf("failed to tokenize content after retries")
}
