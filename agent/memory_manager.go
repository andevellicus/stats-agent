package agent

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"stats-agent/config"
	"stats-agent/rag"
	"stats-agent/web/format"
	"stats-agent/web/types"
	"time"

	"go.uber.org/zap"
)

// MemoryManager handles token counting, context window management, and history archival.
type MemoryManager struct {
	cfg    *config.Config
	rag    *rag.RAG
	logger *zap.Logger
}

// NewMemoryManager creates a new memory manager instance.
func NewMemoryManager(cfg *config.Config, rag *rag.RAG, logger *zap.Logger) *MemoryManager {
	return &MemoryManager{
		cfg:    cfg,
		rag:    rag,
		logger: logger,
	}
}

// CountTokens returns the token count for the given text using the LLM's tokenize endpoint.
func (m *MemoryManager) CountTokens(ctx context.Context, text string) (int, error) {
	reqBody := TokenizeRequest{
		Content: text,
	}
	jsonBody, err := json.Marshal(reqBody)
	if err != nil {
		return 0, fmt.Errorf("failed to marshal tokenize request body: %w", err)
	}

	url := fmt.Sprintf("%s/tokenize", m.cfg.MainLLMHost)
	var resp *http.Response

	// Retry up to 5 times if the endpoint is loading
	for range 5 {
		req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(jsonBody))
		if err != nil {
			return 0, fmt.Errorf("failed to create tokenize request: %w", err)
		}
		req.Header.Set("Content-Type", "application/json")

		client := &http.Client{}
		resp, err = client.Do(req)
		if err != nil {
			return 0, fmt.Errorf("failed to send tokenize request to llama.cpp server: %w", err)
		}

		if resp.StatusCode != http.StatusServiceUnavailable {
			break
		}

		resp.Body.Close()
		m.logger.Warn("Tokenize endpoint is loading, retrying", zap.Duration("retry_delay", 2*time.Second))
		time.Sleep(2 * time.Second)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		bodyBytes, _ := io.ReadAll(resp.Body)
		return 0, fmt.Errorf("llama.cpp server returned non-200 status for tokenize: %s, body: %s", resp.Status, string(bodyBytes))
	}

	var tokenizeResponse TokenizeResponse
	if err := json.NewDecoder(resp.Body).Decode(&tokenizeResponse); err != nil {
		return 0, fmt.Errorf("failed to decode tokenize response body: %w", err)
	}

	return len(tokenizeResponse.Tokens), nil
}

// CalculateHistorySize returns the total token count for the entire message history.
func (m *MemoryManager) CalculateHistorySize(ctx context.Context, history []types.AgentMessage) (int, error) {
	var totalTokens int
	for i := range history {
		message := &history[i]
		if message.TokenCountComputed {
			totalTokens += message.TokenCount
			continue
		}

		tokens, err := m.countTokensWithRetry(ctx, message.Content)
		if err != nil {
			return 0, fmt.Errorf("failed to count tokens for message: %w", err)
		}

		message.TokenCount = tokens
		message.TokenCountComputed = true
		totalTokens += tokens
	}
	return totalTokens, nil
}

func (m *MemoryManager) countTokensWithRetry(ctx context.Context, text string) (int, error) {
	const maxAttempts = 3
	var lastErr error
	for attempt := 1; attempt <= maxAttempts; attempt++ {
		tokens, err := m.CountTokens(ctx, text)
		if err == nil {
			return tokens, nil
		}

		lastErr = err
		if m.logger != nil {
			m.logger.Warn("Token count attempt failed, retrying", zap.Int("attempt", attempt), zap.Error(err))
		}

		if attempt == maxAttempts {
			break
		}

		select {
		case <-ctx.Done():
			return 0, ctx.Err()
		case <-time.After(500 * time.Millisecond):
		}
	}

	if lastErr != nil {
		return 0, lastErr
	}
	return 0, fmt.Errorf("token count retry loop exhausted")
}

// IsOverThreshold checks if the history size exceeds 75% of the context window.
func (m *MemoryManager) IsOverThreshold(ctx context.Context, history []types.AgentMessage) (bool, error) {
	totalTokens, err := m.CalculateHistorySize(ctx, history)
	if err != nil {
		return false, err
	}

	contextWindowThreshold := m.cfg.ContextSoftLimitTokens()
	return totalTokens > contextWindowThreshold, nil
}

// ManageHistory checks if history exceeds threshold and archives older messages to RAG.
// It operates on a pointer to the history slice and modifies it in place.
func (m *MemoryManager) ManageHistory(ctx context.Context, sessionID string, history *[]types.AgentMessage, stream *Stream) error {
	totalTokens, err := m.CalculateHistorySize(ctx, *history)
	if err != nil {
		return fmt.Errorf("failed to calculate history size: %w", err)
	}

	contextWindowThreshold := m.cfg.ContextSoftLimitTokens()

	if totalTokens <= contextWindowThreshold {
		return nil // No action needed
	}

	if stream != nil {
		_ = stream.Status("Archiving older messages....")
	}

	// Cut history in half
	cutoff := len(*history) / 2

	// Ensure we don't split assistant-tool message pairs
	if cutoff > 0 && cutoff < len(*history) {
		lastMessageInBatch := (*history)[cutoff-1]
		firstMessageOutOfBatch := (*history)[cutoff]

		// If we're about to split an assistant-tool pair, move cutoff forward
		if lastMessageInBatch.Role == "assistant" && format.HasTag(lastMessageInBatch.Content, format.PythonTag) && firstMessageOutOfBatch.Role == "tool" {
			cutoff++
			m.logger.Info("Adjusted memory cutoff to prevent splitting an assistant-tool pair")
		}
	}

	if cutoff == 0 {
		return nil // Nothing to archive
	}

	messagesToStore := (*history)[:cutoff]

	// Async archive - fire and forget
	go func(sessionID string, messages []types.AgentMessage) {
		const maxAttempts = 3
		for attempt := 0; attempt < maxAttempts; attempt++ {
			ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
			err := m.rag.AddMessagesToStore(ctx, sessionID, messages)
			cancel()

			if err == nil {
				m.logger.Info("Archived messages to long-term RAG store",
					zap.Int("messages_moved", len(messages)))
				return
			}

			if attempt < maxAttempts-1 {
				time.Sleep(time.Second * time.Duration(attempt+1))
				continue
			}

			m.logger.Error("RAG storage failed after retries - messages lost",
				zap.Error(err),
				zap.String("session_id", sessionID),
				zap.Int("messages_count", len(messages)))
		}
	}(sessionID, messagesToStore)

	// Remove archived messages from history immediately
	*history = (*history)[cutoff:]

	m.logger.Info("Memory management complete",
		zap.Int("remaining_messages", len(*history)),
		zap.Int("total_tokens", totalTokens))

	return nil
}
