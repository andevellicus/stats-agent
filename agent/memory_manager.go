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
	for _, message := range history {
		tokens, err := m.CountTokens(ctx, message.Content)
		if err != nil {
			m.logger.Warn("Could not count tokens for a message, falling back to character count", zap.Error(err))
			// Fallback: rough estimate of 4 chars per token
			totalTokens += len(message.Content) / 4
		} else {
			totalTokens += tokens
		}
	}
	return totalTokens, nil
}

// IsOverThreshold checks if the history size exceeds 75% of the context window.
func (m *MemoryManager) IsOverThreshold(ctx context.Context, history []types.AgentMessage) (bool, error) {
	totalTokens, err := m.CalculateHistorySize(ctx, history)
	if err != nil {
		return false, err
	}

	contextWindowThreshold := int(float64(m.cfg.ContextLength) * 0.75)
	return totalTokens > contextWindowThreshold, nil
}

// ManageHistory checks if history exceeds threshold and archives older messages to RAG.
// It operates on a pointer to the history slice and modifies it in place.
func (m *MemoryManager) ManageHistory(ctx context.Context, sessionID string, history *[]types.AgentMessage, stream *Stream) error {
	totalTokens, err := m.CalculateHistorySize(ctx, *history)
	if err != nil {
		return fmt.Errorf("failed to calculate history size: %w", err)
	}

	contextWindowThreshold := int(float64(m.cfg.ContextLength) * 0.75)

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

	// Archive to RAG - non-critical, log warning if fails but continue
	if err := m.rag.AddMessagesToStore(ctx, sessionID, messagesToStore); err != nil {
		m.logger.Warn("Failed to archive messages to RAG, they will be lost from long-term memory",
			zap.Error(err),
			zap.Int("messages_count", len(messagesToStore)))
		// Don't return error - continue with memory management to prevent context overflow
	} else {
		m.logger.Info("Archived messages to long-term RAG store",
			zap.Int("messages_moved", len(messagesToStore)))
	}

	// Remove archived messages from history
	*history = (*history)[cutoff:]

	m.logger.Info("Memory management complete",
		zap.Int("remaining_messages", len(*history)),
		zap.Int("total_tokens", totalTokens))

	return nil
}
