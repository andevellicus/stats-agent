package agent

import (
	"context"
	"fmt"
	"stats-agent/config"
	"stats-agent/llmclient"
	"stats-agent/web/format"
	"stats-agent/web/types"
	"time"

	"go.uber.org/zap"
)

// MemoryManager handles token counting, context window management, and history trimming.
type MemoryManager struct {
	cfg    *config.Config
	logger *zap.Logger
}

// NewMemoryManager creates a new memory manager instance.
func NewMemoryManager(cfg *config.Config, logger *zap.Logger) *MemoryManager {
	return &MemoryManager{
		cfg:    cfg,
		logger: logger,
	}
}

// CountTokens returns the token count for the given text using the LLM's tokenize endpoint.
func (m *MemoryManager) CountTokens(ctx context.Context, text string) (int, error) {
	client := llmclient.New(m.cfg, m.logger)
	return client.Tokenize(ctx, m.cfg.MainLLMHost, text)
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

// ManageHistory checks if history exceeds threshold and trims older messages.
// It operates on a pointer to the history slice and modifies it in place.
// Note: Messages are stored to RAG proactively during conversation, so this only trims history.
func (m *MemoryManager) ManageHistory(ctx context.Context, sessionID string, history *[]types.AgentMessage, stream *Stream) error {
	totalTokens, err := m.CalculateHistorySize(ctx, *history)
	if err != nil {
		return fmt.Errorf("failed to calculate history size: %w", err)
	}

	contextWindowThreshold := m.cfg.ContextSoftLimitTokens()

	if totalTokens <= contextWindowThreshold {
		return nil // No action needed
	}

	// Cut history in half
	cutoff := len(*history) / 2

	// Ensure we don't split assistant-tool message pairs
	if cutoff > 0 && cutoff < len(*history) {
		lastMessageInBatch := (*history)[cutoff-1]
		firstMessageOutOfBatch := (*history)[cutoff]

		// If we're about to split an assistant-tool pair, move cutoff forward
		if lastMessageInBatch.Role == "assistant" && format.HasCodeBlock(lastMessageInBatch.Content) && firstMessageOutOfBatch.Role == "tool" {
			cutoff++
			m.logger.Info("Adjusted memory cutoff to prevent splitting an assistant-tool pair")
		}
	}

	if cutoff == 0 {
		return nil // Nothing to trim
	}

	// Remove old messages from history immediately
	// Note: Messages are already stored to RAG proactively during conversation,
	// so memory manager only needs to trim the in-memory history
	removedCount := cutoff
	*history = (*history)[cutoff:]

	m.logger.Info("Memory management complete - trimmed old messages from history",
		zap.Int("messages_removed", removedCount),
		zap.Int("remaining_messages", len(*history)),
		zap.Int("total_tokens", totalTokens))

	return nil
}
