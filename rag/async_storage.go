package rag

import (
	"context"
	"stats-agent/web/types"
	"time"

	"go.uber.org/zap"
)

// AddMessagesAsync stores messages to RAG asynchronously with retry logic.
// This method is non-blocking and designed for proactive storage during conversation.
func (r *RAG) AddMessagesAsync(sessionID string, messages []types.AgentMessage) {
	if len(messages) == 0 {
		return
	}

	go func(sessionID string, messages []types.AgentMessage) {
		const maxAttempts = 3
		for attempt := range maxAttempts {
			ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
			err := r.AddMessagesToStore(ctx, sessionID, messages)
			cancel()

			if err == nil {
				r.logger.Info("Stored messages to RAG",
					zap.Int("message_count", len(messages)),
					zap.String("session_id", sessionID))
				return
			}

			if attempt < maxAttempts-1 {
				// Exponential backoff: 1s, 2s
				time.Sleep(time.Second * time.Duration(attempt+1))
				continue
			}

			r.logger.Error("RAG storage failed after retries",
				zap.Error(err),
				zap.String("session_id", sessionID),
				zap.Int("message_count", len(messages)))
		}
	}(sessionID, messages)
}
