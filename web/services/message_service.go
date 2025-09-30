package services

import (
	"bytes"
	"context"
	"fmt"
	"regexp"
	"stats-agent/database"
	"stats-agent/web/format"
	"stats-agent/web/templates/components"
	"stats-agent/web/types"
	"strings"

	"github.com/google/uuid"
	"go.uber.org/zap"
)

type MessageService struct {
	store  *database.PostgresStore
	logger *zap.Logger
}

func NewMessageService(store *database.PostgresStore, logger *zap.Logger) *MessageService {
	return &MessageService{
		store:  store,
		logger: logger,
	}
}

// ParseAndSaveAgentResponse parses the agent's raw response, splits it into assistant and tool messages,
// and saves them to the database. Returns error if save fails.
func (ms *MessageService) ParseAndSaveAgentResponse(ctx context.Context, rawResponse, sessionID, filesHTML string) error {
	// Remove agent_status tags as they're only for UI streaming
	statusRe := regexp.MustCompile(`(?s)<agent_status>.*?</agent_status>`)
	cleanedResponse := statusRe.ReplaceAllString(rawResponse, "")

	// Split by execution_results tags
	re := regexp.MustCompile(`(?s)(<execution_results>.*?</execution_results>)`)
	parts := re.Split(cleanedResponse, -1)
	matches := re.FindAllString(cleanedResponse, -1)

	// Save assistant messages and tool messages alternately
	for i, part := range parts {
		assistantContent := strings.TrimSpace(part)
		if assistantContent != "" {
			assistantRendered, err := ms.processContentForDB(ctx, assistantContent)
			if err != nil {
				return fmt.Errorf("failed to process assistant content: %w", err)
			}

			// Append file HTML only to the last assistant message
			isLastPart := (i == len(parts)-1)
			if isLastPart && filesHTML != "" {
				assistantRendered += filesHTML
			}

			assistantMessage := types.ChatMessage{
				ID:        generateMessageID(),
				SessionID: sessionID,
				Role:      "assistant",
				Content:   assistantContent,
				Rendered:  assistantRendered,
			}
			if err := ms.store.CreateMessage(ctx, assistantMessage); err != nil {
				ms.logger.Error("Failed to save assistant message part", zap.Error(err))
				return fmt.Errorf("failed to save assistant message: %w", err)
			}
		}

		// Save tool message if it exists
		if i < len(matches) {
			toolContentRaw := strings.TrimSpace(matches[i])
			result := strings.TrimSuffix(strings.TrimPrefix(toolContentRaw, "<execution_results>"), "</execution_results>")

			var buf bytes.Buffer
			if err := components.ExecutionResultBlock(result).Render(ctx, &buf); err != nil {
				ms.logger.Error("Failed to render execution result block for DB", zap.Error(err))
				return fmt.Errorf("failed to render tool message: %w", err)
			}

			toolMessage := types.ChatMessage{
				ID:        generateMessageID(),
				SessionID: sessionID,
				Role:      "tool",
				Content:   result,
				Rendered:  buf.String(),
			}
			if err := ms.store.CreateMessage(ctx, toolMessage); err != nil {
				ms.logger.Error("Failed to save tool message", zap.Error(err))
				return fmt.Errorf("failed to save tool message: %w", err)
			}
		}
	}

	return nil
}

// processContentForDB converts agent content (with XML tags) to HTML for database storage.
// Delegates to the format package for consistent conversion logic.
func (ms *MessageService) processContentForDB(ctx context.Context, rawContent string) (string, error) {
	return format.ConvertToHTML(ctx, rawContent)
}

func generateMessageID() string {
	return uuid.New().String()
}
