package services

import (
	"bytes"
	"context"
	"fmt"
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
	return &MessageService{store: store, logger: logger}
}

// SaveAssistantAndTool persists an assistant message and an optional tool message in order.
// filesHTML is appended only to the assistant message if provided (typically on the final flush).
func (ms *MessageService) SaveAssistantAndTool(ctx context.Context, sessionID string, assistant string, tool *string, filesHTML string) (string, error) {
    assistant = strings.TrimSpace(assistant)
    var assistantID string

    if assistant != "" {
        // Remove agent-status sections before persisting (do not store status in DB)
        assistant = format.RemoveTagSections(assistant, format.AgentStatusTag)
        assistant, _ = format.CloseUnbalancedTags(assistant)
        rendered, err := ms.processContentForDB(ctx, assistant)
        if err != nil {
            return "", fmt.Errorf("process assistant content: %w", err)
        }
        if filesHTML != "" {
            rendered += filesHTML
        }

		assistantID = generateMessageID()
		assistantMsg := types.ChatMessage{
			ID:        assistantID,
			SessionID: sessionID,
			Role:      "assistant",
			Content:   assistant,
			Rendered:  rendered,
		}

		if err := ms.store.CreateMessage(ctx, assistantMsg); err != nil {
			ms.logger.Error("Failed to save assistant message", zap.Error(err))
			return "", fmt.Errorf("save assistant message: %w", err)
		}
	}

	if tool != nil {
		result := strings.TrimSpace(*tool)
		if result != "" {
			renderedTool, err := ms.renderToolContent(ctx, result)
			if err != nil {
				ms.logger.Error("Failed to render tool message", zap.Error(err))
				return assistantID, fmt.Errorf("render tool message: %w", err)
			}

			toolMsg := types.ChatMessage{
				ID:        generateMessageID(),
				SessionID: sessionID,
				Role:      "tool",
				Content:   result,
				Rendered:  renderedTool,
			}
			if err := ms.store.CreateMessage(ctx, toolMsg); err != nil {
				ms.logger.Error("Failed to save tool message", zap.Error(err))
				return assistantID, fmt.Errorf("save tool message: %w", err)
			}
		}
	}

	return assistantID, nil
}

// AppendFilesToMessage appends HTML (e.g., uploaded files) to an existing assistant message.
func (ms *MessageService) AppendFilesToMessage(ctx context.Context, messageID string, filesHTML string) error {
	filesHTML = strings.TrimSpace(filesHTML)
	if filesHTML == "" {
		return nil
	}
	if err := ms.store.AppendToMessageRendered(ctx, messageID, filesHTML); err != nil {
		ms.logger.Error("Failed to append HTML to message", zap.Error(err), zap.String("message_id", messageID))
		return fmt.Errorf("append html to message: %w", err)
	}
	return nil
}

func (ms *MessageService) processContentForDB(ctx context.Context, rawContent string) (string, error) {
    // Normalize common LLM quirks (e.g., python> prompts, ```python, curly quotes)
    preprocessed := format.PreprocessAssistantText(rawContent)
    // Remove any agent-status sections defensively (should already be stripped)
    preprocessed = format.RemoveTagSections(preprocessed, format.AgentStatusTag)
    // Ensure tags are balanced before converting to HTML
    preprocessed, _ = format.CloseUnbalancedTags(preprocessed)
    return format.ConvertToHTML(ctx, preprocessed)
}

func (ms *MessageService) renderToolContent(ctx context.Context, result string) (string, error) {
	var buf bytes.Buffer
	if err := components.ExecutionResultBlock(result).Render(ctx, &buf); err != nil {
		return "", fmt.Errorf("render execution result block: %w", err)
	}
	return buf.String(), nil
}

func generateMessageID() string {
	return uuid.New().String()
}
