package agent

import (
	"fmt"
	"stats-agent/config"
	"stats-agent/web/types"
	"strings"

	"go.uber.org/zap"
)

// ResponseHandler manages LLM response collection and message assembly.
type ResponseHandler struct {
	cfg    *config.Config
	logger *zap.Logger
}

// NewResponseHandler creates a new response handler instance.
func NewResponseHandler(cfg *config.Config, logger *zap.Logger) *ResponseHandler {
	return &ResponseHandler{
		cfg:    cfg,
		logger: logger,
	}
}

// BuildMessagesForLLM combines long-term context with current history for LLM input.
// If longTermContext is not empty, it's prepended as a system message.
func (r *ResponseHandler) BuildMessagesForLLM(longTermContext string, history []types.AgentMessage) []types.AgentMessage {
	var messagesForLLM []types.AgentMessage

	if longTermContext != "" {
		messagesForLLM = append(messagesForLLM, types.AgentMessage{
			Role:    "system",
			Content: longTermContext,
		})
	}

	messagesForLLM = append(messagesForLLM, history...)

	return messagesForLLM
}

// CollectStreamedResponse reads chunks from a streaming response channel and builds
// the complete response. It also prints chunks to stdout for real-time display.
func (r *ResponseHandler) CollectStreamedResponse(responseChan <-chan string) string {
	var llmResponseBuilder strings.Builder

	for chunk := range responseChan {
		fmt.Print(chunk)
		llmResponseBuilder.WriteString(chunk)
	}

	llmResponse := llmResponseBuilder.String()

	// Add newline if response doesn't end with one
	if !strings.HasSuffix(llmResponse, "\n") {
		fmt.Println()
	}

	return llmResponse
}

// CollectResponse reads chunks from a response channel and builds the complete response
// without printing it to stdout.
func (r *ResponseHandler) CollectResponse(responseChan <-chan string) string {
	var llmResponseBuilder strings.Builder

	for chunk := range responseChan {
		llmResponseBuilder.WriteString(chunk)
	}

	return llmResponseBuilder.String()
}

// IsEmpty checks if the response is empty or only whitespace.
func (r *ResponseHandler) IsEmpty(response string) bool {
	return strings.TrimSpace(response) == ""
}
