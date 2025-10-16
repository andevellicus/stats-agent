package agent

import (
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

// BuildMessagesForLLM combines retrieved state with current history for LLM input.
// If state is not empty, it's prepended as a system message.
func (r *ResponseHandler) BuildMessagesForLLM(state string, history []types.AgentMessage) []types.AgentMessage {
    var messagesForLLM []types.AgentMessage

    if state != "" {
        messagesForLLM = append(messagesForLLM, types.AgentMessage{
            Role:    "system",
            Content: state,
        })
    }

	messagesForLLM = append(messagesForLLM, history...)

	return messagesForLLM
}

// BuildMessagesForLLMWithEvidence prepends retrieved state and an optional
// ephemeral evidence block (not persisted) before appending history.
func (r *ResponseHandler) BuildMessagesForLLMWithEvidence(state, evidence string, history []types.AgentMessage) []types.AgentMessage {
    var messagesForLLM []types.AgentMessage

    if state != "" {
        messagesForLLM = append(messagesForLLM, types.AgentMessage{Role: "system", Content: state})
    }
    if strings.TrimSpace(evidence) != "" {
        messagesForLLM = append(messagesForLLM, types.AgentMessage{Role: "system", Content: evidence})
    }
    messagesForLLM = append(messagesForLLM, history...)
    return messagesForLLM
}

// CollectStreamedResponse reads chunks from a streaming response channel and builds
// the complete response. It also prints chunks to stdout for real-time display.
func (r *ResponseHandler) CollectStreamedResponse(responseChan <-chan string, stream *Stream) string {
	var llmResponseBuilder strings.Builder
	chunkCount := 0

	for chunk := range responseChan {
		chunkCount++

		if stream != nil {
			_, _ = stream.WriteString(chunk)
		}
		llmResponseBuilder.WriteString(chunk)
	}

	llmResponse := llmResponseBuilder.String()

	// Check if response was stopped mid-code-block (missing closing fence)
	// This happens when stop sequence "\n```\n" triggers
	if strings.Contains(llmResponse, "```python") && !strings.HasSuffix(strings.TrimSpace(llmResponse), "```") {
		// Count opening and closing fences
		openCount := strings.Count(llmResponse, "```python")
		closeCount := strings.Count(llmResponse, "```") - openCount // Total ``` minus opening fences

		if openCount > closeCount {
			// Missing closing fence - add it
			r.logger.Debug("Adding missing closing fence (stopped by stop sequence)")
			llmResponse += "\n```"
			if stream != nil {
				_, _ = stream.WriteString("\n```")
			}
		}
	}

	// Debug log full response to diagnose format issues
	r.logger.Debug("LLM response collected",
		zap.Int("total_chunks", chunkCount),
		zap.Int("total_length", len(llmResponse)),
		zap.String("full_response", llmResponse))

	// Add newline if response doesn't end with one
	if stream != nil && !strings.HasSuffix(llmResponse, "\n") {
		_, _ = stream.WriteString("\n")
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
