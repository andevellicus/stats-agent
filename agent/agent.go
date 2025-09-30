package agent

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"stats-agent/config"
	"stats-agent/rag"
	"stats-agent/tools"
	"stats-agent/web/types"
	"strings"
	"time"

	"io"

	"go.uber.org/zap"
)

type Agent struct {
	cfg        *config.Config
	pythonTool *tools.StatefulPythonTool
	rag        *rag.RAG
	logger     *zap.Logger
}

type TokenizeRequest struct {
	Content string `json:"content"`
}

type TokenizeResponse struct {
	Tokens []int `json:"tokens"`
}

func NewAgent(cfg *config.Config, pythonTool *tools.StatefulPythonTool, rag *rag.RAG, logger *zap.Logger) *Agent {
	logger.Info("Agent initialized", zap.Int("context_window_size", cfg.ContextLength))
	return &Agent{
		cfg:        cfg,
		pythonTool: pythonTool,
		rag:        rag,
		logger:     logger,
	}
}

func (a *Agent) InitializeSession(ctx context.Context, sessionID string, uploadedFiles []string) (string, error) {
	return a.pythonTool.InitializeSession(ctx, sessionID, uploadedFiles)
}

func (a *Agent) CleanupSession(sessionID string) {
	a.pythonTool.CleanupSession(sessionID)
}

// Run now accepts the current session's message history, making it stateless.
func (a *Agent) Run(ctx context.Context, input string, sessionID string, history []types.AgentMessage) {
	currentHistory := history
	currentHistory = append(currentHistory, types.AgentMessage{Role: "user", Content: input})

	longTermContext, err := a.rag.Query(ctx, input, a.cfg.RAGResults)
	if err != nil {
		a.logger.Error("Error querying RAG for long-term context", zap.Error(err))
	}

	if longTermContext != "" {
		contextTokens, err := a.countTokens(ctx, longTermContext)
		if err == nil && contextTokens > int(float64(a.cfg.ContextLength)*0.75) {
			a.logger.Info("Proactive check: RAG context is too large, summarizing", zap.Int("context_tokens", contextTokens))
			fmt.Printf("<agent_status>Compressing memory....</agent_status>")
			summarizedContext, summaryErr := a.rag.SummarizeLongTermMemory(ctx, longTermContext)
			if summaryErr == nil {
				longTermContext = summarizedContext
			}
		}
	}

	consecutiveErrors := 0

	for turn := 0; turn < a.cfg.MaxTurns; turn++ {
		a.manageMemory(ctx, &currentHistory)

		if consecutiveErrors >= a.cfg.ConsecutiveErrors {
			a.logger.Warn("Agent produced consecutive errors, breaking loop to request user feedback", zap.Int("consecutive_errors", a.cfg.ConsecutiveErrors))
			fmt.Printf("<agent_status>Consecutive errors, user feedback needed.</agent_status>")
			break
		}

		var messagesForLLM []types.AgentMessage
		if longTermContext != "" {
			messagesForLLM = append(messagesForLLM, types.AgentMessage{Role: "system", Content: longTermContext})
		}
		messagesForLLM = append(messagesForLLM, currentHistory...)

		var llmResponseBuilder strings.Builder
		isFirstChunk := true

		responseChan, err := getLLMResponse(ctx, a.cfg.MainLLMHost, messagesForLLM, a.cfg, a.logger)
		if err != nil {
			a.logger.Error("Error getting LLM response channel", zap.Error(err))
			break
		}

		for chunk := range responseChan {
			if isFirstChunk && !strings.Contains(chunk, "<python>") {
				fmt.Print("Agent: ")
			}
			isFirstChunk = false
			fmt.Print(chunk)
			llmResponseBuilder.WriteString(chunk)
		}

		llmResponse := llmResponseBuilder.String()
		if !strings.HasSuffix(llmResponse, "\n") {
			fmt.Println()
		}

		// Convert markdown code blocks to XML tags for Python execution
		llmResponse = convertMarkdownToXMLTags(llmResponse)

		if llmResponse == "" {
			a.logger.Warn("LLM response was empty, likely due to a context window error. Attempting to summarize context")
			fmt.Printf("<agent_status>Compressing memory due to a context window error...</agent_status>")
			summarizedContext, summaryErr := a.rag.SummarizeLongTermMemory(ctx, longTermContext)
			if summaryErr != nil {
				a.logger.Error("Recovery failed: Could not summarize RAG context. Aborting turn", zap.Error(summaryErr))
				break
			}
			longTermContext = summarizedContext
			continue
		}

		_, execResult, wasCodeExecuted := a.pythonTool.ExecutePythonCode(ctx, llmResponse, sessionID)

		if wasCodeExecuted {
			currentHistory = append(currentHistory, types.AgentMessage{Role: "assistant", Content: llmResponse})
			executionMessage := fmt.Sprintf("<execution_results>\n%s\n</execution_results>", execResult)
			//fmt.Print(executionMessage)
			toolMessage := types.AgentMessage{Role: "tool", Content: executionMessage}
			currentHistory = append(currentHistory, toolMessage)

			if strings.Contains(execResult, "Error:") {
				fmt.Printf("<agent_status>Error - attempting to self-correct</agent_status>")
				consecutiveErrors++
			} else {
				consecutiveErrors = 0
			}
		} else {
			currentHistory = append(currentHistory, types.AgentMessage{Role: "assistant", Content: llmResponse})
			return
		}
	}
}

func (a *Agent) countTokens(ctx context.Context, text string) (int, error) {
	reqBody := TokenizeRequest{
		Content: text,
	}
	jsonBody, err := json.Marshal(reqBody)
	if err != nil {
		return 0, fmt.Errorf("failed to marshal tokenize request body: %w", err)
	}

	url := fmt.Sprintf("%s/tokenize", a.cfg.MainLLMHost)
	var resp *http.Response

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
		a.logger.Warn("Tokenize endpoint is loading, retrying", zap.Duration("retry_delay", 2*time.Second))
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

// manageMemory now operates on a pointer to a history slice.
func (a *Agent) manageMemory(ctx context.Context, history *[]types.AgentMessage) {
	var totalTokens int
	for _, msg := range *history {
		tokens, err := a.countTokens(ctx, msg.Content)
		if err != nil {
			a.logger.Warn("Could not count tokens for a message, falling back to character count", zap.Error(err))
			totalTokens += len(msg.Content)
		} else {
			totalTokens += tokens
		}
	}

	contextWindowThreshold := int(float64(a.cfg.ContextLength) * 0.75)

	if totalTokens > contextWindowThreshold {
		fmt.Printf("<agent_status>Archiving older messages....</agent_status>")
		cutoff := len(*history) / 2

		if cutoff > 0 && cutoff < len(*history) {
			lastMessageInBatch := (*history)[cutoff-1]
			firstMessageOutOfBatch := (*history)[cutoff]

			if lastMessageInBatch.Role == "assistant" && strings.Contains(lastMessageInBatch.Content, "<python>") && firstMessageOutOfBatch.Role == "tool" {
				cutoff++
				a.logger.Info("Adjusted memory cutoff to prevent splitting an assistant-tool pair")
			}
		}

		if cutoff == 0 {
			return
		}

		messagesToStore := (*history)[:cutoff]

		err := a.rag.AddMessagesToStore(ctx, messagesToStore)
		if err != nil {
			a.logger.Error("Error adding messages to long-term memory", zap.Error(err))
		}

		*history = (*history)[cutoff:]
		a.logger.Info("Memory threshold reached. Moved messages to long-term RAG store", zap.Int("messages_moved", len(messagesToStore)))
	}
}

// convertMarkdownToXMLTags converts markdown code blocks (```python) to XML tags (<python>)
func convertMarkdownToXMLTags(text string) string {
	// Replace ```python with <python>
	text = strings.ReplaceAll(text, "```python", "<python>")

	// Replace closing ``` with </python>
	// We need to be careful here - only replace ``` that comes after a <python> tag
	// Use a simple state machine approach
	var result strings.Builder
	inCodeBlock := false

	lines := strings.Split(text, "\n")
	for i, line := range lines {
		trimmed := strings.TrimSpace(line)

		// Check if we're entering a code block
		if strings.Contains(line, "<python>") {
			inCodeBlock = true
			result.WriteString(line)
		} else if inCodeBlock && trimmed == "```" {
			// We're in a code block and found closing ```
			result.WriteString("</python>")
			inCodeBlock = false
		} else {
			result.WriteString(line)
		}

		// Add newline except for last line
		if i < len(lines)-1 {
			result.WriteString("\n")
		}
	}

	return result.String()
}
