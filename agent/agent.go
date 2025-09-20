package agent

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"stats-agent/config"
	"stats-agent/rag"
	"stats-agent/tools"
	"strings"
	"time"

	"io"

	"github.com/ollama/ollama/api"
)

type Agent struct {
	cfg        *config.Config
	pythonTool *tools.StatefulPythonTool
	rag        *rag.RAG
	history    []api.Message
}

type TokenizeRequest struct {
	Content string `json:"content"`
}

type TokenizeResponse struct {
	Tokens []int `json:"tokens"`
}

func NewAgent(cfg *config.Config, pythonTool *tools.StatefulPythonTool, rag *rag.RAG) *Agent {
	log.Printf("Using context window size: %d", cfg.ContextLength)
	return &Agent{
		cfg:        cfg,
		pythonTool: pythonTool,
		rag:        rag,
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
			break // Success or non-retryable error
		}

		resp.Body.Close()
		log.Printf("Tokenize endpoint is loading, retrying in %v...", 2*time.Second)
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
func (a *Agent) manageMemory(ctx context.Context) {
	var totalTokens int
	for _, msg := range a.history {
		tokens, err := a.countTokens(ctx, msg.Content)
		if err != nil {
			log.Printf("Warning: could not count tokens for a message, falling back to character count. Error: %v", err)
			totalTokens += len(msg.Content) // Fallback
		} else {
			totalTokens += tokens
		}
	}

	contextWindowThreshold := int(float64(a.cfg.ContextLength) * 0.5)

	if totalTokens > contextWindowThreshold {
		cutoff := len(a.history) / 2

		// Intelligent Cutoff Logic to prevent splitting an assistant-tool pair.
		if cutoff > 0 && cutoff < len(a.history) {
			lastMessageInBatch := a.history[cutoff-1]
			firstMessageOutOfBatch := a.history[cutoff]

			if lastMessageInBatch.Role == "assistant" && strings.Contains(lastMessageInBatch.Content, "<python>") && firstMessageOutOfBatch.Role == "tool" {
				// If we're splitting a pair, move the tool message into the batch as well.
				cutoff++
				log.Println("--- Adjusted memory cutoff to prevent splitting an assistant-tool pair. ---")
			}
		}

		if cutoff == 0 {
			return
		}

		messagesToStore := a.history[:cutoff]

		err := a.rag.AddMessagesToStore(ctx, messagesToStore)
		if err != nil {
			log.Printf("Error adding messages to long-term memory: %v", err)
		}

		// Trim the in-memory history.
		a.history = a.history[cutoff:]
		log.Printf("--- Memory threshold reached. Moved %d oldest messages to long-term RAG store. ---", len(messagesToStore))
	}
}

// Run starts the agent's interaction loop for a given user input
func (a *Agent) Run(ctx context.Context, input string) {
	a.manageMemory(ctx)
	a.history = append(a.history, api.Message{Role: "user", Content: input})

	longTermContext, err := a.rag.Query(ctx, input, a.cfg.RAGResults)
	if err != nil {
		log.Println("Error querying RAG for long-term context:", err)
	}

	if longTermContext != "" {
		contextTokens, err := a.countTokens(ctx, longTermContext)
		if err == nil && contextTokens > int(float64(a.cfg.ContextLength)*0.75) {
			log.Printf("--- Proactive check: RAG context is too large (%d tokens). Summarizing... ---", contextTokens)
			summarizedContext, summaryErr := a.rag.SummarizeLongTermMemory(ctx, longTermContext)
			if summaryErr == nil {
				longTermContext = summarizedContext
			}
		}
	}

	for turn := 0; turn < a.cfg.MaxTurns; turn++ {
		messagesForLLM := []api.Message{}
		if longTermContext != "" {
			messagesForLLM = append(messagesForLLM, api.Message{Role: "system", Content: longTermContext})
		}
		messagesForLLM = append(messagesForLLM, a.history...)

		var llmResponseBuilder strings.Builder
		isFirstChunk := true

		// **NEW**: Call the concurrent function and read from the channel
		responseChan, err := getLLMResponse(ctx, a.cfg.MainLLMHost, messagesForLLM, a.cfg)
		if err != nil {
			log.Println("Error getting LLM response channel:", err)
			break
		}

		// Read the streaming response from the channel
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

		if llmResponse == "" {
			log.Println("LLM response was empty, likely due to a context window error. Attempting to summarize context.")
			// This is our reactive fallback
			summarizedContext, summaryErr := a.rag.SummarizeLongTermMemory(ctx, longTermContext)
			if summaryErr != nil {
				log.Println("--- Recovery failed: Could not summarize RAG context. Aborting turn. ---")
				break
			}
			longTermContext = summarizedContext
			continue // Retry the turn
		}

		a.history = append(a.history, api.Message{Role: "assistant", Content: llmResponse})

		_, execResult, wasCodeExecuted := a.pythonTool.ExecutePythonCode(ctx, llmResponse)

		if !wasCodeExecuted {
			return
		}

		executionMessage := fmt.Sprintf("<execution_results>\n%s\n</execution_results>", execResult)
		toolMessage := api.Message{Role: "tool", Content: executionMessage}
		a.history = append(a.history, toolMessage)

		if strings.Contains(execResult, "Error:") {
			fmt.Println("\n--- Agent observed an error, attempting to self-correct ---")
		} else {
			break
		}
	}
}
