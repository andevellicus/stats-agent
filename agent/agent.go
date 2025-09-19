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

	for i := 0; i < 5; i++ {
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

	contextWindowThreshold := int(float64(a.cfg.ContextLength) * 0.8)

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

	messagesForLLM := []api.Message{}
	if longTermContext != "" {
		messagesForLLM = append(messagesForLLM, api.Message{Role: "system", Content: longTermContext})
	}
	messagesForLLM = append(messagesForLLM, a.history...)

	for turn := 0; turn < a.cfg.MaxTurns; turn++ {
		fmt.Print("Agent: ")
		var llmResponseBuilder strings.Builder
		err := getLLMResponse(ctx, a.cfg.MainLLMHost, messagesForLLM, func(chunk string) {
			fmt.Print(chunk)
			llmResponseBuilder.WriteString(chunk)
		})

		// Add a newline after the streaming is complete
		fmt.Println()

		if err != nil {
			log.Println("Error getting LLM response:", err)
			break
		}
		llmResponse := llmResponseBuilder.String()
		a.history = append(a.history, api.Message{Role: "assistant", Content: llmResponse})

		_, execResult, wasCodeExecuted := a.pythonTool.ExecutePythonCode(ctx, llmResponse)

		if !wasCodeExecuted {
			break // The final answer has been streamed, so we can exit.
		}

		executionMessage := fmt.Sprintf("<execution_results>\n%s\n</execution_results>", execResult)
		toolMessage := api.Message{Role: "tool", Content: executionMessage}
		a.history = append(a.history, toolMessage)

		messagesForLLM = a.history

		if strings.Contains(execResult, "Error:") {
			fmt.Println("\n--- Agent observed an error, attempting to self-correct ---")
		}
	}
}
