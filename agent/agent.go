package agent

import (
	"context"
	"fmt"
	"log"
	"strings"

	"stats-agent/rag"
	"stats-agent/tools"

	"github.com/ollama/ollama/api"
)

// Agent struct holds the state for our agent
type Agent struct {
	llmClient     *api.Client
	pythonTool    *tools.StatefulPythonTool
	rag           *rag.RAG
	history       []api.Message // This is the short-term memory
	maxTurns      int
	contextLength int
	ragResults    int
}

// NewAgent creates a new agent
func NewAgent(ctx context.Context, llmClient *api.Client, pythonTool *tools.StatefulPythonTool, rag *rag.RAG, modelName string, maxTurns int, ragResults int) *Agent {
	showReq := &api.ShowRequest{Model: modelName}
	showResp, err := llmClient.Show(ctx, showReq)
	if err != nil {
		log.Fatalf("Failed to get model info: %v", err)
	}

	var contextLength int
	if val, ok := showResp.ModelInfo["llama.context_length"]; ok {
		if floatVal, ok := val.(float64); ok {
			contextLength = int(floatVal)
		}
	}

	if contextLength == 0 {
		log.Printf("Warning: Could not determine 'llama.context_length' from model info. Using default of 4096.")
		contextLength = 4096
	}

	contextLength = 10 // hamstrung for testing

	log.Printf("Using context window size: %d", contextLength)

	return &Agent{
		llmClient:     llmClient,
		pythonTool:    pythonTool,
		rag:           rag,
		maxTurns:      maxTurns,
		contextLength: contextLength,
		ragResults:    ragResults,
	}
}

// manageMemory checks if the short-term history is approaching the context limit.
// It now includes logic to avoid splitting assistant-tool message pairs.
func (a *Agent) manageMemory(ctx context.Context) {
	var totalContentLength int
	for _, msg := range a.history {
		totalContentLength += len(msg.Content)
	}

	contextWindowThreshold := int(float64(a.contextLength) * 0.8)

	if totalContentLength > contextWindowThreshold {
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

	longTermContext, err := a.rag.Query(ctx, input, a.ragResults)
	if err != nil {
		log.Println("Error querying RAG for long-term context:", err)
	}

	messagesForLLM := []api.Message{}
	if longTermContext != "" {
		messagesForLLM = append(messagesForLLM, api.Message{Role: "system", Content: longTermContext})
	}
	messagesForLLM = append(messagesForLLM, a.history...)

	for turn := 0; turn < a.maxTurns; turn++ {
		fmt.Print("Agent: ")

		llmResponse, err := getLLMResponse(ctx, a.llmClient, messagesForLLM)
		if err != nil {
			log.Println("Error getting LLM response:", err)
			break
		}
		a.history = append(a.history, api.Message{Role: "assistant", Content: llmResponse})

		_, execResult, wasCodeExecuted := a.pythonTool.ExecutePythonCode(ctx, llmResponse)

		if !wasCodeExecuted {
			break
		}

		executionMessage := fmt.Sprintf("<execution_results>\n%s\n</execution_results>", execResult)
		toolMessage := api.Message{Role: "tool", Content: executionMessage}
		a.history = append(a.history, toolMessage)

		// Always update the context for the next turn
		messagesForLLM = a.history

		if strings.Contains(execResult, "Error:") {
			fmt.Println("\n--- Agent observed an error, attempting to self-correct ---")
			// The loop will continue, using the updated messagesForLLM
		}
	}

	// After the loop, generate a final summary response to the user.
	finalMessages := append(messagesForLLM, api.Message{Role: "user", Content: "Based on the last execution, what is the answer to my original question? Summarize your findings."})
	finalResponse, err := getLLMResponse(ctx, a.llmClient, finalMessages)
	if err != nil {
		log.Println("Error getting final LLM response:", err)
	} else {
		a.history = append(a.history, api.Message{Role: "assistant", Content: finalResponse})
	}

	fmt.Println()
}
