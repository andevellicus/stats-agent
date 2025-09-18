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
	contextLength = 10 // Temporary override for testing

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
// If it is, it moves the oldest half of the history to the long-term RAG store.
func (a *Agent) manageMemory(ctx context.Context) {
	var totalContentLength int
	for _, msg := range a.history {
		totalContentLength += len(msg.Content)
	}

	contextWindowThreshold := int(float64(a.contextLength) * 0.8)

	if totalContentLength > contextWindowThreshold {
		// If we're over the limit, move the oldest half of the messages to long-term storage.
		cutoff := len(a.history) / 2
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
	// Manage memory at the start of each turn.
	a.manageMemory(ctx)

	// Add the new user input to the short-term history.
	a.history = append(a.history, api.Message{Role: "user", Content: input})

	// Query RAG for long-term memories relevant to the latest input.
	longTermContext, err := a.rag.Query(ctx, input, a.ragResults)
	if err != nil {
		log.Println("Error querying RAG for long-term context:", err)
	}

	// The messages for the LLM will be the RAG context plus the entire short-term memory.
	messagesForLLM := []api.Message{}
	if longTermContext != "" {
		messagesForLLM = append(messagesForLLM, api.Message{Role: "system", Content: longTermContext})
	}
	messagesForLLM = append(messagesForLLM, a.history...)

	for turn := 0; turn < a.maxTurns; turn++ {
		fmt.Print("Agent: ")

		currentMessages := messagesForLLM
		if turn > 0 {
			// On correction loops, just use the short-term history.
			currentMessages = a.history
		}

		llmResponse, err := getLLMResponse(ctx, a.llmClient, currentMessages)
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
		a.history = append(a.history, api.Message{Role: "tool", Content: executionMessage})

		if strings.Contains(execResult, "Error:") {
			fmt.Println("\n--- Agent observed an error, attempting to self-correct ---")
			// Update the prompt for the correction loop to include the error context.
			messagesForLLM = a.history
		}
	}
	fmt.Println()
}
