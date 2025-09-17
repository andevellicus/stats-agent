package agent

import (
	"context"
	"fmt"
	"log"
	"strings"

	"github.com/ollama/ollama/api"
)

// Agent struct holds the state for our agent
type Agent struct {
	llmClient  *api.Client
	pythonTool *StatefulPythonTool
	history    []api.Message
	maxTurns   int
}

// NewAgent creates a new agent
func NewAgent(llmClient *api.Client, pythonTool *StatefulPythonTool, maxTurns int) *Agent {
	return &Agent{
		llmClient:  llmClient,
		pythonTool: pythonTool,
		maxTurns:   maxTurns,
	}
}

// Run starts the agent's interaction loop for a given user input
func (a *Agent) Run(ctx context.Context, input string) {
	a.history = append(a.history, api.Message{Role: "user", Content: input})

	for range a.maxTurns {
		fmt.Print("Agent: ")
		llmResponse, err := getLLMResponse(ctx, a.llmClient, a.history)
		if err != nil {
			log.Println("Error getting LLM response:", err)
			break
		}
		a.history = append(a.history, api.Message{Role: "assistant", Content: llmResponse})

		_, execResult, wasCodeExecuted := executePythonCode(ctx, a.pythonTool, llmResponse)

		if !wasCodeExecuted {
			break
		}

		executionMessage := fmt.Sprintf("<execution_results>\n%s\n</execution_results>", execResult)
		a.history = append(a.history, api.Message{Role: "tool", Content: executionMessage})

		if strings.Contains(execResult, "Error:") {
			fmt.Println("\n--- Agent observed an error, attempting to self-correct ---")
		}
	}
	fmt.Println()
}
