package main

import (
	"bufio"
	"context"
	"fmt"
	"log"
	"os"
	"strings"

	"github.com/ollama/ollama/api"
)

const maxTurns = 5 // Maximum number of self-correction attempts

// getLLMResponse sends the conversation history to the Ollama API and gets a streaming response
func getLLMResponse(ctx context.Context, client *api.Client, messages []api.Message) (string, error) {
	systemMessage := api.Message{
		Role: "system",
		Content: `
You are an expert AI data analyst. Your primary goal is to help users with statistical analysis by writing and executing Python code.

Core Directives
Think Step-by-Step: Break down the user's request into a sequence of logical steps.
Write Code: Use the Python tool for all data exploration, analysis, and visualization.
Use Correct Syntax: You MUST enclose all Python code in <python> and </python> tags. This is the only way the system will execute it. Do not use markdown backticks.
Self-Correct: If your code produces an error, analyze the error message and rewrite the code to fix the problem. You have 5 attempts per task.

Agentic Workflow

Clarify the Goal: First, make sure you understand the user's request.

Explore the Data: Before performing any analysis on a new dataset, you MUST explore it to understand its contents and structure.
To see available files, run:

<python>
import os
print(os.listdir('/app/workspace'))
</python>
To inspect a CSV file, run:
<python>
import pandas as pd
df = pd.read_csv('/app/workspace/data.csv')
print("First 5 rows:")
print(df.head())
print("\nColumn names:")
print(df.columns)
</python>
3. Execute Code Iteratively: Write and execute one logical block of Python code at a time. Analyze the output before deciding on the next step.
4. Summarize Findings: Once the task is complete, provide a concise summary of the results to the user in plain text. Do not output any more code in your final summary.

Execution Environment
Working Directory: /app/workspace
Python Libraries: The environment includes pandas, numpy, matplotlib, scikit-learn, and seaborn.
State Persistence: The Python session is stateful. Variables, functions, and imports are preserved between code executions.

Final Output Guidelines
Text Summary: For the final answer, provide a clear, conversational summary of your findings.
Plots & Visualizations: If you generate a plot, save it as a file (e.g., plot.png) in the /app/workspace directory and inform the user of the filename.
`,
	}

	chatMessages := append([]api.Message{systemMessage}, messages...)

	req := &api.ChatRequest{
		//Model: "qwen3:32b",
		//Model:    "phi4-reasoning:14b",
		Model:    "devstral:24b",
		Messages: chatMessages,
		Stream:   &[]bool{true}[0],
	}

	var fullResponse strings.Builder
	err := client.Chat(ctx, req, func(resp api.ChatResponse) error {
		if resp.Message.Content != "" {
			fmt.Print(resp.Message.Content)
			fullResponse.WriteString(resp.Message.Content)
		}
		return nil
	})

	return fullResponse.String(), err
}

// executePythonCode extracts code from a string, executes it, and returns the result
func executePythonCode(ctx context.Context, pythonTool *StatefulPythonTool, text string) (string, string, bool) {
	startTag := "<python>"
	endTag := "</python>"

	startIdx := strings.Index(text, startTag)
	if startIdx == -1 {
		return "", "", false
	}

	endIdx := strings.Index(text[startIdx:], endTag)
	if endIdx == -1 {
		return "", "", false
	}

	codeStart := startIdx + len(startTag)
	codeEnd := startIdx + endIdx
	pythonCode := strings.TrimSpace(text[codeStart:codeEnd])

	if pythonCode == "" {
		return "", "", false
	}

	fmt.Println("\n--- Executing Python Code ---")
	fmt.Printf("Code to execute:\n%s\n", pythonCode)
	fmt.Println("--- Execution Output ---")

	execResult, err := pythonTool.Call(ctx, pythonCode)
	if err != nil {
		fmt.Printf("Error executing Python: %v\n", err)
		execResult = "Error: " + err.Error()
	} else {
		fmt.Print(execResult)
	}
	fmt.Println("\n--- End Execution ---")

	// Return the code, the result, and a flag indicating code was executed
	return pythonCode, execResult, true
}

func main() {
	ctx := context.Background()

	pythonTool, err := NewStatefulPythonTool(ctx, "localhost:9999")
	if err != nil {
		log.Fatalf("Failed to connect to Python container. Is it running? Error: %v", err)
	}
	defer pythonTool.Close()

	ollamaClient, err := api.ClientFromEnvironment()
	if err != nil {
		log.Fatalf("Failed to create Ollama client: %v", err)
	}

	var conversationHistory []api.Message

	scanner := bufio.NewScanner(os.Stdin)
	for {
		fmt.Print("You: ")
		if !scanner.Scan() {
			break
		}
		input := scanner.Text()
		if input == "exit" {
			break
		}

		conversationHistory = append(conversationHistory, api.Message{Role: "user", Content: input})

		// --- Autonomous Agent Loop ---
		for turn := 0; turn < maxTurns; turn++ {
			fmt.Print("Agent: ")
			llmResponse, err := getLLMResponse(ctx, ollamaClient, conversationHistory)
			if err != nil {
				log.Println("Error getting LLM response:", err)
				break
			}
			conversationHistory = append(conversationHistory, api.Message{Role: "assistant", Content: llmResponse})

			_, execResult, wasCodeExecuted := executePythonCode(ctx, pythonTool, llmResponse)

			if !wasCodeExecuted {
				// If no code was run, the agent is either finished or chatting.
				// Break the autonomous loop and wait for the next user input.
				break
			}

			// Add the execution result to the history for the next turn
			executionMessage := fmt.Sprintf("<execution_results>\n%s\n</execution_results>", execResult)
			conversationHistory = append(conversationHistory, api.Message{Role: "tool", Content: executionMessage})

			if strings.Contains(execResult, "Error:") {
				fmt.Println("\n--- Agent observed an error, attempting to self-correct ---")
				// The loop will now continue, feeding the error back to the LLM
			} else {
				// If the code ran successfully, let the agent decide if it's done.
				// We continue the loop to let it generate a final summary or next step.
			}
		}
		// --- End of Autonomous Loop ---
		fmt.Println()
	}

	if err := scanner.Err(); err != nil {
		log.Fatal(err)
	}
}
