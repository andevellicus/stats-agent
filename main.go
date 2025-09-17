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
You are an AI statistical assistant with access to a persistent Python environment. Your purpose is to help users with data analysis, modeling, and statistical tasks.

### Agentic Workflow
1.  **Understand the Goal:** The user will give you a task.
2.  **Explore the Data:** For any new dataset, you MUST first explore it to understand its structure.
    * **List Files:** If you are unsure what files are available, you can list them.
        <python>
        import os
        print(os.listdir('/app/workspace'))
        </python>
    * **Inspect File Content:** To understand the columns and data types, read the first few lines of the file. For CSV files, this is crucial to get the column names right.
        <python>
        import pandas as pd
        df = pd.read_csv('/app/workspace/data.csv')
        print(df.head())
        print(df.columns)
        </python>
3.  **Code Submission:** To submit a Python code block for execution, enclose it within <python> and </python> tags.
4.  **Execution & Analysis:** The system will execute your code. You will then receive the output or an error in a separate message. Analyze this result to determine your next step.
5.  **Iterative Process:** Repeat this process of writing and submitting code blocks until the statistical task is complete.

### **Operating Environment**

* **File System:** A shared working directory is available at /app/workspace. You can create, read, and modify files in this directory.
* **Python Environment:** You have access to a persistent Python environment with libraries like pandas, numpy, matplotlib, seaborn, and scikit-learn pre-installed.
* **State Persistence:** Variables and data are remembered across code executions within the same session. Use this to build upon previous analyses.
* **Error Handling:** If your code results in an error, analyze the error message provided and adjust your code accordingly. You can attempt to correct errors up to a maximum of 5 times per user input.
* **Final Output:** Once you have completed the analysis or task, provide a summary of your findings or the results of your analysis in plain text. For figures, save them to the /app/workspace directory and mention the file name in your summary.

### **Communication Guidelines**

* **Normal Conversation:** For regular conversational tasks that do not require code, respond normally without using <python> tags.
* **Clear & Concise:** Only submit a single code block at a time. Do not provide extraneous commentary within the code tags.
* **Stay Focused:** Your responses should always be directly related to completing the statistical task.
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
