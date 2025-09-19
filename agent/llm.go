package agent

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"log"
	"net/http"
	"strings"
	"time"

	"github.com/ollama/ollama/api"
)

// ... (constants and structs remain the same) ...
const (
	maxRetries = 5
	retryDelay = 2 * time.Second
)

type LlamaCppStreamChoice struct {
	Delta struct {
		Content string `json:"content"`
	} `json:"delta"`
	Index int `json:"index"`
}
type LlamaCppStreamResponse struct {
	Choices []LlamaCppStreamChoice `json:"choices"`
}
type LlamaCppChatRequest struct {
	Messages []api.Message `json:"messages"`
	Stream   bool          `json:"stream"`
}

func getLLMResponse(ctx context.Context, llamaCppHost string, messages []api.Message, streamHandler func(string)) error {
	systemMessage := api.Message{
		Role: "system",
		Content: `
You are an expert AI data analyst. Your primary goal is to help users with statistical analysis by writing and executing Python code.

### Long-Term Memory & Context
- At the start of the conversation, you may be provided with a <memory>...</memory> block.
- This block contains your memory of past actions, code executions, and findings.
- You can use this information to answer questions directly when appropriate.

### Core Directives
Think Step-by-Step: Break down the user's request into a sequence of logical steps.
Write Code: Use the Python tool for all data exploration, analysis, and visualization.
Use Correct Syntax: You MUST enclose all Python code in <python> and </python> tags. This is the only way the system will execute it. Do not use markdown backticks.
Self-Correct: If your code produces an error, analyze the error message and rewrite the code to fix the problem. You have 5 attempts per task.

### Agentic Workflow
1. Clarify the Goal: First, make sure you understand the user's request.
2. Explore the Data: Before performing any analysis on a new dataset, you MUST explore it to understand its contents and structure.
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
4. Summarize Findings: Once the entire multi-step task is complete, provide a concise summary of the results to the user in plain text. Do not output any more code in your final summary.

### Execution Environment
Working Directory: /app/workspace
Python Libraries: The environment includes pandas, numpy, matplotlib, scikit-learn, and seaborn.
State Persistence: The Python session is stateful. Variables, functions, and imports are preserved between code executions.

### Final Output Guidelines
Text Summary: For the final answer, provide a clear, conversational summary of your findings.
Be Concise: Do not summarize your findings after every single step. Wait until the full request is complete before providing a comprehensive summary.
Plots & Visualizations: If you generate a plot, save it as a file (e.g., plot.png) in the /app/workspace directory and inform the user of the filename. DO NOT use plot.show()
`,
	}

	chatMessages := append([]api.Message{systemMessage}, messages...)

	// ... (rest of the function remains the same) ...
	reqBody := LlamaCppChatRequest{
		Messages: chatMessages,
		Stream:   true, // Enable streaming
	}

	jsonBody, err := json.Marshal(reqBody)
	if err != nil {
		return fmt.Errorf("failed to marshal request body: %w", err)
	}

	url := fmt.Sprintf("%s/v1/chat/completions", llamaCppHost)
	var resp *http.Response

	for i := 0; i < maxRetries; i++ {
		req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(jsonBody))
		if err != nil {
			return fmt.Errorf("failed to create request: %w", err)
		}
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("Accept", "text/event-stream")
		req.Header.Set("Cache-Control", "no-cache")
		req.Header.Set("Connection", "keep-alive")

		client := &http.Client{}
		resp, err = client.Do(req)
		if err != nil {
			return fmt.Errorf("failed to send request: %w", err)
		}

		if resp.StatusCode != http.StatusServiceUnavailable {
			break
		}

		resp.Body.Close()
		log.Printf("Model is loading, retrying in %v...", retryDelay)
		time.Sleep(retryDelay)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		bodyBytes, _ := ioutil.ReadAll(resp.Body)
		return fmt.Errorf("llama.cpp server returned non-200 status: %s, body: %s", resp.Status, string(bodyBytes))
	}

	// Read the streaming response
	scanner := bufio.NewScanner(resp.Body)
	for scanner.Scan() {
		line := scanner.Text()
		if strings.HasPrefix(line, "data: ") {
			data := strings.TrimPrefix(line, "data: ")
			if data == "[DONE]" {
				break
			}

			var streamResp LlamaCppStreamResponse
			if err := json.Unmarshal([]byte(data), &streamResp); err != nil {
				log.Printf("Error unmarshalling stream data: %v", err)
				continue
			}

			if len(streamResp.Choices) > 0 {
				streamHandler(streamResp.Choices[0].Delta.Content)
			}
		}
	}

	if err := scanner.Err(); err != nil {
		return fmt.Errorf("error reading stream: %w", err)
	}

	return nil
}
