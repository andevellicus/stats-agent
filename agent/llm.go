package agent

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"stats-agent/config"
	"strings"
	"time"

	"github.com/ollama/ollama/api"
	"go.uber.org/zap"
)

// Define a specific error type for context window issues
var ErrContextWindowExceeded = errors.New("context window exceeded")

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

// getLLMResponse now includes a client-side timeout to prevent freezing on empty streams.
func getLLMResponse(ctx context.Context, llamaCppHost string, messages []api.Message, cfg *config.Config, logger *zap.Logger) (<-chan string, error) {
	systemMessage := api.Message{
		Role: "system",
		Content: `
You are an expert statistical data analyst. Your primary goal is to conduct analysis through a persistent Python session.

### PRIMARY DIRECTIVES & RULES
1.  **Code Execution:** ALL executable Python code MUST be enclosed in <python> tags.
2.  **Final Summary:** Your final summary at the end of the analysis MUST be plain text and **NOT** inside a <python> block.
3.  **Image Display:** In your final summary, you MUST use an <image> tag for each plot you generated. For example: <image>/app/workspace/plot.png</image>. **DO NOT** emit <image> tags at any other time.
4.  **Statistical Rigor:**
    * Always report sample sizes (N=...), percentages with counts (e.g., 45.2%, n=134/296), test statistics with exact p-values (e.g., t=2.34, p=0.021), and effect sizes with confidence intervals.
	* Use df.head(3) for previews
	* Round floats to 3 decimal places
    * State and verify assumptions (e.g., normality) BEFORE choosing a statistical test.
	* Justify test selection based on data characteristics and assumptions.

### SESSION AND ENVIRONMENT
-   **Working Directory:** /app/workspace/
-   **Persistence:** Variables, functions, and dataframes persist between turns. Do not reload data or re-import libraries unless necessary.
-   **Output:** You will receive output from the Python environment in <execution_results> blocks. Do not write these blocks yourself.

### Long-Term Memory & Context
When you see a <memory> block at conversation start, it contains previous analyses and findings. Reload data files if you need to continue the analysis.

### EXAMPLE WORKFLOW
1.  **Explore Data:**
    <python>
    import pandas as pd
    import os
    df = pd.read_csv('/app/workspace/data.csv')
    print(f"Shape: {df.shape}")
    print(f"Missing Values:\n{df.isnull().sum()}")
    </python>
2.  **Analyze and Visualize:**
    "I will now test for a difference in age and visualize the distribution."
    <python>
    import matplotlib.pyplot as plt
    # ... analysis and plotting code ...
    plt.title(f'Age Distribution (N={len(df)})')
    plt.savefig('/app/workspace/age_distribution.png', dpi=100)
    plt.close()
    print("Saved: age_distribution.png")
    </python>
3.  **Final Summary:**
    "**Key Findings:**
    - There was no significant difference in age (p=0.327).
    **Saved Files:**
    - age_distribution.png"
    <image>/app/workspace/age_distribution.png</image>
`,
	}
	chatMessages := append([]api.Message{systemMessage}, messages...)
	reqBody := LlamaCppChatRequest{
		Messages: chatMessages,
		Stream:   true,
	}
	jsonBody, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request body: %w", err)
	}

	url := fmt.Sprintf("%s/v1/chat/completions", llamaCppHost)

	// Create the channel that will stream the response
	responseChan := make(chan string)

	go func() {
		defer close(responseChan)

		// This entire block now runs in the background
		var resp *http.Response
		var err error

		for i := 0; i < cfg.MaxRetries; i++ {
			req, reqErr := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(jsonBody))
			if reqErr != nil {
				logger.Error("Error creating request", zap.Error(reqErr))
				return
			}
			req.Header.Set("Content-Type", "application/json")
			req.Header.Set("Accept", "text/event-stream")
			req.Header.Set("Cache-Control", "no-cache")
			req.Header.Set("Connection", "keep-alive")

			client := &http.Client{
				Timeout: 30 * time.Second,
			}

			resp, err = client.Do(req)
			if err != nil {
				if netErr, ok := err.(net.Error); ok && netErr.Timeout() {
					logger.Warn("Request timed out")
					return
				}
				logger.Error("Error sending request", zap.Error(err))
				return
			}

			if resp.StatusCode != http.StatusServiceUnavailable {
				break
			}
			resp.Body.Close()
			time.Sleep(cfg.RetryDelaySeconds)
		}
		defer resp.Body.Close()

		if resp.StatusCode != http.StatusOK {
			bodyBytes, _ := io.ReadAll(resp.Body)
			if strings.Contains(string(bodyBytes), "exceeds the available context size") {
				logger.Error("Context window exceeded")
			} else {
				logger.Error("LLM server returned non-200 status", zap.String("status", resp.Status))
			}
			return
		}

		scanner := bufio.NewScanner(resp.Body)
		for scanner.Scan() {
			line := scanner.Text()
			if strings.HasPrefix(line, "data: ") {
				data := strings.TrimPrefix(line, "data: ")
				if data == "[DONE]" {
					break
				}

				var streamResp LlamaCppStreamResponse
				if err := json.Unmarshal([]byte(data), &streamResp); err == nil {
					if len(streamResp.Choices) > 0 {
						responseChan <- streamResp.Choices[0].Delta.Content
					}
				}
			}
		}

		if err := scanner.Err(); err != nil {
			logger.Error("Error reading stream", zap.Error(err))
		}
	}()

	return responseChan, nil
}
