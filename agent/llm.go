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

func buildSystemPrompt(uploadedFiles []string) string {
	var builder strings.Builder
	primaryFile := "uploaded_file.csv" // Default fallback

	builder.WriteString("You are an expert statistical data analyst with Python.\n")

	// Only add the file context block if files were actually uploaded.
	if len(uploadedFiles) > 0 {
		primaryFile = uploadedFiles[0]
		fileList := strings.Join(uploadedFiles, "\n  - ")
		fmt.Fprintf(&builder, `

## CORE METHODOLOGY: OBSERVE, PLAN, ACT
You must follow this loop for every turn:
1.  **OBSERVE**: Look at the last <execution_results></execution_results>. Is there an error? Does the output match what you expected?
2.  **PLAN**: Based on your observation and the user's overall goal, briefly state your plan for the single next step.
3.  **ACT**: Write one small, focused <python></python> block to execute that step. You MUST write the code in <python></python> tags.

You must continue this loop until you have gathered enough information to fully answer the user's request.

## STRICT RULES - MUST FOLLOW
- **Goal-Oriented**: Your primary objective is to answer the user's request. All actions must work towards this goal.
- **One Step at a Time**: Each <python></python> block must perform only ONE logical action.
- **Error Handling**: If you encounter an error, your immediate next step MUST be to debug and fix the issue.
- **Visualizations**: NEVER use plt.show(). ALWAYS save plots to a file with plt.savefig('descriptive_filename.png') and then use plt.close().
- **Final Summary**: Once you have the answer, stop writing code and provide a complete summary in plain text. You MUST display any plots you generated in this summary using the <image>filename.png</image> tag.

## STATISTICAL RIGOR
- Always report sample sizes (N=...), percentages with counts (e.g., 45.2%, n=134/296), test statistics with exact p-values (e.g., t=2.34, p=0.021), and effect sizes with confidence intervals.
- Use df.head(3) for previews
- Round floats to 3 decimal places
- State and verify assumptions (e.g., normality) BEFORE choosing a statistical test.
- Justify test selection based on data characteristics and assumptions.

📁 **FILES UPLOADED BY USER:**
  - %s
  - You MUST use these exact filenames. The primary file is: %s
  - Do NOT use example filenames like 'data.csv' or 'filename.csv'.
`, fileList, primaryFile)
	}

	// Use fmt.Fprintf to safely inject the primaryFile into the rest of the prompt.
	fmt.Fprintf(&builder, `
## VISUALIZATION & SUMMARY EXAMPLE
This is the ONLY way to create and show a plot.

✅ **CORRECT plot block:**
<python>
plt.figure(figsize=(10, 6))
sns.histplot(df['age'], bins=20)
plt.title('Distribution of Age (N=...)')
plt.savefig('age_distribution.png')
plt.close() # IMPORTANT: Close the plot to free memory
print("Saved plot: age_distribution.png")
</python>

✅ **CORRECT final summary:**
## Analysis Complete
Here are the key findings from the analysis.

**Files Generated:**
<image>age_distribution.png</image>

## REQUIRED WORKFLOW PATTERN

Each step in a SEPARATE code block:

Step 1: Import libraries (around 5 lines)
Step 2: List available files (around 3 lines)  
Step 3: Load ONLY the uploaded file: '%s' (around 3 lines)
Step 4: Check shape and columns (around 4 line)
Step 5: Inspect first few rows (around 3 lines)
Step 6: Check for missing data (around 5 lines)
Step 7: Perform analysis (around 10-15 lines per concept)
Step 8: Create visualizations (around 10-15 lines per plot)

## CODE BLOCK ENFORCEMENT

❌ WRONG - Too many operations:
<python>
import pandas as pd
df = pd.read_csv('file.csv')
display(df.head())
display(df.describe())
# ... more code
plt.show()
</python>

✅ CORRECT - Separated operations:
<python>
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
</python>

<python>
# Load the specific file
df = pd.read_csv('%s')
print(f"Loaded: {df.shape}")
</python>

<python>
# Now inspect structure
print(df.head(3))
</python>

## OUTPUT FORMAT

- Code in <python></python> tags (MAX 15 lines each)
- Explain before each code block
- Final summary as plain text
- Images as <image>filename.png</image> in the final summary ONLY.

Remember: SMALL blocks, CHECK output, ITERATE carefully.`, primaryFile, primaryFile)

	return builder.String()
}

func getLLMResponse(ctx context.Context, llamaCppHost string, messages []api.Message, cfg *config.Config, logger *zap.Logger, uploadedFiles []string) (<-chan string, error) {
	systemMessage := api.Message{
		Role:    "system",
		Content: buildSystemPrompt(uploadedFiles),
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
	responseChan := make(chan string)

	go func() {
		defer close(responseChan)

		var resp *http.Response
		var err error

		// This retry loop is for handling "model is loading" scenarios.
		for i := 0; i < cfg.MaxRetries; i++ {
			req, reqErr := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(jsonBody))
			if reqErr != nil {
				logger.Error("Error creating request", zap.Error(reqErr))
				// Propagate critical errors through the channel.
				responseChan <- fmt.Sprintf("ERROR: Could not create request: %v", reqErr)
				return
			}
			req.Header.Set("Content-Type", "application/json")
			req.Header.Set("Accept", "text/event-stream")

			client := &http.Client{} // No need for long timeout in streaming
			resp, err = client.Do(req)
			if err != nil {
				if netErr, ok := err.(net.Error); ok && netErr.Timeout() {
					logger.Warn("Request timed out")
				} else {
					logger.Error("Error sending request", zap.Error(err))
				}
				// Propagate network errors through the channel.
				responseChan <- fmt.Sprintf("ERROR: Network request failed: %v", err)
				return
			}

			if resp.StatusCode == http.StatusOK {
				break // Success, exit retry loop
			}

			// If model is loading, wait and retry.
			if resp.StatusCode == http.StatusServiceUnavailable {
				resp.Body.Close()
				logger.Warn("LLM service unavailable, retrying...", zap.Int("attempt", i+1))
				time.Sleep(cfg.RetryDelaySeconds)
				continue
			}

			// For any other error, break the loop and handle it below.
			break
		}

		if resp == nil {
			logger.Error("No response received after retries")
			responseChan <- "ERROR: No response from LLM server after retries."
			return
		}
		defer resp.Body.Close()

		if resp.StatusCode != http.StatusOK {
			bodyBytes, _ := io.ReadAll(resp.Body)
			bodyString := string(bodyBytes)
			// Check for the context error and propagate it via the channel.
			if strings.Contains(bodyString, "exceeds the available context size") {
				logger.Error("Context window exceeded", zap.String("response", bodyString))
				responseChan <- fmt.Sprintf("ERROR: %s", ErrContextWindowExceeded.Error())
			} else {
				logger.Error("LLM server returned non-200 status", zap.String("status", resp.Status), zap.String("response", bodyString))
				responseChan <- fmt.Sprintf("ERROR: LLM server returned status %s", resp.Status)
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
