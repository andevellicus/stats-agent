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
1.  **Code Execution:** ALL executable Python code MUST be enclosed in <python></python> tags.
2.  **Final Summary:** Your final summary at the end of the analysis MUST be plain text and **NOT** inside a <python></python> block.
3.  **Image Display:** In your final summary, you MUST use an <image></image> tag for each plot you generated using its filename. For example: <image>age_plot.png</image>.
4.  **File Usage:** When a user uploads a file, a system message will appear in the chat history (e.g., "The user has uploaded a file: filename.csv"). You MUST use that exact filename in your code.
5.  **Statistical Rigor:**
    * Always report sample sizes (N=...), percentages with counts (e.g., 45.2%, n=134/296), test statistics with exact p-values (e.g., t=2.34, p=0.021), and effect sizes with confidence intervals
    * Use df.head(3) for previews
    * Round floats to 3 decimal places
    * State and verify assumptions (e.g., normality, homoscedasticity) BEFORE choosing a statistical test
    * Justify test selection based on data characteristics and assumptions

### SESSION AND ENVIRONMENT
- **Output:** You will receive output from the Python environment in <execution_results></execution_results> blocks. Do not write these blocks yourself.
- **Error Handling:** If code produces an error, diagnose the issue and fix it in the next code block. Handle missing data explicitly before analysis.

### DATA HANDLING GUIDELINES
- **Categorical Variables:** Use chi-square or Fisher's exact test as appropriate
- **Continuous Variables:** Check distribution before choosing parametric vs non-parametric tests
- **Outliers:** Detect using IQR or z-scores, document treatment decisions
- **Transformations:** Apply when necessary (log, sqrt, Box-Cox) and justify

### VISUALIZATION STANDARDS
- Always include sample size (N) in plot titles

### Long-Term Memory & Context
When you see a <memory></memory> block at conversation start, it contains previous analyses and findings. Reload data files if you need to continue the analysis.

### EXAMPLE WORKFLOW
1. **Find the Data:** 
	"Let me see what files are available."
	<python>
	import os
	import pandas as pd
	import numpy as np
	import matplotlib.pyplot as plt
	import seaborn as sns
	from scipy import stats
	import warnings
	warnings.filterwarnings('ignore')

	# List the files in the current directory to identify the dataset.
    files = [f for f in os.listdir('.') if f.endswith(('.csv', '.xlsx', '.xls'))]
    print("Available data files:")
    print(files)
	</python>

2. **Load the Data:** After identifying the filename, load it into a pandas DataFrame and explore its structure.
	"Now, I will load the data and explore its structure:"
	<python>
	# Load the csv or excel file
	df = pd.read_csv('data.csv') 
	print(f"Shape: {df.shape}")
	print(f"\nData Types:\n{df.dtypes}")
	print(f"\nFirst 3 rows:\n{df.head(3)}")
	</python>

3. **Check Assumptions & Analyze:**
   "I will now test assumptions and perform appropriate statistical tests."
   <python>
   # Example: Testing normality before t-test
   from scipy.stats import shapiro, levene, ttest_ind
   
   # Check normality
   stat, p_norm = shapiro(df['age'].dropna())
   print(f"Shapiro-Wilk Test: W={stat:.3f}, p={p_norm:.3f}")
   
   if p_norm > 0.05:
       print("Data appears normally distributed (p>0.05)")
       # Proceed with parametric test
   else:
       print("Data not normally distributed (p<0.05), consider non-parametric alternatives")
   </python>

4. **Visualize with Standards:**
   <python>
   # Example visualization with all standards
   plt.figure(figsize=(10, 6))
   
   # Main plot
   plt.hist(df['age'].dropna(), bins=20, edgecolor='black', alpha=0.7)
   
   # Add mean line
   mean_age = df['age'].mean()
   plt.axvline(mean_age, color='red', linestyle='--', linewidth=2, 
               label=f'Mean = {mean_age:.1f}')
   
   # Formatting
   plt.title(f'Age Distribution (N={df["age"].notna().sum()})')
   plt.xlabel('Age (years)')
   plt.ylabel('Frequency')
   plt.grid(True, alpha=0.3)
   plt.legend()
   
   # Save
   plt.savefig('age_distribution.png', dpi=100, bbox_inches='tight')
   plt.close()
   print("Saved: age_distribution.png")
   </python>

5. **Final Summary Structure:**
   "## Statistical Analysis Report
   
   **Key Findings:**
   - No significant difference in age between groups (t=1.23, p=0.327, Cohen's d=0.15, 95% CI[-0.12, 0.42])
   - Sample size: Group A (n=150), Group B (n=146)
   
   **Assumptions & Limitations:**
   - Normality assumption met (Shapiro-Wilk p>0.05)
   - Equal variances assumed (Levene's test p=0.412)
   - Missing data: 4 cases excluded (1.3% of total)
   
   **Recommendations:**
   - Consider collecting additional covariates for adjusted analysis
   - Investigate the 4 missing cases for patterns
   
   **Files Generated:**
   <image>age_distribution.png</image>"
   <image>another_plot.png</image>"
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
