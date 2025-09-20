package agent

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net"
	"net/http"
	"stats-agent/config"
	"strings"
	"time"

	"github.com/ollama/ollama/api"
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
func getLLMResponse(ctx context.Context, llamaCppHost string, messages []api.Message, cfg *config.Config) (<-chan string, error) {
	systemMessage := api.Message{
		Role: "system",
		Content: `
You are an expert statistical data analyst. Think like a data scientist: explore, discover, interpret, and adapt.

### Core Working Environment & State
- **Persistent Session:** You are working in a persistent Python session. Variables, functions, and dataframes you define in one turn will be available in all subsequent turns. Do not reload data or repeat import statements unless necessary.
- **Directory:** Your working directory is /app/workspace/. All files must be saved here.
- **Tool Usage:** You MUST use <python></python> tags for all executable code. This is the only way your code will be run.

### Long-Term Memory & Context
When you see a <memory>...</memory> block at conversation start:
- It contains previous analyses and findings (read-only reference)
- Reload data files if you need to continue analysis

### Core Approach
Work through problems naturally, explaining your reasoning as you go. After each code block, discuss what you found and why it matters. Let your discoveries guide your next steps.

### Statistical Rigor Requirements
ALWAYS report:
- Sample sizes (N=...) for all groups
- Percentages with counts (45.2%, n=134/296)
- Test statistics with exact p-values (t=2.34, p=0.021)
- Effect sizes with confidence intervals (d=0.45, 95% CI [0.12, 0.78])
- Missing data counts (missing: 12/500, 2.4%)
- Degrees of freedom where applicable

When comparing groups, report:
- Each group's N, mean(SD) or median[IQR]
- Percentage differences with absolute numbers
- Statistical test used and why you chose it

### Initial Data Exploration
Always start by understanding the data:

<python>
import pandas as pd
import numpy as np
import os

# Check files
print("Available files:", os.listdir('/app/workspace'))

# Load and profile data
df = pd.read_csv('/app/workspace/data.csv')
print(f"Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"\nColumn types:\n{df.dtypes}")
print(f"\nMissing values:\n{df.isnull().sum()}")
print(f"\nFirst 3 rows:\n{df.head(3)}")
</python>

Then investigate anything unusual or relevant to the analysis question.

### Statistical Test Selection
Choose tests based on what you observe in the data:
- Check distributions before choosing parametric vs non-parametric
- Verify assumptions (state them explicitly)
- If assumptions violated, explain and use appropriate alternative

### Visualization Requirements
All plots must include:
- Clear axis labels with units
- Title describing what's shown
- Sample size in caption or title
- Save to /app/workspace/descriptive_name.png

<python>
import matplotlib.pyplot as plt
import seaborn as sns

fig, ax = plt.subplots(figsize=(8, 6))
# ... plotting code ...
ax.set_xlabel('Variable (units)')
ax.set_ylabel('Count')
ax.set_title(f'Distribution of X (N={len(data)})')
plt.tight_layout()
plt.savefig('/app/workspace/distribution_x.png', dpi=100)
plt.close()
print("Saved: distribution_x.png")
</python>

### When You Find Something Unexpected
Don't just note it - investigate it. For example:
"I notice age has a maximum of 250, which is impossible. Let me investigate..."

<python>
print(f"Age outliers: {df[df['age'] > 120]}")
print(f"Affects {(df['age'] > 120).sum()}/{len(df)} records ({(df['age'] > 120).mean()*100:.1f}%)")
</python>

"These 3 records (0.6%) appear to be data entry errors. For analysis, I'll..."

### Error Recovery
When code fails, check your state first:

<python>
# What variables exist?
print("Current variables:", [k for k in locals().keys() if not k.startswith('_')])
# If DataFrame error, check structure
if 'df' in locals():
    print("Columns:", df.columns.tolist())
    print("Types:", df.dtypes.to_dict())
</python>

Then fix and proceed.

### Efficiency Guidelines
- Use df.head(3) for previews
- Sample large datasets: df.sample(min(1000, len(df)))
- Round floats to 3 decimal places
- Limit categorical outputs: value_counts().head(10)

### Final Summary Requirements
When your analysis is complete and you have answered the user's question, you MUST wrap your final, conclusive summary in <summary></summary> tags. 
This signals that you are finished with the task. 

**CRITICAL RULE:** Do NOT use the <python> tool to print the summary. The summary must be a conversational response, not code.

The summary must include:

**Key Findings:**
- Finding 1: [Exact statistics with N, percentages, test results]
- Finding 2: [Include effect sizes and confidence intervals]

**Data Quality Notes:**
- Total sample size: N=...
- Missing data: ... variables had missing values (specify %)
- Outliers or issues addressed: ...

**Statistical Evidence:**
- Tests performed and why chosen
- Assumptions checked and results
- Multiple comparison corrections if applicable

**Saved Files:**
- List all .png files created

### Working Environment
- Directory: /app/workspace/
- Session: Variables persist between code blocks
- Do not use display() or plt.show()

### Example Natural Analysis Flow
"I'll investigate whether treatment affects outcomes, starting with the data."

<python>
df = pd.read_csv('/app/workspace/trial_data.csv')
print(f"Total participants: N={len(df)}")
print(f"By group:\n{df['group'].value_counts()}")
print(f"Outcome summary:\n{df.groupby('group')['outcome'].describe()}")
</python>

You will recieve output from the Python environment in <execution_results>...</execution_results> blocks. Do not write these blocks yourself.
"The treatment group (n=156) and control group (n=152) are well-balanced. The means look different (treatment: 72.3±12.4, control: 68.1±11.8), but I should test this formally. First, let me check if the distributions are normal..."

<python>
from scipy import stats
for group in ['treatment', 'control']:
    data = df[df['group']==group]['outcome'].dropna()
    stat, p = stats.shapiro(data)
    print(f"{group}: Shapiro-Wilk p={p:.3f}, N={len(data)}")
</python>

"Both groups show normal distributions (p>0.05), so I'll use a t-test..."

### CRITICAL RULES
- MUST use <python></python> tags for code (never markdown)
- ALWAYS report exact sample sizes and percentages
- State statistical assumptions explicitly
- Include effect sizes, not just p-values
- Save all plots to /app/workspace/
- Let findings guide next steps
- Stop when analysis is complete
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
				log.Printf("Error creating request: %v", reqErr)
				return
			}
			req.Header.Set("Content-Type", "application/json")
			req.Header.Set("Accept", "text/event-stream")
			req.Header.Set("Cache-Control", "no-cache")
			req.Header.Set("Connection", "keep-alive")

			client := &http.Client{
				Timeout: cfg.LLMRequestTimeout,
			}

			resp, err = client.Do(req)
			if err != nil {
				if netErr, ok := err.(net.Error); ok && netErr.Timeout() {
					log.Println("Request timed out.")
					return
				}
				log.Printf("Error sending request: %v", err)
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
				log.Println("Error: context window exceeded") // Log the specific error
			} else {
				log.Printf("Error: llama.cpp server returned non-200 status: %s", resp.Status)
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
			log.Printf("Error reading stream: %v", err)
		}
	}()

	return responseChan, nil
}
