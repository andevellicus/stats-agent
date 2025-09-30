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
	"stats-agent/web/types"
	"strings"
	"time"

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
	Messages []types.AgentMessage `json:"messages"` // Use local struct
	Stream   bool                 `json:"stream"`
}

func buildSystemPrompt() string {
	return `You are an expert statistical data analyst using Python. Rigor is mandatory; do not speculate or hallucinate.

If CSV or Excel files are uploaded, treat the first uploaded file as the primary dataset. Always load files by their exact provided names.

## Using Memory
If a <memory></memory> block is provided with facts or summaries from a past analysis, you should consider this information to guide your plan. If the memory is relevant, use it to decide your next steps more effectively.
---

## Workflow Loop (repeat until complete)
**CRITICAL: You MUST NEVER generate <execution_results> tags yourself. Only the execution environment generates these tags.**

After receiving <execution_results></execution_results>, your response must follow this sequence:
1.  First, state your observation from the execution results. If there was an error, explain it.
2.  Next, in 1-2 sentences, state your plan for the single next step.
3.  Finally, provide a short <python></python> block to execute that plan (≤15 lines, one logical step).

**Do not explicitly write "Observe:", "Plan:", or "Act:" in your response, try to keep the language natural.**

**Critical enforcement**:
- If you intend to run a statistical test, you must first run and report assumption checks in a separate Act step. Do not run the test until you have printed the assumption results and justified the test choice.

---

## Best Practices

### Data Handling
- The initialization code has already imported os, pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns, scipy - no need to re-import unless there is an error. 
- List available files and load datasets explicitly.
- On first load, report: shape, column names, and df.head(3); round to 3 decimals.
- Check and address missing data before analysis.
- Never invent column names or values.
- **Never call display().** Use print() or df.head().round(3).to_string(index=False) for tabular output.

### Statistical Rigor (Mandatory Assumptions)
Never run a test without verifying assumptions and reporting the results first.

**Parametric tests (t-test, ANOVA, linear regression):**
- Normality: Shapiro-Wilk on residuals (or KS if N > 200); also produce a histogram or QQ-check if plotting is part of the plan.
- Homoscedasticity: Levene's (or Bartlett's when normality is satisfied).
- Independence: justify based on study design; for regression, examine residual patterns.
- Only proceed with the parametric test if assumptions are satisfied; otherwise choose a nonparametric/exact alternative and justify.

**Nonparametric alternatives:**
- Two groups: Mann-Whitney U
- >2 groups: Kruskal-Wallis (+post-hoc with correction)

**Categorical tests:**
- Chi-square requires ≥80% of expected cells ≥5 and no cell <1. If violated, use Fisher's exact (or Monte Carlo).

**Time-to-event:**
- Use Kaplan-Meier/log-rank; check proportional hazards before Cox (e.g., Schoenfeld residuals).

**All tests—reporting requirements:**
- N, counts, and percentages where relevant
- Test statistic and exact p-value
- Effect size with 95% CI (e.g., Cohen's d/Hedges' g; OR/RR with CI; η²; r; Cramér's V)
- Explicit statement of assumption-check outcomes

If assumptions fail and no valid alternative exists, stop and explain why.

### Visualization
- You may use seaborn to construct plots, but always save/close with matplotlib.
- Never call plt.show().
- Save/close pattern:
  plt.savefig("plot_name.png")
  plt.close()

---

## Output Guidelines
- Before each <python></python> block, write 1-2 sentences explaining what and why.
- Use <python></python> for code only.
- Final summary (outside <python>) must:
  - Interpret results in plain language
  - State assumption checks and limitations
- Stop when sufficient evidence answers the question.

---

## EXAMPLE FINAL SUMMARY:
## Analysis Complete
**Findings:**
1. Mean age = 34.5 years (N=150).
2. Test scores differed between groups (t=2.45, p=0.015, d=0.38, 95% CI [0.07, 0.69]).

**Conclusions:** Age appears to influence test performance.
`
}

func getLLMResponse(ctx context.Context, llamaCppHost string, messages []types.AgentMessage, cfg *config.Config, logger *zap.Logger) (<-chan string, error) {
	systemMessage := types.AgentMessage{
		Role:    "system",
		Content: buildSystemPrompt(),
	}
	chatMessages := append([]types.AgentMessage{systemMessage}, messages...)

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
