package rag

import (
	"context"
	"encoding/json"
	"fmt"
	"sort"
	"strings"

	"stats-agent/llmclient"
	"stats-agent/web/types"

	"go.uber.org/zap"
)

type memoryFact struct {
	Type             string   `json:"type"`
	Dataset          string   `json:"dataset"`
	Variables        []string `json:"variables"`
	Finding          string   `json:"finding"`
	Evidence         string   `json:"evidence"`
	AssumptionChecks string   `json:"assumption_checks"`
}

type memoryFactsEnvelope struct {
	Facts []memoryFact `json:"facts"`
}

func (r *RAG) SummarizeLongTermMemory(ctx context.Context, context, latestUserMessage string) (string, error) {
	latestUserMessage = strings.TrimSpace(latestUserMessage)

	systemPrompt := `Extract statistical facts as JSON with this shape:
{
  "facts": [
    {
      "type": "descriptive|test|model",
      "dataset": "filename.csv",
      "variables": ["var1", "var2"],
      "finding": "age and income are positively correlated",
      "evidence": "Pearson r=0.67, p<0.001",
      "assumption_checks": "normality satisfied via Shapiro-Wilk"
    }
  ]
}

Rules:
- Only include facts supported by statistical evidence (results, statistics, or metrics).
- Omit how-to instructions, failed attempts, and irrelevant conversation.
- Preserve concrete numbers, dataset names, variable names, and test names.
- You may return up to three facts. If none exist, return {"facts": []}.
- Respond with valid JSON only, no additional commentary.`

	if latestUserMessage == "" {
		latestUserMessage = "(no specific question provided)"
	}

	userPrompt := fmt.Sprintf(`User's current question:
"%s"

Conversation history to extract from:
%s

Extract relevant facts following the rules above:`, latestUserMessage, context)

	messages := []types.AgentMessage{
		{Role: "system", Content: systemPrompt},
		{Role: "user", Content: userPrompt},
	}

	summary, err := llmclient.New(r.cfg, r.logger).Chat(ctx, r.cfg.SummarizationLLMHost, messages)
	if err != nil {
		return "", fmt.Errorf("llm chat call failed for memory summary: %w", err)
	}

	summary = strings.TrimSpace(summary)
	if summary == "" {
		return "", fmt.Errorf("llm returned an empty summary for memory")
	}

	facts, err := parseMemoryFacts(summary)
	if err != nil {
		// Try to clean up common JSON issues
		cleaned := strings.TrimSpace(summary)
		cleaned = strings.TrimPrefix(cleaned, "```json")
		cleaned = strings.TrimPrefix(cleaned, "```")
		cleaned = strings.TrimSuffix(cleaned, "```")
		cleaned = strings.TrimSpace(cleaned)

		facts, err = parseMemoryFacts(cleaned)
		if err != nil {
			r.logger.Warn("JSON parsing failed even after cleanup, using fallback",
				zap.Error(err),
				zap.String("summary", summary))
			// Return generic fallback instead of potentially malformed content
			return "<memory>\nFact: Prior analysis available but could not be parsed.\n</memory>", nil
		}
	}

	if len(facts) == 0 {
		return "<memory>\nFact: No relevant prior analysis found.\n</memory>", nil
	}

	var lines []string
	for _, fact := range facts {
		line := formatMemoryFact(fact)
		if line != "" {
			lines = append(lines, "Fact: "+line)
		}
	}

	if len(lines) == 0 {
		lines = append(lines, "Fact: No relevant prior analysis found.")
	}

	return fmt.Sprintf("<memory>\n%s\n</memory>", strings.Join(lines, "\n")), nil
}

func parseMemoryFacts(summary string) ([]memoryFact, error) {
	var envelope memoryFactsEnvelope
	if err := json.Unmarshal([]byte(summary), &envelope); err != nil {
		return nil, err
	}
	return envelope.Facts, nil
}

func formatMemoryFact(f memoryFact) string {
	finding := strings.TrimSpace(f.Finding)
	if finding == "" {
		return ""
	}

	var meta []string
	if trimmed := strings.TrimSpace(f.Dataset); trimmed != "" {
		meta = append(meta, fmt.Sprintf("dataset=%s", trimmed))
	}
	if len(f.Variables) > 0 {
		vars := make([]string, 0, len(f.Variables))
		for _, v := range f.Variables {
			v = strings.TrimSpace(v)
			if v != "" {
				vars = append(vars, v)
			}
		}
		if len(vars) > 0 {
			sort.Strings(vars)
			meta = append(meta, fmt.Sprintf("variables=%s", strings.Join(vars, ",")))
		}
	}
	if trimmed := strings.TrimSpace(f.Evidence); trimmed != "" {
		meta = append(meta, fmt.Sprintf("evidence: %s", trimmed))
	}
	if trimmed := strings.TrimSpace(f.AssumptionChecks); trimmed != "" {
		meta = append(meta, fmt.Sprintf("assumptions: %s", trimmed))
	}
	if trimmed := strings.TrimSpace(f.Type); trimmed != "" {
		meta = append(meta, fmt.Sprintf("type=%s", trimmed))
	}

	if len(meta) == 0 {
		return finding
	}

	return fmt.Sprintf("%s (%s)", finding, strings.Join(meta, "; "))
}

func (r *RAG) generateFactSummary(ctx context.Context, code, result string) (string, error) {
	finalResult := result
	if strings.Contains(result, "Error:") {
		finalResult = compressMiddle(result, 800, 200, 200)
	}

	systemPrompt := `You are an expert at extracting statistical facts from code execution results. Your task is to create searchable, information-dense summaries that preserve methodological details and numerical results. Focus on what was done, what was found, and what it means statistically.

Extract a statistical fact from the following code and output. Follow these rules:

RULES:
1. Start with "Fact:"
2. Maximum 200 words (be concise but complete)
3. Include specific names (test names, variable names, column names)
4. Preserve key numbers (p-values, effect sizes, R², coefficients, sample sizes)
5. State statistical conclusions when present (e.g., "significant at α=0.05", "violates normality assumption")
6. If multiple steps, use 2-3 sentences maximum
7. For errors, state what failed and why

WHAT TO CAPTURE:
- Statistical test names (Shapiro-Wilk, t-test, ANOVA, etc.)
- Variables/columns analyzed
- Key parameters (significance levels, degrees of freedom)
- Numerical results with context
- Data characteristics (sample size, distributions, missing values)
- Transformations or preprocessing applied
- Assumption check results
- Model performance metrics

EXAMPLES:

Example 1 - Normality Test:
Code: from scipy import stats; stat, p = stats.shapiro(df['residuals']); print(f"Shapiro-Wilk: W={stat:.4f}, p={p:.4f}")
Output: Shapiro-Wilk: W=0.9234, p=0.0156
Good Fact: Fact: Shapiro-Wilk normality test on residuals yielded W=0.9234, p=0.0156, indicating violation of normality assumption at α=0.05.
Bad Fact: Fact: A normality test was performed on the data.

Example 2 - Descriptive Statistics:
Code: print(df[['age', 'income', 'score']].describe())
Output: age: count=150, mean=34.23, std=8.91, min=18, max=65; income: mean=52340.12, std=12450.67; score: mean=78.45, std=12.34
Good Fact: Fact: Dataset contains 150 observations with variables age (M=34.23, SD=8.91, range 18-65), income (M=52340.12, SD=12450.67), and score (M=78.45, SD=12.34).
Bad Fact: Fact: Descriptive statistics were calculated for the dataframe.

Example 3 - Regression Model:
Code: model = LinearRegression(); model.fit(X_train, y_train); r2 = model.score(X_test, y_test); print(f"R²={r2:.3f}, Coefficients: {model.coef_}")
Output: R²=0.734, Coefficients: [2.34, -1.56, 0.89]
Good Fact: Fact: Linear regression model trained with R²=0.734 on test set, yielding coefficients [2.34, -1.56, 0.89] for predictor variables.
Bad Fact: Fact: A regression model was fitted to the training data.

Example 4 - Data Transformation:
Code: df['log_income'] = np.log(df['income'])
Output: Success: Code executed with no output.
Good Fact: Fact: Created log-transformed variable log_income from income column for normalization.
Bad Fact: Fact: A transformation was applied to the income variable.

Now extract the fact from this code and output:

Code:
{code}

Output:
{output}

Respond with only the fact, starting with "Fact:"`

	userPrompt := fmt.Sprintf(`Code:
%s

Output:
%s
`, code, finalResult)

	messages := []types.AgentMessage{
		{Role: "system", Content: systemPrompt},
		{Role: "user", Content: userPrompt},
	}

	summary, err := llmclient.New(r.cfg, r.logger).Chat(ctx, r.cfg.SummarizationLLMHost, messages)
	if err != nil {
		return "", fmt.Errorf("llm chat call failed for summary: %w", err)
	}
	if summary == "" {
		return "", fmt.Errorf("llm returned an empty summary")
	}
	return strings.TrimSpace(summary), nil
}

func (r *RAG) generateSearchableSummary(ctx context.Context, content string) (string, error) {
	systemPrompt := `You are an expert at creating concise, searchable summaries of user messages. Your task is to distill the user's message into a single sentence that captures the core question, action, or intent.`

	userPrompt := fmt.Sprintf(`Create a single-sentence summary of the following user message. Focus on key entities, variable names, and statistical concepts.

**User Message:**
"%s"

**Summary:**
`, content)

	messages := []types.AgentMessage{
		{Role: "system", Content: systemPrompt},
		{Role: "user", Content: userPrompt},
	}

	summary, err := llmclient.New(r.cfg, r.logger).Chat(ctx, r.cfg.SummarizationLLMHost, messages)
	if err != nil {
		return "", fmt.Errorf("llm chat call failed for searchable summary: %w", err)
	}

	if summary == "" {
		return "", fmt.Errorf("llm returned an empty searchable summary")
	}

	return strings.TrimSpace(summary), nil
}
