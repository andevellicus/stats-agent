package rag

import (
	"context"
	"fmt"
	"strings"

	"stats-agent/llmclient"
	"stats-agent/web/types"

	"go.uber.org/zap"
)

func (r *RAG) SummarizeLongTermMemory(ctx context.Context, context, latestUserMessage string) (string, error) {
	latestUserMessage = strings.TrimSpace(latestUserMessage)

	systemPrompt := `You are a technical summarization expert specializing in data analysis and statistics.

Your task: Extract key facts from conversation history that are relevant to the user's current question.

CRITICAL RULES:
1. Focus on DATA FINDINGS, not process descriptions
2. ALWAYS preserve: numbers, statistical measures, column names, file names, variable names
3. Extract MULTIPLE facts if relevant (1-3 facts maximum)
4. Each fact should be ONE concise sentence
5. Start each fact with "Fact:" on its own line
6. Ignore: instructions, system messages, casual chat, failed attempts
7. If no relevant facts exist, output: "Fact: No relevant prior analysis found."

WHAT TO EXTRACT:
- Statistical results (correlations, p-values, test results, effect sizes)
- Data characteristics (sample size, distributions, outliers)
- Analysis decisions (which test used, which variables, transformations applied)
- Key findings (patterns discovered, significant differences, trends)
- File/dataset information (which data was analyzed)

DON'T EXTRACT:
- How-to instructions or explanations
- Error messages or debugging steps
- Casual conversation or greetings
- Process descriptions ("I used pandas to...")

EXAMPLES:

Example 1:
Memory: "Let me help you analyze that. First, I loaded the data. Then I checked for missing values - found 23 missing values in the income column. I dropped those rows and proceeded with the t-test. Independent samples t-test comparing male vs female salaries showed t=2.34, df=198, p=0.021, Cohen's d=0.33."
User question: "What were the results of the gender salary comparison?"
Output:
Fact: Dataset had 23 missing values in income column which were removed.
Fact: Independent t-test found statistically significant salary difference between genders (t=2.34, df=198, p=0.021) with small effect size (d=0.33).

Example 2:
Memory: "Hi! How can I help you today? What kind of analysis would you like to perform?"
User question: "What did we discuss about regression?"
Output:
Fact: No relevant prior analysis found.

Remember: Be SPECIFIC. Include actual numbers, variable names, and technical details. Vague summaries are useless.`

	if latestUserMessage == "" {
		latestUserMessage = "(no specific question provided)"
	}

	userPrompt := fmt.Sprintf(`User's current question:
"%s"

Conversation history to extract from:
%s

Extract relevant facts following the rules and examples above:`, latestUserMessage, context)

	messages := []types.AgentMessage{
		{Role: "system", Content: systemPrompt},
		{Role: "user", Content: userPrompt},
	}

	// Non-streaming summarization
	summary, err := llmclient.New(r.cfg, r.logger).Chat(ctx, r.cfg.SummarizationLLMHost, messages)
	if err != nil {
		return "", fmt.Errorf("llm chat call failed for memory summary: %w", err)
	}

	summary = strings.TrimSpace(summary)
	if summary == "" {
		return "", fmt.Errorf("llm returned an empty summary for memory")
	}

	// Post-process: ensure it starts with "Fact:"
	if !strings.HasPrefix(summary, "Fact:") {
		r.logger.Warn("Summary didn't start with 'Fact:', attempting to fix",
			zap.String("summary", summary))

		// Try to salvage it by prepending "Fact:"
		summary = "Fact: " + summary
	}

	// Wrap the summary in memory tags
	return fmt.Sprintf("<memory>\n%s\n</memory>", summary), nil
}

func (r *RAG) generateFactSummary(ctx context.Context, code, result string) (string, error) {
	finalResult := result
	if strings.Contains(result, "Error:") {
		finalResult = compressMiddle(result, 800, 200, 200)
	}

	systemPrompt := `You are an expert at extracting statistical facts from code execution results. Your task is to create searchable, information-dense summaries that preserve methodological details and numerical results. Focus on what was done, what was found, and what it means statistically.`
	userPrompt := fmt.Sprintf(`
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
%s

Output:
%s

Respond with only the fact, starting with "Fact:"
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
