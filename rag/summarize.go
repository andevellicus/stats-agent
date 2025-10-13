package rag

import (
    "context"
    "fmt"
    "strings"

    "stats-agent/llmclient"
    "stats-agent/prompts"
    "stats-agent/web/types"

	"go.uber.org/zap"
)

func buildMetadataTagsForPrompt(metadata map[string]string) string {
	if len(metadata) == 0 {
		return ""
	}

	var tags []string

	// Priority order for tag inclusion
	if test := metadata["primary_test"]; test != "" {
		tags = append(tags, fmt.Sprintf("test:%s", test))
	}
	if metadata["sig_at_05"] == "true" {
		tags = append(tags, "p<0.05:yes")
	} else if metadata["sig_at_05"] == "false" {
		tags = append(tags, "p<0.05:no")
	}
	if metadata["sig_at_01"] == "true" {
		tags = append(tags, "p<0.01:yes")
	}
	if stage := metadata["analysis_stage"]; stage != "" {
		tags = append(tags, fmt.Sprintf("stage:%s", stage))
	}
	if vars := metadata["variables"]; vars != "" {
		tags = append(tags, fmt.Sprintf("variables:%s", vars))
	}
	if dataset := metadata["dataset"]; dataset != "" {
		tags = append(tags, fmt.Sprintf("dataset:%s", dataset))
	}

	if len(tags) == 0 {
		return ""
	}

	return fmt.Sprintf("Based on the extracted metadata, include these tags in your response: [%s]", strings.Join(tags, " | "))
}

func (r *RAG) SummarizeLongTermMemory(ctx context.Context, context, latestUserMessage string) (string, error) {
	latestUserMessage = strings.TrimSpace(latestUserMessage)

    systemPrompt := prompts.SummarizeMemory()

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

	// Non-streaming summarization (use server default temperature)
	summary, err := llmclient.New(r.cfg, r.logger).Chat(ctx, r.cfg.SummarizationLLMHost, messages, nil)
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

    // Wrap the summary in a markdown section header
    return fmt.Sprintf("### Memory Context\n%s", summary), nil
}

func (r *RAG) generateFactSummary(ctx context.Context, code, result string, metadata map[string]string) (string, error) {
	finalResult := result
	if strings.Contains(result, "Error:") {
		finalResult = compressMiddle(result, 800, 200, 200)
	}

    systemPrompt := prompts.FactSummary()

	// Build metadata tags string for the prompt
	metadataTags := buildMetadataTagsForPrompt(metadata)

	userPrompt := fmt.Sprintf(`
Extract a statistical fact from the following code and output. Follow these rules:

RULES:
1. Maximum 200 words (be concise but complete)
2. Include specific names (test names, variable names, column names)
3. Preserve key numbers (p-values, effect sizes, R², coefficients, sample sizes)
4. State statistical conclusions when present (e.g., "significant at α=0.05", "violates normality assumption")
5. If multiple steps, use 2-3 sentences maximum
6. For errors, state what failed and why
7. END your fact with inline metadata tags in square brackets

METADATA TAGS FORMAT:
End your fact with relevant metadata in this format: [key1:value1 | key2:value2 | ...]
Only include tags that are relevant to the analysis. Common tags:
- test: test name (e.g., shapiro-wilk, t-test, anova, pearson-correlation)
- p<0.05: yes/no (significance at α=0.05)
- p<0.01: yes/no (significance at α=0.01)
- stage: assumption_check, hypothesis_test, modeling, descriptive, post_hoc
- variables: comma-separated variable names
- dataset: filename being analyzed

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
Good Fact: Shapiro-Wilk normality test on residuals yielded W=0.9234, p=0.0156, violating normality assumption at α=0.05 [test:shapiro-wilk | p<0.05:yes | stage:assumption_check | variables:residuals]
Bad Fact: A normality test was performed on the data.

Example 2 - Descriptive Statistics:
Code: print(df[['age', 'income', 'score']].describe())
Output: age: count=150, mean=34.23, std=8.91, min=18, max=65; income: mean=52340.12, std=12450.67; score: mean=78.45, std=12.34
Good Fact: Dataset contains 150 observations with variables age (M=34.23, SD=8.91, range 18-65), income (M=52340.12, SD=12450.67), and score (M=78.45, SD=12.34) [stage:descriptive | variables:age,income,score]
Bad Fact: Descriptive statistics were calculated for the dataframe.

Example 3 - Regression Model:
Code: model = LinearRegression(); model.fit(X_train, y_train); r2 = model.score(X_test, y_test); print(f"R²={r2:.3f}, Coefficients: {model.coef_}")
Output: R²=0.734, Coefficients: [2.34, -1.56, 0.89]
Good Fact: Linear regression model trained with R²=0.734 on test set, yielding coefficients [2.34, -1.56, 0.89] for predictor variables [test:linear-regression | stage:modeling]
Bad Fact: A regression model was fitted to the training data.

Example 4 - Data Transformation:
Code: df['log_income'] = np.log(df['income'])
Output: Success: Code executed with no output.
Good Fact: Created log-transformed variable log_income from income column for normalization [variables:income,log_income]
Bad Fact: A transformation was applied to the income variable.

%s

Now extract the fact from this code and output:

Code:
%s

Output:
%s

Respond with only the fact, ending with metadata tags in square brackets.
`, metadataTags, code, finalResult)

	messages := []types.AgentMessage{
		{Role: "system", Content: systemPrompt},
		{Role: "user", Content: userPrompt},
	}

	summary, err := llmclient.New(r.cfg, r.logger).Chat(ctx, r.cfg.SummarizationLLMHost, messages, nil)
	if err != nil {
		return "", fmt.Errorf("llm chat call failed for summary: %w", err)
	}
	if summary == "" {
		return "", fmt.Errorf("llm returned an empty summary")
	}
	return strings.TrimSpace(summary), nil
}

func (r *RAG) generateSearchableSummary(ctx context.Context, content string) (string, error) {
    systemPrompt := prompts.SearchableSummary()

	userPrompt := fmt.Sprintf(`Create a single-sentence summary of the following user message. Focus on key entities, variable names, and statistical concepts.

**User Message:**
"%s"

**Summary:**
`, content)

	messages := []types.AgentMessage{
		{Role: "system", Content: systemPrompt},
		{Role: "user", Content: userPrompt},
	}

	summary, err := llmclient.New(r.cfg, r.logger).Chat(ctx, r.cfg.SummarizationLLMHost, messages, nil)
	if err != nil {
		return "", fmt.Errorf("llm chat call failed for searchable summary: %w", err)
	}

	if summary == "" {
		return "", fmt.Errorf("llm returned an empty searchable summary")
	}

	return strings.TrimSpace(summary), nil
}

// SummarizePDFKeyFacts produces a short, searchable "Key Facts" summary from page 1 text.
// It is generic across document types and avoids hallucinating missing fields.
func (r *RAG) SummarizePDFKeyFacts(ctx context.Context, filename string, pageOneText string) (string, error) {
    filename = strings.TrimSpace(filename)
    pageOneText = strings.TrimSpace(pageOneText)
    if pageOneText == "" {
        return "", fmt.Errorf("page 1 text is empty")
    }

    system := prompts.PDFKeyFacts()

    var user strings.Builder
    if filename != "" {
        user.WriteString("Document: ")
        user.WriteString(filename)
        user.WriteString("\n")
    }
    user.WriteString("Page 1 text (verbatim):\n")
    user.WriteString(pageOneText)
    user.WriteString("\n\nReturn only the Key Facts.")

    msgs := []types.AgentMessage{
        {Role: "system", Content: system},
        {Role: "user", Content: user.String()},
    }

    summary, err := llmclient.New(r.cfg, r.logger).Chat(ctx, r.cfg.SummarizationLLMHost, msgs, nil)
    if err != nil {
        return "", fmt.Errorf("llm chat for pdf key facts failed: %w", err)
    }
    summary = strings.TrimSpace(summary)
    if summary == "" {
        return "", fmt.Errorf("empty key facts summary")
    }
    return summary, nil
}
