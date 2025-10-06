package rag

import (
	"context"
	"fmt"
	"regexp"
	"strings"
	"time"

	"stats-agent/llmclient"
	"stats-agent/web/types"

	"go.uber.org/zap"
)

func (r *RAG) SummarizeLongTermMemory(ctx context.Context, context, latestUserMessage string) (string, error) {
	latestUserMessage = strings.TrimSpace(latestUserMessage)

	// Extremely simple, constrained prompt for fast models like Mistral
	systemPrompt := `Extract 1-3 key facts from conversation history relevant to the user's question. Each fact must be ONE sentence starting with "Fact:".

Rules:
- ONE sentence per fact
- Include numbers, variable names, test results
- NO explanations, NO commentary
- If no relevant facts exist: "Fact: No prior analysis found."

Example:
Fact: Dataset has 150 rows with Age, Gender, and Score columns.
Fact: Independent t-test found p=0.023, Cohen's d=0.42 between groups.`

	userPrompt := fmt.Sprintf(`Question: %s

History:
%s

Extract 1-3 relevant facts (one sentence each):`, latestUserMessage, context)

	messages := []types.AgentMessage{
		{Role: "system", Content: systemPrompt},
		{Role: "user", Content: userPrompt},
	}

	// Non-streaming summarization with operation-specific timeout
	start := time.Now()
	client := llmclient.NewWithTimeout(r.cfg, r.logger, r.cfg.SummarizationTimeout)
	summary, err := client.Chat(ctx, r.cfg.SummarizationLLMHost, messages)
	elapsed := time.Since(start)

	if err != nil {
		r.logger.Error("LLM chat call failed for memory summary",
			zap.Error(err),
			zap.Duration("elapsed", elapsed),
			zap.Duration("timeout", r.cfg.SummarizationTimeout))
		return "", fmt.Errorf("llm chat call failed for memory summary: %w", err)
	}

	if elapsed > r.cfg.SummarizationTimeout/2 {
		r.logger.Warn("Memory summarization took longer than expected",
			zap.Duration("elapsed", elapsed),
			zap.Duration("timeout", r.cfg.SummarizationTimeout))
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

func (r *RAG) generateFactSummary(ctx context.Context, result string, metadata map[string]string) (string, error) {
	// Build structured context from metadata
	var context []string

	if test := metadata["primary_test"]; test != "" {
		context = append(context, "Test: "+test)
	}
	if vars := metadata["variables"]; vars != "" {
		context = append(context, "Variables: "+vars)
	}
	if dataset := metadata["dataset"]; dataset != "" {
		context = append(context, "Dataset: "+dataset)
	}

	contextStr := ""
	if len(context) > 0 {
		contextStr = "\nContext: " + strings.Join(context, ", ")
	}

	resultPreview := truncateResult(result, 800)

	systemPrompt := `Create a one-sentence fact from analysis output. Include all numbers and variable names.

Format: "Fact: [concise finding with numbers]"

Good: "Fact: Shapiro-Wilk test on residuals showed W=0.923, p=0.016, violating normality."
Good: "Fact: Dataset has 12 columns matching 'age'; Days_relative_to_drainage mean=-28.2, SD=25.1."

Extract the finding:`

	userPrompt := fmt.Sprintf(`%s

Output:
%s

Fact:`, contextStr, resultPreview)

	// Check token count BEFORE sending
	fullPrompt := systemPrompt + "\n\n" + userPrompt
	tokenCount, err := r.countTokensForEmbedding(ctx, fullPrompt)

	// Leave 500 tokens for generation, so max input = 3500 tokens
	const maxInputTokens = 3500

	if err == nil && tokenCount > maxInputTokens {
		r.logger.Warn("Prompt too large, truncating more aggressively",
			zap.Int("tokens", tokenCount),
			zap.Int("max", maxInputTokens))

		// Calculate safe result length
		overageTokens := tokenCount - maxInputTokens
		overageChars := overageTokens * 4                        // rough estimate
		safeResultLen := len(resultPreview) - overageChars - 100 // safety margin

		if safeResultLen > 200 {
			resultPreview = resultPreview[:safeResultLen] + "..."
			userPrompt = fmt.Sprintf(`%sOutput:
%s

Fact:`, contextStr, resultPreview)
		} else {
			// Result is fundamentally too large, use fallback
			return r.generateFallbackFact(metadata, result), nil
		}
	}

	messages := []types.AgentMessage{
		{Role: "system", Content: systemPrompt},
		{Role: "user", Content: userPrompt},
	}

	summary, err := llmclient.New(r.cfg, r.logger).Chat(ctx, r.cfg.SummarizationLLMHost, messages)
	if err != nil {
		return "", fmt.Errorf("llm chat call failed for summary: %w", err)
	}

	summary = strings.TrimSpace(summary)
	if summary == "" {
		return "", fmt.Errorf("llm returned empty summary")
	}

	// Validation for small models
	if !strings.HasPrefix(summary, "Fact:") {
		summary = "Fact: " + summary
	}

	// Detect and fix hallucinated repetition
	if strings.Count(summary, "Fact:") > 1 {
		r.logger.Warn("Multiple 'Fact:' prefixes detected, cleaning",
			zap.String("original", summary))
		parts := strings.Split(summary, "Fact:")
		for _, part := range parts {
			trimmed := strings.TrimSpace(part)
			if trimmed != "" {
				summary = "Fact: " + trimmed
				break
			}
		}
	}

	// Length guard
	if len(summary) > 500 {
		r.logger.Warn("Fact too long, truncating",
			zap.Int("length", len(summary)))
		summary = summary[:497] + "..."
	}

	// Word count check (small models sometimes ramble)
	words := strings.Fields(strings.TrimPrefix(summary, "Fact:"))
	if len(words) > 60 {
		r.logger.Warn("Fact exceeds word limit, truncating",
			zap.Int("word_count", len(words)))
		summary = "Fact: " + strings.Join(words[:60], " ") + "..."
	}

	return summary, nil
}

func (r *RAG) generateFallbackFact(metadata map[string]string, result string) string {
	// Extract key numbers from result with regex
	// Build template-based fact without LLM

	fact := "Fact: "
	if test := metadata["primary_test"]; test != "" {
		fact += strings.Title(test) + " "
	}

	// Extract first number (often a test statistic)
	if match := regexp.MustCompile(`[\d.]+`).FindString(result); match != "" {
		fact += "result=" + match
	}

	// Extract p-value
	if pval := metadata["p_value"]; pval != "" {
		fact += ", p=" + pval
	}

	if fact == "Fact: " {
		fact = "Fact: Analysis completed on " + metadata["dataset"]
	}

	return fact + "."
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

	start := time.Now()
	client := llmclient.NewWithTimeout(r.cfg, r.logger, r.cfg.SummarizationTimeout)
	summary, err := client.Chat(ctx, r.cfg.SummarizationLLMHost, messages)
	elapsed := time.Since(start)

	if err != nil {
		r.logger.Error("LLM chat call failed for searchable summary",
			zap.Error(err),
			zap.Duration("elapsed", elapsed),
			zap.Duration("timeout", r.cfg.SummarizationTimeout))
		return "", fmt.Errorf("llm chat call failed for searchable summary: %w", err)
	}

	if elapsed > r.cfg.SummarizationTimeout/2 {
		r.logger.Warn("Searchable summary generation took longer than expected",
			zap.Duration("elapsed", elapsed),
			zap.Duration("timeout", r.cfg.SummarizationTimeout))
	}

	if summary == "" {
		return "", fmt.Errorf("llm returned an empty searchable summary")
	}

	return strings.TrimSpace(summary), nil
}

func truncateResult(result string, maxLen int) string {
	// Handle errors specially
	if strings.Contains(result, "Error:") {
		return compressMiddle(result, maxLen, maxLen/3, maxLen/3)
	}

	// For normal results, take the first portion
	if len(result) <= maxLen {
		return result
	}

	// Try to break at a newline
	truncated := result[:maxLen]
	if lastNewline := strings.LastIndex(truncated, "\n"); lastNewline > maxLen/2 {
		return result[:lastNewline] + "\n[...]"
	}

	return truncated + "..."
}
