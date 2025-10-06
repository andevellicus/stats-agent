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

	// Updated prompt: Removed the "Fact:" prefix requirement.
	systemPrompt := `Extract 1-3 key findings from the conversation history relevant to the user's question.

Rules:
- Each finding must be a single, concise sentence.
- Include numbers, variable names, and test results.
- Provide NO explanations or commentary.
- If no relevant facts exist, state: "No prior relevant analysis was found."

Example:
The dataset has 150 rows with columns for Age, Gender, and Score.
An independent t-test found a significant difference between groups (p=0.023, Cohen's d=0.42).`

	userPrompt := fmt.Sprintf(`Question: %s

History:
%s

Extract 1-3 relevant findings (one sentence each):`, latestUserMessage, context)

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
	// Add other key values from metadata
	if pval := metadata["p_value"]; pval != "" {
		context = append(context, "p-value: "+pval)
	}
	if stat := metadata["test_statistic"]; stat != "" {
		context = append(context, "Test Statistic: "+stat)
	}
	if es := metadata["effect_size"]; es != "" {
		context = append(context, "Effect Size: "+es)
	}
	if sig, ok := metadata["sig_at_05"]; ok {
		context = append(context, fmt.Sprintf("Significant at p<0.05: %s", sig))
	}

	contextStr := ""
	if len(context) > 0 {
		contextStr = "\nContext: " + strings.Join(context, ", ")
	}

	//resultPreview := truncateResult(result, 800)

	systemPrompt := `Create a one-sentence fact from analysis output. Include all numbers and variable names.

Format: "Fact: [concise finding with numbers]"

Good: "Fact: Shapiro-Wilk test on residuals showed W=0.923, p=0.016, violating normality."
Good: "Fact: Dataset has 12 columns matching 'age'; Days_relative_to_drainage mean=-28.2, SD=25.1."

Extract the finding:`

	userPrompt := fmt.Sprintf(`%s

Output:
%s

Fact:`, contextStr, result)

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
		overageChars := overageTokens * 4                 // rough estimate
		safeResultLen := len(result) - overageChars - 100 // safety margin

		if safeResultLen > 200 {
			result = truncateResult(result, safeResultLen)
			userPrompt = fmt.Sprintf(`**Analysis Context:**
%s

**Tool Output to Summarize (for reference):**
%s

**Your Summary:**
`, contextStr, result)
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

	/*
		// Length guard
		if len(summary) > 1000 {
			r.logger.Warn("Fact too long, truncating",
				zap.Int("length", len(summary)))
			summary = summary[:997] + "..."
		}

		// Word count check (small models sometimes ramble)
		words := strings.Fields(summary)
		if len(words) > 100 {
			r.logger.Warn("Fact exceeds word limit, truncating",
				zap.Int("word_count", len(words)))
			summary = strings.Join(words[:100], " ") + "..."
		}
	*/

	return summary, nil
}

func (r *RAG) generateFallbackFact(metadata map[string]string, result string) string {
	// Build template-based fact without LLM and without "Fact:" prefix
	fact := ""
	if test := metadata["primary_test"]; test != "" {
		fact += strings.Title(strings.ReplaceAll(test, "-", " ")) + " "
	}

	// Extract first number (often a test statistic)
	if match := regexp.MustCompile(`[\d.]+`).FindString(result); match != "" {
		fact += "resulted in " + match
	}

	// Extract p-value
	if pval := metadata["p_value"]; pval != "" {
		fact += " with a p-value of " + pval
	}

	if fact == "" {
		fact = "An analysis was completed on the " + metadata["dataset"] + " dataset"
	}

	return fact + "."
}

func truncateResult(result string, maxLen int) string {
	// Handle errors specially
	if strings.Contains(result, "Error:") {
		// Preserve more from the beginning and end of errors
		return compressMiddle(result, maxLen, maxLen/2, maxLen/3)
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
