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
	// Try with detailed prompt first
	summary, err := r.generateFactWithPrompt(ctx, result, metadata, false)
	if err != nil {
		return "", err
	}

	// Validate the generated fact
	if !r.validateFactCoherence(ctx, summary, metadata) {
		r.logger.Info("First fact attempt failed validation, retrying with simplified prompt",
			zap.String("failed_fact", summary))

		// Retry with simplified prompt
		simpleSummary, retryErr := r.generateFactWithPrompt(ctx, result, metadata, true)
		if retryErr != nil {
			// If retry fails, use deterministic fallback
			r.logger.Warn("Retry also failed, using deterministic fallback",
				zap.Error(retryErr))
			return r.generateDeterministicFact(metadata), nil
		}

		// Validate retry attempt
		if !r.validateFactCoherence(ctx, simpleSummary, metadata) {
			r.logger.Warn("Retry fact also failed validation, using deterministic fallback",
				zap.String("retry_fact", simpleSummary))
			return r.generateDeterministicFact(metadata), nil
		}

		summary = simpleSummary
	}

	return summary, nil
}

// generateFactWithPrompt creates a fact summary using either detailed or simplified prompt
func (r *RAG) generateFactWithPrompt(ctx context.Context, result string, metadata map[string]string, simplified bool) (string, error) {
	// Build structured context from metadata - PRIMARY SOURCE OF TRUTH
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
		contextStr = strings.Join(context, ", ")
	}

	var systemPrompt, userPrompt string

	if simplified {
		// Ultra-simple prompt for retry
		systemPrompt = `Create a one-sentence statistical summary. Include ALL items from the data list below.`

		userPrompt = fmt.Sprintf(`Data: %s

Write one sentence that mentions these items:`, contextStr)
	} else {
		// Detailed prompt
		systemPrompt = `You are a statistical fact generator. Create a single-sentence summary that accurately represents the analysis performed.

CRITICAL RULES:
1. ALWAYS include the dataset name, variables, and test name if provided below
2. Use ONLY numbers from the "Required Data" section - NEVER approximate or invent
3. Format as ONE complete sentence without "Fact:" prefix
4. If a value is in Required Data, you MUST mention it in your summary

Examples:
- Good: "Shapiro-Wilk test on age in patients.csv showed W=0.923, p=0.016"
- Good: "Descriptive statistics for score variable in study.csv showed mean=45.2, SD=12.3"
- Bad: "No age variable found" (when age IS in Required Data)
- Bad: "The test showed results" (missing specifics)

Your summary MUST be consistent with the Required Data below.`

		userPrompt = fmt.Sprintf(`REQUIRED DATA (must appear in your summary):
%s

ADDITIONAL CONTEXT (tool output for reference):
%s

Create a one-sentence summary that includes the dataset, variables, and test from Required Data:`, contextStr, result)
	}

	// Check token count BEFORE sending
	fullPrompt := systemPrompt + "\n\n" + userPrompt
	tokenCount, err := r.countTokensForEmbedding(ctx, fullPrompt)

	// Leave 500 tokens for generation, so max input = 3500 tokens
	const maxInputTokens = 3500

	if err == nil && tokenCount > maxInputTokens {
		r.logger.Warn("Prompt too large, truncating result",
			zap.Int("tokens", tokenCount),
			zap.Int("max", maxInputTokens),
			zap.Bool("simplified", simplified))

		// Calculate safe result length
		overageTokens := tokenCount - maxInputTokens
		overageChars := overageTokens * 4                 // rough estimate
		safeResultLen := len(result) - overageChars - 100 // safety margin

		if safeResultLen > 200 {
			result = truncateResult(result, safeResultLen)
			if simplified {
				userPrompt = fmt.Sprintf(`Data: %s

Write one sentence that mentions these items:`, contextStr)
			} else {
				userPrompt = fmt.Sprintf(`REQUIRED DATA (must appear in your summary):
%s

ADDITIONAL CONTEXT (tool output for reference):
%s

Create a one-sentence summary that includes the dataset, variables, and test from Required Data:`, contextStr, result)
			}
		} else {
			// Result is fundamentally too large, use fallback
			return r.generateDeterministicFact(metadata), nil
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

// validateFactCoherence checks if the generated fact is semantically consistent with the metadata.
// It uses the main LLM to verify the fact doesn't contradict the provided data.
func (r *RAG) validateFactCoherence(ctx context.Context, fact string, metadata map[string]string) bool {
	// Build metadata summary
	var metaParts []string
	if test := metadata["primary_test"]; test != "" {
		metaParts = append(metaParts, "test="+test)
	}
	if vars := metadata["variables"]; vars != "" {
		metaParts = append(metaParts, "variables="+vars)
	}
	if dataset := metadata["dataset"]; dataset != "" {
		metaParts = append(metaParts, "dataset="+dataset)
	}

	if len(metaParts) == 0 {
		// No metadata to validate against
		return true
	}

	metaSummary := strings.Join(metaParts, ", ")

	systemPrompt := `You are a fact checker. Determine if a statistical summary is consistent with the provided metadata.

Answer ONLY with "Yes" or "No".

Inconsistent examples:
- Fact: "No age variable found" / Metadata: variables=age → Answer: No
- Fact: "Test on patients.csv" / Metadata: dataset=study.csv → Answer: No

Consistent examples:
- Fact: "Shapiro-Wilk test on age" / Metadata: test=shapiro-wilk, variables=age → Answer: Yes
- Fact: "Descriptive stats for score in data.csv" / Metadata: variables=score, dataset=data.csv → Answer: Yes`

	userPrompt := fmt.Sprintf(`Fact: "%s"
Metadata: %s

Is the fact consistent with the metadata? Answer Yes or No:`, fact, metaSummary)

	messages := []types.AgentMessage{
		{Role: "system", Content: systemPrompt},
		{Role: "user", Content: userPrompt},
	}

	// Use main LLM for validation with short timeout
	validationCtx, cancel := context.WithTimeout(ctx, 10*time.Second)
	defer cancel()

	response, err := llmclient.New(r.cfg, r.logger).Chat(validationCtx, r.cfg.MainLLMHost, messages)
	if err != nil {
		r.logger.Warn("Fact validation failed, assuming valid",
			zap.Error(err),
			zap.String("fact", fact))
		return true // Assume valid if validation fails
	}

	response = strings.TrimSpace(strings.ToLower(response))
	isValid := strings.Contains(response, "yes")

	if !isValid {
		r.logger.Info("Fact validation failed semantic check",
			zap.String("fact", fact),
			zap.String("metadata", metaSummary),
			zap.String("validation_response", response))
	}

	return isValid
}

// generateDeterministicFact creates a fact using only metadata without LLM.
// This is used for routine operations like assumption checks and descriptive stats
// where structured data is sufficient and hallucination risk must be eliminated.
func (r *RAG) generateDeterministicFact(metadata map[string]string) string {
	var parts []string

	// Start with test name
	if test := metadata["primary_test"]; test != "" {
		testName := strings.Title(strings.ReplaceAll(test, "-", " "))
		parts = append(parts, testName)
	}

	// Add variables if present
	if vars := metadata["variables"]; vars != "" {
		parts = append(parts, "on "+vars)
	}

	// Add test statistic
	if stat := metadata["test_statistic"]; stat != "" {
		parts = append(parts, "resulted in "+stat)
	}

	// Add p-value with significance context
	if pval := metadata["p_value"]; pval != "" {
		sigText := "p=" + pval
		if sig := metadata["sig_at_05"]; sig == "true" {
			sigText += " (significant at α=0.05)"
		} else if sig == "false" {
			sigText += " (not significant)"
		}
		parts = append(parts, sigText)
	}

	// Add effect size if present
	if es := metadata["effect_size"]; es != "" {
		parts = append(parts, "with "+es)
	}

	// Add dataset context if no other info
	if len(parts) == 0 {
		if dataset := metadata["dataset"]; dataset != "" {
			parts = append(parts, "Analysis completed on "+dataset)
		} else {
			parts = append(parts, "Statistical analysis completed")
		}
	}

	return strings.Join(parts, " ") + "."
}

// normalizeNumber converts a number string to a normalized form for comparison.
// This treats "15", "15.0", and "15.00" as equivalent.
func normalizeNumber(num string) string {
	// Parse as float to get canonical representation
	var f float64
	if _, err := fmt.Sscanf(num, "%f", &f); err != nil {
		return num // If parsing fails, return original
	}
	// Format with enough precision to preserve scientific notation and decimals
	// but eliminate trailing zeros
	normalized := strings.TrimRight(strings.TrimRight(fmt.Sprintf("%.10f", f), "0"), ".")
	return normalized
}

// verifyNumericAccuracy checks if all numbers in the LLM-generated fact
// exist in either the original metadata or the tool output. Returns true if verification passes.
func (r *RAG) verifyNumericAccuracy(fact string, metadata map[string]string, toolOutput string) bool {
	// Extract all numbers from the fact (including decimals and scientific notation)
	// Use word boundaries to avoid capturing sentence-ending punctuation
	numberPattern := regexp.MustCompile(`\b\d+\.?\d*(?:e[+-]?\d+)?\b`)
	factNumbers := numberPattern.FindAllString(fact, -1)

	if len(factNumbers) == 0 {
		// No numbers to verify
		return true
	}

	// Build a set of all normalized numbers from metadata AND tool output
	validNumbers := make(map[string]bool)

	// Add numbers from metadata
	for _, value := range metadata {
		nums := numberPattern.FindAllString(value, -1)
		for _, num := range nums {
			validNumbers[normalizeNumber(num)] = true
		}
	}

	// Add numbers from tool output (the actual source the LLM reads!)
	if toolOutput != "" {
		nums := numberPattern.FindAllString(toolOutput, -1)
		for _, num := range nums {
			validNumbers[normalizeNumber(num)] = true
		}
	}

	// Check each number in the fact (normalized)
	for _, num := range factNumbers {
		if !validNumbers[normalizeNumber(num)] {
			r.logger.Warn("Numeric hallucination detected in LLM fact",
				zap.String("hallucinated_number", num),
				zap.String("fact", fact))
			return false
		}
	}

	return true
}

// generateStructuredContentForBM25 creates normalized metadata text for BM25 keyword search.
// This provides exact-match capabilities for statistical values, test names, and variables.
//
// Example output: "test:shapiro-wilk stage:assumption_check p:0.016 stat:W=0.923 sig:true vars:residuals dataset:data.csv"
func (r *RAG) generateStructuredContentForBM25(metadata map[string]string) string {
	if len(metadata) == 0 {
		return ""
	}

	var parts []string

	// Priority order: most searchable fields first
	searchableFields := []string{
		"primary_test",
		"analysis_stage",
		"p_value",
		"test_statistic",
		"effect_size",
		"variables",
		"dataset",
		"sig_at_05",
		"sig_at_01",
		"sample_size",
	}

	// Add high-priority fields with normalized keys
	for _, field := range searchableFields {
		if value, ok := metadata[field]; ok && strings.TrimSpace(value) != "" {
			normalizedKey := normalizeFieldKeyForBM25(field)
			normalizedValue := normalizeFieldValueForBM25(value)
			parts = append(parts, fmt.Sprintf("%s:%s", normalizedKey, normalizedValue))
		}
	}

	// Add any remaining metadata fields not in the priority list
	for key, value := range metadata {
		// Skip if already processed or if it's an internal field
		if contains(searchableFields, key) || strings.HasPrefix(key, "_") {
			continue
		}
		if value != "" {
			normalizedKey := normalizeFieldKeyForBM25(key)
			normalizedValue := normalizeFieldValueForBM25(value)
			parts = append(parts, fmt.Sprintf("%s:%s", normalizedKey, normalizedValue))
		}
	}

	return strings.Join(parts, " ")
}

// normalizeFieldKeyForBM25 converts metadata keys to short, searchable forms
func normalizeFieldKeyForBM25(key string) string {
	// Map long field names to short abbreviations for more compact BM25 text
	keyMap := map[string]string{
		"primary_test":   "test",
		"analysis_stage": "stage",
		"test_statistic": "stat",
		"effect_size":    "effect",
		"p_value":        "p",
		"sample_size":    "n",
		"sig_at_05":      "sig",
		"sig_at_01":      "sig01",
		"sig_at_001":     "sig001",
		"variable_count": "nvars",
	}

	if short, ok := keyMap[key]; ok {
		return short
	}

	// Remove underscores and convert to lowercase
	return strings.ToLower(strings.ReplaceAll(key, "_", ""))
}

// normalizeFieldValueForBM25 cleans values for better BM25 matching
func normalizeFieldValueForBM25(value string) string {
	// Keep alphanumeric, dots, hyphens, and equals signs
	// Remove parentheses, quotes, and other noise
	value = strings.TrimSpace(value)

	// Replace spaces with hyphens for multi-word values
	value = strings.ReplaceAll(value, " ", "-")

	// Remove quotes
	value = strings.ReplaceAll(value, "\"", "")
	value = strings.ReplaceAll(value, "'", "")

	// Keep commas for variable lists, but remove other punctuation
	// This preserves "age,gender,score" while removing other noise

	return value
}

// contains checks if a slice contains a string
func contains(slice []string, item string) bool {
	for _, s := range slice {
		if s == item {
			return true
		}
	}
	return false
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
