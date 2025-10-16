package agent

import (
	"context"
	"fmt"
	"regexp"
	"strings"

	"stats-agent/config"
	"stats-agent/rag"
	"stats-agent/web/format"
	"stats-agent/web/types"

	"go.uber.org/zap"
)

// QueryBuilder constructs structured RAG queries by combining fact summaries,
// metadata tokens, statistical values, error context, and synonym expansion.
type QueryBuilder struct {
	cfg    *config.Config
	rag    *rag.RAG
	logger *zap.Logger
}

// NewQueryBuilder creates a new query builder instance.
func NewQueryBuilder(cfg *config.Config, ragInstance *rag.RAG, logger *zap.Logger) *QueryBuilder {
	return &QueryBuilder{
		cfg:    cfg,
		rag:    ragInstance,
		logger: logger,
	}
}

// BuildRAGQuery constructs a structured query for RAG retrieval based on conversation state.
//
// For turn 0 (first turn), returns only the user input.
// For turn 1+, builds a multi-component query with:
//   1. Last fact summary anchor (1 sentence from most recent assistant+tool pair)
//   2. Metadata filter tokens (dataset:X primary_test:Y analysis_stage:Z vars:A,B)
//   3. Statistical values anchor (p=0.031 t=2.2 d=0.45)
//   4. Error context (error:<first exception line> if applicable)
//   5. Lightweight synonym expansion (top 1-2 statistical terms)
//   6. User message (always last)
func (qb *QueryBuilder) BuildRAGQuery(
	ctx context.Context,
	userInput string,
	sessionID string,
	history []types.AgentMessage,
	turn int,
) string {
	// First turn: use only user input (no prior context to leverage)
	if turn == 0 {
		return userInput
	}

	var parts []string

	// 1. Last fact summary anchor - provides 1-sentence context of most recent analysis
	if factSummary := qb.getLastFactSummary(history); factSummary != "" {
		parts = append(parts, fmt.Sprintf("[fact] %s", factSummary))
	}

	// 2. Metadata filter tokens - structured key:value pairs for exact matching
	if metadata := qb.extractLastFactMetadata(history); len(metadata) > 0 {
		if metadataStr := qb.formatMetadataTokens(metadata); metadataStr != "" {
			parts = append(parts, metadataStr)
		}
	}

	// 3. Statistical values anchor - hard numbers for fact matching
	if statValues := qb.extractStatisticalValues(history); statValues != "" {
		parts = append(parts, statValues)
	}

	// 4. Error context - helps retrieve relevant debugging information
	if errorCtx := qb.extractErrorContext(history); errorCtx != "" {
		parts = append(parts, errorCtx)
	}

	// 5. Lightweight synonym expansion - top 1-2 statistical terms from user input
	if synonyms := qb.expandKeyTerms(userInput, 2); synonyms != "" {
		parts = append(parts, synonyms)
	}

	// 6. User message - always preserved at the end
	parts = append(parts, userInput)

	query := strings.Join(parts, " ")

	qb.logger.Debug("Built structured RAG query",
		zap.Int("turn", turn),
		zap.Int("components", len(parts)),
		zap.String("query", query))

	return query
}

// getLastFactSummary extracts the embedded summary from the most recent assistant+tool pair.
// Facts are stored with summaries embedded inline (e.g., "Fact: description [dataset:X primary_test:Y]").
// We extract just the descriptive sentence without metadata tags for semantic retrieval.
func (qb *QueryBuilder) getLastFactSummary(history []types.AgentMessage) string {
	// Search backwards for most recent assistant+tool pair
	for i := len(history) - 1; i >= 0; i-- {
		if history[i].Role == "assistant" && i+1 < len(history) && history[i+1].Role == "tool" {
			assistantContent := history[i].Content
			toolContent := history[i+1].Content

			// Skip if this is an error case (tool output starts with "Error:")
			if strings.HasPrefix(strings.TrimSpace(toolContent), "Error:") {
				continue
			}

			// Extract fact summary from assistant message
			// Facts typically contain code blocks - we want the LLM's description, not the code
			if format.HasCodeBlock(assistantContent) {
				// Remove code blocks to get just the commentary
				cleaned := qb.removeCodeBlocks(assistantContent)
				cleaned = strings.TrimSpace(cleaned)

				// Take first sentence as summary (up to first period, question mark, or exclamation)
				summary := qb.extractFirstSentence(cleaned)
				if summary != "" && len(summary) > 20 {
					return summary
				}
			}

			break // Only check most recent pair
		}
	}

	return ""
}

// extractLastFactMetadata parses statistical metadata from the most recent fact in history.
// This metadata is embedded inline by the fact generation process.
func (qb *QueryBuilder) extractLastFactMetadata(history []types.AgentMessage) map[string]string {
	// Search backwards for most recent assistant+tool pair
	for i := len(history) - 1; i >= 0; i-- {
		if history[i].Role == "assistant" && i+1 < len(history) && history[i+1].Role == "tool" {
			assistantContent := history[i].Content
			toolContent := history[i+1].Content

			// Skip error cases
			if strings.HasPrefix(strings.TrimSpace(toolContent), "Error:") {
				continue
			}

			// Extract code and result for metadata extraction
			if format.HasCodeBlock(assistantContent) {
				code, _ := format.ExtractCodeContent(assistantContent)
				metadata := rag.ExtractStatisticalMetadata(code, toolContent)
				return metadata
			}

			break // Only check most recent pair
		}
	}

	return nil
}

// formatMetadataTokens converts metadata map to structured token format.
// Output format: "dataset:patients.csv primary_test:t-test analysis_stage:hypothesis_test vars:age,group"
func (qb *QueryBuilder) formatMetadataTokens(metadata map[string]string) string {
	if len(metadata) == 0 {
		return ""
	}

	var tokens []string

	// Order matters: most important filters first
	if dataset := metadata["dataset"]; dataset != "" {
		tokens = append(tokens, fmt.Sprintf("dataset:%s", dataset))
	}

	if primaryTest := metadata["primary_test"]; primaryTest != "" {
		tokens = append(tokens, fmt.Sprintf("primary_test:%s", primaryTest))
	}

	if analysisStage := metadata["analysis_stage"]; analysisStage != "" {
		tokens = append(tokens, fmt.Sprintf("analysis_stage:%s", analysisStage))
	}

	if vars := metadata["variables"]; vars != "" {
		tokens = append(tokens, fmt.Sprintf("vars:%s", vars))
	}

	return strings.Join(tokens, " ")
}

// extractStatisticalValues parses numerical values (p-values, test statistics, effect sizes)
// from the most recent tool output.
func (qb *QueryBuilder) extractStatisticalValues(history []types.AgentMessage) string {
	// Search backwards for most recent tool message (execution result)
	for i := len(history) - 1; i >= 0; i-- {
		if history[i].Role == "tool" {
			result := history[i].Content

			// Skip error outputs
			if strings.HasPrefix(strings.TrimSpace(result), "Error:") {
				continue
			}

			var values []string

			// Extract p-value
			pValueRegex := regexp.MustCompile(`(?i)p\s*[=:]\s*([\d.]+(?:e-?\d+)?)`)
			if matches := pValueRegex.FindStringSubmatch(result); len(matches) > 1 {
				values = append(values, fmt.Sprintf("p=%s", matches[1]))
			}

			// Extract test statistic (t, F, chi2, z, r, etc.)
			testStatPatterns := map[string]*regexp.Regexp{
				"t":    regexp.MustCompile(`(?i)(?:^|\s)t\s*[=:]\s*([-\d.]+)`),
				"F":    regexp.MustCompile(`(?i)(?:^|\s)F\s*[=:]\s*([\d.]+)`),
				"chi2": regexp.MustCompile(`(?i)chi2?\s*[=:]\s*([\d.]+)`),
				"z":    regexp.MustCompile(`(?i)(?:^|\s)z\s*[=:]\s*([-\d.]+)`),
				"r":    regexp.MustCompile(`(?i)(?:^|\s)r\s*[=:]\s*([-\d.]+)`),
			}

			for statName, pattern := range testStatPatterns {
				if matches := pattern.FindStringSubmatch(result); len(matches) > 1 {
					values = append(values, fmt.Sprintf("%s=%s", statName, matches[1]))
					break // Only include first test statistic found
				}
			}

			// Extract effect size (Cohen's d, eta squared, etc.)
			effectPatterns := map[string]*regexp.Regexp{
				"d":  regexp.MustCompile(`(?i)Cohen'?s?\s*d\s*[=:]\s*([-\d.]+)`),
				"η²": regexp.MustCompile(`(?i)eta\^?2\s*[=:]\s*([\d.]+)`),
			}

			for effectName, pattern := range effectPatterns {
				if matches := pattern.FindStringSubmatch(result); len(matches) > 1 {
					values = append(values, fmt.Sprintf("%s=%s", effectName, matches[1]))
					break // Only include first effect size found
				}
			}

			if len(values) > 0 {
				return strings.Join(values, " ")
			}

			break // Only check most recent tool output
		}
	}

	return ""
}

// extractErrorContext detects error messages in recent tool outputs and formats them.
// Returns "error:<first exception line>" for targeted error retrieval.
func (qb *QueryBuilder) extractErrorContext(history []types.AgentMessage) string {
	// Check last 2 tool messages for errors
	toolCount := 0
	for i := len(history) - 1; i >= 0 && toolCount < 2; i-- {
		if history[i].Role == "tool" {
			toolCount++
			content := strings.TrimSpace(history[i].Content)

			if strings.HasPrefix(content, "Error:") {
				// Extract first line of error message
				lines := strings.Split(content, "\n")
				if len(lines) > 0 {
					errorLine := strings.TrimSpace(strings.TrimPrefix(lines[0], "Error:"))
					// Truncate if very long
					if len(errorLine) > 100 {
						errorLine = errorLine[:100]
					}
					return fmt.Sprintf("error:%s", errorLine)
				}
			}
		}
	}

	return ""
}

// expandKeyTerms identifies top N statistical terms in the user input and adds compact synonyms.
// Uses the existing synonym map from rag/query_expand.go.
// Output format: "t-test (ttest OR \"Student's t\")"
func (qb *QueryBuilder) expandKeyTerms(userInput string, maxTerms int) string {
	lower := strings.ToLower(userInput)

	// Statistical terms to detect (from query_expand.go synonym map keys)
	// We focus on the most common and important terms to avoid noise
	priorityTerms := []string{
		"t-test", "anova", "chi-square", "correlation", "regression",
		"normality", "mann-whitney", "wilcoxon", "kruskal-wallis",
		"pearson", "spearman", "logistic regression", "linear regression",
		"p-value", "significant", "effect size",
	}

	// Map of terms to their compact synonym lists (abbreviated from full map)
	compactSynonyms := map[string][]string{
		"t-test":             {"ttest", "Student's t"},
		"anova":              {"analysis of variance", "F-test"},
		"chi-square":         {"chi2", "χ²"},
		"correlation":        {"relationship", "r value"},
		"regression":         {"linear model"},
		"normality":          {"normal distribution", "Gaussian"},
		"mann-whitney":       {"mann whitney u", "wilcoxon rank sum"},
		"wilcoxon":           {"wilcoxon signed rank"},
		"kruskal-wallis":     {"kruskal wallis", "nonparametric anova"},
		"pearson":            {"pearson r", "pearson correlation"},
		"spearman":           {"spearman rho", "rank correlation"},
		"logistic regression":{"logit", "binary regression"},
		"linear regression":  {"ols", "least squares"},
		"p-value":            {"p value", "significance level"},
		"significant":        {"sig", "p<0.05"},
		"effect size":        {"Cohen's d", "magnitude"},
	}

	// Find matching terms in user input
	var foundTerms []string
	for _, term := range priorityTerms {
		if strings.Contains(lower, term) {
			foundTerms = append(foundTerms, term)
			if len(foundTerms) >= maxTerms {
				break
			}
		}
	}

	if len(foundTerms) == 0 {
		return ""
	}

	// Build compact synonym expansion
	var expansions []string
	for _, term := range foundTerms {
		if synonyms, ok := compactSynonyms[term]; ok && len(synonyms) > 0 {
			// Format: "term (syn1 OR syn2)"
			orList := strings.Join(synonyms, " OR ")
			expansion := fmt.Sprintf("%s (%s)", term, orList)
			expansions = append(expansions, expansion)
		}
	}

	return strings.Join(expansions, " ")
}

// removeCodeBlocks strips markdown code fences from text.
func (qb *QueryBuilder) removeCodeBlocks(text string) string {
	// Remove ```python ... ``` blocks
	codeBlockRegex := regexp.MustCompile("(?s)" + "```[a-z]*\\n.*?\\n```")
	return codeBlockRegex.ReplaceAllString(text, "")
}

// extractFirstSentence returns the first sentence from text (up to . ! or ?).
func (qb *QueryBuilder) extractFirstSentence(text string) string {
	text = strings.TrimSpace(text)
	if text == "" {
		return ""
	}

	// Find first sentence boundary
	sentenceEnders := []string{". ", "! ", "? ", ".\n", "!\n", "?\n"}
	minIndex := len(text)

	for _, ender := range sentenceEnders {
		if idx := strings.Index(text, ender); idx != -1 && idx < minIndex {
			minIndex = idx + 1 // Include the punctuation
		}
	}

	if minIndex < len(text) {
		return strings.TrimSpace(text[:minIndex])
	}

	// No sentence boundary found, return full text (truncated if too long)
	if len(text) > 200 {
		return text[:200]
	}

	return text
}
