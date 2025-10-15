package rag

import (
	"context"
	"fmt"
	"strings"

	"stats-agent/llmclient"
	"stats-agent/prompts"
	"stats-agent/web/types"
)

func buildMetadataContext(metadata map[string]string) string {
	if len(metadata) == 0 {
		return ""
	}

	var parts []string

	// Build a context block that provides the metadata to the LLM without instructing it to copy
	if test := metadata["primary_test"]; test != "" {
		parts = append(parts, fmt.Sprintf("Test detected: %s", test))
	}
	if metadata["sig_at_05"] == "true" {
		parts = append(parts, "Result significant at α=0.05")
	} else if metadata["sig_at_05"] == "false" {
		parts = append(parts, "Result not significant at α=0.05")
	}
	if metadata["sig_at_01"] == "true" {
		parts = append(parts, "Result significant at α=0.01")
	}
	if stage := metadata["analysis_stage"]; stage != "" {
		parts = append(parts, fmt.Sprintf("Analysis stage: %s", stage))
	}
	if vars := metadata["variables"]; vars != "" {
		parts = append(parts, fmt.Sprintf("Variables: %s", vars))
	}
	if dataset := metadata["dataset"]; dataset != "" {
		parts = append(parts, fmt.Sprintf("Dataset: %s", dataset))
	}

	if len(parts) == 0 {
		return ""
	}

	return "Extracted metadata:\n" + strings.Join(parts, "\n")
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

	// Wrap the summary in memory tags
	return fmt.Sprintf("<memory>\n%s\n</memory>", summary), nil
}

func (r *RAG) generateFactSummary(ctx context.Context, code, result string, metadata map[string]string) (string, error) {
	finalResult := result
	if strings.Contains(result, "Error:") {
		finalResult = compressMiddle(result, 800, 200, 200)
	}

	systemPrompt := prompts.FactSummary()

	// Build metadata context (provides info to LLM without instructing it to copy)
	metadataContext := buildMetadataContext(metadata)

	// Build user prompt with code, output, and metadata context
	var userPrompt strings.Builder
	if metadataContext != "" {
		userPrompt.WriteString(metadataContext)
		userPrompt.WriteString("\n\n")
	}

	userPrompt.WriteString("Code:\n")
	userPrompt.WriteString(code)
	userPrompt.WriteString("\n\nOutput:\n")
	userPrompt.WriteString(finalResult)
	userPrompt.WriteString("\n\nExtract the fact following the rules and examples above. Respond with only the fact, ending with metadata tags in square brackets.")

	messages := []types.AgentMessage{
		{Role: "system", Content: systemPrompt},
		{Role: "user", Content: userPrompt.String()},
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
