package format

import (
	"bytes"
	"context"
	"fmt"
	"regexp"
	"stats-agent/web/templates/components"
	"strings"

	"github.com/gomarkdown/markdown"
)

// ConvertToHTML converts text with XML tags to HTML for storage.
// It processes markdown text and renders custom XML tags as HTML components.
// This function is ONLY called when saving to the database, NOT during streaming.
func ConvertToHTML(ctx context.Context, rawContent string) (string, error) {
	// Combined regex to find all custom tags
	tagPattern := `(?s)(<python>.*?</python>|<agent_status>.*?</agent_status>)`
	re := regexp.MustCompile(tagPattern)

	// Step 1: Find all custom tags and their positions
	type tagInfo struct {
		match    string
		startIdx int
		endIdx   int
		id       string
	}

	var tags []tagInfo
	matches := re.FindAllStringIndex(rawContent, -1)
	for i, loc := range matches {
		tags = append(tags, tagInfo{
			match:    rawContent[loc[0]:loc[1]],
			startIdx: loc[0],
			endIdx:   loc[1],
			id:       fmt.Sprintf("{{COMPONENT_%d}}", i),
		})
	}

	// Step 2: Replace custom tags with unique placeholder IDs
	// Wrap placeholders in newlines to ensure they're treated as block elements
	textWithPlaceholders := rawContent
	for i := len(tags) - 1; i >= 0; i-- { // Reverse order to preserve string indices
		tag := tags[i]
		// Ensure placeholders are on their own lines to maintain markdown structure
		placeholder := "\n\n" + tag.id + "\n\n"
		textWithPlaceholders = textWithPlaceholders[:tag.startIdx] + placeholder + textWithPlaceholders[tag.endIdx:]
	}

	// Step 3: Normalize markdown - ensure lists have blank lines before them
	// This fixes cases where the LLM outputs "**Text:**\n- Item" without a blank line
	textWithPlaceholders = normalizeMarkdownLists(textWithPlaceholders)

	// Step 4: Process entire text as one markdown block (this preserves list structures!)
	md := []byte(textWithPlaceholders)
	html := string(markdown.ToHTML(md, nil, nil))

	// Step 5: Replace placeholder IDs with rendered HTML components
	for _, tag := range tags {
		componentHTML, err := renderTagComponent(ctx, tag.match)
		if err != nil {
			return "", err
		}

		// The markdown processor will wrap the placeholder in <p> tags
		// We need to replace the entire <p> block with our component
		patterns := []string{
			"<p>" + tag.id + "</p>\n",
			"<p>" + tag.id + "</p>",
			tag.id,
		}

		replaced := false
		for _, pattern := range patterns {
			if strings.Contains(html, pattern) {
				html = strings.ReplaceAll(html, pattern, componentHTML)
				replaced = true
				break
			}
		}

		if !replaced {
			// Fallback: just replace the ID directly
			html = strings.ReplaceAll(html, tag.id, componentHTML)
		}
	}

	return html, nil
}

// normalizeMarkdownLists ensures list items have proper spacing for markdown parsing.
// Markdown requires a blank line before lists, but LLMs often forget this.
func normalizeMarkdownLists(text string) string {
	lines := strings.Split(text, "\n")
	var result []string

	for i := 0; i < len(lines); i++ {
		line := lines[i]
		trimmed := strings.TrimSpace(line)

		// Check if this line starts a list (-, *, +, or numbered)
		isListItem := strings.HasPrefix(trimmed, "- ") ||
			strings.HasPrefix(trimmed, "* ") ||
			strings.HasPrefix(trimmed, "+ ") ||
			regexp.MustCompile(`^\d+\.\s`).MatchString(trimmed)

		// If this is a list item and previous line is not blank/list/placeholder
		if isListItem && i > 0 {
			prevLine := strings.TrimSpace(lines[i-1])
			prevIsListItem := strings.HasPrefix(prevLine, "- ") ||
				strings.HasPrefix(prevLine, "* ") ||
				strings.HasPrefix(prevLine, "+ ") ||
				regexp.MustCompile(`^\d+\.\s`).MatchString(prevLine)
			prevIsPlaceholder := strings.Contains(prevLine, "{{COMPONENT_")

			// Add blank line before list if previous line is text (not blank/list/placeholder)
			if prevLine != "" && !prevIsListItem && !prevIsPlaceholder {
				result = append(result, "")
			}
		}

		result = append(result, line)
	}

	return strings.Join(result, "\n")
}

// renderTagComponent renders a tagged content section as an HTML component.
func renderTagComponent(ctx context.Context, taggedContent string) (string, error) {
	var buf bytes.Buffer

	// Determine which tag and render appropriate component
	if after, ok := strings.CutPrefix(taggedContent, PythonTag.OpenTag); ok {
		code := strings.TrimSuffix(after, PythonTag.CloseTag)
		if err := components.PythonCodeBlock(code).Render(ctx, &buf); err != nil {
			return "", fmt.Errorf("failed to render python block: %w", err)
		}
	} else if after, ok := strings.CutPrefix(taggedContent, ToolTag.OpenTag); ok {
		result := strings.TrimSuffix(after, ToolTag.CloseTag)
		if err := components.ExecutionResultBlock(result).Render(ctx, &buf); err != nil {
			return "", fmt.Errorf("failed to render tool block: %w", err)
		}
	} else if after, ok := strings.CutPrefix(taggedContent, AgentStatusTag.OpenTag); ok {
		status := strings.TrimSuffix(after, AgentStatusTag.CloseTag)
		if err := components.AgentStatus(status).Render(ctx, &buf); err != nil {
			return "", fmt.Errorf("failed to render agent status: %w", err)
		}
	}

	return buf.String(), nil
}
