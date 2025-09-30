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
func ConvertToHTML(ctx context.Context, rawContent string) (string, error) {
	// Combined regex to find all custom tags
	tagPattern := `(?s)(<python>.*?</python>|<execution_results>.*?</execution_results>|<agent_status>.*?</agent_status>)`
	re := regexp.MustCompile(tagPattern)

	// Split content by tags
	parts := re.Split(rawContent, -1)
	matches := re.FindAllString(rawContent, -1)

	var finalHTML strings.Builder

	for i, part := range parts {
		// Process plain text with Markdown
		cleanedPart := strings.TrimSpace(part)
		if cleanedPart != "" {
			md := []byte(cleanedPart)
			html := markdown.ToHTML(md, nil, nil)
			finalHTML.WriteString(string(html))
		}

		// Render matching component
		if i < len(matches) {
			componentHTML, err := renderTagComponent(ctx, matches[i])
			if err != nil {
				return "", err
			}
			finalHTML.WriteString(componentHTML)
		}
	}

	return finalHTML.String(), nil
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
	} else if after, ok := strings.CutPrefix(taggedContent, ExecutionResultsTag.OpenTag); ok {
		result := strings.TrimSuffix(after, ExecutionResultsTag.CloseTag)
		if err := components.ExecutionResultBlock(result).Render(ctx, &buf); err != nil {
			return "", fmt.Errorf("failed to render execution result block: %w", err)
		}
	} else if after, ok := strings.CutPrefix(taggedContent, AgentStatusTag.OpenTag); ok {
		status := strings.TrimSuffix(after, AgentStatusTag.CloseTag)
		if err := components.AgentStatus(status).Render(ctx, &buf); err != nil {
			return "", fmt.Errorf("failed to render agent status: %w", err)
		}
	}

	return buf.String(), nil
}

// StreamTransform defines how a tag should be transformed for SSE streaming.
type StreamTransform struct {
	OpenReplace  string // Replacement for opening tag
	CloseReplace string // Replacement for closing tag
}

// GetStreamTransform returns the streaming transformation for a given tag.
// This is used by the stream service to convert XML tags to display format.
func GetStreamTransform(tag Tag) StreamTransform {
	switch tag.Name {
	case TagPython:
		return StreamTransform{
			OpenReplace:  "\n```python\n",
			CloseReplace: "\n```\n",
		}
	case TagExecutionResults:
		return StreamTransform{
			OpenReplace:  "\n```\n",
			CloseReplace: "\n```\n",
		}
	case TagAgentStatus:
		return StreamTransform{
			OpenReplace:  `<div class="agent-status-message">`,
			CloseReplace: `</div>`,
		}
	default:
		return StreamTransform{}
	}
}
