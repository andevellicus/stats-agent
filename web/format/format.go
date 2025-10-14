package format

import (
	"strings"
)

// Tag definitions - single source of truth for all custom XML tags
const (
	TagTool        = "tool"
	TagAgentStatus = "agent_status"
)

// Tag represents a custom XML-like tag used in the application.
type Tag struct {
	Name        string // Internal name
	OpenTag     string // Opening tag string (e.g., "<python>")
	CloseTag    string // Closing tag string (e.g., "</python>")
	MarkdownAlt string // Alternative markdown syntax (e.g., "```python")
}

// Predefined tags used throughout the application
var (
	ToolTag = Tag{
		Name:     TagTool,
		OpenTag:  "<tool>",
		CloseTag: "</tool>",
	}

	AgentStatusTag = Tag{
		Name:     TagAgentStatus,
		OpenTag:  "<agent_status>",
		CloseTag: "</agent_status>",
	}

	// AllTags contains all tags for iteration
	AllTags = []Tag{ToolTag, AgentStatusTag}
)

// HasTag checks if text contains a specific tag (opening or closing).
func HasTag(text string, tag Tag) bool {
	return strings.Contains(text, tag.OpenTag) || strings.Contains(text, tag.CloseTag)
}

// HasOpenTag checks if text contains the opening tag.
func HasOpenTag(text string, tag Tag) bool {
	return strings.Contains(text, tag.OpenTag)
}

// HasCloseTag checks if text contains the closing tag.
func HasCloseTag(text string, tag Tag) bool {
	return strings.Contains(text, tag.CloseTag)
}

// ExtractTagContent extracts content between opening and closing tags.
// Returns the content and true if both tags were found, empty string and false otherwise.
func ExtractTagContent(text string, tag Tag) (content string, found bool) {
	startIdx := strings.Index(text, tag.OpenTag)
	if startIdx == -1 {
		return "", false
	}

	endIdx := strings.Index(text[startIdx:], tag.CloseTag)
	if endIdx == -1 {
		return "", false
	}

	contentStart := startIdx + len(tag.OpenTag)
	contentEnd := startIdx + endIdx

	return strings.TrimSpace(text[contentStart:contentEnd]), true
}

// HasCodeBlock checks if text contains Python code in markdown format.
// Retrained model outputs ```python blocks natively.
func HasCodeBlock(text string) bool {
	return strings.Contains(text, "```python")
}

// ExtractCodeContent extracts Python code from markdown format.
// Returns the code and true if found, empty string and false otherwise.
func ExtractCodeContent(text string) (string, bool) {
	if code := extractMarkdownCodeInternal(text); code != "" {
		return code, true
	}
	return "", false
}

// extractMarkdownCodeInternal extracts code from ```python ... ``` blocks.
// This is an internal helper that matches the logic in tools/python.go
func extractMarkdownCodeInternal(text string) string {
	startMarker := "```python"
	startIdx := strings.Index(text, startMarker)
	if startIdx == -1 {
		return ""
	}

	// Skip past opening marker and optional newline
	codeStart := startIdx + len(startMarker)
	if codeStart < len(text) && text[codeStart] == '\n' {
		codeStart++
	}

	// Find closing ```
	endMarker := "```"
	endIdx := strings.Index(text[codeStart:], endMarker)
	if endIdx == -1 {
		return ""
	}

	code := text[codeStart : codeStart+endIdx]
	return strings.TrimSpace(code)
}

// StripTag removes a specific tag (both opening and closing) from text.
func StripTag(text string, tag Tag) string {
	text = strings.ReplaceAll(text, tag.OpenTag, "")
	text = strings.ReplaceAll(text, tag.CloseTag, "")
	return text
}

// StripAllTags removes all known tags from text.
func StripAllTags(text string) string {
	for _, tag := range AllTags {
		text = StripTag(text, tag)
	}
	return text
}

// CloseUnbalancedTags appends missing closing tags for any tags left open in the text.
// It returns the balanced text along with the list of tags that were closed (in the order they were appended).
func CloseUnbalancedTags(text string) (string, []Tag) {
	type stackEntry struct {
		tag Tag
	}

	stack := make([]stackEntry, 0)

	for i := 0; i < len(text); {
		matched := false
		for _, tag := range AllTags {
			if strings.HasPrefix(text[i:], tag.OpenTag) {
				stack = append(stack, stackEntry{tag: tag})
				i += len(tag.OpenTag)
				matched = true
				break
			}
			if strings.HasPrefix(text[i:], tag.CloseTag) {
				if len(stack) > 0 && stack[len(stack)-1].tag.Name == tag.Name {
					stack = stack[:len(stack)-1]
				}
				i += len(tag.CloseTag)
				matched = true
				break
			}
		}
		if !matched {
			i++
		}
	}

	if len(stack) == 0 {
		return text, nil
	}

	var builder strings.Builder
	builder.Grow(len(text) + len(stack)*8)
	builder.WriteString(text)

	closers := make([]Tag, 0, len(stack))
	for i := len(stack) - 1; i >= 0; i-- {
		tag := stack[i].tag
		builder.WriteString(tag.CloseTag)
		closers = append(closers, tag)
	}

	return builder.String(), closers
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
	case TagTool:
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
