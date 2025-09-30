package format

import "strings"

// Tag definitions - single source of truth for all custom XML tags
const (
	TagPython           = "python"
	TagExecutionResults = "execution_results"
	TagAgentStatus      = "agent_status"
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
	PythonTag = Tag{
		Name:        TagPython,
		OpenTag:     "<python>",
		CloseTag:    "</python>",
		MarkdownAlt: "```python",
	}

	ExecutionResultsTag = Tag{
		Name:     TagExecutionResults,
		OpenTag:  "<execution_results>",
		CloseTag: "</execution_results>",
	}

	AgentStatusTag = Tag{
		Name:     TagAgentStatus,
		OpenTag:  "<agent_status>",
		CloseTag: "</agent_status>",
	}

	// AllTags contains all tags for iteration
	AllTags = []Tag{PythonTag, ExecutionResultsTag, AgentStatusTag}
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
