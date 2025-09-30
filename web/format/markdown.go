package format

import "strings"

// MarkdownToXML converts markdown code blocks to XML tags.
// Specifically handles: ```python → <python>
// This is used when the LLM outputs markdown instead of the expected XML format.
func MarkdownToXML(text string) string {
	// Replace ```python with <python>
	text = strings.ReplaceAll(text, "```python", PythonTag.OpenTag)

	// Use state machine to replace closing ``` only after <python>
	// This prevents incorrectly converting ``` that aren't part of python blocks
	var result strings.Builder
	inCodeBlock := false

	lines := strings.Split(text, "\n")
	for i, line := range lines {
		trimmed := strings.TrimSpace(line)

		// Check if we're entering a code block
		if strings.Contains(line, PythonTag.OpenTag) {
			inCodeBlock = true
			result.WriteString(line)
		} else if inCodeBlock && trimmed == "```" {
			// We're in a code block and found closing ```
			result.WriteString(PythonTag.CloseTag)
			inCodeBlock = false
		} else {
			result.WriteString(line)
		}

		// Add newline except for last line
		if i < len(lines)-1 {
			result.WriteString("\n")
		}
	}

	return result.String()
}

// XMLToMarkdown converts XML tags to markdown for display.
// Only works for tags that have a markdown alternative defined.
func XMLToMarkdown(text string, tag Tag) string {
	if tag.MarkdownAlt == "" {
		return text // No markdown equivalent
	}

	text = strings.ReplaceAll(text, tag.OpenTag, "\n"+tag.MarkdownAlt+"\n")
	text = strings.ReplaceAll(text, tag.CloseTag, "\n```\n")
	return text
}
