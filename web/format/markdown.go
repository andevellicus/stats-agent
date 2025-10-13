package format

import (
    "strings"
)

// MarkdownToXML is deprecated but kept for backward compatibility.
// Modern code should work with markdown blocks natively.
func MarkdownToXML(text string) string {
	// No-op: we now support markdown natively, no conversion needed
	return text
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

// PreprocessAssistantText normalizes LLM output.
// Performs basic text cleanup for better readability.
func PreprocessAssistantText(text string) string {
    if text == "" {
        return text
    }

    // Replace curly quotes (helps readability)
    text = strings.NewReplacer(
        "\u201c", "\"", // "
        "\u201d", "\"", // "
        "\u2018", "'",  // '
        "\u2019", "'",  // '
    ).Replace(text)

    return text
}
