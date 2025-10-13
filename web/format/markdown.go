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
		"\u2018", "'", // '
		"\u2019", "'", // '
	).Replace(text)

	// Normalize occasional XML-style python tags that may leak from the model
	// into markdown fences for consistency in both streaming and DB rendering.
	text = strings.ReplaceAll(text, "<python>", "\n```python\n")
	text = strings.ReplaceAll(text, "</python>", "\n```\n")

	// Normalize common fence-typos for python language tags
	// 1) Single backtick language line -> proper fenced block opener
	if strings.HasPrefix(text, "`python\r\n") {
		text = "```python\n" + text[len("`python\r\n"):]
	}
	if strings.HasPrefix(text, "`python\n") {
		text = "```python\n" + text[len("`python\n"):]
	}
	text = strings.ReplaceAll(text, "\n`python\r\n", "\n```python\n")
	text = strings.ReplaceAll(text, "\n`python\n", "\n```python\n")
	// 2) Triple backticks with space/tab before language -> tighten
	text = strings.ReplaceAll(text, "``` python", "```python")
	text = strings.ReplaceAll(text, "```\tpython", "```python")

    // Heuristic: sometimes the model emits ".python\n" appended to a sentence
    // to signal the start of code. Convert ".python" at line-end into a fenced block.
    // Keep the preceding period and start a new line before the fence.
    text = strings.ReplaceAll(text, ".python\r\n", ".\n```python\n")
    text = strings.ReplaceAll(text, ".python\n", ".\n```python\n")

    // Remove any bare double-backtick lines (preserve triple backticks)
    // This avoids stray paragraphs rendered as `` while keeping real fences intact
    {
        lines := strings.Split(text, "\n")
        filtered := make([]string, 0, len(lines))
        for _, ln := range lines {
            trimmed := strings.TrimSpace(ln)
            if trimmed == "``" { // drop only the double-backtick line
                continue
            }
            filtered = append(filtered, ln)
        }
        text = strings.Join(filtered, "\n")
    }

    return text
}
