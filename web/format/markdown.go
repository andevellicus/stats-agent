package format

import (
    "regexp"
    "strings"
)

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

// PreprocessAssistantText normalizes common LLM output quirks into our XML tag format
// so that downstream HTML rendering can format code and status blocks correctly.
// It is safe to call on arbitrary assistant text; no-ops when nothing matches.
func PreprocessAssistantText(text string) string {
    if text == "" {
        return text
    }

    // 1) Convert markdown ```python blocks to <python> ... </python>
    text = MarkdownToXML(text)

    // 2) Normalize malformed tag spacings
    replacements := map[string]string{
        "< python>":   "<python>",
        "<python >":   "<python>",
        "< python >":  "<python>",
        "</ python>":  "</python>",
        "</python >":  "</python>",
        "</ python >": "</python>",
        "<python\n":   "<python>\n",
        "\npython>":   "\n<python>",
        "\n/python>":  "\n</python>",
        "</python\n":  "</python>\n",
    }
    for old, new := range replacements {
        text = strings.ReplaceAll(text, old, new)
    }

    // 3) Replace curly quotes with straight quotes (helps readability and any inline code)
    text = strings.NewReplacer(
        "“", "\"",
        "”", "\"",
        "‘", "'",
        "’", "'",
    ).Replace(text)

    // 4) Open python blocks on common prompt lines
    // a) A sentence ends with ".python"
    reDotPython := regexp.MustCompile(`(?m)\.python\s*\n`)
    text = reDotPython.ReplaceAllString(text, ".\n<python>\n")

    // b) A label line like "python:" before code
    reLabelPython := regexp.MustCompile(`(?mi)^[\t ]*python:\s*\n`)
    text = reLabelPython.ReplaceAllString(text, "<python>\n")

    // c) A standalone line "python" before code
    reLinePython := regexp.MustCompile(`(?m)^[\t ]*python\s*\n`)
    text = reLinePython.ReplaceAllString(text, "<python>\n")

    // d) Prompt-style: "python>" or "analysis.python>"
    rePrompt := regexp.MustCompile(`(?mi)^[\t ]*(?:analysis\.)?python>\s*`)
    replacedFirst := false
    text = rePrompt.ReplaceAllStringFunc(text, func(_ string) string {
        if !replacedFirst {
            replacedFirst = true
            return "<python>\n"
        }
        return ""
    })

    return text
}
