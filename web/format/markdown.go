package format

import (
    "strings"
)

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
