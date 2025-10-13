package services

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"stats-agent/web/format"
	"strings"
	"sync"

	"go.uber.org/zap"
)

type StreamData struct {
	Type    string `json:"type"`
	Content string `json:"content,omitempty"`
}

type StreamService struct {
	logger *zap.Logger
}

func NewStreamService(logger *zap.Logger) *StreamService {
	return &StreamService{
		logger: logger,
	}
}

// WriteSSEData is a helper to write SSE formatted data safely.
func (ss *StreamService) WriteSSEData(ctx context.Context, w http.ResponseWriter, data StreamData, mu *sync.Mutex) error {
	mu.Lock()
	defer mu.Unlock()

	select {
	case <-ctx.Done():
		return ctx.Err()
	default:
	}

	jsonData, err := json.Marshal(data)
	if err != nil {
		return err
	}

	_, err = fmt.Fprintf(w, "data: %s\n\n", jsonData)
	if err != nil {
		return err
	}

	if flusher, ok := w.(http.Flusher); ok {
		flusher.Flush()
	}
	return nil
}

// ProcessStreamByWord reads from an io.Reader and processes output word-by-word,
// converting XML tags to markdown-style formatting for SSE streaming.
// Uses the format package for consistent tag handling.
func (ss *StreamService) ProcessStreamByWord(ctx context.Context, r io.Reader, writeFunc func(StreamData) error) {
    reader := bufio.NewReader(r)
    var currentWord strings.Builder
    var openTagStack []format.Tag
    // Buffer for partial backtick fences (e.g., "`" or "``") across tokens
    var tickBuf string

    // Helper: whether token is composed only of backticks and optional whitespace/newlines
    isOnlyBackticks := func(s string) bool {
        if s == "" { return false }
        trimmed := strings.Trim(s, " \t\r\n")
        if trimmed == "" { return false }
        for i := 0; i < len(trimmed); i++ {
            if trimmed[i] != '`' { return false }
        }
        return true
    }

    // Helper: strip leading backticks from s and return remainder and count
    stripLeadingBackticks := func(s string) (string, int) {
        i := 0
        for i < len(s) && s[i] == '`' { i++ }
        return s[i:], i
    }

    var processToken func(string)
    processToken = func(token string) {
        if token == "" {
            return
        }

        // Handle any pending partial backticks first
        if tickBuf != "" {
            afterTicks, lead := stripLeadingBackticks(token)
            totalTicks := len(strings.Trim(tickBuf, " \t\r\n")) + lead
            lowerAfter := strings.ToLower(afterTicks)
            if totalTicks >= 3 {
                // Fence detected
                if strings.HasPrefix(lowerAfter, "python") {
                    // Opening fence
                    writeFunc(StreamData{Type: "chunk", Content: "\n```python\n"})
                    // Emit remainder after 'python'
                    rem := afterTicks[len("python"):]
                    if rem != "" {
                        writeFunc(StreamData{Type: "chunk", Content: rem})
                    }
                    tickBuf = ""
                    return
                }
                // Closing fence
                writeFunc(StreamData{Type: "chunk", Content: "\n```\n"})
                tickBuf = ""
                if lead > 0 {
                    if afterTicks != "" {
                        writeFunc(StreamData{Type: "chunk", Content: afterTicks})
                    }
                    return
                }
                // Fallthrough to process token normally if no leading ticks
            } else {
                // Not enough ticks to decide
                if lead > 0 && isOnlyBackticks(token) {
                    tickBuf += token
                    return
                }
                // Flush buffered ticks as literals
                writeFunc(StreamData{Type: "chunk", Content: tickBuf})
                tickBuf = ""
                // Continue with current token
            }
        }

		// Check each known tag for opening tags
		for _, tag := range format.AllTags {
            if strings.Contains(token, tag.OpenTag) {
                parts := strings.SplitN(token, tag.OpenTag, 2)
                transform := format.GetStreamTransform(tag)
                writeFunc(StreamData{Type: "chunk", Content: parts[0]})
                writeFunc(StreamData{Type: "chunk", Content: transform.OpenReplace})
                if ss.logger != nil {
                    ss.logger.Debug("sse transform open",
                        zap.String("tag", tag.Name),
                        zap.String("emit", transform.OpenReplace))
                }
                openTagStack = append(openTagStack, tag)
                processToken(parts[1])
                return
            }
        }

        // Check each known tag for closing tags
        for _, tag := range format.AllTags {
            if strings.Contains(token, tag.CloseTag) {
                parts := strings.SplitN(token, tag.CloseTag, 2)
                transform := format.GetStreamTransform(tag)
                // Emit a close replacement only if a matching open tag is on the stack
                if len(openTagStack) > 0 && openTagStack[len(openTagStack)-1].Name == tag.Name {
                    writeFunc(StreamData{Type: "chunk", Content: parts[0]})
                    writeFunc(StreamData{Type: "chunk", Content: transform.CloseReplace})
                    if ss.logger != nil {
                        ss.logger.Debug("sse transform close",
                            zap.String("tag", tag.Name),
                            zap.String("emit", transform.CloseReplace))
                    }
                    openTagStack = openTagStack[:len(openTagStack)-1]
                    processToken(parts[1])
                    return
                }
                // No matching opener found; pass the literal token through unchanged
                writeFunc(StreamData{Type: "chunk", Content: token})
                return
            }
        }

        // No XML tags found – normalize common LLM quirks when not already inside python:
        // 1) sentence-ending ".python\n" -> open fenced block
        // 2) line-ending "`python\n" -> open fenced block
        if len(openTagStack) == 0 || openTagStack[len(openTagStack)-1].Name != format.TagPython {
            if strings.HasSuffix(token, ".python\n") {
                // Emit prefix (up to the '.') and then open the python fence
                prefix := strings.TrimSuffix(token, ".python\n")
                if prefix != "" { writeFunc(StreamData{Type: "chunk", Content: prefix}) }
                transform := format.GetStreamTransform(format.PythonTag)
                writeFunc(StreamData{Type: "chunk", Content: transform.OpenReplace})
                openTagStack = append(openTagStack, format.PythonTag)
                return
            }
            if strings.HasSuffix(token, "`python\n") {
                // Emit prefix (before the backtick) and then open the python fence
                prefix := strings.TrimSuffix(token, "`python\n")
                if prefix != "" { writeFunc(StreamData{Type: "chunk", Content: prefix}) }
                transform := format.GetStreamTransform(format.PythonTag)
                writeFunc(StreamData{Type: "chunk", Content: transform.OpenReplace})
                openTagStack = append(openTagStack, format.PythonTag)
                return
            }
        }

        // Treat raw markdown fence closers as closing any heuristic python block we opened
        if len(openTagStack) > 0 && openTagStack[len(openTagStack)-1].Name == format.TagPython {
            if strings.Contains(token, "```") {
                // Pop the python tag to avoid appending an extra closing fence at EOF
                openTagStack = openTagStack[:len(openTagStack)-1]
            }
        }

        // If token is only backticks, buffer and wait for the next token
        if isOnlyBackticks(token) {
            tickBuf = token
            return
        }

        // Otherwise, write token as-is
        if ss.logger != nil {
            if strings.Contains(token, "```") || strings.Contains(token, "python") {
                ss.logger.Debug("sse emit token", zap.String("token", token))
            }
        }
        writeFunc(StreamData{Type: "chunk", Content: token})
    }

	for {
		select {
		case <-ctx.Done():
			return
		default:
			char, _, err := reader.ReadRune()
            if err != nil {
                if currentWord.Len() > 0 {
                    processToken(currentWord.String())
                }
                // Drop any pending partial backticks (do not emit stray "``")
                if tickBuf != "" {
                    trimmed := strings.Trim(tickBuf, " \t\r\n")
                    // Only emit if this is a full fence (>=3 backticks). Otherwise, drop.
                    allTicks := true
                    for i := 0; i < len(trimmed); i++ { if trimmed[i] != '`' { allTicks = false; break } }
                    if allTicks && len(trimmed) >= 3 {
                        writeFunc(StreamData{Type: "chunk", Content: "\n```\n"})
                    }
                    tickBuf = ""
                }
                for i := len(openTagStack) - 1; i >= 0; i-- {
                    tag := openTagStack[i]
                    transform := format.GetStreamTransform(tag)
                    writeFunc(StreamData{Type: "chunk", Content: transform.CloseReplace})
                }
                return
            }

			// If we encounter '<' that could be a tag start, check if it's one of our known tags
			// This prevents tags from being concatenated while not breaking on comparison operators like "5 < 3"
			if char == '<' && currentWord.Len() > 0 {
				// Peek ahead to see if this could be a known tag
				currentWordStr := currentWord.String()

				// Check if we're potentially at the start of a known tag
				couldBeTag := false
				for _, tag := range format.AllTags {
					// Check both opening and closing tags
					if len(tag.OpenTag) > 0 && tag.OpenTag[0] == '<' {
						couldBeTag = true
						break
					}
				}

				// Only split if this could be a tag (all our tags start with '<')
				// This allows normal '<' in code like "if x < 5" to pass through
				if couldBeTag {
					// Check if the current word ends in a way that suggests a tag boundary
					// (e.g., ends with '>' from a closing tag, or is at start of line)
					if strings.HasSuffix(currentWordStr, ">") || len(currentWordStr) == 0 {
						processToken(currentWordStr)
						currentWord.Reset()
					}
				}
			}

			currentWord.WriteRune(char)

			if char == ' ' || char == '\n' {
				processToken(currentWord.String())
				currentWord.Reset()
			}
		}
	}
}
