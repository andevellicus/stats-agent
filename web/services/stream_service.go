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

	var processToken func(string)
	processToken = func(token string) {
		if token == "" {
			return
		}

		// Check each known tag for opening tags
		for _, tag := range format.AllTags {
			if strings.Contains(token, tag.OpenTag) {
				parts := strings.SplitN(token, tag.OpenTag, 2)
				transform := format.GetStreamTransform(tag)
				writeFunc(StreamData{Type: "chunk", Content: parts[0]})
				writeFunc(StreamData{Type: "chunk", Content: transform.OpenReplace})
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
				writeFunc(StreamData{Type: "chunk", Content: parts[0]})
				writeFunc(StreamData{Type: "chunk", Content: transform.CloseReplace})
				if len(openTagStack) > 0 && openTagStack[len(openTagStack)-1].Name == tag.Name {
					openTagStack = openTagStack[:len(openTagStack)-1]
				}
				processToken(parts[1])
				return
			}
		}

		// No tags found, write token as-is
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
