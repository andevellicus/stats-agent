package services

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
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

// ProcessStreamByWord reads from an io.Reader and processes output word-by-word for SSE streaming.
// Simplified version that just passes through content with minimal processing.
func (ss *StreamService) ProcessStreamByWord(ctx context.Context, r io.Reader, writeFunc func(StreamData) error) {
	reader := bufio.NewReader(r)
	var currentWord strings.Builder

	for {
		select {
		case <-ctx.Done():
			return
		default:
			char, _, err := reader.ReadRune()
			if err != nil {
				// Flush any remaining content
				if currentWord.Len() > 0 {
					writeFunc(StreamData{Type: "chunk", Content: currentWord.String()})
				}
				return
			}

			currentWord.WriteRune(char)

			// Emit on word boundaries (space or newline)
			if char == ' ' || char == '\n' {
				writeFunc(StreamData{Type: "chunk", Content: currentWord.String()})
				currentWord.Reset()
			}
		}
	}
}
