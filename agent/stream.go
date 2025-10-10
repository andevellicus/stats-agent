package agent

import (
    "fmt"
    "io"
    "strings"
    "sync"

    "stats-agent/web/format"
)

// FlushHandler receives an assistant segment and an optional tool result.
type FlushHandler func(assistant string, tool *string)

// Stream captures assistant output and tool results while forwarding data to the client in real time.
type Stream struct {
	mu           sync.Mutex
	logWriter    io.Writer
	streamWriter io.Writer
	flush        FlushHandler
	segment      strings.Builder
}

// NewStream constructs a stream that duplicates assistant output to logWriter and streamWriter,
// and notifies flush whenever an assistant segment completes (typically just before a tool result).
func NewStream(logWriter, streamWriter io.Writer, flush FlushHandler) *Stream {
	return &Stream{
		logWriter:    logWriter,
		streamWriter: streamWriter,
		flush:        flush,
	}
}

// Write appends data to the current assistant segment while writing to the provided writers.
func (s *Stream) Write(p []byte) (int, error) {
    s.mu.Lock()
    defer s.mu.Unlock()

    if s.logWriter != nil {
        if _, err := s.logWriter.Write(p); err != nil {
            return 0, err
        }
    }
    if s.streamWriter != nil {
        // Apply lightweight preprocessing so the live stream shows code/status formatting
        // (DB persistence will still process the raw segment for canonical HTML.)
        toStream := p
        if len(p) > 0 {
            transformed := format.PreprocessAssistantText(string(p))
            toStream = []byte(transformed)
        }
        if _, err := s.streamWriter.Write(toStream); err != nil {
            return 0, err
        }
    }
    s.segment.Write(p)
    return len(p), nil
}

// WriteString convenience wrapper over Write.
func (s *Stream) WriteString(str string) (int, error) {
	return s.Write([]byte(str))
}

// Status streams a status message to the client.
func (s *Stream) Status(message string) error {
	_, err := s.WriteString(fmt.Sprintf("<agent_status>%s</agent_status>", message))
	return err
}

// ExecutionResult delegates to Tool for backward compatibility.
func (s *Stream) ExecutionResult(result string) error {
	return s.Tool(result)
}

// Tool finalizes the current assistant segment, emits it via the flush handler alongside the tool result,
// and streams the tool output to the client in markdown code fences.
func (s *Stream) Tool(result string) error {
	assistant := s.popSegment()
	trimmed := strings.TrimSpace(result)

	if s.flush != nil {
		var toolPtr *string
		if trimmed != "" {
			toolCopy := trimmed
			toolPtr = &toolCopy
		}
		s.flush(assistant, toolPtr)
	}

	if s.streamWriter == nil || trimmed == "" {
		return nil
	}

	formatted := fmt.Sprintf("\n```\n%s\n```\n", trimmed)
	s.mu.Lock()
	defer s.mu.Unlock()
	_, err := s.streamWriter.Write([]byte(formatted))
	return err
}

// Finalize flushes any remaining assistant output (without an accompanying tool message).
func (s *Stream) Finalize() {
	assistant := s.popSegment()
	if assistant != "" && s.flush != nil {
		s.flush(assistant, nil)
	}
}

func (s *Stream) popSegment() string {
	s.mu.Lock()
	defer s.mu.Unlock()
	assistant := strings.TrimSpace(s.segment.String())
	s.segment.Reset()
	return assistant
}
