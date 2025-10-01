package agent

import (
	"fmt"
	"io"
	"sync"
)

// Stream provides a concurrency-safe writer for agent output and helper methods for common tags.
type Stream struct {
	mu     sync.Mutex
	writer io.Writer
}

// NewStream wraps the provided writer for safe concurrent use by the agent components.
func NewStream(w io.Writer) *Stream {
	return &Stream{writer: w}
}

// Write implements io.Writer, guarding concurrent access to the underlying writer.
func (s *Stream) Write(p []byte) (int, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.writer.Write(p)
}

// WriteString writes the provided string to the underlying writer.
func (s *Stream) WriteString(str string) (int, error) {
	return s.Write([]byte(str))
}

// Status writes an agent status message wrapped in the expected XML tag.
func (s *Stream) Status(message string) error {
	_, err := s.WriteString(fmt.Sprintf("<agent_status>%s</agent_status>", message))
	return err
}

// ExecutionResult writes execution output wrapped in the expected XML tag.
func (s *Stream) ExecutionResult(result string) error {
	_, err := s.WriteString(fmt.Sprintf("<execution_results>%s</execution_results>", result))
	return err
}
