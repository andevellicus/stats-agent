package main

import (
	"context"
	"fmt"
	"io"
	"net"

	"github.com/google/uuid" // Make sure to add this import
	"github.com/tmc/langchaingo/tools"
)

// StatefulPythonTool now has a session ID.
type StatefulPythonTool struct {
	conn      net.Conn
	sessionID string
}

// NewStatefulPythonTool now creates a unique session ID for each instance.
func NewStatefulPythonTool(ctx context.Context, address string) (*StatefulPythonTool, error) {
	conn, err := net.Dial("tcp", address)
	if err != nil {
		return nil, fmt.Errorf("could not connect to Python executor: %w", err)
	}

	// Each tool instance gets a unique ID to sandbox its state.
	sessionID := uuid.New().String()

	return &StatefulPythonTool{conn: conn, sessionID: sessionID}, nil
}

func (t *StatefulPythonTool) Name() string {
	return "Stateful Python Environment"
}

func (t *StatefulPythonTool) Description() string {
	return "Executes Python code in a persistent, sandboxed session. Variables and data are remembered across calls for the same agent."
}

func (t *StatefulPythonTool) Call(ctx context.Context, input string) (string, error) {
	// Prepend the session ID to the code before sending.
	message := fmt.Sprintf("%s|%s", t.sessionID, input)

	_, err := t.conn.Write([]byte(message))
	if err != nil {
		return "", fmt.Errorf("failed to send code to Python server: %w", err)
	}

	buf := make([]byte, 4096)
	n, err := t.conn.Read(buf)
	if err != nil {
		if err == io.EOF {
			return "", fmt.Errorf("connection closed by Python server")
		}
		return "", fmt.Errorf("failed to read result from Python server: %w", err)
	}

	return string(buf[:n]), nil
}

func (t *StatefulPythonTool) Close() {
	if t.conn != nil {
		t.conn.Close()
	}
}

var _ tools.Tool = &StatefulPythonTool{}
