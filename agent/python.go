package agent

import (
	"context"
	"fmt"
	"io"
	"net"
	"strings"

	"github.com/google/uuid"
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

// executePythonCode extracts code from a string, executes it, and returns the result
func executePythonCode(ctx context.Context, pythonTool *StatefulPythonTool, text string) (string, string, bool) {
	startTag := "<python>"
	endTag := "</python>"

	startIdx := strings.Index(text, startTag)
	if startIdx == -1 {
		return "", "", false
	}

	endIdx := strings.Index(text[startIdx:], endTag)
	if endIdx == -1 {
		return "", "", false
	}

	codeStart := startIdx + len(startTag)
	codeEnd := startIdx + endIdx
	pythonCode := strings.TrimSpace(text[codeStart:codeEnd])

	if pythonCode == "" {
		return "", "", false
	}

	fmt.Println("\n--- Executing Python Code ---")
	fmt.Printf("Code to execute:\n%s\n", pythonCode)
	fmt.Println("--- Execution Output ---")

	execResult, err := pythonTool.Call(ctx, pythonCode)
	if err != nil {
		fmt.Printf("Error executing Python: %v\n", err)
		execResult = "Error: " + err.Error()
	} else {
		fmt.Print(execResult)
	}
	fmt.Println("\n--- End Execution ---")

	// Return the code, the result, and a flag indicating code was executed
	return pythonCode, execResult, true
}
