package tools

import (
	"bufio"
	"context"
	"fmt"
	"net"
	"strings"

	"github.com/google/uuid"
	"go.uber.org/zap"
)

const EOM_TOKEN = "<|EOM|>"

// StatefulPythonTool now has a session ID.
type StatefulPythonTool struct {
	conn      net.Conn
	sessionID string
	logger    *zap.Logger
}

// NewStatefulPythonTool now creates a unique session ID for each instance.
func NewStatefulPythonTool(ctx context.Context, address string, logger *zap.Logger) (*StatefulPythonTool, error) {
	conn, err := net.Dial("tcp", address)
	if err != nil {
		return nil, fmt.Errorf("could not connect to Python executor: %w", err)
	}

	sessionID := uuid.New().String()
	logger.Info("Python tool initialized", zap.String("session_id", sessionID), zap.String("address", address))

	return &StatefulPythonTool{conn: conn, sessionID: sessionID, logger: logger}, nil
}

func (t *StatefulPythonTool) Name() string {
	return "Stateful Python Environment"
}

func (t *StatefulPythonTool) Description() string {
	return "Executes Python code in a persistent, sandboxed session."
}

// Call now reads from the connection until it sees the EOM_TOKEN.
func (t *StatefulPythonTool) Call(ctx context.Context, input string) (string, error) {
	message := fmt.Sprintf("%s|%s", t.sessionID, input)

	_, err := t.conn.Write([]byte(message))
	if err != nil {
		return "", fmt.Errorf("failed to send code to Python server: %w", err)
	}

	// Use a buffered reader to read until the EOM token is found.
	reader := bufio.NewReader(t.conn)
	fullResponse, err := reader.ReadString('>')
	if err != nil {
		return "", fmt.Errorf("failed to read result from Python server: %w", err)
	}

	// Check if the response ends with the EOM token.
	// This handles cases where the token itself might be split across reads.
	for !strings.HasSuffix(fullResponse, EOM_TOKEN) {
		nextChunk, err := reader.ReadString('>')
		if err != nil {
			return "", fmt.Errorf("failed to read full response from Python server: %w", err)
		}
		fullResponse += nextChunk
	}

	// Trim the EOM token from the final response.
	return strings.TrimSuffix(fullResponse, EOM_TOKEN), nil
}

func (t *StatefulPythonTool) Close() {
	if t.conn != nil {
		t.conn.Close()
	}
}

// ExecutePythonCode extracts code from a string, executes it, and returns the result.
func (t *StatefulPythonTool) ExecutePythonCode(ctx context.Context, text string) (string, string, bool) {
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

	t.logger.Info("Executing Python code", zap.String("code", pythonCode))
	fmt.Println("\n--- Executing Python Code ---")
	fmt.Printf("Code to execute:\n%s\n", pythonCode)
	fmt.Println("--- Execution Output ---")

	execResult, err := t.Call(ctx, pythonCode)
	if err != nil {
		t.logger.Error("Error executing Python code", zap.Error(err))
		fmt.Printf("Error executing Python: %v\n", err)
		execResult = "Error: " + err.Error()
	} else {
		t.logger.Debug("Python code executed successfully", zap.String("result_preview", execResult[:min(100, len(execResult))]))
		fmt.Print(execResult)
	}
	fmt.Println("\n--- End Execution ---")

	return pythonCode, execResult, true
}
