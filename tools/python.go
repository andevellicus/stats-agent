package tools

import (
	"bufio"
	"context"
	"fmt"
	"net"
	"strings"

	"go.uber.org/zap"
)

const EOM_TOKEN = "<|EOM|>"

// StatefulPythonTool no longer holds a sessionID
type StatefulPythonTool struct {
	conn   net.Conn
	logger *zap.Logger
}

// NewStatefulPythonTool no longer creates a session ID.
func NewStatefulPythonTool(ctx context.Context, address string, logger *zap.Logger) (*StatefulPythonTool, error) {
	conn, err := net.Dial("tcp", address)
	if err != nil {
		return nil, fmt.Errorf("could not connect to Python executor: %w", err)
	}

	logger.Info("Python tool initialized", zap.String("address", address))

	return &StatefulPythonTool{conn: conn, logger: logger}, nil
}

func (t *StatefulPythonTool) Name() string {
	return "Stateful Python Environment"
}

func (t *StatefulPythonTool) Description() string {
	return "Executes Python code in a persistent, sandboxed session."
}

// Call now accepts the sessionID for each execution.
func (t *StatefulPythonTool) Call(ctx context.Context, input string, sessionID string) (string, error) {
	message := fmt.Sprintf("%s|%s", sessionID, input)

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

// ExecutePythonCode now requires a sessionID to be passed.
func (t *StatefulPythonTool) ExecutePythonCode(ctx context.Context, text string, sessionID string) (string, string, bool) {
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

	t.logger.Info("Executing Python code", zap.String("code", pythonCode), zap.String("session_id", sessionID))

	execResult, err := t.Call(ctx, pythonCode, sessionID)
	if err != nil {
		t.logger.Error("Error executing Python code", zap.Error(err))
		execResult = "Error: " + err.Error()
	} else {
		t.logger.Debug("Python code executed successfully", zap.String("result_preview", execResult[:min(100, len(execResult))]))
	}

	// ONLY print the execution result, wrapped in tags.
	// This is the only output from this function that goes to the web UI stream.
	fmt.Printf("<execution_result>%s</execution_result>", execResult)

	return pythonCode, execResult, true
}
