package tools

import (
	"bufio"
	"context"
	"fmt"
	"io"
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

func (t *StatefulPythonTool) InitializeSession(ctx context.Context, sessionID string, uploadedFiles []string) (string, error) {
	if len(uploadedFiles) == 0 {
		return "", nil
	}

	// Build initialization code that lists files
	filesList := strings.Join(uploadedFiles, "', '")
	initCode := fmt.Sprintf(`
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Session initialized with uploaded files
uploaded_files = ['%s']
print("="*50)
print("POCKET STATISTICIAN SESSION INITIALIZED")
print("="*50)
print(f"Uploaded files detected: {len(uploaded_files)}")
for f in uploaded_files:
    if os.path.exists(f):
        size = os.path.getsize(f) / 1024  # Size in KB
        print(f"  ✓ {f} ({size:.1f} KB)")
    else:
        print(f"  ✗ {f} (not found)")
print("="*50)
print(f"Primary file for analysis: {uploaded_files[0]}")
print("Ready for statistical analysis!")
print("="*50)
`, filesList)

	return t.Call(ctx, initCode, sessionID)
}

func (t *StatefulPythonTool) Name() string {
	return "Stateful Python Environment"
}

func (t *StatefulPythonTool) Description() string {
	return "Executes Python code in a persistent, sandboxed session."
}

// Call now accepts the sessionID for each execution.
func (t *StatefulPythonTool) Call(ctx context.Context, input string, sessionID string) (string, error) {
	message := fmt.Sprintf("%s|%s%s", sessionID, input, EOM_TOKEN)

	_, err := t.conn.Write([]byte(message))
	if err != nil {
		return "", fmt.Errorf("failed to send code to Python server: %w", err)
	}

	// Use a buffered reader to read until the EOM token is found.
	reader := bufio.NewReader(t.conn)
	var fullResponse strings.Builder
	buffer := make([]byte, 1024) // Read in chunks

	for {
		n, err := reader.Read(buffer)
		if err != nil {
			if err == io.EOF {
				break
			}
			return "", fmt.Errorf("failed to read result from Python server: %w", err)
		}

		fullResponse.Write(buffer[:n])

		// Check if the complete EOM token is now in our collected response
		if strings.HasSuffix(fullResponse.String(), EOM_TOKEN) {
			break
		}
	}

	// Trim the EOM token from the final response.
	return strings.TrimSuffix(fullResponse.String(), EOM_TOKEN), nil
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
	fmt.Printf("<execution_results>%s</execution_results>", execResult)

	return pythonCode, execResult, true
}
