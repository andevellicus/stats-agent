package tools

import (
	"bufio"
	"context"
	"errors"
	"fmt"
	"io"
	"net"
	"strings"
	"time"

	"go.uber.org/zap"
)

const EOM_TOKEN = "<|EOM|>"

type StatefulPythonTool struct {
	addr        string
	logger      *zap.Logger
	dialTimeout time.Duration
	ioTimeout   time.Duration
}

// NewStatefulPythonTool no longer creates a session ID.
func NewStatefulPythonTool(ctx context.Context, address string, logger *zap.Logger) (*StatefulPythonTool, error) {
	d := &net.Dialer{Timeout: 3 * time.Second}
	conn, err := d.DialContext(ctx, "tcp", address)
	if err != nil {
		return nil, fmt.Errorf("could not connect to Python executor: %w", err)
	}
	_ = conn.Close()

	logger.Info("Python tool initialized", zap.String("address", address))

	return &StatefulPythonTool{
		addr:        address,
		logger:      logger,
		dialTimeout: 3 * time.Second,
		ioTimeout:   60 * time.Second,
	}, nil
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

func (t *StatefulPythonTool) Call(ctx context.Context, input string, sessionID string) (string, error) {
	// 1) Open a new connection for this call (server closes after one message)
	d := &net.Dialer{Timeout: t.dialTimeout}
	conn, err := d.DialContext(ctx, "tcp", t.addr) // e.g., "executor:9999"
	if err != nil {
		return "", fmt.Errorf("dial python server: %w", err)
	}
	defer conn.Close()

	// Optional: avoid hangs
	deadline := time.Now().Add(t.ioTimeout)
	_ = conn.SetDeadline(deadline)

	// 2) Frame: sessionID|code<EOM>
	// Use bytes write (not Fprintf) so '%' inside code is not treated as a format verb
	payload := sessionID + "|" + input + EOM_TOKEN
	if _, err := conn.Write([]byte(payload)); err != nil {
		return "", fmt.Errorf("send code: %w", err)
	}

	// 3) Read until we see EOM anywhere in the accumulated buffer
	reader := bufio.NewReader(conn)
	var b strings.Builder
	buf := make([]byte, 4096)

	for {
		n, err := reader.Read(buf)
		if n > 0 {
			b.Write(buf[:n])
			s := b.String()
			if strings.Contains(s, EOM_TOKEN) {
				out := strings.ReplaceAll(s, EOM_TOKEN, "")
				return strings.TrimSpace(out), nil
			}
		}
		if err != nil {
			if errors.Is(err, io.EOF) {
				// If server closed immediately after sending, we may still have EOM
				s := b.String()
				if strings.Contains(s, EOM_TOKEN) {
					out := strings.ReplaceAll(s, EOM_TOKEN, "")
					return strings.TrimSpace(out), nil
				}
			}
			return "", fmt.Errorf("read result: %w", err)
		}
	}
}

func (t *StatefulPythonTool) Close() {
	// Connections are opened per-call, so there is nothing persistent to close.
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
