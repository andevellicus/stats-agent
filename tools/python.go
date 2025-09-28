package tools

import (
	"bufio"
	"context"
	"errors"
	"fmt"
	"io"
	"net"
	"strings"
	"sync"
	"time"

	"go.uber.org/zap"
)

const (
	EOM_TOKEN               = "<|EOM|>"
	defaultExecutorCooldown = 5 * time.Second
)

type executorNode struct {
	address    string
	retryAfter time.Time
}

type executorPool struct {
	nodes    []*executorNode
	mu       sync.Mutex
	next     int
	cooldown time.Duration
}

func newExecutorPool(addresses []string, cooldown time.Duration) (*executorPool, error) {
	unique := make(map[string]struct{}, len(addresses))
	nodes := make([]*executorNode, 0, len(addresses))
	for _, addr := range addresses {
		addr = strings.TrimSpace(addr)
		if addr == "" {
			continue
		}
		if _, exists := unique[addr]; exists {
			continue
		}
		unique[addr] = struct{}{}
		nodes = append(nodes, &executorNode{address: addr})
	}
	if len(nodes) == 0 {
		return nil, errors.New("no valid python executor addresses provided")
	}
	return &executorPool{
		nodes:    nodes,
		cooldown: cooldown,
	}, nil
}

func (p *executorPool) Next() (string, error) {
	p.mu.Lock()
	defer p.mu.Unlock()
	if len(p.nodes) == 0 {
		return "", errors.New("no python executors configured")
	}
	now := time.Now()
	checked := 0
	for checked < len(p.nodes) {
		idx := p.next
		p.next = (p.next + 1) % len(p.nodes)
		node := p.nodes[idx]
		checked++
		if now.After(node.retryAfter) {
			return node.address, nil
		}
	}
	return "", errors.New("no healthy python executors available")
}

func (p *executorPool) MarkFailure(address string) {
	p.mu.Lock()
	defer p.mu.Unlock()
	now := time.Now().Add(p.cooldown)
	for _, node := range p.nodes {
		if node.address == address {
			node.retryAfter = now
			return
		}
	}
}

func (p *executorPool) MarkSuccess(address string) {
	p.mu.Lock()
	defer p.mu.Unlock()
	for _, node := range p.nodes {
		if node.address == address {
			node.retryAfter = time.Time{}
			return
		}
	}
}

func (p *executorPool) Size() int {
	p.mu.Lock()
	defer p.mu.Unlock()
	return len(p.nodes)
}

func (p *executorPool) Addresses() []string {
	p.mu.Lock()
	defer p.mu.Unlock()
	addrs := make([]string, 0, len(p.nodes))
	for _, node := range p.nodes {
		addrs = append(addrs, node.address)
	}
	return addrs
}

type StatefulPythonTool struct {
	pool        *executorPool
	logger      *zap.Logger
	dialTimeout time.Duration
	ioTimeout   time.Duration
	sessionMu   sync.RWMutex
	sessionAddr map[string]string
}

// NewStatefulPythonTool no longer creates a session ID.
func NewStatefulPythonTool(ctx context.Context, addresses []string, logger *zap.Logger) (*StatefulPythonTool, error) {
	pool, err := newExecutorPool(addresses, defaultExecutorCooldown)
	if err != nil {
		return nil, err
	}
	tool := &StatefulPythonTool{
		pool:        pool,
		logger:      logger,
		dialTimeout: 3 * time.Second,
		ioTimeout:   60 * time.Second,
		sessionAddr: make(map[string]string),
	}
	if err := tool.ensureInitialConnectivity(ctx); err != nil {
		return nil, err
	}
	if tool.logger != nil {
		tool.logger.Info("Python tool initialized", zap.Strings("addresses", tool.pool.Addresses()))
	}
	return tool, nil
}

func (t *StatefulPythonTool) ensureInitialConnectivity(ctx context.Context) error {
	addresses := t.pool.Addresses()
	var lastErr error
	for _, addr := range addresses {
		conn, err := t.dial(ctx, addr)
		if err != nil {
			t.pool.MarkFailure(addr)
			lastErr = err
			if t.logger != nil {
				t.logger.Warn("Initial executor health check failed", zap.String("address", addr), zap.Error(err))
			}
			continue
		}
		_ = conn.Close()
		t.pool.MarkSuccess(addr)
		return nil
	}
	if lastErr != nil {
		return fmt.Errorf("unable to reach any python executor: %w", lastErr)
	}
	return errors.New("no python executors available")
}

func (t *StatefulPythonTool) dial(ctx context.Context, address string) (net.Conn, error) {
	d := &net.Dialer{Timeout: t.dialTimeout}
	return d.DialContext(ctx, "tcp", address)
}

func (t *StatefulPythonTool) execute(conn net.Conn, input string, sessionID string) (string, error) {
	deadline := time.Now().Add(t.ioTimeout)
	_ = conn.SetDeadline(deadline)
	payload := sessionID + "|" + input + EOM_TOKEN
	if _, err := conn.Write([]byte(payload)); err != nil {
		return "", fmt.Errorf("send code: %w", err)
	}

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

func (t *StatefulPythonTool) InitializeSession(ctx context.Context, sessionID string, uploadedFiles []string) (string, error) {
	quoted := make([]string, len(uploadedFiles))
	for i, f := range uploadedFiles {
		sanitized := strings.ReplaceAll(f, "'", "\\'")
		quoted[i] = fmt.Sprintf("'%s'", sanitized)
	}
	filesLiteral := strings.Join(quoted, ", ")

	initCode := fmt.Sprintf(`
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

workspace_path = os.getcwd()
uploaded_files = [%s]

print("=" * 50)
print("POCKET STATISTICIAN SESSION INITIALIZED")
print("=" * 50)

if uploaded_files:
    print(f"Uploaded files detected: {len(uploaded_files)}")
    for f in uploaded_files:
        file_path = os.path.join(workspace_path, f)
        if os.path.exists(file_path):
            size = os.path.getsize(file_path) / 1024  # Size in KB
            print(f"  \u2713 {f} ({size:.1f} KB)")
        else:
            print(f"  \u2717 {f} (not found)")
    print("=" * 50)
    print(f"Primary file for analysis: {uploaded_files[0]}")
else:
    print("No uploaded files detected yet. You can upload CSV or Excel files at any time.")
    print("=" * 50)

print("Ready for statistical analysis!")
print("=" * 50)
`, filesLiteral)

	return t.Call(ctx, initCode, sessionID)
}

func (t *StatefulPythonTool) Name() string {
	return "Stateful Python Environment"
}

func (t *StatefulPythonTool) Description() string {
	return "Executes Python code in a persistent, sandboxed session."
}

func (t *StatefulPythonTool) Call(ctx context.Context, input string, sessionID string) (string, error) {
	total := t.pool.Size()
	if total == 0 {
		return "", errors.New("no python executors configured")
	}

	tried := make(map[string]struct{})

	// Try the previously assigned executor first, if any.
	if sessionID != "" {
		t.sessionMu.RLock()
		boundAddr, ok := t.sessionAddr[sessionID]
		t.sessionMu.RUnlock()
		if ok {
			if result, err := t.callExecutor(ctx, boundAddr, input, sessionID); err == nil {
				return result, nil
			}
			tried[boundAddr] = struct{}{}
			t.sessionMu.Lock()
			delete(t.sessionAddr, sessionID)
			t.sessionMu.Unlock()
		}
	}

	var lastErr error
	for attempts := 0; attempts < total; attempts++ {
		addr, err := t.pool.Next()
		if err != nil {
			if lastErr != nil {
				return "", fmt.Errorf("no healthy python executors available: %w", lastErr)
			}
			return "", err
		}
		if _, seen := tried[addr]; seen {
			continue
		}
		tried[addr] = struct{}{}

		result, execErr := t.callExecutor(ctx, addr, input, sessionID)
		if execErr == nil {
			t.sessionMu.Lock()
			t.sessionAddr[sessionID] = addr
			t.sessionMu.Unlock()
			return result, nil
		}
		lastErr = execErr
	}

	if lastErr != nil {
		return "", fmt.Errorf("all python executors failed: %w", lastErr)
	}
	return "", errors.New("no healthy python executors available")
}

func (t *StatefulPythonTool) callExecutor(ctx context.Context, addr, input, sessionID string) (string, error) {
	conn, err := t.dial(ctx, addr)
	if err != nil {
		t.pool.MarkFailure(addr)
		if t.logger != nil {
			t.logger.Warn("Failed to connect to python executor", zap.String("address", addr), zap.Error(err))
		}
		return "", fmt.Errorf("dial python server %s: %w", addr, err)
	}

	result, execErr := t.execute(conn, input, sessionID)
	_ = conn.Close()
	if execErr != nil {
		t.pool.MarkFailure(addr)
		if t.logger != nil {
			t.logger.Warn("Python executor call failed", zap.String("address", addr), zap.Error(execErr))
		}
		return "", fmt.Errorf("executor %s: %w", addr, execErr)
	}

	t.pool.MarkSuccess(addr)
	if t.logger != nil {
		t.logger.Debug("Python code executed", zap.String("address", addr), zap.String("session_id", sessionID))
	}
	return result, nil
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
