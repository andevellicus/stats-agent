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

	"stats-agent/config"

	"go.uber.org/zap"
)

const (
    EOM_TOKEN = "<|EOM|>"
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

type connPool struct {
	address string
	idle    chan net.Conn
	sem     chan struct{}
	dial    func(context.Context) (net.Conn, error)
}

func newConnPool(address string, maxSize int, dial func(context.Context) (net.Conn, error)) *connPool {
	if maxSize <= 0 {
		maxSize = 1
	}
	return &connPool{
		address: address,
		idle:    make(chan net.Conn, maxSize),
		sem:     make(chan struct{}, maxSize),
		dial:    dial,
	}
}

func (p *connPool) Get(ctx context.Context) (net.Conn, error) {
	for {
		select {
		case conn := <-p.idle:
			if conn != nil {
				return conn, nil
			}
		default:
		}

		select {
		case p.sem <- struct{}{}:
			conn, err := p.dial(ctx)
			if err != nil {
				select {
				case <-p.sem:
				default:
				}
				return nil, err
			}
			return conn, nil
		case conn := <-p.idle:
			if conn != nil {
				return conn, nil
			}
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	}
}

func (p *connPool) Put(conn net.Conn) {
	if conn == nil {
		return
	}
	select {
	case p.idle <- conn:
	default:
		_ = conn.Close()
		select {
		case <-p.sem:
		default:
		}
	}
}

func (p *connPool) Discard(conn net.Conn) {
	if conn != nil {
		_ = conn.Close()
	}
	select {
	case <-p.sem:
	default:
	}
}

func (p *connPool) Close() {
	for {
		select {
		case conn := <-p.idle:
			if conn != nil {
				_ = conn.Close()
				select {
				case <-p.sem:
				default:
				}
			}
		default:
			return
		}
	}
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
	pool                      *executorPool
	logger                    *zap.Logger
	dialTimeout               time.Duration
	ioTimeout                 time.Duration
	sessionMu                 sync.RWMutex
	sessionAddr               map[string]string
	connPoolsMu               sync.RWMutex
	connPools                 map[string]*connPool
	maxConnectionsPerExecutor int
}

// NewStatefulPythonTool no longer creates a session ID.
func NewStatefulPythonTool(ctx context.Context, cfg *config.Config, logger *zap.Logger) (*StatefulPythonTool, error) {
	if cfg == nil {
		return nil, errors.New("config is required")
	}
	addresses := cfg.PythonExecutorAddresses
    cooldown := cfg.PythonExecutorCooldownSeconds
	pool, err := newExecutorPool(addresses, cooldown)
	if err != nil {
		return nil, err
	}
    dialTimeout := cfg.PythonExecutorDialTimeoutSeconds
    ioTimeout := cfg.PythonExecutorIOTimeoutSeconds
    maxConnections := cfg.PythonExecutorMaxConnections
	tool := &StatefulPythonTool{
		pool:                      pool,
		logger:                    logger,
		dialTimeout:               dialTimeout,
		ioTimeout:                 ioTimeout,
		sessionAddr:               make(map[string]string),
		connPools:                 make(map[string]*connPool),
		maxConnectionsPerExecutor: maxConnections,
	}
	if err := tool.ensureInitialConnectivity(ctx); err != nil {
		return nil, err
	}
	if tool.logger != nil {
		tool.logger.Info("Python tool initialized", zap.Strings("addresses", tool.pool.Addresses()))
	}
	return tool, nil
}

func (t *StatefulPythonTool) getConnPool(address string) *connPool {
	t.connPoolsMu.RLock()
	pool := t.connPools[address]
	t.connPoolsMu.RUnlock()
	if pool != nil {
		return pool
	}

	t.connPoolsMu.Lock()
	defer t.connPoolsMu.Unlock()
	if pool = t.connPools[address]; pool == nil {
		pool = newConnPool(address, t.maxConnectionsPerExecutor, func(ctx context.Context) (net.Conn, error) {
			return t.dial(ctx, address)
		})
		t.connPools[address] = pool
	}
	return pool
}

func (t *StatefulPythonTool) ensureInitialConnectivity(ctx context.Context) error {
	addresses := t.pool.Addresses()
	var lastErr error
	for _, addr := range addresses {
		cp := t.getConnPool(addr)
		conn, err := cp.Get(ctx)
		if err != nil {
			t.pool.MarkFailure(addr)
			lastErr = err
			if t.logger != nil {
				t.logger.Warn("Initial executor health check failed", zap.String("address", addr), zap.Error(err))
			}
			continue
		}
		cp.Put(conn)
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
import scipy
import warnings
from scipy import stats
warnings.filterwarnings('ignore')

pd.set_option('display.precision', 3)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 0)
pd.set_option('display.max_colwidth', None)

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
	cp := t.getConnPool(addr)
	conn, err := cp.Get(ctx)
	if err != nil {
		t.pool.MarkFailure(addr)
		if t.logger != nil {
			t.logger.Warn("Failed to connect to python executor", zap.String("address", addr), zap.Error(err))
		}
		return "", fmt.Errorf("dial python server %s: %w", addr, err)
	}

	result, execErr := t.execute(conn, input, sessionID)
	if execErr != nil {
		cp.Discard(conn)
		t.pool.MarkFailure(addr)
		if t.logger != nil {
			t.logger.Warn("Python executor call failed", zap.String("address", addr), zap.Error(execErr))
		}
		return "", fmt.Errorf("executor %s: %w", addr, execErr)
	}

	cp.Put(conn)
	t.pool.MarkSuccess(addr)
	if t.logger != nil {
		t.logger.Debug("Python code executed", zap.String("address", addr), zap.String("session_id", sessionID))
	}
	return result, nil
}

func (t *StatefulPythonTool) Close() {
	t.connPoolsMu.Lock()
	defer t.connPoolsMu.Unlock()
	for addr, pool := range t.connPools {
		if pool != nil {
			pool.Close()
		}
		delete(t.connPools, addr)
	}
}

// CleanupSession removes the session binding from the executor pool
func (t *StatefulPythonTool) CleanupSession(sessionID string) {
	t.sessionMu.Lock()
	defer t.sessionMu.Unlock()
	delete(t.sessionAddr, sessionID)
	if t.logger != nil {
		t.logger.Info("Python session cleaned up", zap.String("session_id", sessionID))
	}
}

// ExecutePythonCode now requires a sessionID to be passed.
// Supports markdown code blocks (```python) only.
func (t *StatefulPythonTool) ExecutePythonCode(ctx context.Context, text string, sessionID string, output io.Writer) (string, string, bool) {
	pythonCode := extractMarkdownCode(text)
	if pythonCode == "" {
		return "", "", false
	}

	// Log execution without full code (which could contain sensitive data)
	codeLines := strings.Count(pythonCode, "\n") + 1
	t.logger.Info("Executing Python code",
		zap.String("session_id", sessionID),
		zap.Int("code_lines", codeLines))

	execResult, err := t.Call(ctx, pythonCode, sessionID)
	if err != nil {
		t.logger.Error("Error executing Python code", zap.Error(err), zap.String("session_id", sessionID))
		execResult = "Error: " + err.Error()
	} else {
		// Only log result preview in debug mode, and sanitize it
		if t.logger.Core().Enabled(zap.DebugLevel) {
			preview := sanitizeLogOutput(execResult, 100)
			t.logger.Debug("Python code executed successfully",
				zap.String("session_id", sessionID),
				zap.String("result_preview", preview))
		}
	}

	return pythonCode, execResult, true
}

// sanitizeLogOutput truncates and removes potentially sensitive patterns from log output
func sanitizeLogOutput(s string, maxLen int) string {
	// Truncate to max length
	if len(s) > maxLen {
		s = s[:maxLen] + "..."
	}

	// Remove common patterns that might contain secrets
	// (passwords, tokens, keys, etc.)
	sensitive := []string{
		"password", "passwd", "pwd",
		"token", "api_key", "apikey", "secret",
		"credentials", "auth",
	}

	lower := strings.ToLower(s)
	for _, pattern := range sensitive {
		if strings.Contains(lower, pattern) {
			return "[Output contains potentially sensitive data - not logged]"
		}
	}

	return s
}

// extractMarkdownCode extracts Python code from markdown code blocks (```python ... ```)
func extractMarkdownCode(text string) string {
    // Preferred path: explicit ```python fences
    if startIdx := strings.Index(text, "```python"); startIdx != -1 {
        codeStart := startIdx + len("```python")
        if codeStart < len(text) && text[codeStart] == '\n' {
            codeStart++
        }
        if endRel := strings.Index(text[codeStart:], "```"); endRel != -1 {
            code := text[codeStart : codeStart+endRel]
            return strings.TrimSpace(code)
        }
        return ""
    }

    // Fallback: generic ``` fences that look like Python
    // Find first generic opening fence
    open := "```"
    gStart := strings.Index(text, open)
    if gStart == -1 {
        return ""
    }
    codeStart := gStart + len(open)
    if codeStart < len(text) && text[codeStart] == '\n' {
        codeStart++
    }
    // Find its closing fence
    gEndRel := strings.Index(text[codeStart:], open)
    if gEndRel == -1 {
        return ""
    }
    candidate := strings.TrimSpace(text[codeStart : codeStart+gEndRel])
    if looksLikePython(candidate) {
        return candidate
    }
    return ""
}

// looksLikePython returns true if the snippet contains pythonic tokens.
func looksLikePython(code string) bool {
    lc := strings.ToLower(code)
    // Common Python/data-science tokens
    tokens := []string{
        "import ", "from ", "pd.", "plt.", "sns.", "stats.",
        "df =", "print(", "def ", "for ", "in ", "os.", "np.",
    }
    for _, t := range tokens {
        if strings.Contains(lc, t) {
            return true
        }
    }
    return false
}
