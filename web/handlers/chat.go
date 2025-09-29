package handlers

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"stats-agent/agent"
	"stats-agent/web/templates/components"
	"stats-agent/web/templates/pages"
	"stats-agent/web/types"
	"strings"
	"sync"
	"time"

	"github.com/a-h/templ"
	"github.com/gin-gonic/gin"
	"github.com/google/uuid"
	"go.uber.org/zap"
)

type ChatHandler struct {
	agent    *agent.Agent
	logger   *zap.Logger
	sessions map[string]*ChatSession
	mu       sync.RWMutex
}

type ChatSession struct {
	ID            string
	Messages      []types.ChatMessage
	LastAccess    time.Time
	mu            sync.RWMutex
	RenderedFiles map[string]bool
}

type ChatRequest struct {
	Message   string `json:"message" form:"message"`
	SessionID string `json:"session_id" form:"session_id"`
}

type StreamData struct {
	Type    string `json:"type"`
	Content string `json:"content,omitempty"`
}

func NewChatHandler(agent *agent.Agent, logger *zap.Logger) *ChatHandler {
	return &ChatHandler{
		agent:    agent,
		logger:   logger,
		sessions: make(map[string]*ChatSession),
	}
}

func (h *ChatHandler) Index(c *gin.Context) {
	sessionID := generateSessionID()

	// Create session workspace
	workspaceDir := filepath.Join("workspaces", sessionID)
	if err := os.MkdirAll(workspaceDir, 0755); err != nil {
		h.logger.Error("Failed to create workspace directory", zap.Error(err))
		c.String(http.StatusInternalServerError, "Could not create workspace.")
		return
	}

	// Create new session
	h.mu.Lock()
	h.sessions[sessionID] = &ChatSession{
		ID:            sessionID,
		Messages:      []types.ChatMessage{},
		LastAccess:    time.Now(),
		RenderedFiles: make(map[string]bool),
	}
	h.mu.Unlock()

	component := pages.ChatPage(sessionID)
	component.Render(c.Request.Context(), c.Writer)
}

func (h *ChatHandler) SendMessage(c *gin.Context) {
	var req ChatRequest
	if err := c.ShouldBind(&req); err != nil {
		h.logger.Error("Failed to bind chat request", zap.Error(err))
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid request"})
		return
	}

	if req.Message == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Message cannot be empty"})
		return
	}

	// Get or create session
	h.mu.Lock()
	session, exists := h.sessions[req.SessionID]
	if !exists {
		session = &ChatSession{
			ID:            req.SessionID,
			Messages:      []types.ChatMessage{},
			RenderedFiles: make(map[string]bool),
		}
		h.sessions[req.SessionID] = session
	}
	session.LastAccess = time.Now()
	h.mu.Unlock()

	// Handle potential file upload
	file, err := c.FormFile("file")
	if err == nil {
		// **1. Sanitize the original filename**
		sanitizedFilename := sanitizeFilename(file.Filename)
		if sanitizedFilename == "" {
			c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid or unsafe filename."})
			return
		}

		// A file was included, process it
		ext := strings.ToLower(filepath.Ext(file.Filename))
		if ext != ".csv" && ext != ".xlsx" && ext != ".xls" {
			c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid file type. Please upload a CSV or Excel file."})
			return
		}

		workspaceDir := filepath.Join("workspaces", req.SessionID)
		dst := filepath.Join(workspaceDir, file.Filename)
		if err := c.SaveUploadedFile(file, dst); err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": "Could not save file"})
			return
		}

		// **2. Verify that the sanitized file now exists on disk**
		if !verifyFileExists(workspaceDir, sanitizedFilename) {
			c.JSON(http.StatusInternalServerError, gin.H{"error": "File verification failed after upload."})
			return
		}

		// Modify the user's message to include file info prominently
		if strings.TrimSpace(req.Message) == "" {
			req.Message = fmt.Sprintf("I've uploaded %s. Please analyze this dataset and provide statistical insights.", file.Filename)
		} else {
			req.Message = fmt.Sprintf("[📎 File uploaded: %s]\n\n%s", file.Filename, req.Message)
		}

		h.logger.Info("File uploaded", zap.String("filename", file.Filename), zap.String("session_id", req.SessionID))

		// Mark the uploaded file as rendered
		session.mu.Lock()
		session.RenderedFiles[file.Filename] = true
		session.mu.Unlock()
	}

	// Add user's message to session (now includes file info if uploaded)
	userMessage := types.ChatMessage{
		Role:      "user",
		Content:   req.Message,
		ID:        generateMessageID(),
		SessionID: req.SessionID,
	}

	session.mu.Lock()
	session.Messages = append(session.Messages, userMessage)
	session.mu.Unlock()

	h.logger.Info("Processing chat message", zap.String("session_id", req.SessionID), zap.String("message", req.Message))

	// Return the user message immediately for HTMX to display
	component := components.UserMessage(userMessage)
	c.Header("Content-Type", "text/html")
	component.Render(c.Request.Context(), c.Writer)
}
func (h *ChatHandler) UploadFile(c *gin.Context) {
	sessionID := c.PostForm("session_id")
	if sessionID == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Session ID is required"})
		return
	}

	h.mu.RLock()
	session, exists := h.sessions[sessionID]
	h.mu.RUnlock()

	if !exists {
		c.JSON(http.StatusNotFound, gin.H{"error": "Session not found"})
		return
	}
	session.LastAccess = time.Now()

	file, err := c.FormFile("file")
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "File upload error"})
		return
	}

	// **Server-side validation for file type**
	ext := strings.ToLower(filepath.Ext(file.Filename))
	if ext != ".csv" && ext != ".xlsx" && ext != ".xls" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid file type. Please upload a CSV or Excel file."})
		return
	}

	// Save the file to the session's workspace
	workspaceDir := filepath.Join("workspaces", sessionID)
	dst := filepath.Join(workspaceDir, file.Filename)
	if err := c.SaveUploadedFile(file, dst); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Could not save file"})
		return
	}

	// Add a system message to inform the agent of the new file.
	systemMessage := types.ChatMessage{
		Role:      "system",
		Content:   fmt.Sprintf("The user has uploaded a file: %s. Unless specified otherwise, use this file for your analysis.", file.Filename),
		ID:        generateMessageID(),
		SessionID: sessionID,
	}
	session.mu.Lock()
	session.Messages = append(session.Messages, systemMessage)
	session.mu.Unlock()

	c.JSON(http.StatusOK, gin.H{"message": fmt.Sprintf("File %s uploaded successfully.", file.Filename)})
}

func (h *ChatHandler) StreamResponse(c *gin.Context) {
	sessionID := c.Query("session_id")
	userMessageID := c.Query("user_message_id")

	if sessionID == "" || userMessageID == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Session ID and user message ID required"})
		return
	}

	// Set SSE headers
	c.Header("Content-Type", "text/event-stream")
	c.Header("Cache-Control", "no-cache")
	c.Header("Connection", "keep-alive")
	c.Header("Access-Control-Allow-Origin", "*")

	// Test connection first - send as JSON
	data := StreamData{Type: "connection_established"}
	jsonData, _ := json.Marshal(data)
	fmt.Fprintf(c.Writer, "data: %s\n\n", jsonData)
	c.Writer.(http.Flusher).Flush()

	h.mu.RLock()
	session, exists := h.sessions[sessionID]
	h.mu.RUnlock()

	if !exists {
		fmt.Fprintf(c.Writer, "data: Session not found\n\n")
		c.Writer.(http.Flusher).Flush()
		return
	}
	session.LastAccess = time.Now()

	// Create a context for this streaming session
	ctx := c.Request.Context()

	// Find the user message
	session.mu.RLock()
	var userMessage *types.ChatMessage
	for _, msg := range session.Messages {
		if msg.ID == userMessageID {
			userMessage = &msg
			break
		}
	}
	session.mu.RUnlock()

	if userMessage == nil {
		fmt.Fprintf(c.Writer, "data: User message not found\n\n")
		c.Writer.(http.Flusher).Flush()
		return
	}

	// Start streaming the agent response
	h.streamAgentResponse(ctx, c.Writer, userMessage.Content, userMessageID, sessionID)
}

// processStreamByWord reads from the stream rune by rune, buffers them into words,
// and processes each word for tags before sending it to the client. This version is stateful.
func (h *ChatHandler) processStreamByWord(ctx context.Context, r io.Reader, writeSSEData func(StreamData) error) {
	reader := bufio.NewReader(r)
	var currentWord strings.Builder

	// processToken is a recursive function that handles tags within a word/token.
	var processToken func(string)
	processToken = func(token string) {
		if token == "" {
			return
		}

		// Check for our special tags
		switch {
		case strings.Contains(token, "<python>"):
			parts := strings.SplitN(token, "<python>", 2)
			writeSSEData(StreamData{Type: "chunk", Content: parts[0]})
			writeSSEData(StreamData{Type: "chunk", Content: "\n```python\n"})
			processToken(parts[1])
		case strings.Contains(token, "</python>"):
			parts := strings.SplitN(token, "</python>", 2)
			writeSSEData(StreamData{Type: "chunk", Content: parts[0]})
			writeSSEData(StreamData{Type: "chunk", Content: "\n```\n"})
			processToken(parts[1])
		case strings.Contains(token, "<execution_results>"):
			parts := strings.SplitN(token, "<execution_results>", 2)
			writeSSEData(StreamData{Type: "chunk", Content: parts[0]})
			writeSSEData(StreamData{Type: "chunk", Content: "\n```\n"})
			processToken(parts[1])
		case strings.Contains(token, "</execution_results>"):
			parts := strings.SplitN(token, "</execution_results>", 2)
			writeSSEData(StreamData{Type: "chunk", Content: parts[0]})
			writeSSEData(StreamData{Type: "chunk", Content: "\n```\n"})
			processToken(parts[1])
		case strings.Contains(token, "<agent_status>"):
			parts := strings.SplitN(token, "<agent_status>", 2)
			writeSSEData(StreamData{Type: "chunk", Content: parts[0]})
			writeSSEData(StreamData{Type: "chunk", Content: `<div class="agent-status-message">`})
			processToken(parts[1])
		case strings.Contains(token, "</agent_status>"):
			parts := strings.SplitN(token, "</agent_status>", 2)
			writeSSEData(StreamData{Type: "chunk", Content: parts[0]})
			writeSSEData(StreamData{Type: "chunk", Content: `</div>`})
			processToken(parts[1])
		default:
			writeSSEData(StreamData{Type: "chunk", Content: token})
		}
	}

	for {
		select {
		case <-ctx.Done():
			return
		default:
			char, _, err := reader.ReadRune()
			if err != nil {
				if currentWord.Len() > 0 {
					processToken(currentWord.String())
				}
				return
			}

			currentWord.WriteRune(char)

			if char == ' ' || char == '\n' {
				processToken(currentWord.String())
				currentWord.Reset()
			}
		}
	}
}

func (h *ChatHandler) streamAgentResponse(ctx context.Context, w http.ResponseWriter, input string, userMessageID string, sessionID string) {
	agentMessageID := generateMessageID()
	var writeMu sync.Mutex

	writeSSEData := func(data StreamData) error {
		writeMu.Lock()
		defer writeMu.Unlock()
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}
		jsonData, err := json.Marshal(data)
		if err != nil {
			return err
		}
		_, err = fmt.Fprintf(w, "data: %s\n\n", jsonData)
		if err != nil {
			return err
		}
		if flusher, ok := w.(http.Flusher); ok {
			flusher.Flush()
		}
		return nil
	}

	if err := writeSSEData(StreamData{Type: "remove_loader", Content: "loading-" + userMessageID}); err != nil {
		h.logger.Error("Failed to send remove loader message", zap.Error(err))
		return
	}

	if err := writeSSEData(StreamData{Type: "create_container", Content: agentMessageID}); err != nil {
		h.logger.Error("Failed to send create container message", zap.Error(err))
		return
	}

	r, pipeW, err := os.Pipe()
	if err != nil {
		h.logger.Error("Failed to create pipe", zap.Error(err))
		writeSSEData(StreamData{Type: "error", Content: "Internal server error"})
		return
	}

	originalStdout := os.Stdout
	os.Stdout = pipeW
	log.SetOutput(pipeW)

	agentDone := make(chan error, 1)
	streamDone := make(chan struct{})

	// Goroutine to stream the captured output using our new word-by-word processor
	go func() {
		defer close(streamDone)
		h.processStreamByWord(ctx, r, func(data StreamData) error {
			if err := writeSSEData(data); err != nil {
				h.logger.Error("Error writing chunk data", zap.Error(err))
				return err
			}
			return nil
		})
	}()

	// Run the agent in a separate goroutine
	go func() {
		defer func() {
			os.Stdout = originalStdout
			log.SetOutput(originalStdout)
			pipeW.Close()
			r.Close()
		}()
		h.agent.Run(ctx, input, sessionID)
		close(agentDone)
	}()

	select {
	case <-ctx.Done():
		h.logger.Info("Context cancelled, closing SSE connection")
	case <-agentDone:
		// Agent finished, wait for the stream processing to complete.
		<-streamDone

		// Scan for and stream any new files.
		h.streamNewFiles(ctx, writeSSEData, sessionID)
		// Send the final end-of-stream message.
		if err := writeSSEData(StreamData{Type: "end"}); err != nil {
			h.logger.Error("Failed to send end message", zap.Error(err))
		}
	}
}

func (h *ChatHandler) CleanupWorkspaces(maxAge time.Duration, logger *zap.Logger) {
	h.mu.Lock()
	defer h.mu.Unlock()

	for sessionID, session := range h.sessions {
		if time.Since(session.LastAccess) > maxAge {
			workspaceDir := filepath.Join("workspaces", sessionID)
			logger.Info("Cleaning up stale workspace", zap.String("session_id", sessionID), zap.String("workspace", workspaceDir))
			os.RemoveAll(workspaceDir)
			delete(h.sessions, sessionID)
		}
	}
}

func (h *ChatHandler) streamNewFiles(ctx context.Context, writeSSEData func(StreamData) error, sessionID string) {
	session, exists := h.sessions[sessionID]
	if !exists {
		h.logger.Error("Session not found when streaming new files", zap.String("session_id", sessionID))
		return
	}

	workspaceDir := filepath.Join("workspaces", session.ID)
	files, err := os.ReadDir(workspaceDir)
	if err != nil {
		h.logger.Error("Failed to read workspace directory", zap.Error(err))
		return
	}

	session.mu.Lock()
	defer session.mu.Unlock()

	for _, file := range files {
		if !file.IsDir() {
			fileName := file.Name()
			if _, rendered := session.RenderedFiles[fileName]; !rendered {
				webPath := filepath.ToSlash(filepath.Join("/workspaces", session.ID, fileName))
				var buf bytes.Buffer
				var component templ.Component

				switch strings.ToLower(filepath.Ext(fileName)) {
				case ".png", ".jpg", ".jpeg", ".gif":
					component = components.ImageBlock(webPath)
				case ".csv", ".xlsx", ".xls":
					component = components.FileBlock(webPath)
				}

				if component != nil {
					if err := component.Render(ctx, &buf); err == nil {
						writeSSEData(StreamData{Type: "file", Content: buf.String()})
						session.RenderedFiles[fileName] = true
					}
				}
			}
		}
	}
}

// Sanitize the filename to prevent security issues like directory traversal.
func sanitizeFilename(filename string) string {
	// Trim whitespace and periods from the ends
	sanitized := strings.Trim(filename, " .")

	// Replace known dangerous sequences.
	sanitized = strings.ReplaceAll(sanitized, "..", "")

	// Define a regex for allowed characters: letters, numbers, underscore, hyphen, period, space.
	// This is a whitelist approach, which is more secure.
	reg := regexp.MustCompile(`[^a-zA-Z0-9._\s-]`)
	sanitized = reg.ReplaceAllString(sanitized, "")

	// Limit the length of the filename to a reasonable size.
	if len(sanitized) > 255 {
		sanitized = sanitized[:255]
	}

	return sanitized
}

// Verify that the file exists in the session's workspace.
func verifyFileExists(workspaceDir, filename string) bool {
	// Create the full, safe path to the file.
	safePath := filepath.Join(workspaceDir, filename)

	// Check if the file exists and is not a directory.
	info, err := os.Stat(safePath)
	if os.IsNotExist(err) {
		return false // File does not exist
	}
	if info.IsDir() {
		return false // It's a directory, not a file
	}
	return true
}

func generateSessionID() string {
	return "session_" + uuid.New().String()
}

func generateMessageID() string {
	return "msg_" + uuid.New().String()
}
