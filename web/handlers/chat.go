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
	"stats-agent/database"
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
	agent  *agent.Agent
	logger *zap.Logger
	store  *database.PostgresStore
}

type ChatSession struct {
	ID            string
	Messages      []types.ChatMessage
	LastAccess    time.Time
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

func NewChatHandler(agent *agent.Agent, logger *zap.Logger, store *database.PostgresStore) *ChatHandler {
	return &ChatHandler{
		agent:  agent,
		logger: logger,
		store:  store,
	}
}

func (h *ChatHandler) Index(c *gin.Context) {
	sessionID := c.MustGet("sessionID").(uuid.UUID)

	// Create session workspace if it doesn't exist
	workspaceDir := filepath.Join("workspaces", sessionID.String())
	if err := os.MkdirAll(workspaceDir, 0755); err != nil {
		h.logger.Error("Failed to create workspace directory", zap.Error(err))
		c.String(http.StatusInternalServerError, "Could not create workspace.")
		return
	}

	sessions, _ := h.store.GetSessions(c.Request.Context(), nil)
	messages, _ := h.store.GetMessagesBySession(c.Request.Context(), sessionID)

	component := pages.ChatPage(sessionID, sessions, messages)
	component.Render(c.Request.Context(), c.Writer)
}

func (h *ChatHandler) LoadSession(c *gin.Context) {
	sessionID, err := uuid.Parse(c.Param("sessionID"))
	if err != nil {
		c.String(http.StatusBadRequest, "Invalid session ID")
		return
	}

	// In a real app, you'd also check if the user has permission to view this session
	sessions, _ := h.store.GetSessions(c.Request.Context(), nil) // Assuming GetSessions exists
	messages, _ := h.store.GetMessagesBySession(c.Request.Context(), sessionID)

	pages.ChatPage(sessionID, sessions, messages).Render(c.Request.Context(), c.Writer)
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

	sessionID, err := uuid.Parse(req.SessionID)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid session ID"})
		return
	}

	// Handle potential file upload
	file, err := c.FormFile("file")
	if err == nil {
		sanitizedFilename := sanitizeFilename(file.Filename)
		if sanitizedFilename == "" {
			c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid or unsafe filename."})
			return
		}

		ext := strings.ToLower(filepath.Ext(file.Filename))
		if ext != ".csv" && ext != ".xlsx" && ext != ".xls" {
			c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid file type. Please upload a CSV or Excel file."})
			return
		}

		workspaceDir := filepath.Join("workspaces", req.SessionID)
		dst := filepath.Join(workspaceDir, sanitizedFilename)
		if err := c.SaveUploadedFile(file, dst); err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": "Could not save file"})
			return
		}

		if !verifyFileExists(workspaceDir, sanitizedFilename) {
			c.JSON(http.StatusInternalServerError, gin.H{"error": "File verification failed after upload."})
			return
		}

		if strings.TrimSpace(req.Message) == "" {
			req.Message = fmt.Sprintf("I've uploaded %s. Please analyze this dataset and provide statistical insights.", file.Filename)
		} else {
			req.Message = fmt.Sprintf("[📎 File uploaded: %s]\n\n%s", file.Filename, req.Message)
		}

		h.logger.Info("File uploaded", zap.String("filename", file.Filename), zap.String("session_id", req.SessionID))
	}

	userMessage := types.ChatMessage{
		Role:      "user",
		Content:   req.Message,
		ID:        generateMessageID(),
		SessionID: sessionID.String(),
	}

	if err := h.store.CreateMessage(c.Request.Context(), userMessage); err != nil {
		h.logger.Error("Failed to save user message", zap.Error(err))
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Could not save message"})
		return
	}

	h.logger.Info("Processing chat message", zap.String("session_id", req.SessionID), zap.String("message", req.Message))

	component := components.UserMessage(userMessage)
	c.Header("Content-Type", "text/html")
	component.Render(c.Request.Context(), c.Writer)
}
func (h *ChatHandler) UploadFile(c *gin.Context) {
	sessionIDStr := c.PostForm("session_id")
	if sessionIDStr == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Session ID is required"})
		return
	}

	sessionID, err := uuid.Parse(sessionIDStr)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid session ID"})
		return
	}

	file, err := c.FormFile("file")
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "File upload error"})
		return
	}

	ext := strings.ToLower(filepath.Ext(file.Filename))
	if ext != ".csv" && ext != ".xlsx" && ext != ".xls" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid file type. Please upload a CSV or Excel file."})
		return
	}

	workspaceDir := filepath.Join("workspaces", sessionID.String())
	dst := filepath.Join(workspaceDir, file.Filename)
	if err := c.SaveUploadedFile(file, dst); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Could not save file"})
		return
	}

	systemMessage := types.ChatMessage{
		Role:      "system",
		Content:   fmt.Sprintf("The user has uploaded a file: %s. Unless specified otherwise, use this file for your analysis.", file.Filename),
		ID:        generateMessageID(),
		SessionID: sessionID.String(),
	}

	if err := h.store.CreateMessage(c.Request.Context(), systemMessage); err != nil {
		h.logger.Error("Failed to save system message", zap.Error(err))
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Could not save message"})
		return
	}

	c.JSON(http.StatusOK, gin.H{"message": fmt.Sprintf("File %s uploaded successfully.", file.Filename)})
}

func (h *ChatHandler) StreamResponse(c *gin.Context) {
	sessionIDStr := c.Query("session_id")
	userMessageID := c.Query("user_message_id")

	if sessionIDStr == "" || userMessageID == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Session ID and user message ID required"})
		return
	}
	sessionID, err := uuid.Parse(sessionIDStr)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid Session ID"})
		return
	}

	c.Header("Content-Type", "text/event-stream")
	c.Header("Cache-Control", "no-cache")
	c.Header("Connection", "keep-alive")
	c.Header("Access-Control-Allow-Origin", "*")

	data := StreamData{Type: "connection_established"}
	jsonData, _ := json.Marshal(data)
	fmt.Fprintf(c.Writer, "data: %s\n\n", jsonData)
	c.Writer.(http.Flusher).Flush()

	ctx := c.Request.Context()

	messages, err := h.store.GetMessagesBySession(ctx, sessionID)
	if err != nil {
		fmt.Fprintf(c.Writer, "data: Error fetching messages\n\n")
		c.Writer.(http.Flusher).Flush()
		return
	}

	var userMessage *types.ChatMessage
	for i := range messages {
		if messages[i].ID == userMessageID {
			userMessage = &messages[i]
			break
		}
	}

	if userMessage == nil {
		fmt.Fprintf(c.Writer, "data: User message not found\n\n")
		c.Writer.(http.Flusher).Flush()
		return
	}

	h.streamAgentResponse(ctx, c.Writer, userMessage.Content, userMessageID, sessionID.String())
}

// processStreamByWord reads from the stream rune by rune, buffers them into words,
// and processes each word for tags before sending it to the client. This version is stateful.
func (h *ChatHandler) processStreamByWord(ctx context.Context, r io.Reader, writeSSEData func(StreamData) error) {
	reader := bufio.NewReader(r)
	var currentWord strings.Builder

	var processToken func(string)
	processToken = func(token string) {
		if token == "" {
			return
		}

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
		<-streamDone
		h.streamNewFiles(ctx, writeSSEData, sessionID)
		if err := writeSSEData(StreamData{Type: "end"}); err != nil {
			h.logger.Error("Failed to send end message", zap.Error(err))
		}
	}
}

func (h *ChatHandler) streamNewFiles(ctx context.Context, writeSSEData func(StreamData) error, sessionID string) {
	sessionUUID, err := uuid.Parse(sessionID)
	if err != nil {
		h.logger.Error("Invalid session ID for streaming new files", zap.Error(err))
		return
	}

	workspaceDir := filepath.Join("workspaces", sessionID)
	files, err := os.ReadDir(workspaceDir)
	if err != nil {
		h.logger.Error("Failed to read workspace directory", zap.Error(err), zap.String("session_id", sessionID))
		return
	}

	// Fetch the set of already rendered files from the database
	renderedFiles, err := h.store.GetRenderedFiles(ctx, sessionUUID)
	if err != nil {
		h.logger.Error("Failed to get rendered files from DB", zap.Error(err), zap.String("session_id", sessionID))
		return
	}

	for _, file := range files {
		if !file.IsDir() {
			fileName := file.Name()
			if _, rendered := renderedFiles[fileName]; !rendered {
				webPath := filepath.ToSlash(filepath.Join("/workspaces", sessionID, fileName))
				var buf bytes.Buffer
				var component templ.Component

				// Determine which component to use based on file type
				switch strings.ToLower(filepath.Ext(fileName)) {
				case ".png", ".jpg", ".jpeg", ".gif":
					component = components.ImageBlock(webPath)
				default:
					component = components.FileBlock(webPath)
				}

				if component != nil {
					if err := component.Render(ctx, &buf); err == nil {
						writeSSEData(StreamData{Type: "file", Content: buf.String()})
						// Add the file to the database of rendered files
						if err := h.store.AddRenderedFile(ctx, sessionUUID, fileName); err != nil {
							h.logger.Error("Failed to mark file as rendered in DB", zap.Error(err), zap.String("filename", fileName))
						}
					}
				}
			}
		}
	}
}

// Sanitize the filename to prevent security issues like directory traversal.
func sanitizeFilename(filename string) string {
	sanitized := strings.Trim(filename, " .")
	sanitized = strings.ReplaceAll(sanitized, "..", "")
	reg := regexp.MustCompile(`[^a-zA-Z0-9._\s-]`)
	sanitized = reg.ReplaceAllString(sanitized, "")
	if len(sanitized) > 255 {
		sanitized = sanitized[:255]
	}
	return sanitized
}

// Verify that the file exists in the session's workspace.
func verifyFileExists(workspaceDir, filename string) bool {
	safePath := filepath.Join(workspaceDir, filename)
	info, err := os.Stat(safePath)
	if os.IsNotExist(err) {
		return false
	}
	if info.IsDir() {
		return false
	}
	return true
}

func generateMessageID() string {
	return uuid.New().String()
}
