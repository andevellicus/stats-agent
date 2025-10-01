package handlers

import (
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"stats-agent/database"
	"stats-agent/web/middleware"
	"stats-agent/web/services"
	"stats-agent/web/templates/components"
	"stats-agent/web/templates/pages"
	"stats-agent/web/types"
	"strings"

	"github.com/gin-gonic/gin"
	"github.com/google/uuid"
	"go.uber.org/zap"
)

type ChatHandler struct {
	chatService *services.ChatService
	logger      *zap.Logger
	store       *database.PostgresStore
}

type ChatRequest struct {
	Message   string `json:"message" form:"message"`
	SessionID string `json:"session_id" form:"session_id"`
}

func NewChatHandler(chatService *services.ChatService, logger *zap.Logger, store *database.PostgresStore) *ChatHandler {
	return &ChatHandler{
		chatService: chatService,
		logger:      logger,
		store:       store,
	}
}

func (h *ChatHandler) NewChat(c *gin.Context) {
	c.Header("Cache-Control", "no-cache, no-store, must-revalidate") // Add this line
	// By setting the cookie's max age to -1, we tell the browser to delete it.
	c.SetCookie(middleware.SessionCookieName, "", -1, "/", "", false, true)
	// Redirect to the home page. The session middleware will now see no cookie and create a new session.
	c.Redirect(http.StatusFound, "/")
}

func (h *ChatHandler) DeleteSession(c *gin.Context) {
	sessionIDStr := c.Param("sessionID")
	sessionID, err := uuid.Parse(sessionIDStr)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "invalid session ID"})
		return
	}

	// Get session info before deleting (to get workspace path)
	session, err := h.store.GetSessionByID(c.Request.Context(), sessionID)
	if err != nil {
		h.logger.Error("Failed to get session for deletion", zap.Error(err), zap.String("session_id", sessionIDStr))
		c.JSON(http.StatusNotFound, gin.H{"error": "Session not found"})
		return
	}

	// Delete from database (this cascades to messages)
	if err := h.store.DeleteSession(c.Request.Context(), sessionID); err != nil {
		h.logger.Error("Failed to delete session from database", zap.Error(err), zap.String("session_id", sessionIDStr))
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to delete session"})
		return
	}

	// Cleanup Python executor session binding
	h.chatService.CleanupSession(sessionIDStr)

	// Delete workspace directory
	workspaceDir := session.WorkspacePath
	if workspaceDir != "" {
		if err := os.RemoveAll(workspaceDir); err != nil {
			h.logger.Warn("Failed to delete workspace directory", zap.Error(err), zap.String("path", workspaceDir))
		} else {
			h.logger.Info("Workspace directory deleted", zap.String("path", workspaceDir))
		}
	}

	h.logger.Info("Session deleted successfully", zap.String("session_id", sessionIDStr))

	// Check if this was the current session
	currentSessionID, exists := c.Get("sessionID")
	if exists && currentSessionID.(uuid.UUID) == sessionID {
		// Deleting the current session - clear cookie and redirect to create new session
		c.SetCookie(middleware.SessionCookieName, "", -1, "/", "", false, true)
	}

	// Always redirect to home page to refresh the UI
	// The HX-Redirect header tells HTMX to perform a full page redirect
	c.Header("HX-Redirect", "/")
	c.Status(http.StatusOK)
}

func (h *ChatHandler) Index(c *gin.Context) {
	sessionID, exists := c.Get("sessionID")
	if !exists {
		h.logger.Error("Session ID not found in context")
		c.String(http.StatusInternalServerError, "Session not found")
		return
	}
	sessionUUID := sessionID.(uuid.UUID)

	userID, userExists := c.Get("userID")
	var userUUIDPtr *uuid.UUID
	if userExists {
		userUUID := userID.(uuid.UUID)
		userUUIDPtr = &userUUID
	}

	workspaceDir := filepath.Join("workspaces", sessionUUID.String())
	if err := os.MkdirAll(workspaceDir, 0755); err != nil {
		h.logger.Error("Failed to create workspace directory",
			zap.Error(err),
			zap.String("session_id", sessionUUID.String()))
		c.String(http.StatusInternalServerError, "Could not create workspace")
		return
	}

	// Get sessions - non-critical, show empty sidebar if fails
	sessions, err := h.store.GetSessions(c.Request.Context(), userUUIDPtr)
	if err != nil {
		h.logger.Error("Failed to get sessions for sidebar",
			zap.Error(err),
			zap.String("session_id", sessionUUID.String()))
		sessions = []types.Session{} // Empty sidebar
	}

	// Get messages - critical for page render
	messages, err := h.store.GetMessagesBySession(c.Request.Context(), sessionUUID)
	if err != nil {
		h.logger.Error("Failed to get messages for session",
			zap.Error(err),
			zap.String("session_id", sessionUUID.String()))
		c.String(http.StatusInternalServerError, "Could not load conversation history")
		return
	}

	messageGroups := groupMessages(messages)
	component := pages.ChatPage(sessionUUID, sessions, messageGroups)
	component.Render(c.Request.Context(), c.Writer)
}

func (h *ChatHandler) LoadSession(c *gin.Context) {
	sessionID, err := uuid.Parse(c.Param("sessionID"))
	if err != nil {
		c.String(http.StatusBadRequest, "invalid session ID")
		return
	}

	userID, userExists := c.Get("userID")
	var userUUIDPtr *uuid.UUID
	if userExists {
		userUUID := userID.(uuid.UUID)
		userUUIDPtr = &userUUID
	}

	// Verify the session belongs to this user
	session, err := h.store.GetSessionByID(c.Request.Context(), sessionID)
	if err != nil {
		h.logger.Error("Failed to get session",
			zap.Error(err),
			zap.String("session_id", sessionID.String()))
		c.String(http.StatusNotFound, "Session not found")
		return
	}

	// Check ownership - security check
	if userUUIDPtr != nil && (session.UserID == nil || *session.UserID != *userUUIDPtr) {
		h.logger.Warn("Attempted to access session belonging to different user",
			zap.String("session_id", sessionID.String()),
			zap.String("user_id", userUUIDPtr.String()))
		c.Redirect(http.StatusFound, "/")
		return
	}

	// Get sessions - non-critical, show empty sidebar if fails
	sessions, err := h.store.GetSessions(c.Request.Context(), userUUIDPtr)
	if err != nil {
		h.logger.Error("Failed to get sessions for sidebar",
			zap.Error(err),
			zap.String("session_id", sessionID.String()))
		sessions = []types.Session{} // Empty sidebar
	}

	// Get messages - critical for page render
	messages, err := h.store.GetMessagesBySession(c.Request.Context(), sessionID)
	if err != nil {
		h.logger.Error("Failed to get messages for session",
			zap.Error(err),
			zap.String("session_id", sessionID.String()))
		c.String(http.StatusInternalServerError, "Could not load conversation history")
		return
	}

	messageGroups := groupMessages(messages)
	pages.ChatPage(sessionID, sessions, messageGroups).Render(c.Request.Context(), c.Writer)
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
			h.logger.Error("Failed to save uploaded file",
				zap.Error(err),
				zap.String("filename", sanitizedFilename),
				zap.String("session_id", req.SessionID))
			c.JSON(http.StatusInternalServerError, gin.H{"error": "Could not save file"})
			return
		}

		if !verifyFileExists(workspaceDir, sanitizedFilename) {
			h.logger.Error("File verification failed after upload",
				zap.String("filename", sanitizedFilename),
				zap.String("workspace", workspaceDir))
			c.JSON(http.StatusInternalServerError, gin.H{"error": "File verification failed after upload"})
			return
		}

		if strings.TrimSpace(req.Message) == "" {
			req.Message = fmt.Sprintf("I've uploaded %s. Please analyze this dataset and provide statistical insights.", file.Filename)
		} else {
			req.Message = fmt.Sprintf("[📎 File uploaded: %s]\n\n%s", file.Filename, req.Message)
		}

		// Mark file as rendered - non-critical, log if fails
		if err := h.store.AddRenderedFile(c.Request.Context(), sessionID, sanitizedFilename); err != nil {
			h.logger.Warn("Failed to mark file as rendered",
				zap.Error(err),
				zap.String("filename", sanitizedFilename),
				zap.String("session_id", req.SessionID))
		}

		h.logger.Info("File uploaded successfully",
			zap.String("filename", file.Filename),
			zap.String("session_id", req.SessionID),
			zap.Int64("size_bytes", file.Size))
	}

	userMessage := types.ChatMessage{
		Role:      "user",
		Content:   req.Message,
		ID:        generateMessageID(),
		SessionID: sessionID.String(),
	}

	// Save user message - critical operation
	if err := h.store.CreateMessage(c.Request.Context(), userMessage); err != nil {
		h.logger.Error("Failed to save user message",
			zap.Error(err),
			zap.String("session_id", req.SessionID))
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Could not save message"})
		return
	}

	h.logger.Info("Processing chat message", zap.String("session_id", req.SessionID), zap.String("message", req.Message))

	// This is the crucial change. When a new message is sent, we now render a component
	// that includes the SSE loader. This ensures only new messages trigger the agent.
	component := components.UserMessageWithLoader(userMessage)
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
		c.JSON(http.StatusBadRequest, gin.H{"error": "invalid session ID"})
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
		c.JSON(http.StatusBadRequest, gin.H{"error": "invalid session ID"})
		return
	}

	c.Header("Content-Type", "text/event-stream")
	c.Header("Cache-Control", "no-cache")
	c.Header("Connection", "keep-alive")
	c.Header("Access-Control-Allow-Origin", "*")

	data := services.StreamData{Type: "connection_established"}
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

	// Check if this is the first message in the session to trigger initialization
	if len(messages) == 1 {
		if err := h.chatService.InitializeSession(ctx, sessionID.String()); err != nil {
			h.logger.Error("Failed to initialize session", zap.Error(err))
		}
		// Re-fetch messages to include the initialization message
		messages, err = h.store.GetMessagesBySession(ctx, sessionID)
		if err != nil {
			fmt.Fprintf(c.Writer, "data: Error fetching messages after initialization\n\n")
			c.Writer.(http.Flusher).Flush()
			return
		}
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

	// Convert messages to agent history format
	agentHistory := toAgentMessages(messages)

	// Stream agent response using ChatService
	h.chatService.StreamAgentResponse(ctx, c.Writer, userMessage.Content, userMessageID, sessionID.String(), agentHistory)
}

// Helper functions that remain in the handler for presentation logic

func groupMessages(messages []types.ChatMessage) []types.MessageGroup {
	if len(messages) == 0 {
		return nil
	}
	var groups []types.MessageGroup
	i := 0
	for i < len(messages) {
		message := messages[i]
		switch message.Role {
		case "user":
			groups = append(groups, types.MessageGroup{PrimaryRole: "user", Messages: []types.ChatMessage{message}})
			i++
		case "system":
			i++ // Skip system messages
		case "assistant", "tool":
			var agentMessages []types.ChatMessage
			for i < len(messages) && (messages[i].Role == "assistant" || messages[i].Role == "tool") {
				agentMessages = append(agentMessages, messages[i])
				i++
			}
			if len(agentMessages) > 0 {
				groups = append(groups, types.MessageGroup{PrimaryRole: "agent", Messages: agentMessages})
			}
		default:
			i++
		}
	}
	return groups
}

func toAgentMessages(messages []types.ChatMessage) []types.AgentMessage {
	var agentMessages []types.AgentMessage
	for _, message := range messages {
		if message.Role == "user" || message.Role == "assistant" || message.Role == "tool" {
			agentMessages = append(agentMessages, types.AgentMessage{
				Role:    message.Role,
				Content: message.Content,
			})
		}
	}
	return agentMessages
}

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
