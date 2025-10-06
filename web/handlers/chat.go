package handlers

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"stats-agent/agent"
	"stats-agent/config"
	"stats-agent/database"
	"stats-agent/utils"
	"stats-agent/web/middleware"
	"stats-agent/web/services"
	"stats-agent/web/templates/components"
	"stats-agent/web/templates/pages"
	"stats-agent/web/types"
	"strings"
	"sync"

	"github.com/gin-gonic/gin"
	"github.com/google/uuid"
	"go.uber.org/zap"
)

type ChatHandler struct {
	chatService            *services.ChatService
	streamService          *services.StreamService
	pdfService             *services.PDFService
	uploadService          *services.UploadService
	messageGroupingService *services.MessageGroupingService
	agent                  AgentInterface
	cfg                    *config.Config
	logger                 *zap.Logger
	store                  *database.PostgresStore
}

// AgentInterface defines the subset of agent methods we need
type AgentInterface interface {
	GetMemoryManager() *agent.MemoryManager
}

type ChatRequest struct {
	Message   string `json:"message" form:"message"`
	SessionID string `json:"session_id" form:"session_id"`
}

func NewChatHandler(
	chatService *services.ChatService,
	streamService *services.StreamService,
	pdfService *services.PDFService,
	uploadService *services.UploadService,
	agent AgentInterface,
	cfg *config.Config,
	logger *zap.Logger,
	store *database.PostgresStore,
) *ChatHandler {
	return &ChatHandler{
		chatService:            chatService,
		streamService:          streamService,
		pdfService:             pdfService,
		uploadService:          uploadService,
		messageGroupingService: services.NewMessageGroupingService(),
		agent:                  agent,
		cfg:                    cfg,
		logger:                 logger,
		store:                  store,
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
		respondWithClientError(c, http.StatusBadRequest, "Session expired. Please refresh the page.")
		return
	}

	// Get session info before deleting (to get workspace path)
	session, err := h.store.GetSessionByID(c.Request.Context(), sessionID)
	if err != nil {
		respondWithError(c, http.StatusNotFound, err, "Unable to find conversation.", h.logger,
			zap.String("session_id", sessionIDStr))
		return
	}

	if deleted, err := h.store.DeleteRAGDocumentsBySession(c.Request.Context(), sessionID); err != nil {
		h.logger.Warn("Failed to delete RAG documents for session",
			zap.Error(err),
			zap.String("session_id", sessionIDStr))
	} else if deleted > 0 {
		h.logger.Debug("Deleted RAG documents for session",
			zap.String("session_id", sessionIDStr),
			zap.Int64("documents_deleted", deleted))
	}

	// Delete from database (this cascades to messages)
	if err := h.store.DeleteSession(c.Request.Context(), sessionID); err != nil {
		respondWithError(c, http.StatusInternalServerError, err, "Unable to delete conversation. Please try again.", h.logger,
			zap.String("session_id", sessionIDStr))
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
		respondWithError(c, http.StatusInternalServerError, fmt.Errorf("session ID not in context"),
			"Session expired. Please refresh the page.", h.logger)
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
		respondWithError(c, http.StatusInternalServerError, err,
			"Unable to initialize workspace. Please refresh the page.", h.logger,
			zap.String("session_id", sessionUUID.String()))
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
		respondWithError(c, http.StatusInternalServerError, err,
			"Unable to load this conversation. Please try again.", h.logger,
			zap.String("session_id", sessionUUID.String()))
		return
	}

	messageGroups := h.messageGroupingService.GroupMessages(messages)
	component := pages.ChatPage(sessionUUID, sessions, messageGroups)
	component.Render(c.Request.Context(), c.Writer)
}

func (h *ChatHandler) LoadSession(c *gin.Context) {
	sessionID, err := uuid.Parse(c.Param("sessionID"))
	if err != nil {
		respondWithClientError(c, http.StatusBadRequest, "Session expired. Please refresh the page.")
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
		if errors.Is(err, sql.ErrNoRows) {
			h.logger.Info("Requested session not found, creating new one",
				zap.String("requested_session_id", sessionID.String()))
			newSessionID, createErr := h.store.CreateSession(c.Request.Context(), userUUIDPtr)
			if createErr != nil {
				respondWithError(c, http.StatusInternalServerError, createErr,
					"Unable to create new session. Please try again.", h.logger)
				return
			}
			c.SetCookie(middleware.SessionCookieName, newSessionID.String(), middleware.CookieMaxAge, "/", "", false, true)
			c.Redirect(http.StatusFound, fmt.Sprintf("/chat/%s", newSessionID.String()))
			return
		}
		respondWithError(c, http.StatusInternalServerError, err,
			"Unable to load session. Please try again.", h.logger,
			zap.String("session_id", sessionID.String()))
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
		respondWithError(c, http.StatusInternalServerError, err,
			"Unable to load conversation history. Please try again.", h.logger,
			zap.String("session_id", sessionID.String()))
		return
	}

	messageGroups := h.messageGroupingService.GroupMessages(messages)
	pages.ChatPage(sessionID, sessions, messageGroups).Render(c.Request.Context(), c.Writer)
}

func (h *ChatHandler) SendMessage(c *gin.Context) {
	var req ChatRequest
	if err := c.ShouldBind(&req); err != nil {
		respondWithError(c, http.StatusBadRequest, err, "Invalid request format.", h.logger)
		return
	}

	if req.Message == "" {
		respondWithClientError(c, http.StatusBadRequest, "Message cannot be empty.")
		return
	}

	sessionID, err := uuid.Parse(req.SessionID)
	if err != nil {
		respondWithClientError(c, http.StatusBadRequest, "Session expired. Please refresh the page.")
		return
	}

	// Handle potential file upload
	file, err := c.FormFile("file")
	if err == nil {
		// Validate upload
		sanitizedFilename, err := h.uploadService.ValidateUpload(file)
		if err != nil {
			respondWithClientError(c, http.StatusBadRequest, err.Error())
			return
		}

		// Save file to workspace
		workspaceDir := filepath.Join("workspaces", req.SessionID)
		dst := filepath.Join(workspaceDir, sanitizedFilename)
		if err := c.SaveUploadedFile(file, dst); err != nil {
			respondWithError(c, http.StatusInternalServerError, err,
				"Unable to save file. Please try again.", h.logger,
				zap.String("filename", sanitizedFilename),
				zap.String("session_id", req.SessionID))
			return
		}

		// Track file in database
		if err := h.uploadService.TrackUploadedFile(c.Request.Context(), file, req.SessionID, sanitizedFilename, dst); err != nil {
			// Log but don't fail - file was saved successfully
			h.logger.Warn("Failed to track file in database, continuing", zap.Error(err))
		}

		// Process PDF content if applicable
		fileInfo := utils.GetFileInfo(file.Filename)
		var pdfContent string
		if fileInfo.Type == utils.FileTypePDF {
			truncConfig := services.TruncationConfig{
				MaxTokens:                h.cfg.ContextLength,
				TokenThreshold:           h.cfg.PDFTokenThreshold,
				FirstPagesPrio:           h.cfg.PDFFirstPagesPriority,
				EnableTableDetection:     h.cfg.PDFEnableTableDetection,
				SentenceBoundaryTruncate: h.cfg.PDFSentenceBoundaryTruncate,
			}
			memManager := h.agent.GetMemoryManager()
			pdfContent, _ = h.uploadService.ProcessPDFContent(c.Request.Context(), dst, file.Filename, truncConfig, memManager)
		}

		// Prepare message with file content
		req.Message = h.uploadService.PrepareMessageWithFile(
			req.Message,
			file.Filename,
			pdfContent,
			fileInfo.Type == utils.FileTypePDF,
		)

		h.logger.Info("File uploaded successfully",
			zap.String("filename", file.Filename),
			zap.String("session_id", req.SessionID),
			zap.Int64("size_bytes", file.Size))
	}

	userMessage := types.ChatMessage{
		Role:      "user",
		Content:   req.Message,
		ID:        utils.GenerateMessageID(),
		SessionID: sessionID.String(),
	}

	// Save user message - critical operation
	if err := h.store.CreateMessage(c.Request.Context(), userMessage); err != nil {
		respondWithError(c, http.StatusInternalServerError, err,
			"Unable to save message. Please try again.", h.logger,
			zap.String("session_id", req.SessionID))
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
		respondWithClientError(c, http.StatusBadRequest, "Session ID is required.")
		return
	}

	sessionID, err := uuid.Parse(sessionIDStr)
	if err != nil {
		respondWithClientError(c, http.StatusBadRequest, "Session expired. Please refresh the page.")
		return
	}

	file, err := c.FormFile("file")
	if err != nil {
		respondWithClientError(c, http.StatusBadRequest, "No file provided. Please select a file to upload.")
		return
	}

	ext := strings.ToLower(filepath.Ext(file.Filename))
	if ext != ".csv" && ext != ".xlsx" && ext != ".xls" && ext != ".pdf" {
		respondWithClientError(c, http.StatusBadRequest, "Invalid file type. Please upload CSV, Excel, or PDF files.")
		return
	}

	// Limit PDF size to 10MB
	if ext == ".pdf" && file.Size > 10*1024*1024 {
		respondWithClientError(c, http.StatusBadRequest, "PDF file too large. Maximum size is 10MB.")
		return
	}

	workspaceDir := filepath.Join("workspaces", sessionID.String())
	dst := filepath.Join(workspaceDir, file.Filename)
	if err := c.SaveUploadedFile(file, dst); err != nil {
		respondWithError(c, http.StatusInternalServerError, err,
			"Unable to save file. Please try again.", h.logger,
			zap.String("filename", file.Filename),
			zap.String("session_id", sessionIDStr))
		return
	}

	systemMessage := types.ChatMessage{
		Role:      "system",
		Content:   fmt.Sprintf("The user has uploaded a file: %s. Unless specified otherwise, use this file for your analysis.", file.Filename),
		ID:        utils.GenerateMessageID(),
		SessionID: sessionID.String(),
	}

	if err := h.store.CreateMessage(c.Request.Context(), systemMessage); err != nil {
		respondWithError(c, http.StatusInternalServerError, err,
			"Unable to save message. Please try again.", h.logger,
			zap.String("session_id", sessionIDStr))
		return
	}

	c.JSON(http.StatusOK, gin.H{"message": fmt.Sprintf("File %s uploaded successfully.", file.Filename)})
}

func (h *ChatHandler) StreamResponse(c *gin.Context) {
	sessionIDStr := c.Query("session_id")
	userMessageID := c.Query("user_message_id")

	if sessionIDStr == "" || userMessageID == "" {
		respondWithClientError(c, http.StatusBadRequest, "Session ID and message ID are required.")
		return
	}
	sessionID, err := uuid.Parse(sessionIDStr)
	if err != nil {
		respondWithClientError(c, http.StatusBadRequest, "Session expired. Please refresh the page.")
		return
	}

	c.Header("Content-Type", "text/event-stream")
	c.Header("Cache-Control", "no-cache")
	c.Header("Connection", "keep-alive")
	c.Header("Access-Control-Allow-Origin", "*")

	// Use a single mutex for all SSE writes in this request
	var mu sync.Mutex

	ctx := c.Request.Context()

	// Use the service layer method
	h.streamService.WriteSSEData(ctx, c.Writer, services.StreamData{Type: "connection_established"}, &mu)

	messages, err := h.store.GetMessagesBySession(ctx, sessionID)
	if err != nil {
		h.streamService.WriteSSEData(ctx, c.Writer, services.StreamData{Type: "error", Content: "Error fetching messages"}, &mu)
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
		h.streamService.WriteSSEData(ctx, c.Writer, services.StreamData{Type: "error", Content: "User message not found"}, &mu)
		return
	}

	// Check if this is the first message in the session to trigger initialization and title generation
	if len(messages) == 1 {
		// Pass the service method to the goroutine
		go h.chatService.GenerateAndSetTitle(context.Background(), sessionID, userMessage.Content, func(data services.StreamData) error {
			return h.streamService.WriteSSEData(context.Background(), c.Writer, data, &mu)
		})

		if err := h.chatService.InitializeSession(ctx, sessionID.String()); err != nil {
			h.logger.Error("Failed to initialize session", zap.Error(err))
		}
		// Re-fetch messages to include the initialization message for the agent's context
		messages, err = h.store.GetMessagesBySession(ctx, sessionID)
		if err != nil {
			h.streamService.WriteSSEData(ctx, c.Writer, services.StreamData{Type: "error", Content: "Error fetching messages after initialization"}, &mu)
			return
		}
	}

	// Convert messages to agent history format
	agentHistory := h.messageGroupingService.ToAgentMessages(messages)

	// Stream agent response using ChatService
	h.chatService.StreamAgentResponse(ctx, c.Writer, userMessage.Content, userMessageID, sessionID.String(), agentHistory)
}

func (h *ChatHandler) StopGeneration(c *gin.Context) {
	sessionID := c.Param("sessionID")
	if _, err := uuid.Parse(sessionID); err != nil {
		respondWithClientError(c, http.StatusBadRequest, "Invalid session ID.")
		return
	}

	h.logger.Info("Received request to stop generation for session", zap.String("session_id", sessionID))
	h.chatService.StopSessionRun(sessionID)

	c.Status(http.StatusNoContent)
}
