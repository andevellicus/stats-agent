package handlers

import (
    "fmt"
    "net/http"
    "os"
    "path/filepath"
    "stats-agent/agent"
    "stats-agent/config"
    "stats-agent/database"
    "stats-agent/rag"
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
	chatService    *services.ChatService
	streamService  *services.StreamService
	sessionService *services.SessionService
	uploadService  *services.UploadService
	agent          AgentInterface
	cfg            *config.Config
	logger         *zap.Logger
	store          *database.PostgresStore
}

// AgentInterface defines the subset of agent methods we need
type AgentInterface interface {
	GetMemoryManager() *agent.MemoryManager
	GetRAG() *rag.RAG
}

type ChatRequest struct {
	Message   string `json:"message" form:"message"`
	SessionID string `json:"session_id" form:"session_id"`
}

func NewChatHandler(
	chatService *services.ChatService,
	streamService *services.StreamService,
	sessionService *services.SessionService,
	uploadService *services.UploadService,
	agent AgentInterface,
	cfg *config.Config,
	logger *zap.Logger,
	store *database.PostgresStore,
) *ChatHandler {
	return &ChatHandler{
		chatService:    chatService,
		streamService:  streamService,
		sessionService: sessionService,
		uploadService:  uploadService,
		agent:          agent,
		cfg:            cfg,
		logger:         logger,
		store:          store,
	}
}

func (h *ChatHandler) NewChat(c *gin.Context) {
	c.Header("Cache-Control", "no-cache, no-store, must-revalidate")
	// By setting the cookie's max age to -1, we tell the browser to delete it.
	middleware.SetSecureCookie(c, middleware.SessionCookieName, "", -1)
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
	if exists && currentSessionID != nil {
		// SessionMiddleware stores *uuid.UUID in context
		if currentSessionIDPtr, ok := currentSessionID.(*uuid.UUID); ok && currentSessionIDPtr != nil {
			if *currentSessionIDPtr == sessionID {
				// Deleting the current session - clear cookie and redirect to create new session
				middleware.SetSecureCookie(c, middleware.SessionCookieName, "", -1)
			}
		}
	}

	// Always redirect to home page to refresh the UI
	// The HX-Redirect header tells HTMX to perform a full page redirect
	c.Header("HX-Redirect", "/")
	c.Status(http.StatusOK)
}

// Legacy session claiming removed - no longer needed
func (h *ChatHandler) Index(c *gin.Context) {
	sessionIDPtr, exists := c.Get("sessionID")
	var sessionUUID uuid.UUID
	var userUUIDPtr *uuid.UUID

	// Check if session exists
	if exists && sessionIDPtr != nil {
		sessionUUIDPtr := sessionIDPtr.(*uuid.UUID)
		if sessionUUIDPtr != nil {
			sessionUUID = *sessionUUIDPtr
		}
	}

	// Get user if it exists
	if userIDPtr, userExists := c.Get("userID"); userExists && userIDPtr != nil {
		userUUIDPtr = userIDPtr.(*uuid.UUID)
	}

	// If no session exists yet, render empty page - session will be created on first message
	if sessionUUID == uuid.Nil {
		// Create a temporary/placeholder ID for the frontend (not persisted yet)
		sessionUUID = uuid.New()
		c.Set("sessionID", &sessionUUID)
		sessions := h.sessionService.GetSessionsForSidebar(c.Request.Context(), userUUIDPtr)
		component := pages.ChatPage(sessionUUID, sessions, nil)
		component.Render(c.Request.Context(), c.Writer)
		return
	}

	// Create workspace using service
	if err := h.sessionService.CreateWorkspace(sessionUUID); err != nil {
		c.String(http.StatusInternalServerError, "Could not create workspace")
		return
	}

	// Get sessions for sidebar using service
	sessions := h.sessionService.GetSessionsForSidebar(c.Request.Context(), userUUIDPtr)

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

    userIDPtr, userExists := c.Get("userID")
    var userUUIDPtr *uuid.UUID
    if userExists && userIDPtr != nil {
        userUUIDPtr = userIDPtr.(*uuid.UUID)
    }

    // Validate and get session using service
    session, shouldCreate, err := h.sessionService.ValidateAndGetSession(c.Request.Context(), sessionID, userUUIDPtr)
    if err != nil {
        h.logger.Error("Session validation failed",
            zap.Error(err),
            zap.String("session_id", sessionID.String()))
        c.Redirect(http.StatusFound, "/")
        return
    }

	// If session not found, don't create - redirect to home
	// Session will be created on first message if needed
	if shouldCreate {
		h.logger.Info("Requested session not found, redirecting to home",
			zap.String("requested_session_id", sessionID.String()))
		c.Redirect(http.StatusFound, "/")
		return
	}

	// Get sessions for sidebar using service
	sessions := h.sessionService.GetSessionsForSidebar(c.Request.Context(), userUUIDPtr)

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
    _ = session // Mark as used
}

func (h *ChatHandler) SendMessage(c *gin.Context) {
	h.logger.Debug("SendMessage called")

	var req ChatRequest
	if err := c.ShouldBind(&req); err != nil {
		h.logger.Error("Failed to bind chat request", zap.Error(err))
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid request"})
		return
	}
	h.logger.Debug("Request bound successfully", zap.String("session_id", req.SessionID), zap.String("message", req.Message))

	if req.Message == "" {
		h.logger.Error("Empty message received")
		c.JSON(http.StatusBadRequest, gin.H{"error": "Message cannot be empty"})
		return
	}

	sessionID, err := uuid.Parse(req.SessionID)
	if err != nil {
		h.logger.Error("Failed to parse session ID", zap.Error(err), zap.String("session_id", req.SessionID))
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid session ID"})
		return
	}
	h.logger.Debug("Session ID parsed", zap.String("session_id", sessionID.String()))

	// Check if user and session need to be created
	userIDPtr, userExists := c.Get("userID")
	var userUUID *uuid.UUID
	if userExists && userIDPtr != nil {
		userUUID = userIDPtr.(*uuid.UUID)
		if userUUID != nil {
			h.logger.Debug("Existing user found", zap.String("user_id", userUUID.String()))
		} else {
			h.logger.Debug("User pointer is nil")
		}
	} else {
		h.logger.Debug("No user in context", zap.Bool("exists", userExists))
	}

	sessionIDPtr, sessionExists := c.Get("sessionID")
	var persistedSessionID *uuid.UUID
	if sessionExists && sessionIDPtr != nil {
		persistedSessionID = sessionIDPtr.(*uuid.UUID)
		if persistedSessionID != nil {
			h.logger.Debug("Existing session found", zap.String("session_id", persistedSessionID.String()))
		} else {
			h.logger.Debug("Session pointer is nil")
		}
	} else {
		h.logger.Debug("No session in context", zap.Bool("exists", sessionExists))
	}

	// Create user if doesn't exist
	if userUUID == nil {
		h.logger.Debug("Creating new user")
		var creationErr error
		newUserID, creationErr := h.store.CreateUser(c.Request.Context())
		if creationErr != nil {
			h.logger.Error("Failed to create user on first message",
				zap.Error(creationErr),
				zap.String("session_id", req.SessionID))
			c.JSON(http.StatusInternalServerError, gin.H{"error": "Could not create user account"})
			return
		}
		userUUID = &newUserID
		middleware.SetSecureCookie(c, middleware.UserCookieName, newUserID.String(), middleware.CookieMaxAge)
		h.logger.Info("User created on first message", zap.String("user_id", newUserID.String()))
	} else {
		h.logger.Debug("User already exists, skipping creation")
	}

	// Create session if doesn't exist (or placeholder needs to be persisted)
	sessionNeedsCreation := true
	if persistedSessionID != nil && *persistedSessionID == sessionID {
		sessionNeedsCreation = false
	}
	h.logger.Debug("Session needs creation check",
		zap.Bool("needs_creation", sessionNeedsCreation),
		zap.Bool("persisted_is_nil", persistedSessionID == nil))

	if sessionNeedsCreation {
		h.logger.Debug("Creating new session", zap.String("user_id", userUUID.String()))
		var creationErr error
		newSessionID, creationErr := h.store.CreateSession(c.Request.Context(), userUUID)
		if creationErr != nil {
			h.logger.Error("Failed to create session on first message",
				zap.Error(creationErr),
				zap.String("user_id", userUUID.String()))
			c.JSON(http.StatusInternalServerError, gin.H{"error": "Could not create session"})
			return
		}
		h.logger.Debug("Session created successfully", zap.String("new_session_id", newSessionID.String()), zap.String("requested_session_id", sessionID.String()))

		// If the frontend sent a placeholder ID, update the reference
		if sessionID != newSessionID {
			h.logger.Debug("Updating session ID from placeholder to persisted",
				zap.String("placeholder", sessionID.String()),
				zap.String("persisted", newSessionID.String()))
			sessionID = newSessionID
			req.SessionID = newSessionID.String()
		}
		middleware.SetSecureCookie(c, middleware.SessionCookieName, newSessionID.String(), middleware.CookieMaxAge)
		h.logger.Info("Session created on first message",
			zap.String("session_id", newSessionID.String()),
			zap.String("user_id", userUUID.String()))

		// Create workspace after session is persisted
		if err := h.sessionService.CreateWorkspace(newSessionID); err != nil {
			h.logger.Error("Failed to create workspace",
				zap.Error(err),
				zap.String("session_id", newSessionID.String()))
			// Don't fail the entire request, workspace creation is less critical
		}
	} else {
		h.logger.Debug("Session already exists, skipping creation")
	}

	// Handle potential file upload using upload service
	var displayMessage string // Track what to display to the user
	file, err := c.FormFile("file")
	if err == nil {
		// Get file extension for mode detection
		ext := strings.ToLower(filepath.Ext(file.Filename))

		// Detect and set session mode based on first file upload
		_, err := h.sessionService.DetectAndSetMode(c.Request.Context(), sessionID, ext)
		if err != nil {
			h.logger.Warn("Failed to update session mode", zap.Error(err))
			// Continue - mode update is not critical
		}

		// Process the upload using upload service
		uploadResult, err := h.uploadService.ProcessUpload(c.Request.Context(), file, sessionID, req.Message)
		if err != nil {
			h.logger.Error("File upload failed",
				zap.Error(err),
				zap.String("filename", file.Filename),
				zap.String("session_id", req.SessionID))
			c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
			return
		}

		// Use the formatted messages from upload service
		req.Message = uploadResult.ContentMessage
		displayMessage = uploadResult.DisplayMessage
	}

	userMessage := types.ChatMessage{
		Role:        "user",
		Content:     req.Message,
		Rendered:    displayMessage, // If empty, template will use Content
		ContentHash: rag.ComputeMessageContentHash("user", req.Message),
		ID:          generateMessageID(),
		SessionID:   sessionID.String(),
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
	if ext != ".csv" && ext != ".xlsx" && ext != ".xls" && ext != ".pdf" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid file type. Please upload CSV, Excel, or PDF files."})
		return
	}

	// Limit PDF size to 10MB
	if ext == ".pdf" && file.Size > 10*1024*1024 {
		c.JSON(http.StatusBadRequest, gin.H{"error": "PDF file too large. Maximum size is 10MB."})
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

func (h *ChatHandler) StopAgent(c *gin.Context) {
	sessionIDStr := c.Query("session_id")
	if sessionIDStr == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Session ID required"})
		return
	}

	h.chatService.StopSessionRun(sessionIDStr)
	h.logger.Info("Agent execution stopped by user", zap.String("session_id", sessionIDStr))
	c.Status(http.StatusOK)
}

// Status returns whether the session currently has an active agent run,
// and if so, the user message ID that initiated it (to allow SSE reattach).
func (h *ChatHandler) Status(c *gin.Context) {
	sessionIDStr := c.Query("session_id")
	if sessionIDStr == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Session ID required"})
		return
	}

	running, userMsgID := h.chatService.GetActiveRun(sessionIDStr)
	c.JSON(http.StatusOK, gin.H{
		"running":         running,
		"user_message_id": userMsgID,
	})
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
        // Use the request context for both the background title generation and SSE writes.
        // This prevents writes after the client disconnects, avoiding nil deref in Flush.
        go h.chatService.GenerateAndSetTitle(ctx, sessionID, userMessage.Content, func(data services.StreamData) error {
            return h.streamService.WriteSSEData(ctx, c.Writer, data, &mu)
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

    // Document-ready gating: if user asks a document question but no PDF embeddings exist yet,
	// return a short assistant response and do not start the agent. Applies only to PDFs.
	// Check if session has any tracked PDFs
	files, _ := h.store.GetFilesBySession(ctx, sessionID)
	hasPDF := false
	for _, f := range files {
		if strings.EqualFold(f.FileType, "pdf") || strings.HasSuffix(strings.ToLower(f.Filename), ".pdf") {
			hasPDF = true
			break
		}
	}
	if hasPDF {
		ready, err := h.store.HasSessionPDFEmbeddings(ctx, sessionID)
		if err != nil {
			h.logger.Warn("Failed to check PDF embedding readiness", zap.Error(err), zap.String("session_id", sessionID.String()))
		}
		if !ready && isDocumentQuestion(userMessage.Content) {
			// Create and persist a brief assistant message
			assistantID := uuid.New().String()
			content := "I’m still indexing your PDF. Please wait a few seconds and ask again. I’ll use the document once it’s ready."
			if err := h.store.CreateMessage(ctx, types.ChatMessage{
				ID:        assistantID,
				SessionID: sessionID.String(),
				Role:      "assistant",
				Content:   content,
				Rendered:  content,
			}); err != nil {
				h.logger.Warn("Failed to persist gating assistant message", zap.Error(err))
			}
			// Stream minimal response to replace loader and show message
			h.streamService.WriteSSEData(ctx, c.Writer, services.StreamData{Type: "remove_loader", Content: "loading-" + userMessageID}, &mu)
			h.streamService.WriteSSEData(ctx, c.Writer, services.StreamData{Type: "create_container", Content: assistantID}, &mu)
			h.streamService.WriteSSEData(ctx, c.Writer, services.StreamData{Type: "chunk", Content: content}, &mu)
			h.streamService.WriteSSEData(ctx, c.Writer, services.StreamData{Type: "end"}, &mu)
			return
		}
	}

    // Dataset-ready gating: if this is the first turn, no dataset files exist yet,
    // and the user message doesn't look like a stats/data question, reply with a short
    // nudge and do not start the agent.
    if len(messages) == 1 {
        files, _ := h.store.GetFilesBySession(ctx, sessionID)
        hasDataset := false
        for _, f := range files {
            lf := strings.ToLower(f.Filename)
            if strings.HasSuffix(lf, ".csv") || strings.HasSuffix(lf, ".xlsx") || strings.HasSuffix(lf, ".xls") {
                hasDataset = true
                break
            }
        }
        if !hasDataset && !isDatasetQuestion(userMessage.Content) && !isDocumentQuestion(userMessage.Content) {
            assistantID := uuid.New().String()
            content := "I’m your statistics partner. Upload a CSV/Excel file or ask a question about your dataset (e.g., ‘Describe columns’, ‘Compute correlation A vs B’, ‘Check normality’). Once a file is present, I’ll load it and continue step‑by‑step."
            if err := h.store.CreateMessage(ctx, types.ChatMessage{
                ID:        assistantID,
                SessionID: sessionID.String(),
                Role:      "assistant",
                Content:   content,
                Rendered:  content,
            }); err != nil {
                h.logger.Warn("Failed to persist dataset gating assistant message", zap.Error(err))
            }

            h.streamService.WriteSSEData(ctx, c.Writer, services.StreamData{Type: "remove_loader", Content: "loading-" + userMessageID}, &mu)
            h.streamService.WriteSSEData(ctx, c.Writer, services.StreamData{Type: "create_container", Content: assistantID}, &mu)
            h.streamService.WriteSSEData(ctx, c.Writer, services.StreamData{Type: "chunk", Content: content}, &mu)
            h.streamService.WriteSSEData(ctx, c.Writer, services.StreamData{Type: "end"}, &mu)
            return
        }
    }

    // Convert messages to agent history format, excluding the current user message
	// because agent.Run() appends the incoming input again. This prevents
	// duplicating the latest user message in the LLM history.
	filtered := make([]types.ChatMessage, 0, len(messages))
	for _, m := range messages {
		if m.ID == userMessageID {
			continue
		}
		filtered = append(filtered, m)
	}
	agentHistory := toAgentMessages(filtered)

	// Stream agent response using ChatService
	h.chatService.StreamAgentResponse(ctx, c.Writer, userMessage.Content, userMessageID, sessionID.String(), agentHistory)
}

// isDocumentQuestion heuristically detects questions about PDF documents (not datasets).
// It looks for common terms that refer to paper content and structure.
func isDocumentQuestion(s string) bool {
	// Heuristic: only trigger for explicitly document-oriented queries
	// Avoid false positives for dataset analysis (tables, results, etc.).
	ls := strings.ToLower(s)

	// If the message is an upload notice for a non-PDF, do not treat as document question
	if strings.Contains(ls, "[📎 file uploaded:") {
		if strings.Contains(ls, ".csv]") || strings.Contains(ls, ".xlsx]") || strings.Contains(ls, ".xls]") {
			return false
		}
	}

	// Strong document signals
	strong := []string{
		"author", "authors", "title", "abstract", "doi", "journal", "manuscript", "paper", "pdf",
		"in the paper", "from the paper", "in the pdf", "from the pdf", "citation",
	}
	for _, k := range strong {
		if strings.Contains(ls, k) {
			return true
		}
	}

	// Page reference like "page 3" or "on page 10"
	if strings.Contains(ls, "page ") || strings.Contains(ls, "on page ") {
		return true
	}

	return false
}

// isDatasetQuestion heuristically detects questions about datasets/analysis (not PDFs).
func isDatasetQuestion(s string) bool {
    ls := strings.ToLower(s)
    // Strong dataset signals
    strong := []string{
        "data", "dataset", "csv", "excel", "column", "columns", "variable", "variables",
        "describe", "load", "plot", "histogram", "boxplot", "mean", "median", "correlation",
        "regression", "anova", "t-test", "ttest", "p-value", "normality", "shapiro", "levene",
        "chi-square", "chi2", "cramer", "mann-whitney", "wilcoxon", "kruskal-wallis",
    }
    for _, k := range strong {
        if strings.Contains(ls, k) {
            return true
        }
    }
    // Code-ish hints
    codeHints := []string{"```python", "pd.", "df =", "import ", "from ", "print("}
    for _, k := range codeHints {
        if strings.Contains(ls, k) {
            return true
        }
    }
    return false
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
				// Skip tool/assistant messages that have no rendered content (e.g., init banner suppression)
				if strings.TrimSpace(messages[i].Rendered) != "" {
					agentMessages = append(agentMessages, messages[i])
				}
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
				Role:        message.Role,
				Content:     message.Content,
				ContentHash: message.ContentHash,
			})
		}
	}
	return agentMessages
}

func generateMessageID() string {
	return uuid.New().String()
}
