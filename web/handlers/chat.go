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
    "stats-agent/rag"
    "stats-agent/web/middleware"
    "stats-agent/web/services"
    "stats-agent/web/templates/components"
    "stats-agent/web/templates/pages"
    "stats-agent/web/types"
    "strings"
    "sync"
    "time"

    "github.com/gin-gonic/gin"
    "github.com/google/uuid"
    "go.uber.org/zap"
)

type ChatHandler struct {
	chatService   *services.ChatService
	streamService *services.StreamService
	pdfService    *services.PDFService
	agent         AgentInterface
	cfg           *config.Config
	logger        *zap.Logger
	store         *database.PostgresStore
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
	pdfService *services.PDFService,
	agent AgentInterface,
	cfg *config.Config,
	logger *zap.Logger,
	store *database.PostgresStore,
) *ChatHandler {
	return &ChatHandler{
		chatService:   chatService,
		streamService: streamService,
		pdfService:    pdfService,
		agent:         agent,
		cfg:           cfg,
		logger:        logger,
		store:         store,
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
		if errors.Is(err, sql.ErrNoRows) {
			h.logger.Info("Requested session not found, creating new one",
				zap.String("requested_session_id", sessionID.String()))
			newSessionID, createErr := h.store.CreateSession(c.Request.Context(), userUUIDPtr)
			if createErr != nil {
				h.logger.Error("Failed to create replacement session",
					zap.Error(createErr))
				c.String(http.StatusInternalServerError, "Could not create new session")
				return
			}
			c.SetCookie(middleware.SessionCookieName, newSessionID.String(), middleware.CookieMaxAge, "/", "", false, true)
			c.Redirect(http.StatusFound, fmt.Sprintf("/chat/%s", newSessionID.String()))
			return
		}
		h.logger.Error("Failed to get session",
			zap.Error(err),
			zap.String("session_id", sessionID.String()))
		c.String(http.StatusInternalServerError, "Could not load session")
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
	var displayMessage string // Track what to display to the user
	file, err := c.FormFile("file")
	if err == nil {
		sanitizedFilename := sanitizeFilename(file.Filename)
		if sanitizedFilename == "" {
			c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid or unsafe filename."})
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

		// Handle PDF uploads
		if ext == ".pdf" {
			// Set user message for display (without embedding full PDF text)
			if strings.TrimSpace(req.Message) == "" {
				req.Message = fmt.Sprintf("[📎 File uploaded: %s]\n\nPlease analyze the content from this PDF and provide statistical insights.", file.Filename)
				displayMessage = fmt.Sprintf("[📎 PDF uploaded: %s]<br><br>Please analyze the content from this PDF and provide statistical insights.", file.Filename)
			} else {
				displayMessage = fmt.Sprintf("[📎 PDF uploaded: %s]<br><br>%s", file.Filename, req.Message)
				req.Message = fmt.Sprintf("[📎 File uploaded: %s]\n\n%s", file.Filename, req.Message)
			}

			// Extract pages and store in RAG synchronously
			// This ensures the PDF content is available when the agent starts processing
			pdfCtx, pdfCancel := context.WithTimeout(c.Request.Context(), 30*time.Second)
			defer pdfCancel()

			pages, err := h.pdfService.ExtractPages(dst)
			if err != nil {
				h.logger.Error("Failed to extract PDF pages for RAG",
					zap.Error(err),
					zap.String("filename", sanitizedFilename))
				// Continue - user can still ask questions, just without PDF content in RAG
			} else {
				ragInstance := h.agent.GetRAG()
				if ragInstance == nil {
					h.logger.Warn("RAG instance not available for PDF storage")
				} else {
					if err := ragInstance.AddPDFPagesToRAG(pdfCtx, req.SessionID, file.Filename, pages); err != nil {
						h.logger.Error("Failed to store PDF pages in RAG",
							zap.Error(err),
							zap.String("filename", sanitizedFilename),
							zap.String("session_id", req.SessionID))
					} else {
						h.logger.Info("Successfully stored PDF pages in RAG",
							zap.String("filename", sanitizedFilename),
							zap.Int("pages", len(pages)),
							zap.String("session_id", req.SessionID))
					}
				}
			}
		} else {
			// For non-PDF files, use existing message logic
			if strings.TrimSpace(req.Message) == "" {
				req.Message = fmt.Sprintf("I've uploaded %s. Please analyze this dataset and provide statistical insights.", file.Filename)
			} else {
				req.Message = fmt.Sprintf("[📎 File uploaded: %s]\n\n%s", file.Filename, req.Message)
			}
		}

		// Track uploaded file in database - non-critical, log if fails
		webPath := filepath.ToSlash(filepath.Join("/workspaces", req.SessionID, sanitizedFilename))

		// Determine file type
		fileType := "csv"
		if ext == ".pdf" {
			fileType = "pdf"
		}

		fileRecord := database.FileRecord{
			ID:        uuid.New(),
			SessionID: sessionID,
			Filename:  sanitizedFilename,
			FilePath:  webPath,
			FileType:  fileType,
			FileSize:  file.Size,
			CreatedAt: time.Now(),
			MessageID: nil, // Will be associated with user message later if needed
		}
		if _, err := h.store.CreateFile(c.Request.Context(), fileRecord); err != nil {
			h.logger.Warn("Failed to track uploaded file in database",
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
		Rendered:  displayMessage, // If empty, template will use Content
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
        "running":          running,
        "user_message_id":  userMsgID,
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

    // Document-ready gating: if user asks a document question but no PDF embeddings exist yet,
    // return a short assistant response and do not start the agent. Applies only to PDFs.
    if userMessage != nil {
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
				Role:    message.Role,
				Content: message.Content,
			})
		}
	}
	return agentMessages
}

func sanitizeFilename(filename string) string {
	// Trim leading/trailing spaces and dots
	sanitized := strings.Trim(filename, " .")

	// Remove path traversal attempts
	sanitized = strings.ReplaceAll(sanitized, "..", "")

	// Preserve extension
	ext := filepath.Ext(sanitized)
	nameWithoutExt := strings.TrimSuffix(sanitized, ext)

	// Replace special characters with safe alternatives
	nameWithoutExt = replaceSpecialChars(nameWithoutExt)

	// Reconstruct with original extension
	sanitized = nameWithoutExt + ext

	// Limit total length to 255 characters
	if len(sanitized) > 255 {
		// Truncate name portion, preserve extension
		maxNameLen := 255 - len(ext)
		if maxNameLen > 0 {
			sanitized = nameWithoutExt[:maxNameLen] + ext
		} else {
			sanitized = sanitized[:255]
		}
	}

	return sanitized
}

func replaceSpecialChars(s string) string {
	// Replace common special chars with readable alternatives
	s = strings.ReplaceAll(s, "%", "pct")
	s = strings.ReplaceAll(s, "&", "and")
	s = strings.ReplaceAll(s, " ", "_")

	// Replace filesystem-unsafe characters with underscore
	// These are problematic across Windows, Linux, macOS
	unsafeChars := []string{
		"<", ">", ":", "\"", "/", "\\", "|", "?", "*",
		"(", ")", "[", "]", "{", "}", "'", ",", ";", "!",
		"@", "#", "$", "^", "`", "~", "+", "=",
	}

	for _, char := range unsafeChars {
		s = strings.ReplaceAll(s, char, "_")
	}

	// Collapse multiple underscores to single
	for strings.Contains(s, "__") {
		s = strings.ReplaceAll(s, "__", "_")
	}

	// Trim leading/trailing underscores
	s = strings.Trim(s, "_")

	return s
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
