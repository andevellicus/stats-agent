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
	"stats-agent/web/middleware"
	"stats-agent/web/templates/components"
	"stats-agent/web/templates/pages"
	"stats-agent/web/types"
	"strings"
	"sync"
	"time"

	"github.com/gomarkdown/markdown"

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
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid session ID"})
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
	h.agent.CleanupSession(sessionIDStr)

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
		return
	}
	sessionUUID := sessionID.(uuid.UUID)

	workspaceDir := filepath.Join("workspaces", sessionUUID.String())
	if err := os.MkdirAll(workspaceDir, 0755); err != nil {
		h.logger.Error("Failed to create workspace directory", zap.Error(err))
		c.String(http.StatusInternalServerError, "Could not create workspace.")
		return
	}

	sessions, err := h.store.GetSessions(c.Request.Context(), nil)
	if err != nil {
		h.logger.Error("Failed to get sessions", zap.Error(err))
	}
	messages, err := h.store.GetMessagesBySession(c.Request.Context(), sessionUUID)
	if err != nil {
		h.logger.Error("Failed to get messages for session", zap.Error(err))
	}

	messageGroups := groupMessages(messages)
	component := pages.ChatPage(sessionUUID, sessions, messageGroups)
	component.Render(c.Request.Context(), c.Writer)
}

func (h *ChatHandler) LoadSession(c *gin.Context) {
	sessionID, err := uuid.Parse(c.Param("sessionID"))
	if err != nil {
		c.String(http.StatusBadRequest, "Invalid session ID")
		return
	}

	sessions, err := h.store.GetSessions(c.Request.Context(), nil)
	if err != nil {
		h.logger.Error("Failed to get sessions", zap.Error(err))
	}
	messages, err := h.store.GetMessagesBySession(c.Request.Context(), sessionID)
	if err != nil {
		h.logger.Error("Failed to get messages for session", zap.Error(err))
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

		if err := h.store.AddRenderedFile(c.Request.Context(), sessionID, sanitizedFilename); err != nil {
			h.logger.Error("Failed to mark file as rendered", zap.Error(err))
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

	// Check if this is the first message in the session to trigger initialization
	if len(messages) == 1 {
		if err := h.initializeSession(ctx, sessionID.String()); err != nil {
			h.logger.Error("Failed to initialize session", zap.Error(err))
			// Decide if you want to abort or continue
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

	h.streamAgentResponse(ctx, c.Writer, userMessage.Content, userMessageID, sessionID.String())
}

func (h *ChatHandler) initializeSession(ctx context.Context, sessionID string) error {
	h.logger.Info("Initializing new session", zap.String("session_id", sessionID))

	workspaceDir := filepath.Join("workspaces", sessionID)
	files, err := os.ReadDir(workspaceDir)
	if err != nil {
		return fmt.Errorf("could not read workspace directory: %w", err)
	}

	var uploadedFiles []string
	for _, file := range files {
		if !file.IsDir() {
			uploadedFiles = append(uploadedFiles, file.Name())
		}
	}

	initResult, err := h.agent.InitializeSession(ctx, sessionID, uploadedFiles)
	if err != nil {
		return fmt.Errorf("failed to initialize python session: %w", err)
	}

	initMessage := types.ChatMessage{
		ID:        generateMessageID(),
		SessionID: sessionID,
		Role:      "system",
		Content:   initResult,
		Rendered:  fmt.Sprintf("<pre><code>%s</code></pre>", initResult),
	}

	return h.store.CreateMessage(ctx, initMessage)
}

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

	originalStdout := os.Stdout
	r, pipeWriter, _ := os.Pipe()
	os.Stdout = pipeWriter
	log.SetOutput(pipeWriter)

	var agentResponseForDB bytes.Buffer
	teeReader := io.TeeReader(r, &agentResponseForDB)

	agentDone := make(chan struct{})
	streamDone := make(chan struct{})

	go func() {
		defer close(streamDone)
		h.processStreamByWord(ctx, teeReader, writeSSEData)
	}()

	sessionUUID, err := uuid.Parse(sessionID)
	if err != nil {
		h.logger.Error("Invalid session ID for streaming", zap.Error(err))
		return
	}
	messages, err := h.store.GetMessagesBySession(ctx, sessionUUID)
	if err != nil {
		h.logger.Error("Failed to get message history for agent", zap.Error(err))
		return
	}
	agentHistory := toAgentMessages(messages)

	go func() {
		defer func() {
			os.Stdout = originalStdout
			log.SetOutput(originalStdout)
			pipeWriter.Close()
			close(agentDone)
		}()
		h.agent.Run(ctx, input, sessionID, agentHistory)
	}()

	select {
	case <-ctx.Done():
		h.logger.Info("Context cancelled, closing SSE connection")
	case <-agentDone:
		<-streamDone

		// For post-stream tasks, create a new background context.
		// This context will not be cancelled when the client disconnects.
		backgroundCtx := context.Background()

		h.streamNewFiles(backgroundCtx, writeSSEData, sessionID)

		if err := writeSSEData(StreamData{Type: "end"}); err != nil {
			h.logger.Error("Failed to send end message", zap.Error(err))
		}

		rawAgentResponse := agentResponseForDB.String()

		statusRe := regexp.MustCompile(`(?s)<agent_status>.*?</agent_status>`)
		rawAgentResponse = statusRe.ReplaceAllString(rawAgentResponse, "")

		re := regexp.MustCompile(`(?s)(<execution_results>.*?</execution_results>)`)
		parts := re.Split(rawAgentResponse, -1)
		matches := re.FindAllString(rawAgentResponse, -1)

		for i, part := range parts {
			assistantContent := strings.TrimSpace(part)
			if assistantContent != "" {
				// Use the background context for rendering and saving.
				assistantRendered, _ := processAgentContentForDB(backgroundCtx, assistantContent)
				assistantMessage := types.ChatMessage{
					ID:        generateMessageID(),
					SessionID: sessionID,
					Role:      "assistant",
					Content:   assistantContent,
					Rendered:  assistantRendered,
				}
				if err := h.store.CreateMessage(backgroundCtx, assistantMessage); err != nil {
					h.logger.Error("Failed to save assistant message part", zap.Error(err))
				}
			}

			if i < len(matches) {
				toolContentRaw := strings.TrimSpace(matches[i])
				result := strings.TrimSuffix(strings.TrimPrefix(toolContentRaw, "<execution_results>"), "</execution_results>")

				var buf bytes.Buffer
				// Use the background context for rendering the component.
				if err := components.ExecutionResultBlock(result).Render(backgroundCtx, &buf); err != nil {
					h.logger.Error("Failed to render execution result block for DB", zap.Error(err))
				}

				toolMessage := types.ChatMessage{
					ID:        generateMessageID(),
					SessionID: sessionID,
					Role:      "tool",
					Content:   toolContentRaw,
					Rendered:  buf.String(),
				}
				// Use the background context for saving to the database.
				if err := h.store.CreateMessage(backgroundCtx, toolMessage); err != nil {
					h.logger.Error("Failed to save tool message", zap.Error(err))
				}
			}
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
				switch strings.ToLower(filepath.Ext(fileName)) {
				case ".png", ".jpg", ".jpeg", ".gif":
					component = components.ImageBlock(webPath)
				default:
					component = components.FileBlock(webPath)
				}

				if component != nil {
					if err := component.Render(ctx, &buf); err == nil {
						writeSSEData(StreamData{Type: "file", Content: buf.String()})
						if err := h.store.AddRenderedFile(ctx, sessionUUID, fileName); err != nil {
							h.logger.Error("Failed to mark file as rendered in DB", zap.Error(err), zap.String("filename", fileName))
						}
					}
				}
			}
		}
	}
}

func groupMessages(messages []types.ChatMessage) []types.MessageGroup {
	if len(messages) == 0 {
		return nil
	}
	var groups []types.MessageGroup
	i := 0
	for i < len(messages) {
		msg := messages[i]
		switch msg.Role {
		case "user":
			groups = append(groups, types.MessageGroup{PrimaryRole: "user", Messages: []types.ChatMessage{msg}})
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

// New helper function to process agent content before saving to DB
func processAgentContentForDB(ctx context.Context, rawContent string) (string, error) {
	// Combined regex to find all custom tags
	re := regexp.MustCompile(`(?s)(<python>.*?</python>|<execution_results>.*?</execution_results>|<agent_status>.*?</agent_status>)`)

	// Split the content by our custom tags. `parts` will contain the text *between* the tags.
	parts := re.Split(rawContent, -1)
	// `matches` will contain the tags themselves.
	matches := re.FindAllString(rawContent, -1)

	var finalHTML strings.Builder

	for i, part := range parts {
		// Process the plain text part with Markdown

		cleanedPart := strings.TrimSpace(strings.ReplaceAll(part, "Agent: ", ""))
		if cleanedPart != "" {
			md := []byte(cleanedPart)
			html := markdown.ToHTML(md, nil, nil)
			// The markdown library wraps text in <p> tags, which we can trim for cleaner output
			//finalHTML.WriteString(strings.TrimSuffix(strings.TrimPrefix(string(html), "<p>"), "</p>\n"))
			finalHTML.WriteString(string(html))
		}

		// If there is a matching component, render it and append it
		if i < len(matches) {
			match := matches[i]
			var componentHTML string

			if after, ok := strings.CutPrefix(match, "<python>"); ok {
				code := strings.TrimSuffix(after, "</python>")
				var buf bytes.Buffer
				components.PythonCodeBlock(code).Render(ctx, &buf)
				componentHTML = buf.String()
			} else if after0, ok0 := strings.CutPrefix(match, "<execution_results>"); ok0 {
				result := strings.TrimSuffix(after0, "</execution_results>")
				var buf bytes.Buffer
				components.ExecutionResultBlock(result).Render(ctx, &buf)
				componentHTML = buf.String()
			} else if after1, ok1 := strings.CutPrefix(match, "<agent_status>"); ok1 {
				status := strings.TrimSuffix(after1, "</agent_status>")
				var buf bytes.Buffer
				components.AgentStatus(status).Render(ctx, &buf)
				componentHTML = buf.String()
			}
			finalHTML.WriteString(componentHTML)
		}
	}

	return finalHTML.String(), nil
}

func toAgentMessages(messages []types.ChatMessage) []types.AgentMessage {
	var agentMessages []types.AgentMessage
	for _, msg := range messages {
		if msg.Role == "user" || msg.Role == "assistant" || msg.Role == "tool" {
			agentMessages = append(agentMessages, types.AgentMessage{
				Role:    msg.Role,
				Content: msg.Content,
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
