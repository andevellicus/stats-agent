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

	component := pages.ChatPage(sessionUUID, sessions, messages)
	component.Render(c.Request.Context(), c.Writer)
}

func (h *ChatHandler) LoadSession(c *gin.Context) {
	sessionID, err := uuid.Parse(c.Param("sessionID"))
	if err != nil {
		c.String(http.StatusBadRequest, "Invalid session ID")
		return
	}

	sessions, _ := h.store.GetSessions(c.Request.Context(), nil)
	messages, _ := h.store.GetMessagesBySession(c.Request.Context(), sessionID)

	// When loading a session, we render the entire chat history.
	// The key change is that historical messages will not have the SSE loader.
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

// andevellicus/stats-agent/stats-agent-sessions/web/handlers/chat.go

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

		h.streamNewFiles(ctx, writeSSEData, sessionID)

		if err := writeSSEData(StreamData{Type: "end"}); err != nil {
			h.logger.Error("Failed to send end message", zap.Error(err))
		}

		rawAgentResponse := agentResponseForDB.String()

		// Remove all agent status messages so they aren't processed or saved
		statusRe := regexp.MustCompile(`(?s)<agent_status>.*?</agent_status>`)
		rawAgentResponse = statusRe.ReplaceAllString(rawAgentResponse, "")

		// Split the entire response into turns based on the execution results delimiter
		turnDelimiter := "<execution_results>"
		turns := strings.Split(rawAgentResponse, turnDelimiter)

		for i, turn := range turns {
			turn = strings.TrimSpace(turn)
			if turn == "" {
				continue
			}

			// The first part of a split by execution_results is always the assistant's turn.
			// Subsequent parts will start with the result content and end with the next assistant response.
			if i == 0 {
				// This is the first assistant message before any tool calls in this response
				if turn != "" {
					assistantRendered, _ := processAgentContentForDB(ctx, turn)
					assistantMessage := types.ChatMessage{
						ID:        generateMessageID(),
						SessionID: sessionID,
						Role:      "assistant",
						Content:   turn,
						Rendered:  assistantRendered,
					}
					if err := h.store.CreateMessage(context.Background(), assistantMessage); err != nil {
						h.logger.Error("Failed to save initial assistant message", zap.Error(err))
					}
				}
			} else {
				// This block contains a tool result followed by the next assistant thought
				parts := strings.SplitN(turn, "</execution_results>", 2)
				toolContentRaw := strings.TrimSpace(parts[0])

				// Save the tool message
				toolContentForDB := turnDelimiter + toolContentRaw + "</execution_results>"
				toolRendered, _ := processAgentContentForDB(ctx, toolContentForDB)
				toolMessage := types.ChatMessage{
					ID:        generateMessageID(),
					SessionID: sessionID,
					Role:      "tool",
					Content:   toolContentForDB,
					Rendered:  toolRendered,
				}
				if err := h.store.CreateMessage(context.Background(), toolMessage); err != nil {
					h.logger.Error("Failed to save tool message", zap.Error(err))
				}

				// If there's a following assistant message in this turn, save it
				if len(parts) > 1 && strings.TrimSpace(parts[1]) != "" {
					assistantContent := strings.TrimSpace(parts[1])
					assistantRendered, _ := processAgentContentForDB(ctx, assistantContent)
					assistantMessage := types.ChatMessage{
						ID:        generateMessageID(),
						SessionID: sessionID,
						Role:      "assistant",
						Content:   assistantContent,
						Rendered:  assistantRendered,
					}
					if err := h.store.CreateMessage(context.Background(), assistantMessage); err != nil {
						h.logger.Error("Failed to save subsequent assistant message", zap.Error(err))
					}
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
			finalHTML.WriteString(strings.TrimSuffix(strings.TrimPrefix(string(html), "<p>"), "</p>\n"))
		}

		// If there is a matching component, render it and append it
		if i < len(matches) {
			match := matches[i]
			var componentHTML string

			if strings.HasPrefix(match, "<python>") {
				code := strings.TrimSuffix(strings.TrimPrefix(match, "<python>"), "</python>")
				var buf bytes.Buffer
				components.PythonCodeBlock(code).Render(ctx, &buf)
				componentHTML = buf.String()
			} else if strings.HasPrefix(match, "<execution_results>") {
				result := strings.TrimSuffix(strings.TrimPrefix(match, "<execution_results>"), "</execution_results>")
				var buf bytes.Buffer
				components.ExecutionResultBlock(result).Render(ctx, &buf)
				componentHTML = buf.String()
			} else if strings.HasPrefix(match, "<agent_status>") {
				status := strings.TrimSuffix(strings.TrimPrefix(match, "<agent_status>"), "</agent_status>")
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
