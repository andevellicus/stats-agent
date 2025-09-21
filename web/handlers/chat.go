package handlers

import (
	"context"
	"fmt"
	"net/http"
	"stats-agent/agent"
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
	agent    *agent.Agent
	webAgent *WebAgent
	logger   *zap.Logger
	sessions map[string]*ChatSession
	mu       sync.RWMutex
}

type ChatSession struct {
	ID       string
	Messages []types.ChatMessage
	mu       sync.RWMutex
}

type ChatRequest struct {
	Message   string `json:"message" form:"message"`
	SessionID string `json:"session_id" form:"session_id"`
}

func NewChatHandler(agent *agent.Agent, logger *zap.Logger) *ChatHandler {
	return &ChatHandler{
		agent:    agent,
		webAgent: NewWebAgent(agent, logger),
		logger:   logger,
		sessions: make(map[string]*ChatSession),
	}
}

func (h *ChatHandler) Index(c *gin.Context) {
	sessionID := generateSessionID()

	// Create new session
	h.mu.Lock()
	h.sessions[sessionID] = &ChatSession{
		ID:       sessionID,
		Messages: []types.ChatMessage{},
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
			ID:       req.SessionID,
			Messages: []types.ChatMessage{},
		}
		h.sessions[req.SessionID] = session
	}
	h.mu.Unlock()

	// Add user message to session
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

	// Test connection first
	fmt.Fprintf(c.Writer, "data: Connection established\n\n")
	c.Writer.(http.Flusher).Flush()

	h.mu.RLock()
	session, exists := h.sessions[sessionID]
	h.mu.RUnlock()

	if !exists {
		fmt.Fprintf(c.Writer, "data: Session not found\n\n")
		c.Writer.(http.Flusher).Flush()
		return
	}

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
	h.streamAgentResponse(ctx, c.Writer, session, userMessage.Content, sessionID, userMessageID)
}

func (h *ChatHandler) streamAgentResponse(ctx context.Context, w http.ResponseWriter, session *ChatSession, input string, sessionID string, userMessageID string) {
	agentMessageID := generateMessageID()

	// Send initial loading indicator replacement message
	fmt.Fprintf(w, "data: <div id=\"%s\" class=\"flex justify-start\"><div class=\"chat-message chat-message--agent\"><div class=\"font-medium text-xs mb-1 text-gray-500\">Stats Agent</div><div id=\"content-%s\" class=\"whitespace-pre-wrap text-gray-700\"></div></div></div>\n\n", agentMessageID, agentMessageID)
	w.(http.Flusher).Flush()

	// Send script to remove loading indicator
	fmt.Fprintf(w, "data: <script>document.getElementById('loading-%s')?.remove();</script>\n\n", userMessageID)
	w.(http.Flusher).Flush()

	// Buffer to collect streaming content
	var contentBuffer strings.Builder

	// Run the agent with streaming callback
	response, err := h.webAgent.RunForWebStream(ctx, input, func(chunk string) {
		contentBuffer.WriteString(chunk)
		// Send incremental content update
		escapedContent := strings.ReplaceAll(contentBuffer.String(), `"`, `\"`)
		escapedContent = strings.ReplaceAll(escapedContent, "\n", "\\n")
		escapedContent = strings.ReplaceAll(escapedContent, "\r", "")

		fmt.Fprintf(w, "data: <script>document.getElementById('content-%s').textContent = `%s`;</script>\n\n", agentMessageID, escapedContent)
		w.(http.Flusher).Flush()
	})

	if err != nil {
		h.logger.Error("Agent execution failed", zap.Error(err))
		fmt.Fprintf(w, "data: <script>document.getElementById('content-%s').innerHTML = '<div class=\"text-red-500\">Error: %v</div>';</script>\n\n", agentMessageID, err)
		w.(http.Flusher).Flush()
		return
	}

	// Format final response with parsed content
	var responseContent strings.Builder
	if response.Content != "" {
		responseContent.WriteString(response.Content)
	}

	// Add code blocks and outputs
	for i, code := range response.CodeBlocks {
		responseContent.WriteString("\n\n**Python Code:**\n```python\n")
		responseContent.WriteString(code)
		responseContent.WriteString("\n```")

		if i < len(response.OutputBlocks) {
			responseContent.WriteString("\n\n**Output:**\n```\n")
			responseContent.WriteString(response.OutputBlocks[i])
			responseContent.WriteString("\n```")
		}
	}

	// Send final formatted content
	finalContent := responseContent.String()
	if finalContent != "" {
		escapedFinal := strings.ReplaceAll(finalContent, `"`, `\"`)
		escapedFinal = strings.ReplaceAll(escapedFinal, "\n", "\\n")
		escapedFinal = strings.ReplaceAll(escapedFinal, "\r", "")

		fmt.Fprintf(w, "data: <script>document.getElementById('content-%s').textContent = `%s`;</script>\n\n", agentMessageID, escapedFinal)
		w.(http.Flusher).Flush()
	}

	// Add final message to session
	agentMessage := types.ChatMessage{
		Role:      "assistant",
		Content:   finalContent,
		ID:        agentMessageID,
		SessionID: sessionID,
	}

	session.mu.Lock()
	session.Messages = append(session.Messages, agentMessage)
	session.mu.Unlock()
}

func generateSessionID() string {
	return "session_" + uuid.New().String()
}

func generateMessageID() string {
	return "msg_" + uuid.New().String()
}