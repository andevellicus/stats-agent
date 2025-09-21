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

	// Send a simple initial message to establish connection
	fmt.Fprintf(w, "data: %s\n\n", agentMessageID)
	w.(http.Flusher).Flush()

	// Run the agent and collect the response
	response, err := h.webAgent.RunForWeb(ctx, input)
	if err != nil {
		h.logger.Error("Agent execution failed", zap.Error(err))
		fmt.Fprintf(w, "data: ERROR: %v\n\n", err)
		w.(http.Flusher).Flush()
		return
	}

	// Format the response
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

	// Send the final response as a simple agent message component
	finalContent := responseContent.String()
	agentMessage := types.ChatMessage{
		Role:      "assistant",
		Content:   finalContent,
		ID:        agentMessageID,
		SessionID: sessionID,
	}

	// Add to session
	session.mu.Lock()
	session.Messages = append(session.Messages, agentMessage)
	session.mu.Unlock()

	// Render the agent message component
	var buf strings.Builder
	component := components.AgentMessage(agentMessage)
	err = component.Render(ctx, &buf)
	if err != nil {
		h.logger.Error("Failed to render agent message", zap.Error(err))
		fmt.Fprintf(w, "data: <div class=\"text-red-500\">Error rendering response</div>\n\n")
		w.(http.Flusher).Flush()
		return
	}

	// Send the rendered component
	fmt.Fprintf(w, "data: %s\n\n", buf.String())
	w.(http.Flusher).Flush()
}

func generateSessionID() string {
	return "session_" + uuid.New().String()
}

func generateMessageID() string {
	return "msg_" + uuid.New().String()
}