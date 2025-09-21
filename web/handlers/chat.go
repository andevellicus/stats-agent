package handlers

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
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

type StreamData struct {
	Type    string `json:"type"`
	Content string `json:"content,omitempty"`
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

	// Create a mutex to synchronize writes to the response writer
	var writeMu sync.Mutex

	// Helper function to safely write SSE data
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

	// Send message to remove loading indicator
	if err := writeSSEData(StreamData{Type: "remove_loader", Content: "loading-" + userMessageID}); err != nil {
		h.logger.Error("Failed to send remove loader message", zap.Error(err))
		return
	}

	// Send message to create the agent message container
	if err := writeSSEData(StreamData{Type: "create_container", Content: agentMessageID}); err != nil {
		h.logger.Error("Failed to send create container message", zap.Error(err))
		return
	}

	// Pipe to capture stdout
	r, pipeW, err := os.Pipe()
	if err != nil {
		h.logger.Error("Failed to create pipe", zap.Error(err))
		writeSSEData(StreamData{Type: "error", Content: "Internal server error"})
		return
	}

	originalStdout := os.Stdout
	os.Stdout = pipeW
	log.SetOutput(pipeW)

	// Channel to signal when agent is done
	agentDone := make(chan error, 1)
	streamDone := make(chan error, 1)

	// Buffer to accumulate all output for final processing
	var outputBuffer strings.Builder
	var bufferMu sync.Mutex

	// Goroutine to stream the captured output
	go func() {
		defer close(streamDone)

		buf := make([]byte, 1024)
		for {
			select {
			case <-ctx.Done():
				streamDone <- ctx.Err()
				return
			default:
			}

			n, err := r.Read(buf)
			if err != nil {
				if err != io.EOF {
					h.logger.Error("Error reading from stdout pipe", zap.Error(err))
					streamDone <- err
				} else {
					streamDone <- nil
				}
				return
			}

			if n > 0 {
				chunk := string(buf[:n])

				// Add to buffer for final processing
				bufferMu.Lock()
				outputBuffer.WriteString(chunk)
				bufferMu.Unlock()

				// Stream raw chunks for real-time display (without tag processing)
				data := StreamData{Type: "chunk", Content: chunk}
				if err := writeSSEData(data); err != nil {
					h.logger.Error("Error writing chunk data", zap.Error(err))
					streamDone <- err
					return
				}
			}
		}
	}()

	// Run the agent
	go func() {
		defer func() {
			// Restore stdout
			os.Stdout = originalStdout
			log.SetOutput(originalStdout)
			pipeW.Close()
			r.Close()
		}()

		defer close(agentDone)

		h.agent.Run(ctx, input)
		agentDone <- nil
	}()

	// Wait for either agent completion or context cancellation
	select {
	case <-ctx.Done():
		h.logger.Info("Context cancelled, closing SSE connection")
		return
	case err := <-agentDone:
		if err != nil {
			h.logger.Error("Agent execution failed", zap.Error(err))
			writeSSEData(StreamData{Type: "error", Content: "Agent execution failed"})
			return
		}

		// Wait a moment for any remaining output to be streamed
		select {
		case <-streamDone:
		case <-ctx.Done():
			return
		}

		// Process the complete accumulated output for tag conversion
		bufferMu.Lock()
		completeOutput := outputBuffer.String()
		bufferMu.Unlock()

		processedOutput := processAgentOutput(completeOutput)

		// Send the fully processed content
		if err := writeSSEData(StreamData{Type: "final_content", Content: processedOutput}); err != nil {
			h.logger.Error("Failed to send final content", zap.Error(err))
		}

		// Send end of stream message
		if err := writeSSEData(StreamData{Type: "end"}); err != nil {
			h.logger.Error("Failed to send end message", zap.Error(err))
		}
	}
}

func generateSessionID() string {
	return "session_" + uuid.New().String()
}

func generateMessageID() string {
	return "msg_" + uuid.New().String()
}

// StreamBuffer holds partial content for processing streaming tags
type StreamBuffer struct {
	buffer strings.Builder
	mu     sync.Mutex
}

// processAgentOutput converts <python> and <execution_result> tags to markdown code blocks
// This version handles streaming content by maintaining a buffer for complete processing
func processAgentOutput(content string) string {
	// Convert Python code blocks
	content = strings.ReplaceAll(content, "<python>", "\n```python\n")
	content = strings.ReplaceAll(content, "</python>", "\n```\n")

	// Convert execution result blocks
	content = strings.ReplaceAll(content, "<execution_result>", "\n```\n")
	content = strings.ReplaceAll(content, "</execution_result>", "\n```\n")

	return content
}
