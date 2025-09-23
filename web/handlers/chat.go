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
	h.streamAgentResponse(ctx, c.Writer, userMessage.Content, userMessageID)
}

// processStreamByWord reads from the stream rune by rune, buffers them into words,
// and processes each word for tags before sending it to the client. This version is stateful.
func (h *ChatHandler) processStreamByWord(ctx context.Context, r io.Reader, writeSSEData func(StreamData) error) {
	reader := bufio.NewReader(r)
	var currentWord strings.Builder
	var isBufferingImage bool
	var imagePathBuffer strings.Builder

	// processToken is a recursive function that handles tags within a word/token.
	var processToken func(string)
	processToken = func(token string) {
		if token == "" {
			return
		}

		// Handle image buffering state
		if isBufferingImage {
			if strings.Contains(token, "</image>") {
				parts := strings.SplitN(token, "</image>", 2)
				imagePathBuffer.WriteString(parts[0])

				imagePath := strings.TrimSpace(imagePathBuffer.String())
				webPath := strings.Replace(imagePath, "/app/workspace/", "/workspace/", 1)

				// Render the ImageBlock component to a buffer and send
				var buf bytes.Buffer
				component := components.ImageBlock(webPath)
				if err := component.Render(ctx, &buf); err == nil {
					writeSSEData(StreamData{Type: "chunk", Content: buf.String()})
				}

				// Reset state and process the rest of the token
				isBufferingImage = false
				imagePathBuffer.Reset()
				processToken(parts[1])
			} else {
				imagePathBuffer.WriteString(token)
			}
			return
		}

		// Check for our special tags
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
		case strings.Contains(token, "<execution_result>"):
			parts := strings.SplitN(token, "<execution_result>", 2)
			writeSSEData(StreamData{Type: "chunk", Content: parts[0]})
			writeSSEData(StreamData{Type: "chunk", Content: "\n```\n"})
			processToken(parts[1])
		case strings.Contains(token, "</execution_result>"):
			parts := strings.SplitN(token, "</execution_result>", 2)
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
		case strings.Contains(token, "<image>"):
			parts := strings.SplitN(token, "<image>", 2)
			writeSSEData(StreamData{Type: "chunk", Content: parts[0]})
			isBufferingImage = true
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

func (h *ChatHandler) streamAgentResponse(ctx context.Context, w http.ResponseWriter, input string, userMessageID string) {
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

	// Goroutine to stream the captured output using our new word-by-word processor
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

	// Run the agent in a separate goroutine
	go func() {
		defer func() {
			os.Stdout = originalStdout
			log.SetOutput(originalStdout)
			pipeW.Close()
			r.Close()
		}()
		h.agent.Run(ctx, input)
		close(agentDone)
	}()

	select {
	case <-ctx.Done():
		h.logger.Info("Context cancelled, closing SSE connection")
	case <-agentDone:
		// Agent finished, wait for the stream processing to complete.
		<-streamDone
		// Send the final end-of-stream message.
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
