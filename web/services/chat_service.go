package services

import (
	"context"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"stats-agent/agent"
	"stats-agent/database"
	"stats-agent/web/types"
	"sync"
	"time"

	"github.com/google/uuid"
	"go.uber.org/zap"
)

type ChatService struct {
	agent          *agent.Agent
	store          *database.PostgresStore
	logger         *zap.Logger
	fileService    *FileService
	messageService *MessageService
	streamService  *StreamService
}

func NewChatService(
	agent *agent.Agent,
	store *database.PostgresStore,
	logger *zap.Logger,
	fileService *FileService,
	messageService *MessageService,
	streamService *StreamService,
) *ChatService {
	return &ChatService{
		agent:          agent,
		store:          store,
		logger:         logger,
		fileService:    fileService,
		messageService: messageService,
		streamService:  streamService,
	}
}

// InitializeSession initializes a new session by checking for uploaded files
// and running Python initialization code.
func (cs *ChatService) InitializeSession(ctx context.Context, sessionID string) error {
	cs.logger.Info("Initializing new session", zap.String("session_id", sessionID))

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

	initResult, err := cs.agent.InitializeSession(ctx, sessionID, uploadedFiles)
	if err != nil {
		return fmt.Errorf("failed to initialize python session: %w", err)
	}

	initMessage := types.ChatMessage{
		ID:        uuid.New().String(),
		SessionID: sessionID,
		Role:      "system",
		Content:   initResult,
		Rendered:  fmt.Sprintf("<pre><code>%s</code></pre>", initResult),
	}

	return cs.store.CreateMessage(ctx, initMessage)
}

// CleanupSession cleans up agent session bindings (e.g., Python executor bindings).
func (cs *ChatService) CleanupSession(sessionID string) {
	cs.agent.CleanupSession(sessionID)
}

// StreamAgentResponse orchestrates the agent's response streaming via SSE.
// It captures stdout, streams word-by-word, tracks new files, and saves messages to DB.
func (cs *ChatService) StreamAgentResponse(
	ctx context.Context,
	w http.ResponseWriter,
	input string,
	userMessageID string,
	sessionID string,
	history []types.AgentMessage,
) {
	agentMessageID := uuid.New().String()
	var writeMu sync.Mutex

	// Helper function to write SSE data
	writeSSEData := func(data StreamData) error {
		return cs.streamService.WriteSSEData(ctx, w, data, &writeMu)
	}

	// Send initial SSE messages - critical for UI responsiveness
	if err := writeSSEData(StreamData{Type: "remove_loader", Content: "loading-" + userMessageID}); err != nil {
		cs.logger.Error("Failed to send remove loader message, aborting stream",
			zap.Error(err),
			zap.String("session_id", sessionID))
		return
	}

	if err := writeSSEData(StreamData{Type: "create_container", Content: agentMessageID}); err != nil {
		cs.logger.Error("Failed to send create container message, aborting stream",
			zap.Error(err),
			zap.String("session_id", sessionID))
		return
	}

	// Create streaming orchestrator with proper error handling
	orchestrator, err := NewStreamingOrchestrator(cs.logger)
	if err != nil {
		cs.logger.Error("Failed to create streaming orchestrator",
			zap.Error(err),
			zap.String("session_id", sessionID))
		writeSSEData(StreamData{Type: "error", Content: "Failed to initialize streaming"})
		return
	}
	defer orchestrator.StopCapture() // Always restore stdout

	// Start capturing stdout
	orchestrator.StartCapture()

	// Reduced goroutines: Run agent + stream processor in parallel
	agentDone := make(chan struct{})

	// Agent execution goroutine
	go func() {
		defer close(agentDone)
		cs.agent.Run(ctx, input, sessionID, history)
		orchestrator.StopCapture() // Close pipe write end to signal stream processor
	}()

	// Stream processing (runs in current goroutine, blocks until agent done)
	orchestrator.StreamAndWait(ctx, cs.streamService, writeSSEData)

	// Wait for agent to finish
	<-agentDone

	// Use background context for DB operations after request context might be cancelled
	backgroundCtx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	// Discover and mark new files - non-critical, continue if fails
	newFilePaths, err := cs.fileService.GetAndMarkNewFiles(backgroundCtx, sessionID)
	if err != nil {
		cs.logger.Error("Failed to get and mark new file paths",
			zap.Error(err),
			zap.String("session_id", sessionID))
		// Continue - files won't be displayed this time but can be discovered later
	}

	// Stream new files as OOB updates - non-critical
	if len(newFilePaths) > 0 {
		fileContainerID := fmt.Sprintf("file-container-agent-msg-%s", agentMessageID)
		oobHTML, err := cs.fileService.RenderFileOOBWrapper(backgroundCtx, fileContainerID, newFilePaths)
		if err != nil {
			cs.logger.Error("Failed to render file OOB wrapper",
				zap.Error(err),
				zap.Int("file_count", len(newFilePaths)))
		} else {
			if err := writeSSEData(StreamData{Type: "file_append_html", Content: oobHTML}); err != nil {
				cs.logger.Error("Failed to stream file HTML", zap.Error(err))
			}
		}
	}

	// Send end signal - best effort
	if err := writeSSEData(StreamData{Type: "end"}); err != nil {
		cs.logger.Error("Failed to send end message", zap.Error(err))
	}

	// Render file blocks for DB storage - non-critical
	dbFilesHTML, err := cs.fileService.RenderFileBlocksForDB(backgroundCtx, newFilePaths)
	if err != nil {
		cs.logger.Error("Failed to render file blocks for DB",
			zap.Error(err),
			zap.Int("file_count", len(newFilePaths)))
		// Continue without file HTML in DB
		dbFilesHTML = ""
	}

	// Parse and save messages to database - critical for preserving conversation
	rawAgentResponse := orchestrator.GetCapturedContent()
	if err := cs.messageService.ParseAndSaveAgentResponse(backgroundCtx, rawAgentResponse, sessionID, dbFilesHTML); err != nil {
		cs.logger.Error("Failed to parse and save agent response - CONVERSATION DATA MAY BE LOST",
			zap.Error(err),
			zap.String("session_id", sessionID),
			zap.Int("response_length", len(rawAgentResponse)))
	}
}
