package services

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"log"
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

	// Send initial SSE messages
	if err := writeSSEData(StreamData{Type: "remove_loader", Content: "loading-" + userMessageID}); err != nil {
		cs.logger.Error("Failed to send remove loader message", zap.Error(err))
		return
	}

	if err := writeSSEData(StreamData{Type: "create_container", Content: agentMessageID}); err != nil {
		cs.logger.Error("Failed to send create container message", zap.Error(err))
		return
	}

	// Capture agent's stdout for streaming
	originalStdout := os.Stdout
	r, pipeWriter, _ := os.Pipe()
	os.Stdout = pipeWriter
	log.SetOutput(pipeWriter)

	var agentResponseForDB bytes.Buffer
	teeReader := io.TeeReader(r, &agentResponseForDB)

	agentDone := make(chan struct{})
	streamDone := make(chan struct{})

	// Stream processing goroutine
	go func() {
		defer close(streamDone)
		cs.streamService.ProcessStreamByWord(ctx, teeReader, writeSSEData)
	}()

	// Agent execution goroutine
	go func() {
		defer func() {
			os.Stdout = originalStdout
			log.SetOutput(originalStdout)
			pipeWriter.Close()
			close(agentDone)
		}()
		cs.agent.Run(ctx, input, sessionID, history)
	}()

	// Wait for completion and handle post-processing
	select {
	case <-ctx.Done():
		cs.logger.Info("Context cancelled, closing SSE connection")
	case <-agentDone:
		<-streamDone

		// Use background context for DB operations after request context might be cancelled
		backgroundCtx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
		defer cancel()

		// Discover and mark new files
		newFilePaths, err := cs.fileService.GetAndMarkNewFiles(backgroundCtx, sessionID)
		if err != nil {
			cs.logger.Error("Failed to get and mark new file paths", zap.Error(err), zap.String("session_id", sessionID))
		}

		// Stream new files as OOB updates
		if len(newFilePaths) > 0 {
			fileContainerID := fmt.Sprintf("file-container-agent-msg-%s", agentMessageID)
			oobHTML, err := cs.fileService.RenderFileOOBWrapper(backgroundCtx, fileContainerID, newFilePaths)
			if err != nil {
				cs.logger.Error("Failed to render file OOB wrapper", zap.Error(err))
			} else {
				if err := writeSSEData(StreamData{Type: "file_append_html", Content: oobHTML}); err != nil {
					cs.logger.Error("Failed to stream file HTML", zap.Error(err))
				}
			}
		}

		// Send end signal
		if err := writeSSEData(StreamData{Type: "end"}); err != nil {
			cs.logger.Error("Failed to send end message", zap.Error(err))
		}

		// Render file blocks for DB storage
		dbFilesHTML, err := cs.fileService.RenderFileBlocksForDB(backgroundCtx, newFilePaths)
		if err != nil {
			cs.logger.Error("Failed to render file blocks for DB", zap.Error(err))
		}

		// Parse and save messages to database
		rawAgentResponse := agentResponseForDB.String()
		if err := cs.messageService.ParseAndSaveAgentResponse(backgroundCtx, rawAgentResponse, sessionID, dbFilesHTML); err != nil {
			cs.logger.Error("Failed to parse and save agent response", zap.Error(err))
		}
	}
}
