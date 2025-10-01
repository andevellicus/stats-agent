package services

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"stats-agent/agent"
	"stats-agent/database"
	"stats-agent/web/templates/components"
	"stats-agent/web/types"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/google/uuid"
	"go.uber.org/zap"
)

type sessionRun struct {
	cancel context.CancelFunc
	token  string
}

type ChatService struct {
	agent          *agent.Agent
	store          *database.PostgresStore
	logger         *zap.Logger
	fileService    *FileService
	messageService *MessageService
	streamService  *StreamService
	activeRunsMu   sync.Mutex
	activeRuns     map[string]sessionRun
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
		activeRuns:     make(map[string]sessionRun),
	}
}

func (cs *ChatService) registerRun(sessionID string, cancel context.CancelFunc) string {
	token := uuid.New().String()
	var previous context.CancelFunc

	cs.activeRunsMu.Lock()
	if existing, ok := cs.activeRuns[sessionID]; ok {
		previous = existing.cancel
	}
	cs.activeRuns[sessionID] = sessionRun{cancel: cancel, token: token}
	cs.activeRunsMu.Unlock()

	if previous != nil {
		cs.logger.Info("Cancelling previous active run for session", zap.String("session_id", sessionID))
		previous()
	}

	return token
}

func (cs *ChatService) deregisterRun(sessionID, token string) {
	cs.activeRunsMu.Lock()
	defer cs.activeRunsMu.Unlock()
	if existing, ok := cs.activeRuns[sessionID]; ok && existing.token == token {
		delete(cs.activeRuns, sessionID)
	}
}

func (cs *ChatService) StopSessionRun(sessionID string) {
	cs.activeRunsMu.Lock()
	run, ok := cs.activeRuns[sessionID]
	if ok {
		delete(cs.activeRuns, sessionID)
	}
	cs.activeRunsMu.Unlock()

	if ok {
		cs.logger.Info("Cancelling active run for session", zap.String("session_id", sessionID))
		run.cancel()
	}
}

// InitializeSession initializes a new session by checking for uploaded files
// and running Python initialization code.
func (cs *ChatService) InitializeSession(ctx context.Context, sessionID string) error {
	cs.logger.Info("Initializing new session", zap.String("session_id", sessionID))

	select {
	case <-ctx.Done():
		cs.logger.Debug("Request context cancelled during initialization; continuing in background",
			zap.Error(ctx.Err()),
			zap.String("session_id", sessionID))
	default:
	}

	initCtx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()

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

	initResult, err := cs.agent.InitializeSession(initCtx, sessionID, uploadedFiles)
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

	return cs.store.CreateMessage(initCtx, initMessage)
}

func (cs *ChatService) GenerateAndSetTitle(ctx context.Context, sessionID uuid.UUID, firstMessage string, writeFunc func(StreamData) error) {
	session, err := cs.store.GetSessionByID(ctx, sessionID)
	if err != nil {
		cs.logger.Error("Failed to get session for title update", zap.Error(err), zap.String("session_id", sessionID.String()))
		return
	}

	title, err := cs.agent.GenerateTitle(ctx, firstMessage)
	if err != nil {
		cs.logger.Warn("Failed to generate title", zap.Error(err), zap.String("session_id", sessionID.String()))
		return
	}

	if title != "" && len(strings.Split(title, " ")) <= 10 {
		if err := cs.store.UpdateSessionTitle(ctx, sessionID, title); err != nil {
			cs.logger.Warn("Failed to update session title", zap.Error(err), zap.String("session_id", sessionID.String()))
			return
		}

		// Update the session object with the new title before rendering
		session.Title = title

		// Render the component to a buffer
		var buf bytes.Buffer
		if err := components.SessionLinkOOB(session).Render(ctx, &buf); err != nil {
			cs.logger.Error("Failed to render SessionLinkOOB component", zap.Error(err))
			return
		}

		if err := writeFunc(StreamData{Type: "sidebar_update", Content: buf.String()}); err != nil {
			cs.logger.Warn("Failed to send sidebar update SSE", zap.Error(err))
		}
	}
}

// CleanupSession cleans up agent session bindings (e.g., Python executor bindings).
func (cs *ChatService) CleanupSession(sessionID string) {
	cs.StopSessionRun(sessionID)
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
	runCtx, cancelRun := context.WithCancel(context.Background())
	token := cs.registerRun(sessionID, cancelRun)
	defer func() {
		cancelRun()
		cs.deregisterRun(sessionID, token)
	}()
	var sseActive atomic.Bool
	sseActive.Store(true)

	// Helper function to write SSE data without aborting background work on failure.
	safeWrite := func(data StreamData) {
		if runCtx.Err() != nil {
			return
		}
		if !sseActive.Load() {
			return
		}
		if err := cs.streamService.WriteSSEData(ctx, w, data, &writeMu); err != nil {
			if sseActive.CompareAndSwap(true, false) {
				cs.logger.Info("SSE stream closed, continuing agent in background",
					zap.Error(err),
					zap.String("session_id", sessionID),
					zap.String("user_message_id", userMessageID))
			}
		}
	}

	// Send initial SSE messages - best effort for active clients
	safeWrite(StreamData{Type: "remove_loader", Content: "loading-" + userMessageID})
	safeWrite(StreamData{Type: "create_container", Content: agentMessageID})

	pipeReader, pipeWriter := io.Pipe()
	defer pipeReader.Close()

	var captureBuffer bytes.Buffer
	fanOutWriter := io.MultiWriter(&captureBuffer, pipeWriter)
	agentStream := agent.NewStream(fanOutWriter)

	streamDone := make(chan struct{})
	go func() {
		defer close(streamDone)
		cs.streamService.ProcessStreamByWord(runCtx, pipeReader, func(data StreamData) error {
			safeWrite(data)
			return nil
		})
	}()

	agentDone := make(chan struct{})
	go func() {
		defer close(agentDone)
		cs.agent.Run(runCtx, input, sessionID, history, agentStream)
		_ = pipeWriter.Close()
	}()

	go func() {
		<-runCtx.Done()
		pipeWriter.CloseWithError(runCtx.Err())
	}()

	<-agentDone
	<-streamDone

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
			safeWrite(StreamData{Type: "file_append_html", Content: oobHTML})
		}
	}

	// Send end signal - best effort
	safeWrite(StreamData{Type: "end"})

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
	rawAgentResponse := captureBuffer.String()
	if err := cs.messageService.ParseAndSaveAgentResponse(backgroundCtx, rawAgentResponse, sessionID, dbFilesHTML); err != nil {
		cs.logger.Error("Failed to parse and save agent response - CONVERSATION DATA MAY BE LOST",
			zap.Error(err),
			zap.String("session_id", sessionID),
			zap.Int("response_length", len(rawAgentResponse)))
	}
}
