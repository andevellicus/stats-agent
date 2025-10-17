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
	"stats-agent/rag"
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
	cancel        context.CancelFunc
	token         string
	userMessageID string
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

func (cs *ChatService) registerRun(sessionID string, cancel context.CancelFunc, userMessageID string) string {
	token := uuid.New().String()
	var previous context.CancelFunc

	cs.activeRunsMu.Lock()
	if existing, ok := cs.activeRuns[sessionID]; ok {
		previous = existing.cancel
	}
	cs.activeRuns[sessionID] = sessionRun{cancel: cancel, token: token, userMessageID: userMessageID}
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

// GetActiveRun returns whether a run is active for the session and, if so,
// the user message ID that initiated it (used to reattach SSE).
func (cs *ChatService) GetActiveRun(sessionID string) (bool, string) {
	cs.activeRunsMu.Lock()
	defer cs.activeRunsMu.Unlock()
	if run, ok := cs.activeRuns[sessionID]; ok {
		return true, run.userMessageID
	}
	return false, ""
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
			// Only report CSV and Excel files that the agent can directly process
			// Other files (PDFs, images) are tracked in the database but not auto-loaded
			filename := file.Name()
			ext := filepath.Ext(strings.ToLower(filename))
			if ext == ".csv" || ext == ".xlsx" || ext == ".xls" {
				uploadedFiles = append(uploadedFiles, filename)
			}
		}
	}

	initResult, err := cs.agent.InitializeSession(initCtx, sessionID, uploadedFiles)
	if err != nil {
		return fmt.Errorf("failed to initialize python session: %w", err)
	}

	initMessage := types.ChatMessage{
		ID:          uuid.New().String(),
		SessionID:   sessionID,
		Role:        "tool",
		Content:     initResult,
		ContentHash: rag.ComputeMessageContentHash("tool", initResult),
		// Do not render the Python init banner on reload; keep content for LLM context only.
		Rendered: "",
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
// Routes to either dataset mode (with code execution) or document mode (Q&A only) based on session.
func (cs *ChatService) StreamAgentResponse(
	ctx context.Context,
	w http.ResponseWriter,
	input string,
	userMessageID string,
	sessionID string,
	history []types.AgentMessage,
) {
	// Determine session mode to route to appropriate agent workflow
	sessionUUID, err := uuid.Parse(sessionID)
	if err != nil {
		cs.logger.Error("Invalid session ID format", zap.Error(err), zap.String("session_id", sessionID))
		return
	}

	session, err := cs.store.GetSessionByID(ctx, sessionUUID)
	if err != nil {
		cs.logger.Error("Failed to get session for mode detection", zap.Error(err), zap.String("session_id", sessionID))
		// Default to dataset mode if we can't determine
		session.Mode = types.ModeDataset
	}

	// Route based on mode
	if session.Mode == types.ModeDocument {
		cs.streamDocumentResponse(ctx, w, input, userMessageID, sessionID, history)
	} else {
		cs.streamDatasetResponse(ctx, w, input, userMessageID, sessionID, history)
	}
}

// streamDatasetResponse handles the original agentic workflow with Python code execution
func (cs *ChatService) streamDatasetResponse(
    ctx context.Context,
    w http.ResponseWriter,
    input string,
    userMessageID string,
    sessionID string,
    history []types.AgentMessage,
) {
    cs.streamWithRunner(ctx, w, input, userMessageID, sessionID, history, true,
        func(runCtx context.Context, input string, sessionID string, history []types.AgentMessage, stream *agent.Stream) {
            cs.agent.RunDatasetMode(runCtx, input, sessionID, history, stream)
        })
}

// streamDocumentResponse handles document Q&A mode without code execution
func (cs *ChatService) streamDocumentResponse(
    ctx context.Context,
    w http.ResponseWriter,
    input string,
    userMessageID string,
    sessionID string,
    history []types.AgentMessage,
) {
    cs.streamWithRunner(ctx, w, input, userMessageID, sessionID, history, false,
        func(runCtx context.Context, input string, sessionID string, history []types.AgentMessage, stream *agent.Stream) {
            cs.agent.RunDocumentMode(runCtx, input, sessionID, history, stream)
        })
}

// streamWithRunner wraps common SSE + streaming + persistence logic for both modes.
func (cs *ChatService) streamWithRunner(
    ctx context.Context,
    w http.ResponseWriter,
    input string,
    userMessageID string,
    sessionID string,
    history []types.AgentMessage,
    includeFiles bool,
    runFn func(context.Context, string, string, []types.AgentMessage, *agent.Stream),
) {
    agentMessageID := uuid.New().String()
    var writeMu sync.Mutex
    runCtx, cancelRun := context.WithCancel(context.Background())
    token := cs.registerRun(sessionID, cancelRun, userMessageID)
    defer func() {
        cancelRun()
        cs.deregisterRun(sessionID, token)
    }()
    var sseActive atomic.Bool
    sseActive.Store(true)

    safeWrite := func(data StreamData) {
        if runCtx.Err() != nil {
            return
        }
        if !sseActive.Load() {
            return
        }
        if err := cs.streamService.WriteSSEData(ctx, w, data, &writeMu); err != nil {
            if sseActive.CompareAndSwap(true, false) {
                cs.logger.Info("SSE stream closed, continuing in background",
                    zap.Error(err),
                    zap.String("session_id", sessionID),
                    zap.String("user_message_id", userMessageID))
            }
        }
    }

    // Initial SSE events
    safeWrite(StreamData{Type: "remove_loader", Content: "loading-" + userMessageID})
    safeWrite(StreamData{Type: "create_container", Content: agentMessageID})

    pipeReader, pipeWriter := io.Pipe()
    defer pipeReader.Close()

    var captureBuffer bytes.Buffer

    var lastAssistantMu sync.Mutex
    var lastAssistantID string

    persist := func(assistant string, tool *string) {
        assistant = strings.TrimSpace(assistant)
        toolStr := ""
        if tool != nil {
            toolStr = strings.TrimSpace(*tool)
        }
        if assistant == "" && toolStr == "" {
            return
        }

        ctxPersist, cancelPersist := context.WithTimeout(context.Background(), 30*time.Second)
        defer cancelPersist()

        var toolPtr *string
        if toolStr != "" {
            toolPtr = &toolStr
        }

        id, err := cs.messageService.SaveAssistantAndTool(ctxPersist, sessionID, assistant, toolPtr, "")
        if err != nil {
            cs.logger.Error("Incremental message persistence failed",
                zap.Error(err),
                zap.String("session_id", sessionID))
            return
        }
        if id != "" {
            lastAssistantMu.Lock()
            lastAssistantID = id
            lastAssistantMu.Unlock()
        }
    }

    agentStream := agent.NewStream(&captureBuffer, pipeWriter, persist)

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
        runFn(runCtx, input, sessionID, history, agentStream)
        _ = pipeWriter.Close()
    }()

    go func() {
        <-runCtx.Done()
        pipeWriter.CloseWithError(runCtx.Err())
    }()

    <-agentDone
    <-streamDone

    agentStream.Finalize()

    // Post-run work
    backgroundCtx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
    defer cancel()

    if includeFiles {
        newFilePaths, err := cs.fileService.GetAndMarkNewFiles(backgroundCtx, sessionID)
        if err != nil {
            cs.logger.Error("Failed to get and mark new file paths",
                zap.Error(err),
                zap.String("session_id", sessionID))
        }
        if len(newFilePaths) > 0 {
            fileContainerID := fmt.Sprintf("file-container-agent-msg-%s", agentMessageID)
            oobHTML, err := cs.fileService.RenderFileOOBWrapper(backgroundCtx, fileContainerID, newFilePaths)
            if err != nil {
                cs.logger.Warn("Failed to render OOB file wrapper",
                    zap.Error(err),
                    zap.Int("file_count", len(newFilePaths)))
            } else {
                safeWrite(StreamData{Type: "file_append_html", Content: oobHTML})
            }

            dbFilesHTML, err := cs.fileService.RenderFileBlocksForDB(backgroundCtx, newFilePaths)
            if err != nil {
                cs.logger.Error("Failed to render file blocks for DB",
                    zap.Error(err),
                    zap.Int("file_count", len(newFilePaths)))
                dbFilesHTML = ""
            }
            if dbFilesHTML != "" {
                lastAssistantMu.Lock()
                assistantID := lastAssistantID
                lastAssistantMu.Unlock()
                if assistantID != "" {
                    if err := cs.messageService.AppendFilesToMessage(backgroundCtx, assistantID, dbFilesHTML); err != nil {
                        cs.logger.Error("Failed to append files HTML to assistant message",
                            zap.Error(err),
                            zap.String("message_id", assistantID))
                    }
                }
            }
        }
    }

    safeWrite(StreamData{Type: "end"})
}
