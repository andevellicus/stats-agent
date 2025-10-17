package services

import (
    "context"
    "database/sql"
    "errors"
    "fmt"
    "os"
	"path/filepath"
	"stats-agent/database"
	"stats-agent/web/types"

	"github.com/google/uuid"
	"go.uber.org/zap"
)

type SessionService struct {
	store  *database.PostgresStore
	logger *zap.Logger
}

func NewSessionService(store *database.PostgresStore, logger *zap.Logger) *SessionService {
	return &SessionService{
		store:  store,
		logger: logger,
	}
}

// CreateWorkspace creates the workspace directory for a session if it doesn't exist.
func (ss *SessionService) CreateWorkspace(sessionID uuid.UUID) error {
	workspaceDir := filepath.Join("workspaces", sessionID.String())
	if err := os.MkdirAll(workspaceDir, 0755); err != nil {
		ss.logger.Error("Failed to create workspace directory",
			zap.Error(err),
			zap.String("session_id", sessionID.String()))
		return fmt.Errorf("could not create workspace: %w", err)
	}
	return nil
}

// ValidateAndGetSession retrieves a session and validates ownership.
// Returns the session if valid, or creates a new one if session not found.
// Returns error for other failures or ownership violations.
func (ss *SessionService) ValidateAndGetSession(
    ctx context.Context,
    sessionID uuid.UUID,
    userID *uuid.UUID,
) (*types.Session, bool, error) {
    session, err := ss.store.GetSessionByID(ctx, sessionID)
    if err != nil {
        if errors.Is(err, sql.ErrNoRows) {
            // Session not found - caller should create new one
            return nil, true, nil // true = should create new session
        }
        ss.logger.Error("Failed to get session",
            zap.Error(err),
            zap.String("session_id", sessionID.String()))
        return nil, false, fmt.Errorf("could not load session: %w", err)
    }

    // Enforce ownership strictly; no legacy claiming.
    if userID != nil && (session.UserID == nil || *session.UserID != *userID) {
        ss.logger.Warn("Attempted to access session belonging to different user",
            zap.String("session_id", sessionID.String()),
            zap.Any("session_owner_id", session.UserID),
            zap.String("user_id", userID.String()))
        return nil, false, fmt.Errorf("unauthorized access to session")
    }

    return &session, false, nil
}

// DetectAndSetMode sets the session mode based on the first uploaded file type.
// Only sets mode if session has no files yet.
// Returns true if mode was updated.
func (ss *SessionService) DetectAndSetMode(ctx context.Context, sessionID uuid.UUID, fileExt string) (bool, error) {
	// Check if this is the first file upload (determine mode)
	files, err := ss.store.GetFilesBySession(ctx, sessionID)
	if err != nil {
		ss.logger.Warn("Failed to check existing files, continuing with default mode",
			zap.Error(err))
		files = []database.FileRecord{} // Treat as no files
	}

	// If files already exist, mode is already set
	if len(files) > 0 {
		return false, nil
	}

	// Get current session to check if mode needs updating
	session, err := ss.store.GetSessionByID(ctx, sessionID)
	if err != nil {
		return false, fmt.Errorf("failed to get session for mode detection: %w", err)
	}

	// Determine mode based on file type
	var newMode string
	if fileExt == ".pdf" {
		newMode = types.ModeDocument
	} else {
		newMode = types.ModeDataset
	}

	// Update session mode if it differs from current
	if session.Mode != newMode {
		if err := ss.store.UpdateSessionMode(ctx, sessionID, newMode); err != nil {
			ss.logger.Warn("Failed to update session mode, continuing with current mode",
				zap.Error(err),
				zap.String("session_id", sessionID.String()),
				zap.String("new_mode", newMode))
			return false, err
		}
		ss.logger.Info("Session mode set based on first file upload",
			zap.String("session_id", sessionID.String()),
			zap.String("mode", newMode),
			zap.String("file_type", fileExt))
		return true, nil
	}

	return false, nil
}

// GetSessionsForSidebar retrieves all sessions for a user (or all sessions if userID is nil).
// Returns empty slice on error to allow graceful degradation.
func (ss *SessionService) GetSessionsForSidebar(ctx context.Context, userID *uuid.UUID) []types.Session {
	sessions, err := ss.store.GetSessions(ctx, userID)
	if err != nil {
		ss.logger.Error("Failed to get sessions for sidebar", zap.Error(err))
		return []types.Session{} // Empty sidebar on error
	}
	return sessions
}
