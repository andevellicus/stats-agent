package web

import (
	"context"
	"fmt"
	"os"
	"stats-agent/agent"
	"stats-agent/database"
	"time"

	"github.com/google/uuid"
	"go.uber.org/zap"
)

// CleanupService handles session and workspace cleanup operations
type CleanupService struct {
	store  *database.PostgresStore
	agent  *agent.Agent
	logger *zap.Logger
}

// NewCleanupService creates a new cleanup service instance
func NewCleanupService(store *database.PostgresStore, agent *agent.Agent, logger *zap.Logger) *CleanupService {
	return &CleanupService{
		store:  store,
		agent:  agent,
		logger: logger,
	}
}

// CleanupStaleWorkspaces finds and deletes sessions older than maxAge
// Returns the number of sessions deleted and any error encountered
func (cs *CleanupService) CleanupStaleWorkspaces(ctx context.Context, maxAge time.Duration) (int, error) {
	cutoffTime := time.Now().Add(-maxAge)

	cs.logger.Info("Starting stale workspace cleanup",
		zap.Time("cutoff_time", cutoffTime),
		zap.Duration("max_age", maxAge))

	// Get list of stale sessions
	staleSessions, err := cs.store.GetStaleSessions(ctx, cutoffTime)
	if err != nil {
		return 0, fmt.Errorf("failed to get stale sessions: %w", err)
	}

	if len(staleSessions) == 0 {
		cs.logger.Debug("No stale sessions found")
		return 0, nil
	}

	cs.logger.Info("Found stale sessions to clean up",
		zap.Int("count", len(staleSessions)))

	// Delete each stale session
	deletedCount := 0
	for _, sessionID := range staleSessions {
		if err := cs.DeleteSessionAndWorkspace(ctx, sessionID); err != nil {
			cs.logger.Error("Failed to delete stale session",
				zap.Error(err),
				zap.String("session_id", sessionID.String()))
			// Continue with other sessions even if one fails
			continue
		}
		deletedCount++
	}

	cs.logger.Info("Stale workspace cleanup completed",
		zap.Int("sessions_deleted", deletedCount),
		zap.Int("sessions_failed", len(staleSessions)-deletedCount))

	return deletedCount, nil
}

// DeleteSessionAndWorkspace encapsulates the full deletion logic for a session
// This includes database deletion, Python executor cleanup, and workspace directory removal
func (cs *CleanupService) DeleteSessionAndWorkspace(ctx context.Context, sessionID uuid.UUID) error {
	sessionIDStr := sessionID.String()

	// Get session info before deleting (to get workspace path)
	session, err := cs.store.GetSessionByID(ctx, sessionID)
	if err != nil {
		return fmt.Errorf("failed to get session info: %w", err)
	}

	// Delete from database (this cascades to messages)
	if err := cs.store.DeleteSession(ctx, sessionID); err != nil {
		return fmt.Errorf("failed to delete session from database: %w", err)
	}

	// Cleanup Python executor session binding
	cs.agent.CleanupSession(sessionIDStr)

	// Delete workspace directory
	workspaceDir := session.WorkspacePath
	if workspaceDir != "" {
		if err := os.RemoveAll(workspaceDir); err != nil {
			cs.logger.Warn("Failed to delete workspace directory",
				zap.Error(err),
				zap.String("path", workspaceDir),
				zap.String("session_id", sessionIDStr))
			// Don't return error - session already deleted from DB
		} else {
			cs.logger.Debug("Workspace directory deleted",
				zap.String("path", workspaceDir),
				zap.String("session_id", sessionIDStr))
		}
	}

	return nil
}
