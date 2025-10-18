package rag

import (
	"context"
	"fmt"
	"time"

	"github.com/google/uuid"
	"go.uber.org/zap"
)

func (r *RAG) DeleteSessionDocuments(sessionID string) error {
    ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
    defer cancel()

	sessionUUID, err := uuid.Parse(sessionID)
	if err != nil {
		r.logger.Warn("Unable to parse session ID for RAG document cleanup",
			zap.Error(err),
			zap.String("session_id", sessionID))
		return fmt.Errorf("invalid session ID: %w", err)
	}

    deleted, err := r.store.DeleteRAGDocumentsBySession(ctx, sessionUUID)
    if err != nil {
        r.logger.Warn("Failed to delete RAG documents from database",
            zap.Error(err),
            zap.String("session_id", sessionID))
        return fmt.Errorf("failed to delete session documents from database: %w", err)
    }

	if deleted > 0 {
		r.logger.Debug("Deleted RAG documents from database",
			zap.String("session_id", sessionID),
			zap.Int64("documents_deleted", deleted))
	}

    // Cleanup Graph overlays (edges, aliases). Best-effort; ignore errors.
    if r.graph != nil && r.graph.Enabled() {
        if count, err := r.graph.DeleteEdgesBySession(ctx, sessionID); err == nil && count > 0 {
            r.logger.Debug("Deleted graph edges for session", zap.String("session_id", sessionID), zap.Int64("edges_deleted", count))
        } else if err != nil {
            r.logger.Warn("Failed to delete graph edges for session", zap.String("session_id", sessionID), zap.Error(err))
        }
        if count, err := r.graph.DeleteAliasesBySession(ctx, sessionID); err == nil && count > 0 {
            r.logger.Debug("Deleted variable aliases for session", zap.String("session_id", sessionID), zap.Int64("aliases_deleted", count))
        } else if err != nil {
            r.logger.Warn("Failed to delete variable aliases for session", zap.String("session_id", sessionID), zap.Error(err))
        }
    }

    r.clearSessionDataset(sessionID)
    return nil
}
