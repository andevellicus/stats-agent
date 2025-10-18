package graph

import (
    "context"
)

// TouchMetadata upserts a graph_metadata record for the session, updating last_sync_at.
func (g *Graph) TouchMetadata(ctx context.Context, sessionID string) error {
    if !g.Enabled() {
        return nil
    }
    query := `
        INSERT INTO graph_metadata (session_id, last_sync_at, status, rag_document_count, edge_count)
        VALUES ($1, NOW(), 'synced', NULL, NULL)
        ON CONFLICT (session_id)
        DO UPDATE SET last_sync_at = EXCLUDED.last_sync_at, status = EXCLUDED.status
    `
    _, err := g.db.ExecContext(ctx, query, sessionID)
    return err
}

