package graph

import (
    "context"
    "encoding/json"
    "fmt"

	"github.com/google/uuid"
	"go.uber.org/zap"
)

// CreateEdge creates a new edge in the graph.
// This is non-blocking - errors are logged but don't fail the caller.
func (g *Graph) CreateEdge(ctx context.Context, edge StatEdge) error {
	if !g.Enabled() {
		return nil // Graceful no-op when disabled
	}

	// Marshal metadata to JSONB
	metadataJSON, err := json.Marshal(edge.Metadata)
	if err != nil {
		return fmt.Errorf("failed to marshal edge metadata: %w", err)
	}

	query := `
		INSERT INTO stat_edges (from_id, to_id, edge_type, metadata, session_id, dataset)
		VALUES ($1, $2, $3, $4, $5, $6)
	`

	fromUUID, err := uuid.Parse(edge.FromID)
	if err != nil {
		return fmt.Errorf("invalid from_id UUID: %w", err)
	}

	toUUID, err := uuid.Parse(edge.ToID)
	if err != nil {
		return fmt.Errorf("invalid to_id UUID: %w", err)
	}

	sessionUUID, err := uuid.Parse(edge.SessionID)
	if err != nil {
		return fmt.Errorf("invalid session_id UUID: %w", err)
	}

    _, err = g.db.ExecContext(ctx, query, fromUUID, toUUID, edge.EdgeType, metadataJSON, sessionUUID, edge.Dataset)
    if err != nil {
        return fmt.Errorf("failed to insert edge: %w", err)
    }

    g.logger.Debug("Created graph edge",
        zap.String("type", edge.EdgeType),
        zap.String("from", edge.FromID),
        zap.String("to", edge.ToID))

    // Best-effort metadata touch
    _ = g.TouchMetadata(ctx, edge.SessionID)

    return nil
}

// IsSuperseded checks if a document has been superseded by a newer version.
func (g *Graph) IsSuperseded(ctx context.Context, documentID string) (bool, error) {
	if !g.Enabled() {
		return false, nil
	}

	docUUID, err := uuid.Parse(documentID)
	if err != nil {
		return false, fmt.Errorf("invalid document UUID: %w", err)
	}

	query := `
		SELECT EXISTS (
			SELECT 1 FROM stat_edges
			WHERE to_id = $1 AND edge_type = 'supersedes'
		)
	`

	var exists bool
	err = g.db.QueryRowContext(ctx, query, docUUID).Scan(&exists)
	if err != nil {
		return false, fmt.Errorf("failed to check supersession: %w", err)
	}

	return exists, nil
}

// HasIncomingEdgeType checks if there is at least one incoming edge of a given type to the document.
func (g *Graph) HasIncomingEdgeType(ctx context.Context, documentID string, edgeType string) (bool, error) {
    if !g.Enabled() {
        return false, nil
    }
    docUUID, err := uuid.Parse(documentID)
    if err != nil {
        return false, fmt.Errorf("invalid document UUID: %w", err)
    }
    query := `
        SELECT EXISTS (
            SELECT 1 FROM stat_edges
            WHERE to_id = $1 AND edge_type = $2
        )
    `
    var exists bool
    err = g.db.QueryRowContext(ctx, query, docUUID, edgeType).Scan(&exists)
    if err != nil {
        return false, fmt.Errorf("failed to check incoming edge type: %w", err)
    }
    return exists, nil
}

// IsBlocked reports if the given document has an incoming 'blocks' edge.
func (g *Graph) IsBlocked(ctx context.Context, documentID string) (bool, error) {
    return g.HasIncomingEdgeType(ctx, documentID, "blocks")
}

// GetIncomingEdges returns all edges pointing TO the specified document.
func (g *Graph) GetIncomingEdges(ctx context.Context, documentID string, edgeType string) ([]StatEdge, error) {
	if !g.Enabled() {
		return nil, nil
	}

	docUUID, err := uuid.Parse(documentID)
	if err != nil {
		return nil, fmt.Errorf("invalid document UUID: %w", err)
	}

	query := `
		SELECT id, from_id, to_id, edge_type, metadata, session_id, dataset
		FROM stat_edges
		WHERE to_id = $1 AND edge_type = $2
	`

	rows, err := g.db.QueryContext(ctx, query, docUUID, edgeType)
	if err != nil {
		return nil, fmt.Errorf("failed to query incoming edges: %w", err)
	}
	defer rows.Close()

	var edges []StatEdge
	for rows.Next() {
		var edge StatEdge
		var id, fromID, toID, sessionID uuid.UUID
		var metadataJSON []byte

		err := rows.Scan(&id, &fromID, &toID, &edge.EdgeType, &metadataJSON, &sessionID, &edge.Dataset)
		if err != nil {
			return nil, fmt.Errorf("failed to scan edge row: %w", err)
		}

		edge.ID = id.String()
		edge.FromID = fromID.String()
		edge.ToID = toID.String()
		edge.SessionID = sessionID.String()

		if len(metadataJSON) > 0 {
			if err := json.Unmarshal(metadataJSON, &edge.Metadata); err != nil {
				g.logger.Warn("Failed to unmarshal edge metadata", zap.Error(err))
				edge.Metadata = make(map[string]interface{})
			}
		}

		edges = append(edges, edge)
	}

	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("error iterating edge rows: %w", err)
	}

	return edges, nil
}

// GetOutgoingEdges returns all edges FROM the specified document.
func (g *Graph) GetOutgoingEdges(ctx context.Context, documentID string, edgeType string) ([]StatEdge, error) {
	if !g.Enabled() {
		return nil, nil
	}

	docUUID, err := uuid.Parse(documentID)
	if err != nil {
		return nil, fmt.Errorf("invalid document UUID: %w", err)
	}

	query := `
		SELECT id, from_id, to_id, edge_type, metadata, session_id, dataset
		FROM stat_edges
		WHERE from_id = $1 AND edge_type = $2
	`

	rows, err := g.db.QueryContext(ctx, query, docUUID, edgeType)
	if err != nil {
		return nil, fmt.Errorf("failed to query outgoing edges: %w", err)
	}
	defer rows.Close()

	var edges []StatEdge
	for rows.Next() {
		var edge StatEdge
		var id, fromID, toID, sessionID uuid.UUID
		var metadataJSON []byte

		err := rows.Scan(&id, &fromID, &toID, &edge.EdgeType, &metadataJSON, &sessionID, &edge.Dataset)
		if err != nil {
			return nil, fmt.Errorf("failed to scan edge row: %w", err)
		}

		edge.ID = id.String()
		edge.FromID = fromID.String()
		edge.ToID = toID.String()
		edge.SessionID = sessionID.String()

		if len(metadataJSON) > 0 {
			if err := json.Unmarshal(metadataJSON, &edge.Metadata); err != nil {
				g.logger.Warn("Failed to unmarshal edge metadata", zap.Error(err))
				edge.Metadata = make(map[string]interface{})
			}
		}

		edges = append(edges, edge)
	}

	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("error iterating edge rows: %w", err)
	}

	return edges, nil
}

// DeleteEdgesBySession deletes all edges for a given session.
// This is called during session cleanup.
func (g *Graph) DeleteEdgesBySession(ctx context.Context, sessionID string) (int64, error) {
	if !g.Enabled() {
		return 0, nil
	}

	sessionUUID, err := uuid.Parse(sessionID)
	if err != nil {
		return 0, fmt.Errorf("invalid session UUID: %w", err)
	}

	query := `DELETE FROM stat_edges WHERE session_id = $1`
	result, err := g.db.ExecContext(ctx, query, sessionUUID)
	if err != nil {
		return 0, fmt.Errorf("failed to delete edges: %w", err)
	}

	count, err := result.RowsAffected()
	if err != nil {
		return 0, fmt.Errorf("failed to get rows affected: %w", err)
	}

	return count, nil
}
