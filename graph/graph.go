package graph

import (
	"database/sql"

	"go.uber.org/zap"
)

// Graph provides a lightweight index layer on top of RAG storage.
// It maintains relationship edges and variable aliases for enhanced query capabilities.
// RAG remains the source of truth; graph can be rebuilt from RAG metadata.
type Graph struct {
	db      *sql.DB
	logger  *zap.Logger
	enabled bool
}

// StatEdge represents a relationship between two RAG documents.
// Edge types: supersedes, supports, blocks, compares, emitted_in
type StatEdge struct {
	ID        string
	FromID    string                 // RAG document ID
	ToID      string                 // RAG document ID
	EdgeType  string                 // supersedes, supports, blocks, compares, emitted_in
	Metadata  map[string]interface{} // Edge-specific data
	SessionID string
	Dataset   string // Dataset name for scoping
}

// New creates a new Graph instance.
// If enabled is false, all operations will no-op gracefully.
func New(db *sql.DB, logger *zap.Logger, enabled bool) *Graph {
	return &Graph{
		db:      db,
		logger:  logger,
		enabled: enabled,
	}
}

// Enabled returns whether the graph is enabled.
func (g *Graph) Enabled() bool {
	return g != nil && g.enabled
}
