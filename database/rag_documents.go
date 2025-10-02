package database

import (
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/google/uuid"
	"github.com/lib/pq"
)

type StoredRAGDocument struct {
	ID               uuid.UUID
	DocumentID       uuid.UUID
	Content          string
	EmbeddingContent string
	Metadata         map[string]string
	ContentHash      string
	Embedding        []float32
	CreatedAt        time.Time
}

// UpsertRAGDocument stores or updates a RAG document's content and metadata.
func (s *PostgresStore) UpsertRAGDocument(ctx context.Context, documentID uuid.UUID, content string, embeddingContent string, metadata map[string]string, contentHash string, embedding []float32) error {
	metaJSON, err := json.Marshal(metadata)
	if err != nil {
		return fmt.Errorf("failed to marshal metadata for rag document: %w", err)
	}

	hashValue := sql.NullString{String: contentHash, Valid: contentHash != ""}
	var embeddingValue interface{}
	if len(embedding) > 0 {
		embeddingValue = pq.Float32Array(embedding)
	} else {
		embeddingValue = nil
	}

	query := `
	        INSERT INTO rag_documents (id, document_id, content, embedding_content, metadata, content_hash, embedding, created_at)
	        VALUES ($1, $2, $3, $4, $5, $6, $7, NOW())
	        ON CONFLICT (document_id)
	        DO UPDATE SET content = EXCLUDED.content, embedding_content = EXCLUDED.embedding_content, metadata = EXCLUDED.metadata, content_hash = EXCLUDED.content_hash, embedding = EXCLUDED.embedding, created_at = NOW()
	    `

	if _, err := s.DB.ExecContext(ctx, query, documentID, documentID, content, embeddingContent, string(metaJSON), hashValue, embeddingValue); err != nil {
		return fmt.Errorf("failed to upsert rag document: %w", err)
	}
	return nil
}

// ListRAGDocuments returns all persisted RAG documents including their embeddings.
func (s *PostgresStore) ListRAGDocuments(ctx context.Context) ([]StoredRAGDocument, error) {
	const query = `
		SELECT id, document_id, content, embedding_content, metadata, content_hash, embedding, created_at
		FROM rag_documents
		ORDER BY created_at ASC
	`

	rows, err := s.DB.QueryContext(ctx, query)
	if err != nil {
		return nil, fmt.Errorf("failed to list rag documents: %w", err)
	}
	defer rows.Close()

	var documents []StoredRAGDocument
	for rows.Next() {
		var (
			id               uuid.UUID
			documentID       uuid.UUID
			content          string
			embeddingContent sql.NullString
			metadataJSON     []byte
			contentHash      sql.NullString
			embedding        pq.Float32Array
			createdAt        time.Time
		)

		if err := rows.Scan(&id, &documentID, &content, &embeddingContent, &metadataJSON, &contentHash, &embedding, &createdAt); err != nil {
			return nil, fmt.Errorf("failed to scan rag document: %w", err)
		}

		metadata := make(map[string]string)
		if len(metadataJSON) > 0 {
			if err := json.Unmarshal(metadataJSON, &metadata); err != nil {
				return nil, fmt.Errorf("failed to unmarshal rag document metadata: %w", err)
			}
		}

		var embeddingCopy []float32
		if len(embedding) > 0 {
			embeddingCopy = make([]float32, len(embedding))
			copy(embeddingCopy, embedding)
		}

		documents = append(documents, StoredRAGDocument{
			ID:               id,
			DocumentID:       documentID,
			Content:          content,
			EmbeddingContent: embeddingContent.String,
			Metadata:         metadata,
			ContentHash:      contentHash.String,
			Embedding:        embeddingCopy,
			CreatedAt:        createdAt,
		})
	}

	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("error iterating rag documents: %w", err)
	}

	return documents, nil
}

// GetRAGDocumentContent returns the stored content for a given document ID.
func (s *PostgresStore) GetRAGDocumentContent(ctx context.Context, documentID uuid.UUID) (string, error) {
	const query = `SELECT content FROM rag_documents WHERE document_id = $1`

	var content string
	err := s.DB.QueryRowContext(ctx, query, documentID).Scan(&content)
	if err != nil {
		if err == sql.ErrNoRows {
			return "", sql.ErrNoRows
		}
		return "", fmt.Errorf("failed to fetch rag document content: %w", err)
	}
	return content, nil
}

// FindRAGDocumentByHash looks for an existing RAG document using session, role, and hashed content.
// Returns uuid.Nil when no matching record exists or hash is empty.
func (s *PostgresStore) FindRAGDocumentByHash(ctx context.Context, sessionID, role, contentHash string) (uuid.UUID, error) {
	if contentHash == "" {
		return uuid.Nil, nil
	}

	var builder strings.Builder
	builder.WriteString("SELECT document_id FROM rag_documents WHERE content_hash = $1")
	args := []any{contentHash}
	paramIndex := 2

	if sessionID != "" {
		builder.WriteString(fmt.Sprintf(" AND COALESCE(metadata ->> 'session_id', '') = $%d", paramIndex))
		args = append(args, sessionID)
		paramIndex++
	}

	if role != "" {
		builder.WriteString(fmt.Sprintf(" AND COALESCE(metadata ->> 'role', '') = $%d", paramIndex))
		args = append(args, role)
		paramIndex++
	}

	builder.WriteString(" LIMIT 1")

	var documentID uuid.UUID
	row := s.DB.QueryRowContext(ctx, builder.String(), args...)
	if err := row.Scan(&documentID); err != nil {
		if errors.Is(err, sql.ErrNoRows) {
			return uuid.Nil, nil
		}
		return uuid.Nil, fmt.Errorf("failed to lookup rag document by hash: %w", err)
	}

	return documentID, nil
}

// DeleteRAGDocumentsBySession removes all RAG documents associated with the provided session.
func (s *PostgresStore) DeleteRAGDocumentsBySession(ctx context.Context, sessionID uuid.UUID) (int64, error) {
	const query = `DELETE FROM rag_documents WHERE metadata ->> 'session_id' = $1`

	result, err := s.DB.ExecContext(ctx, query, sessionID.String())
	if err != nil {
		return 0, fmt.Errorf("failed to delete rag documents for session %s: %w", sessionID, err)
	}

	rowsAffected, err := result.RowsAffected()
	if err != nil {
		return 0, fmt.Errorf("failed to determine rows deleted for session %s: %w", sessionID, err)
	}

	return rowsAffected, nil
}
