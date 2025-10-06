package database

import (
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"strconv"
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

// BM25SearchResult represents a full-text search hit scored via PostgreSQL's ranking engine.
type BM25SearchResult struct {
	DocumentID       uuid.UUID
	Metadata         map[string]string
	Content          string
	EmbeddingContent string
	BM25Score        float64
	ExactMatchBonus  float64
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

	rowID := uuid.New()
	if _, err := s.DB.ExecContext(ctx, query, rowID, documentID, content, embeddingContent, string(metaJSON), hashValue, embeddingValue); err != nil {
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

// SearchRAGDocumentsBM25 performs a BM25-style full-text search over the stored RAG documents.
// It returns ranked results ordered by their textual relevance to the provided query.
func (s *PostgresStore) SearchRAGDocumentsBM25(ctx context.Context, query string, limit int, sessionID string) ([]BM25SearchResult, error) {
	trimmed := strings.TrimSpace(query)
	if trimmed == "" || limit <= 0 {
		return nil, nil
	}

	const searchableTextExpr = "COALESCE(rd.embedding_content, rd.content) || ' ' || COALESCE(meta.metadata_text, '')"
	const rankExpr = "ts_rank_cd(to_tsvector('english', " + searchableTextExpr + "), websearch_to_tsquery('english', $1))"
	const positionExpr = "position(lower($1) in lower(" + searchableTextExpr + "))"
	const bonusExpr = "CASE WHEN " + positionExpr + " > 0 THEN 0.2 ELSE 0 END"

	var builder strings.Builder
	args := []any{trimmed}

	builder.WriteString("SELECT rd.document_id, rd.metadata, rd.content, rd.embedding_content, ")
	builder.WriteString(rankExpr)
	builder.WriteString(" AS rank, ")
	builder.WriteString(bonusExpr)
	builder.WriteString(" AS exact_bonus FROM rag_documents rd")
	builder.WriteString(" LEFT JOIN LATERAL (SELECT string_agg(replace(j.key, '_', ' ') || ' ' || j.value || ' ' || replace(j.value, '_', ' '), ' ') AS metadata_text FROM jsonb_each_text(rd.metadata) AS j(key, value)) AS meta ON TRUE")

	// Apply session-specific filtering when provided.
	if sessionID != "" {
		builder.WriteString(" WHERE COALESCE(rd.metadata ->> 'session_id', '') = $")
		builder.WriteString(strconv.Itoa(len(args) + 1))
		args = append(args, sessionID)
		builder.WriteString(" AND (" + rankExpr + " > 0 OR " + positionExpr + " > 0)")
	} else {
		builder.WriteString(" WHERE " + rankExpr + " > 0 OR " + positionExpr + " > 0")
	}

	builder.WriteString(" ORDER BY (" + rankExpr + " + " + bonusExpr + ") DESC LIMIT $")
	builder.WriteString(strconv.Itoa(len(args) + 1))
	args = append(args, limit)

	rows, err := s.DB.QueryContext(ctx, builder.String(), args...)
	if err != nil {
		return nil, fmt.Errorf("failed to execute BM25 search: %w", err)
	}
	defer rows.Close()

	var results []BM25SearchResult
	for rows.Next() {
		var (
			documentID       uuid.UUID
			metadataJSON     []byte
			content          string
			embeddingContent sql.NullString
			rank             float64
			exactBonus       float64
		)

		if err := rows.Scan(&documentID, &metadataJSON, &content, &embeddingContent, &rank, &exactBonus); err != nil {
			return nil, fmt.Errorf("failed to scan BM25 search result: %w", err)
		}

		metadata := make(map[string]string)
		if len(metadataJSON) > 0 {
			if err := json.Unmarshal(metadataJSON, &metadata); err != nil {
				return nil, fmt.Errorf("failed to unmarshal BM25 metadata: %w", err)
			}
		}

		results = append(results, BM25SearchResult{
			DocumentID:       documentID,
			Metadata:         metadata,
			Content:          content,
			EmbeddingContent: embeddingContent.String,
			BM25Score:        rank,
			ExactMatchBonus:  exactBonus,
		})
	}

	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("error iterating BM25 search results: %w", err)
	}

	return results, nil
}

// BatchUpsertRAGDocuments efficiently inserts multiple RAG documents in a single transaction.
// This is significantly faster than individual UpsertRAGDocument calls when persisting many documents.
func (s *PostgresStore) BatchUpsertRAGDocuments(ctx context.Context, docs []struct {
	DocumentID       uuid.UUID
	Content          string
	EmbeddingContent string
	Metadata         map[string]string
	ContentHash      string
	Embedding        []float32
}) error {
	if len(docs) == 0 {
		return nil
	}

	tx, err := s.DB.BeginTx(ctx, nil)
	if err != nil {
		return fmt.Errorf("failed to begin transaction for batch upsert: %w", err)
	}
	defer tx.Rollback()

	// Prepare arrays for UNNEST-based batch insert
	ids := make([]uuid.UUID, len(docs))
	documentIDs := make([]uuid.UUID, len(docs))
	contents := make([]string, len(docs))
	embeddingContents := make([]sql.NullString, len(docs))
	metadatas := make([]string, len(docs))
	contentHashes := make([]sql.NullString, len(docs))
	embeddings := make([]interface{}, len(docs))

	for i, doc := range docs {
		ids[i] = uuid.New()
		documentIDs[i] = doc.DocumentID
		contents[i] = doc.Content

		if doc.EmbeddingContent != "" {
			embeddingContents[i] = sql.NullString{String: doc.EmbeddingContent, Valid: true}
		}

		metaJSON, err := json.Marshal(doc.Metadata)
		if err != nil {
			return fmt.Errorf("failed to marshal metadata for document %s: %w", doc.DocumentID, err)
		}
		metadatas[i] = string(metaJSON)

		if doc.ContentHash != "" {
			contentHashes[i] = sql.NullString{String: doc.ContentHash, Valid: true}
		}

		if len(doc.Embedding) > 0 {
			embeddings[i] = pq.Float32Array(doc.Embedding)
		} else {
			embeddings[i] = nil
		}
	}

	// Build multi-row VALUES clause
	var valueClauses []string
	var allArgs []interface{}
	argIndex := 1

	for i := range docs {
		valueClauses = append(valueClauses, fmt.Sprintf(
			"($%d, $%d, $%d, $%d, $%d, $%d, $%d, NOW())",
			argIndex, argIndex+1, argIndex+2, argIndex+3, argIndex+4, argIndex+5, argIndex+6,
		))
		allArgs = append(allArgs, ids[i], documentIDs[i], contents[i], embeddingContents[i], metadatas[i], contentHashes[i], embeddings[i])
		argIndex += 7
	}

	query := `
		INSERT INTO rag_documents (id, document_id, content, embedding_content, metadata, content_hash, embedding, created_at)
		VALUES ` + strings.Join(valueClauses, ", ") + `
		ON CONFLICT (document_id)
		DO UPDATE SET
			content = EXCLUDED.content,
			embedding_content = EXCLUDED.embedding_content,
			metadata = EXCLUDED.metadata,
			content_hash = EXCLUDED.content_hash,
			embedding = EXCLUDED.embedding,
			created_at = NOW()
	`

	if _, err := tx.ExecContext(ctx, query, allArgs...); err != nil {
		return fmt.Errorf("failed to batch upsert rag documents: %w", err)
	}

	if err := tx.Commit(); err != nil {
		return fmt.Errorf("failed to commit batch upsert transaction: %w", err)
	}

	return nil
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
