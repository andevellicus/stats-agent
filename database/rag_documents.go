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
	"github.com/pgvector/pgvector-go"
)

// RAGDocument represents a document in the rag_documents table (content and metadata only).
type RAGDocument struct {
	ID          uuid.UUID
	Content     string
	Metadata    map[string]string
	ContentHash string
	CreatedAt   time.Time
}

// RAGEmbedding represents an embedding window in the rag_embeddings table.
type RAGEmbedding struct {
	ID          uuid.UUID
	DocumentID  uuid.UUID
	WindowIndex int
	WindowStart int
	WindowEnd   int
	WindowText  string
	Embedding   []float32
	CreatedAt   time.Time
}

// StoredRAGDocument is the legacy combined struct (for backwards compatibility).
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

// UpsertDocument stores or updates a RAG document's content and metadata (without embeddings).
// Returns the document ID (either existing or newly created).
func (s *PostgresStore) UpsertDocument(ctx context.Context, documentID uuid.UUID, content string, metadata map[string]string, contentHash string) (uuid.UUID, error) {
	metaJSON, err := json.Marshal(metadata)
	if err != nil {
		return uuid.Nil, fmt.Errorf("failed to marshal metadata for rag document: %w", err)
	}

	hashValue := sql.NullString{String: contentHash, Valid: contentHash != ""}

	query := `
		INSERT INTO rag_documents (id, content, metadata, content_hash, created_at)
		VALUES ($1, $2, $3, $4, NOW())
		ON CONFLICT (id)
		DO UPDATE SET content = EXCLUDED.content, metadata = EXCLUDED.metadata, content_hash = EXCLUDED.content_hash, created_at = NOW()
		RETURNING id
	`

	var returnedID uuid.UUID
	if err := s.DB.QueryRowContext(ctx, query, documentID, content, string(metaJSON), hashValue).Scan(&returnedID); err != nil {
		return uuid.Nil, fmt.Errorf("failed to upsert rag document: %w", err)
	}
	return returnedID, nil
}

// CreateEmbedding stores a single embedding window for a document.
func (s *PostgresStore) CreateEmbedding(ctx context.Context, documentID uuid.UUID, windowIndex, windowStart, windowEnd int, windowText string, embedding []float32) error {
	if len(embedding) == 0 {
		return fmt.Errorf("cannot create embedding with empty vector")
	}

	embeddingVector := pgvector.NewVector(embedding)

	query := `
		INSERT INTO rag_embeddings (id, document_id, window_index, window_start, window_end, window_text, embedding, created_at)
		VALUES ($1, $2, $3, $4, $5, $6, $7, NOW())
		ON CONFLICT (document_id, window_index)
		DO UPDATE SET window_start = EXCLUDED.window_start, window_end = EXCLUDED.window_end, window_text = EXCLUDED.window_text, embedding = EXCLUDED.embedding, created_at = NOW()
	`

	embeddingID := uuid.New()
	if _, err := s.DB.ExecContext(ctx, query, embeddingID, documentID, windowIndex, windowStart, windowEnd, windowText, embeddingVector); err != nil {
		return fmt.Errorf("failed to create embedding for document %s window %d: %w", documentID, windowIndex, err)
	}
	return nil
}

// GetDocumentEmbeddings retrieves all embedding windows for a specific document.
func (s *PostgresStore) GetDocumentEmbeddings(ctx context.Context, documentID uuid.UUID) ([]RAGEmbedding, error) {
	query := `
		SELECT id, document_id, window_index, window_start, window_end, window_text, embedding, created_at
		FROM rag_embeddings
		WHERE document_id = $1
		ORDER BY window_index ASC
	`

	rows, err := s.DB.QueryContext(ctx, query, documentID)
	if err != nil {
		return nil, fmt.Errorf("failed to query embeddings for document %s: %w", documentID, err)
	}
	defer rows.Close()

	var embeddings []RAGEmbedding
	for rows.Next() {
		var (
			id          uuid.UUID
			docID       uuid.UUID
			windowIndex int
			windowStart int
			windowEnd   int
			windowText  string
			embedding   pgvector.Vector
			createdAt   time.Time
		)

		if err := rows.Scan(&id, &docID, &windowIndex, &windowStart, &windowEnd, &windowText, &embedding, &createdAt); err != nil {
			return nil, fmt.Errorf("failed to scan embedding row: %w", err)
		}

		var embeddingCopy []float32
		if embeddingSlice := embedding.Slice(); len(embeddingSlice) > 0 {
			embeddingCopy = make([]float32, len(embeddingSlice))
			copy(embeddingCopy, embeddingSlice)
		}

		embeddings = append(embeddings, RAGEmbedding{
			ID:          id,
			DocumentID:  docID,
			WindowIndex: windowIndex,
			WindowStart: windowStart,
			WindowEnd:   windowEnd,
			WindowText:  windowText,
			Embedding:   embeddingCopy,
			CreatedAt:   createdAt,
		})
	}

	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("error iterating embedding rows: %w", err)
	}

	return embeddings, nil
}

// UpsertRAGDocument is the legacy function (deprecated, kept for backwards compatibility during migration).
func (s *PostgresStore) UpsertRAGDocument(ctx context.Context, documentID uuid.UUID, content string, embeddingContent string, metadata map[string]string, contentHash string, embedding []float32) error {
	// Use new schema: store document first
	if _, err := s.UpsertDocument(ctx, documentID, content, metadata, contentHash); err != nil {
		return err
	}

	// Store single embedding window if provided
	if len(embedding) > 0 {
		windowText := embeddingContent
		if windowText == "" {
			windowText = content
		}
		if err := s.CreateEmbedding(ctx, documentID, 0, 0, len(windowText), windowText, embedding); err != nil {
			return err
		}
	}

	return nil
}

// FindStateDocument returns the most recent state document for a (sessionID, dataset, stage).
func (s *PostgresStore) FindStateDocument(ctx context.Context, sessionID, dataset, stage string) (uuid.UUID, string, map[string]string, error) {
	if sessionID == "" || dataset == "" || stage == "" {
		return uuid.Nil, "", nil, sql.ErrNoRows
	}
	const query = `
        SELECT id, content, metadata
        FROM rag_documents
        WHERE (metadata ->> 'session_id') = $1
          AND (metadata ->> 'type') = 'state'
          AND (metadata ->> 'dataset') = $2
          AND (metadata ->> 'stage') = $3
        ORDER BY created_at DESC
        LIMIT 1`

	var (
		id       uuid.UUID
		content  string
		metaJSON []byte
	)
	err := s.DB.QueryRowContext(ctx, query, sessionID, dataset, stage).Scan(&id, &content, &metaJSON)
	if err != nil {
		return uuid.Nil, "", nil, err
	}
	meta := make(map[string]string)
	if len(metaJSON) > 0 {
		if err := json.Unmarshal(metaJSON, &meta); err != nil {
			return uuid.Nil, "", nil, err
		}
	}
	return id, content, meta, nil
}

// FindStateDocumentWithFilters returns the most recent state document for a (sessionID, dataset, stage, filters_key).
func (s *PostgresStore) FindStateDocumentWithFilters(ctx context.Context, sessionID, dataset, stage, filtersKey string) (uuid.UUID, string, map[string]string, error) {
	if sessionID == "" || dataset == "" || stage == "" {
		return uuid.Nil, "", nil, sql.ErrNoRows
	}
	const query = `
        SELECT id, content, metadata
        FROM rag_documents
        WHERE (metadata ->> 'session_id') = $1
          AND (metadata ->> 'type') = 'state'
          AND (metadata ->> 'dataset') = $2
          AND (metadata ->> 'stage') = $3
          AND COALESCE((metadata ->> 'filters_key'), '') = $4
        ORDER BY created_at DESC
        LIMIT 1`

	var (
		id       uuid.UUID
		content  string
		metaJSON []byte
	)
	err := s.DB.QueryRowContext(ctx, query, sessionID, dataset, stage, strings.TrimSpace(filtersKey)).Scan(&id, &content, &metaJSON)
	if err != nil {
		return uuid.Nil, "", nil, err
	}
	meta := make(map[string]string)
	if len(metaJSON) > 0 {
		if err := json.Unmarshal(metaJSON, &meta); err != nil {
			return uuid.Nil, "", nil, err
		}
	}
	return id, content, meta, nil
}

// ListStateDocuments lists all state documents for a session ordered by newest first.
func (s *PostgresStore) ListStateDocuments(ctx context.Context, sessionID string) ([]RAGDocument, error) {
	const query = `
        SELECT id, content, metadata, content_hash, created_at
        FROM rag_documents
        WHERE (metadata ->> 'session_id') = $1 AND (metadata ->> 'type') = 'state'
        ORDER BY created_at DESC`

	rows, err := s.DB.QueryContext(ctx, query, sessionID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var docs []RAGDocument
	for rows.Next() {
		var (
			id        uuid.UUID
			content   string
			metaJSON  []byte
			hash      sql.NullString
			createdAt time.Time
		)
		if err := rows.Scan(&id, &content, &metaJSON, &hash, &createdAt); err != nil {
			return nil, err
		}
		meta := make(map[string]string)
		if len(metaJSON) > 0 {
			if err := json.Unmarshal(metaJSON, &meta); err != nil {
				return nil, err
			}
		}
		docs = append(docs, RAGDocument{ID: id, Content: content, Metadata: meta, ContentHash: hash.String, CreatedAt: createdAt})
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}
	return docs, nil
}

// DeleteRAGDocument deletes a rag document by id (cascades delete to embeddings via FK).
func (s *PostgresStore) DeleteRAGDocument(ctx context.Context, id uuid.UUID) error {
	_, err := s.DB.ExecContext(ctx, `DELETE FROM rag_documents WHERE id = $1`, id)
	return err
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
			embedding        pgvector.Vector
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
		if embeddingSlice := embedding.Slice(); len(embeddingSlice) > 0 {
			embeddingCopy = make([]float32, len(embeddingSlice))
			copy(embeddingCopy, embeddingSlice)
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

// HasSessionPDFEmbeddings returns true if there is at least one embedding row
// for documents of type 'pdf' associated with the given session.
func (s *PostgresStore) HasSessionPDFEmbeddings(ctx context.Context, sessionID uuid.UUID) (bool, error) {
	// Consider embeddings ready if we have any embeddings for PDF-derived documents:
	// - type = 'pdf' (single-page windows)
	// - type = 'document_chunk' with a filename (chunked PDF pages)
	// - type = 'pdf_summary' (key facts summary)
	const query = `
        SELECT EXISTS (
            SELECT 1
            FROM rag_embeddings e
            JOIN rag_documents d ON d.id = e.document_id
            WHERE (d.metadata ->> 'session_id') = $1
              AND (
                    (d.metadata ->> 'type') IN ('pdf', 'pdf_summary')
                 OR ((d.metadata ->> 'type') = 'document_chunk' AND (d.metadata ->> 'filename') IS NOT NULL AND (d.metadata ->> 'filename') <> '')
              )
        )
    `
	var exists bool
	if err := s.DB.QueryRowContext(ctx, query, sessionID.String()).Scan(&exists); err != nil {
		return false, fmt.Errorf("failed to check session pdf embeddings: %w", err)
	}
	return exists, nil
}

// GetRAGDocumentContent returns the stored content for a given document ID.
func (s *PostgresStore) GetRAGDocumentContent(ctx context.Context, documentID uuid.UUID) (string, error) {
	const query = `SELECT content FROM rag_documents WHERE id = $1`

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

// GetDocumentsBatch returns contents for multiple document IDs using a single query.
// Returns a map of id.String() -> content.
func (s *PostgresStore) GetDocumentsBatch(ctx context.Context, ids []uuid.UUID) (map[string]string, error) {
	result := make(map[string]string)
	if len(ids) == 0 {
		return result, nil
	}

	var b strings.Builder
	b.WriteString("SELECT id, content FROM rag_documents WHERE id IN (")
	args := make([]any, 0, len(ids))
	for i, id := range ids {
		if i > 0 {
			b.WriteString(", ")
		}
		b.WriteString("$")
		b.WriteString(strconv.Itoa(i + 1))
		args = append(args, id)
	}
	b.WriteString(")")

	rows, err := s.DB.QueryContext(ctx, b.String(), args...)
	if err != nil {
		return nil, fmt.Errorf("failed to batch fetch documents: %w", err)
	}
	defer rows.Close()

	for rows.Next() {
		var id uuid.UUID
		var content string
		if err := rows.Scan(&id, &content); err != nil {
			return nil, fmt.Errorf("failed to scan batch document row: %w", err)
		}
		result[id.String()] = content
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("error iterating batch document rows: %w", err)
	}
	return result, nil
}

// GetDocument retrieves a full document record by ID.
func (s *PostgresStore) GetDocument(ctx context.Context, documentID uuid.UUID) (RAGDocument, error) {
	query := `SELECT id, content, metadata, content_hash, created_at FROM rag_documents WHERE id = $1`

	var (
		id           uuid.UUID
		content      string
		metadataJSON []byte
		contentHash  sql.NullString
		createdAt    time.Time
	)

	err := s.DB.QueryRowContext(ctx, query, documentID).Scan(&id, &content, &metadataJSON, &contentHash, &createdAt)
	if err != nil {
		if err == sql.ErrNoRows {
			return RAGDocument{}, sql.ErrNoRows
		}
		return RAGDocument{}, fmt.Errorf("failed to fetch document: %w", err)
	}

	metadata := make(map[string]string)
	if len(metadataJSON) > 0 {
		if err := json.Unmarshal(metadataJSON, &metadata); err != nil {
			return RAGDocument{}, fmt.Errorf("failed to unmarshal document metadata: %w", err)
		}
	}

	return RAGDocument{
		ID:          id,
		Content:     content,
		Metadata:    metadata,
		ContentHash: contentHash.String,
		CreatedAt:   createdAt,
	}, nil
}

// FindRAGDocumentByHash looks for an existing RAG document using session, role, and hashed content.
// Returns uuid.Nil when no matching record exists or hash is empty.
func (s *PostgresStore) FindRAGDocumentByHash(ctx context.Context, sessionID, role, contentHash string) (uuid.UUID, error) {
	if contentHash == "" {
		return uuid.Nil, nil
	}

	var builder strings.Builder
	builder.WriteString("SELECT id FROM rag_documents WHERE content_hash = $1")
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

// FindDocumentIDsByContentHash returns document IDs for given content hashes.
// This is used for post-query pruning to avoid retrieving messages already in history.
// Returns map: content_hash → document_id
func (s *PostgresStore) FindDocumentIDsByContentHash(ctx context.Context, sessionID string, contentHashes []string) (map[string]string, error) {
	if len(contentHashes) == 0 {
		return make(map[string]string), nil
	}

	// Build placeholders for IN clause
	placeholders := make([]string, len(contentHashes))
	args := make([]interface{}, 0, len(contentHashes)+1)
	args = append(args, sessionID)

	for i, hash := range contentHashes {
		placeholders[i] = fmt.Sprintf("$%d", i+2)
		args = append(args, hash)
	}

	query := fmt.Sprintf(`
		SELECT content_hash, id::text as document_id
		FROM rag_documents
		WHERE metadata->>'session_id' = $1
		  AND content_hash IN (%s)
		  AND content_hash IS NOT NULL
	`, strings.Join(placeholders, ","))

	rows, err := s.DB.QueryContext(ctx, query, args...)
	if err != nil {
		return nil, fmt.Errorf("failed to query document IDs by content hash: %w", err)
	}
	defer rows.Close()

	result := make(map[string]string)
	for rows.Next() {
		var hash, docID string
		if err := rows.Scan(&hash, &docID); err != nil {
			return nil, fmt.Errorf("failed to scan document ID: %w", err)
		}
		result[hash] = docID
	}

	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("error iterating document ID results: %w", err)
	}

	return result, nil
}

// SearchRAGDocumentsBM25 performs a BM25-style full-text search over the stored RAG documents.
// It returns ranked results ordered by their textual relevance to the provided query.
func (s *PostgresStore) SearchRAGDocumentsBM25(ctx context.Context, query string, limit int, sessionID string, excludeHashes []string) ([]BM25SearchResult, error) {
	trimmed := strings.TrimSpace(query)
	if trimmed == "" || limit <= 0 {
		return nil, nil
	}

	// Try rich websearch_to_tsquery first, then fallback to simpler plainto_tsquery on error
	results, err := s.searchBM25With(ctx, trimmed, limit, sessionID, excludeHashes, "websearch_to_tsquery")
	if err == nil {
		return results, nil
	}
	// Fallback attempt
	fallback, fbErr := s.searchBM25With(ctx, trimmed, limit, sessionID, excludeHashes, "plainto_tsquery")
	if fbErr == nil {
		return fallback, nil
	}
	return nil, fmt.Errorf("failed BM25 search: %v; fallback failed: %w", err, fbErr)
}

// searchBM25With builds and executes a BM25-like query using the provided tsquery function name
// (e.g., "websearch_to_tsquery" or "plainto_tsquery").
func (s *PostgresStore) searchBM25With(ctx context.Context, trimmed string, limit int, sessionID string, excludeHashes []string, tsFunc string) ([]BM25SearchResult, error) {
	const searchableTextExpr = "rd.content || ' ' || COALESCE(meta.metadata_text, '')"
	rankExpr := "ts_rank_cd(to_tsvector('english', " + searchableTextExpr + "), " + tsFunc + "('english', $1))"
	positionExpr := "position(lower($1) in lower(" + searchableTextExpr + "))"
	bonusExpr := "CASE WHEN " + positionExpr + " > 0 THEN 0.2 ELSE 0 END"

	var builder strings.Builder
	args := []any{trimmed}

	builder.WriteString("SELECT rd.id, rd.metadata, rd.content, ")
	builder.WriteString(rankExpr)
	builder.WriteString(" AS rank, ")
	builder.WriteString(bonusExpr)
	builder.WriteString(" AS exact_bonus FROM rag_documents rd")
	builder.WriteString(" LEFT JOIN LATERAL (SELECT string_agg(replace(j.key, '_', ' ') || ' ' || j.value || ' ' || replace(j.value, '_', ' '), ' ') AS metadata_text FROM jsonb_each_text(rd.metadata) AS j(key, value)) AS meta ON TRUE")

	if sessionID != "" {
		builder.WriteString(" WHERE COALESCE(rd.metadata ->> 'session_id', '') = $")
		builder.WriteString(strconv.Itoa(len(args) + 1))
		args = append(args, sessionID)
		builder.WriteString(" AND (" + rankExpr + " > 0 OR " + positionExpr + " > 0)")
	} else {
		builder.WriteString(" WHERE " + rankExpr + " > 0 OR " + positionExpr + " > 0")
	}

	// Exclude superseded state cards while preserving all other document types
	builder.WriteString(" AND (COALESCE(rd.metadata ->> 'type', '') <> 'state' OR COALESCE(rd.metadata ->> 'state_status', '') <> 'superseded')")

	// Exclude documents with matching content hashes
	if len(excludeHashes) > 0 {
		builder.WriteString(" AND (rd.content_hash IS NULL OR rd.content_hash NOT IN (")
		for i, hash := range excludeHashes {
			if i > 0 {
				builder.WriteString(", ")
			}
			builder.WriteString("$")
			builder.WriteString(strconv.Itoa(len(args) + 1))
			args = append(args, hash)
		}
		builder.WriteString("))")
	}

	builder.WriteString(" ORDER BY (" + rankExpr + " + " + bonusExpr + ") DESC LIMIT $")
	builder.WriteString(strconv.Itoa(len(args) + 1))
	args = append(args, limit)

	rows, err := s.DB.QueryContext(ctx, builder.String(), args...)
	if err != nil {
		return nil, fmt.Errorf("failed to execute BM25 search (%s): %w", tsFunc, err)
	}
	defer rows.Close()

	var results []BM25SearchResult
	for rows.Next() {
		var (
			documentID   uuid.UUID
			metadataJSON []byte
			content      string
			rank         float64
			exactBonus   float64
		)

		if err := rows.Scan(&documentID, &metadataJSON, &content, &rank, &exactBonus); err != nil {
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
			EmbeddingContent: "",
			BM25Score:        rank,
			ExactMatchBonus:  exactBonus,
		})
	}

	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("error iterating BM25 search results: %w", err)
	}

	return results, nil
}

// VectorSearchResult represents a vector similarity search hit from pgvector.
type VectorSearchResult struct {
	DocumentID       uuid.UUID
	Metadata         map[string]string
	Content          string
	EmbeddingContent string
	Similarity       float64
	WindowIndex      int // Which embedding window matched (for multi-vector documents)
	WindowStart      int // Character offset where window starts in full document
	WindowEnd        int // Character offset where window ends in full document
}

// VectorSearchRAGDocuments performs a cosine similarity search using pgvector.
// Returns documents ordered by similarity (highest first), joining embeddings with documents.
func (s *PostgresStore) VectorSearchRAGDocuments(ctx context.Context, queryVector []float32, limit int, sessionID string, excludeHashes []string) ([]VectorSearchResult, error) {
	if len(queryVector) == 0 || limit <= 0 {
		return nil, nil
	}

	// Convert []float32 to pgvector.Vector
	vec := pgvector.NewVector(queryVector)

	var builder strings.Builder
	args := []any{vec}

	builder.WriteString("SELECT rd.id, rd.metadata, rd.content, re.window_text, re.window_index, re.window_start, re.window_end, 1 - (re.embedding <=> $1) AS similarity ")
	builder.WriteString("FROM rag_embeddings re ")
	builder.WriteString("INNER JOIN rag_documents rd ON re.document_id = rd.id ")
	builder.WriteString("WHERE re.embedding IS NOT NULL ")

	// Apply session-specific filtering when provided
	if sessionID != "" {
		builder.WriteString("AND COALESCE(rd.metadata ->> 'session_id', '') = $")
		builder.WriteString(strconv.Itoa(len(args) + 1))
		args = append(args, sessionID)
		builder.WriteString(" ")
	}

	// Exclude superseded state cards while preserving other types
	builder.WriteString("AND (COALESCE(rd.metadata ->> 'type', '') <> 'state' OR COALESCE(rd.metadata ->> 'state_status', '') <> 'superseded') ")

	// Exclude documents with matching content hashes
	if len(excludeHashes) > 0 {
		builder.WriteString("AND (rd.content_hash IS NULL OR rd.content_hash NOT IN (")
		for i, hash := range excludeHashes {
			if i > 0 {
				builder.WriteString(", ")
			}
			builder.WriteString("$")
			builder.WriteString(strconv.Itoa(len(args) + 1))
			args = append(args, hash)
		}
		builder.WriteString(")) ")
	}

	builder.WriteString("ORDER BY re.embedding <=> $1 LIMIT $")
	builder.WriteString(strconv.Itoa(len(args) + 1))
	args = append(args, limit)

	rows, err := s.DB.QueryContext(ctx, builder.String(), args...)
	if err != nil {
		return nil, fmt.Errorf("failed to execute vector search: %w", err)
	}
	defer rows.Close()

	var results []VectorSearchResult
	for rows.Next() {
		var (
			documentID       uuid.UUID
			metadataJSON     []byte
			content          string
			embeddingContent string
			windowIndex      int
			windowStart      int
			windowEnd        int
			similarity       float64
		)

		if err := rows.Scan(&documentID, &metadataJSON, &content, &embeddingContent, &windowIndex, &windowStart, &windowEnd, &similarity); err != nil {
			return nil, fmt.Errorf("failed to scan vector search result: %w", err)
		}

		metadata := make(map[string]string)
		if len(metadataJSON) > 0 {
			if err := json.Unmarshal(metadataJSON, &metadata); err != nil {
				return nil, fmt.Errorf("failed to unmarshal vector search metadata: %w", err)
			}
		}

		results = append(results, VectorSearchResult{
			DocumentID:       documentID,
			Metadata:         metadata,
			Content:          content,
			EmbeddingContent: embeddingContent,
			Similarity:       similarity,
			WindowIndex:      windowIndex,
			WindowStart:      windowStart,
			WindowEnd:        windowEnd,
		})
	}

	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("error iterating vector search results: %w", err)
	}

	return results, nil
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
