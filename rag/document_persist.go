package rag

import (
	"context"
	"fmt"
	"time"

	"stats-agent/web/types"

	"github.com/google/uuid"
	"github.com/philippgille/chromem-go"
	"go.uber.org/zap"
)

// dbDocument represents a document to be batch-persisted to PostgreSQL
type dbDocument struct {
	DocumentID       uuid.UUID
	Content          string
	EmbeddingContent string
	Metadata         map[string]string
	ContentHash      string
	Embedding        []float32
}

// AddMessagesToStore is the main entry point for adding messages to RAG storage.
// It orchestrates document preparation, persistence, and embedding into the vector store.
func (r *RAG) AddMessagesToStore(ctx context.Context, sessionID string, messages []types.AgentMessage) error {
	collection := r.db.GetCollection("long-term-memory", r.embedder)
	if collection == nil {
		return fmt.Errorf("long-term memory collection not found")
	}

	processedIndices := make(map[int]bool)
	var documentsToEmbed []chromem.Document
	var dbDocs []dbDocument

	var sessionFilter map[string]string
	if sessionID != "" {
		sessionFilter = map[string]string{"session_id": sessionID}
	}

	// Prepare all messages first
	for i := range messages {
		if processedIndices[i] {
			continue
		}

		docData, skip, err := r.prepareDocumentForMessage(ctx, sessionID, messages, i, collection, sessionFilter, processedIndices)
		if err != nil {
			r.logger.Warn("Failed to prepare RAG document", zap.Error(err))
			continue
		}
		if skip || docData == nil {
			continue
		}

		embeddableDocs, batchDocs := r.preparePersistenceData(ctx, docData)
		documentsToEmbed = append(documentsToEmbed, embeddableDocs...)
		dbDocs = append(dbDocs, batchDocs...)
	}

	if len(documentsToEmbed) == 0 {
		return nil
	}

	// Batch persist all documents to PostgreSQL
	if len(dbDocs) > 0 {
		batchStructs := make([]struct {
			DocumentID       uuid.UUID
			Content          string
			EmbeddingContent string
			Metadata         map[string]string
			ContentHash      string
			Embedding        []float32
		}, len(dbDocs))

		for i, doc := range dbDocs {
			batchStructs[i].DocumentID = doc.DocumentID
			batchStructs[i].Content = doc.Content
			batchStructs[i].EmbeddingContent = doc.EmbeddingContent
			batchStructs[i].Metadata = doc.Metadata
			batchStructs[i].ContentHash = doc.ContentHash
			batchStructs[i].Embedding = doc.Embedding
		}

		if err := r.store.BatchUpsertRAGDocuments(ctx, batchStructs); err != nil {
			r.logger.Warn("Batch persist failed, falling back to individual inserts", zap.Error(err))

			// Create fresh context for fallback with generous timeout
			fallbackCtx, fallbackCancel := context.WithTimeout(context.Background(), 5*time.Minute)
			defer fallbackCancel()

			// Fallback to individual persists with fresh timeout budget
			for _, doc := range dbDocs {
				if err := r.store.UpsertRAGDocument(fallbackCtx, doc.DocumentID, doc.Content, doc.EmbeddingContent, doc.Metadata, doc.ContentHash, doc.Embedding); err != nil {
					r.logger.Warn("Failed to persist RAG document", zap.Error(err), zap.String("document_id", doc.DocumentID.String()))
				}
			}
		}
	}

	// Add all documents to chromem vector store
	if err := collection.AddDocuments(ctx, documentsToEmbed, 4); err != nil {
		return fmt.Errorf("failed to add documents to collection: %w", err)
	}

	r.logger.Info("Added document chunks to long-term RAG memory", zap.Int("chunks_added", len(documentsToEmbed)))
	return nil
}

// preparePersistenceData handles the embedding and data preparation for a document.
// It returns chromem documents for the vector store and database documents for batch persistence.
func (r *RAG) preparePersistenceData(ctx context.Context, data *ragDocumentData) ([]chromem.Document, []dbDocument) {
	if data == nil {
		return nil, nil
	}

	// Embed content directly (no augmentation - metadata is already inline in fact text)
	embedContent := r.ensureEmbeddingTokenLimit(ctx, data.EmbedContent)

	embedStart := time.Now()
	embeddingVector, embedErr := r.embedder(ctx, embedContent)
	embedElapsed := time.Since(embedStart)

	if embedErr != nil {
		r.logger.Error("Failed to create embedding for RAG persistence",
			zap.Error(embedErr),
			zap.String("document_id", data.Metadata["document_id"]),
			zap.Duration("elapsed", embedElapsed),
			zap.Duration("timeout", r.cfg.EmbeddingTimeout))
	} else if embedElapsed > r.cfg.EmbeddingTimeout/2 {
		r.logger.Warn("Embedding generation took longer than expected",
			zap.Duration("elapsed", embedElapsed),
			zap.Duration("timeout", r.cfg.EmbeddingTimeout),
			zap.String("document_id", data.Metadata["document_id"]))
	}

	// Filter metadata to keep only structural fields for JSONB storage
	structuralMetadata := filterStructuralMetadata(data.Metadata)

	// Prepare main document for database batch
	var dbDocs []dbDocument
	dbDocs = append(dbDocs, dbDocument{
		DocumentID:       data.ID,
		Content:          data.StoredContent,
		EmbeddingContent: embedContent,
		Metadata:         cloneStringMap(structuralMetadata),
		ContentHash:      data.ContentHash,
		Embedding:        embeddingVector,
	})

	// Prepare documents for chromem vector store
	var chromemDocs []chromem.Document
	if len(data.EmbedContent) > r.maxEmbeddingChars {
		// Content is too large - split into chunks
		r.logger.Info("Chunking oversized message for embedding",
			zap.String("role", data.Metadata["role"]),
			zap.Int("length", len(data.EmbedContent)))
		chunkChromemDocs, chunkDbDocs := r.prepareChunks(ctx, structuralMetadata, data.EmbedContent)
		chromemDocs = append(chromemDocs, chunkChromemDocs...)
		dbDocs = append(dbDocs, chunkDbDocs...)
	} else {
		// Content fits within limits - create single document
		doc := chromem.Document{
			ID:       uuid.New().String(),
			Content:  embedContent,
			Metadata: cloneStringMap(structuralMetadata),
		}
		if embedErr == nil && len(embeddingVector) > 0 {
			doc.Embedding = embeddingVector
		}
		chromemDocs = append(chromemDocs, doc)
	}

	// Add summary document if present
	if data.SummaryDoc != nil {
		summaryChromem, summaryDb := r.prepareSummaryDocument(ctx, data.SummaryDoc)
		if summaryChromem != nil {
			chromemDocs = append(chromemDocs, *summaryChromem)
		}
		if summaryDb != nil {
			dbDocs = append(dbDocs, *summaryDb)
		}
	}

	return chromemDocs, dbDocs
}

// prepareSummaryDocument prepares a summary document for both PostgreSQL and chromem.
// Returns nil documents if the summary is invalid.
func (r *RAG) prepareSummaryDocument(ctx context.Context, summaryDoc *chromem.Document) (*chromem.Document, *dbDocument) {
	if summaryDoc == nil {
		return nil, nil
	}

	summaryMetadata := cloneStringMap(summaryDoc.Metadata)
	summaryDoc.Metadata = summaryMetadata
	summaryIDStr, ok := summaryMetadata["document_id"]
	if !ok {
		r.logger.Warn("Summary document missing document_id, skipping persistence")
		return nil, nil
	}

	summaryID, err := uuid.Parse(summaryIDStr)
	if err != nil {
		r.logger.Warn("Summary document has invalid document_id", zap.String("document_id", summaryIDStr), zap.Error(err))
		return nil, nil
	}

	summaryContent := summaryDoc.Content
	summaryHash := hashContent(normalizeForHash(summaryContent))
	if summaryHash != "" {
		summaryMetadata["content_hash"] = summaryHash
	}

	// Filter summary metadata to structural fields only
	structuralSummaryMetadata := filterStructuralMetadata(summaryMetadata)

	// Embed content directly (no augmentation)
	summaryEmbeddingContent := r.ensureEmbeddingTokenLimit(ctx, summaryContent)

	embedStart := time.Now()
	summaryEmbedding, summaryErr := r.embedder(ctx, summaryEmbeddingContent)
	embedElapsed := time.Since(embedStart)

	if summaryErr != nil {
		r.logger.Error("Failed to create embedding for summary document",
			zap.Error(summaryErr),
			zap.String("document_id", summaryIDStr),
			zap.Duration("elapsed", embedElapsed),
			zap.Duration("timeout", r.cfg.EmbeddingTimeout))
	} else if embedElapsed > r.cfg.EmbeddingTimeout/2 {
		r.logger.Warn("Summary embedding took longer than expected",
			zap.Duration("elapsed", embedElapsed),
			zap.Duration("timeout", r.cfg.EmbeddingTimeout),
			zap.String("document_id", summaryIDStr))
	}

	// Prepare database document
	dbDoc := &dbDocument{
		DocumentID:       summaryID,
		Content:          summaryContent,
		EmbeddingContent: summaryEmbeddingContent,
		Metadata:         cloneStringMap(structuralSummaryMetadata),
		ContentHash:      summaryHash,
		Embedding:        summaryEmbedding,
	}

	// Update document with embedding for chromem
	if summaryErr == nil && len(summaryEmbedding) > 0 {
		summaryDoc.Embedding = summaryEmbedding
	}
	summaryDoc.Content = summaryEmbeddingContent

	return summaryDoc, dbDoc
}
