package rag

import (
	"context"
	"fmt"
	"time"

	"github.com/google/uuid"
	"github.com/philippgille/chromem-go"
	"go.uber.org/zap"
)

func (r *RAG) LoadPersistedDocuments(ctx context.Context) error {
	collection := r.db.GetCollection("long-term-memory", r.embedder)
	if collection == nil {
		return fmt.Errorf("long-term memory collection not found")
	}

	documents, err := r.store.ListRAGDocuments(ctx)
	if err != nil {
		return fmt.Errorf("failed to load stored RAG documents: %w", err)
	}

	if len(documents) == 0 {
		return nil
	}

	added := 0
	for _, stored := range documents {
		metadataCopy := make(map[string]string, len(stored.Metadata)+1)
		for k, v := range stored.Metadata {
			metadataCopy[k] = v
		}
		if _, ok := metadataCopy["document_id"]; !ok {
			metadataCopy["document_id"] = stored.DocumentID.String()
		}

		embeddingContent := stored.EmbeddingContent
		if embeddingContent == "" {
			embeddingContent = stored.Content
		}
		if embeddingContent == "" {
			r.logger.Warn("Stored RAG document missing content, skipping",
				zap.String("document_id", stored.DocumentID.String()))
			continue
		}
		embeddingContent = r.ensureEmbeddingTokenLimit(ctx, embeddingContent)

		embeddingVector := stored.Embedding
		if len(embeddingVector) == 0 {
			var embedErr error
			embeddingVector, embedErr = r.embedder(ctx, embeddingContent)
			if embedErr != nil {
				r.logger.Warn("Failed to rebuild embedding for stored document",
					zap.Error(embedErr),
					zap.String("document_id", stored.DocumentID.String()))
				continue
			}
			if err := r.store.UpsertRAGDocument(ctx, stored.DocumentID, stored.Content, embeddingContent, metadataCopy, stored.ContentHash, embeddingVector); err != nil {
				r.logger.Warn("Failed to update stored document with embedding",
					zap.Error(err),
					zap.String("document_id", stored.DocumentID.String()))
			}
		}

		doc := chromem.Document{
			ID:        uuid.New().String(),
			Content:   embeddingContent,
			Metadata:  metadataCopy,
			Embedding: embeddingVector,
		}

		if err := collection.AddDocument(ctx, doc); err != nil {
			r.logger.Warn("Failed to add stored document to collection",
				zap.Error(err),
				zap.String("document_id", stored.DocumentID.String()))
			continue
		}
		added++
	}

	r.logger.Info("Loaded persisted RAG documents", zap.Int("documents", added))
	return nil
}

func (r *RAG) DeleteSessionDocuments(sessionID string) error {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	var resultErr error

	collection := r.db.GetCollection("long-term-memory", r.embedder)
	if collection == nil {
		resultErr = fmt.Errorf("long-term memory collection not found")
		r.logger.Warn("Vector collection missing during session cleanup", zap.String("session_id", sessionID))
	} else {
		if err := collection.Delete(ctx, map[string]string{"session_id": sessionID}, nil); err != nil {
			wrapped := fmt.Errorf("failed to delete session documents from collection: %w", err)
			r.logger.Warn("Failed to delete session documents from collection",
				zap.Error(err),
				zap.String("session_id", sessionID))
			resultErr = wrapped
		} else {
			r.logger.Debug("Removed session documents from RAG collection", zap.String("session_id", sessionID))
		}
	}

	if sessionUUID, err := uuid.Parse(sessionID); err != nil {
		r.logger.Warn("Unable to parse session ID for RAG document cleanup",
			zap.Error(err),
			zap.String("session_id", sessionID))
	} else {
		if deleted, err := r.store.DeleteRAGDocumentsBySession(ctx, sessionUUID); err != nil {
			r.logger.Warn("Failed to delete RAG documents from database",
				zap.Error(err),
				zap.String("session_id", sessionID))
			if resultErr == nil {
				resultErr = fmt.Errorf("failed to delete session documents from database: %w", err)
			}
		} else if deleted > 0 {
			r.logger.Debug("Deleted RAG documents from database",
				zap.String("session_id", sessionID),
				zap.Int64("documents_deleted", deleted))
		}
	}

	r.clearSessionDataset(sessionID)
	return resultErr
}
