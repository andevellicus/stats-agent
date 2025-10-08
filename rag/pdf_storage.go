package rag

import (
	"context"
	"fmt"
	"stats-agent/pdf"

	"github.com/google/uuid"
	"github.com/philippgille/chromem-go"
	"go.uber.org/zap"
)

// AddPDFPagesToRAG stores PDF pages in RAG for retrieval
// Each page is stored as a separate document with metadata
func (r *RAG) AddPDFPagesToRAG(ctx context.Context, sessionID, filename string, pages []pdf.Page) error {
	if len(pages) == 0 {
		return nil
	}

	collection := r.db.GetCollection("long-term-memory", r.embedder)
	if collection == nil {
		return fmt.Errorf("long-term memory collection not found")
	}

	var documentsToEmbed []chromem.Document

	for _, page := range pages {
		if page.Text == "" {
			continue // Skip empty pages
		}

		// Create document ID and content hash
		docID := uuid.New()
		contentHash := hashContent(fmt.Sprintf("pdf:%s:page:%d:%s", filename, page.PageNumber, page.Text))

		// Prepare metadata
		metadata := map[string]string{
			"session_id":  sessionID,
			"document_id": docID.String(),
			"role":        "pdf_page",
			"source":      "pdf",
			"filename":    filename,
			"page_number": fmt.Sprintf("%d", page.PageNumber),
		}

		// Content to embed includes page context
		embedContent := fmt.Sprintf("PDF: %s (Page %d)\n\n%s", filename, page.PageNumber, page.Text)

		// Ensure content is within embedding limits
		embedContent = r.ensureEmbeddingTokenLimit(ctx, embedContent)

		// Create embedding
		embeddingVector, err := r.embedder(ctx, embedContent)
		if err != nil {
			r.logger.Warn("Failed to create embedding for PDF page",
				zap.Error(err),
				zap.String("filename", filename),
				zap.Int("page", page.PageNumber))
			continue
		}

		// Filter metadata for JSONB storage
		structuralMetadata := filterStructuralMetadata(metadata)

		// Store in database
		if err := r.store.UpsertRAGDocument(ctx, docID, page.Text, embedContent, structuralMetadata, contentHash, embeddingVector); err != nil {
			r.logger.Warn("Failed to persist PDF page to database",
				zap.Error(err),
				zap.String("filename", filename),
				zap.Int("page", page.PageNumber))
			continue
		}

		// Check if we need to chunk this page
		if len(embedContent) > r.maxEmbeddingChars {
			r.logger.Info("Chunking large PDF page",
				zap.String("filename", filename),
				zap.Int("page", page.PageNumber),
				zap.Int("length", len(embedContent)))
			chunks := r.persistChunks(ctx, structuralMetadata, embedContent)
			documentsToEmbed = append(documentsToEmbed, chunks...)
		} else {
			// Single document for this page
			doc := chromem.Document{
				ID:       uuid.New().String(),
				Content:  embedContent,
				Metadata: metadata,
			}
			documentsToEmbed = append(documentsToEmbed, doc)
		}
	}

	if len(documentsToEmbed) == 0 {
		r.logger.Warn("No PDF pages could be embedded", zap.String("filename", filename))
		return nil
	}

	// Add all documents to the collection
	if err := collection.AddDocuments(ctx, documentsToEmbed, 4); err != nil {
		return fmt.Errorf("failed to add PDF pages to RAG collection: %w", err)
	}

	r.logger.Info("Added PDF pages to RAG",
		zap.String("filename", filename),
		zap.Int("pages", len(pages)),
		zap.Int("chunks", len(documentsToEmbed)))

	return nil
}
