package rag

import (
	"context"
	"fmt"
	"stats-agent/pdf"

	"github.com/google/uuid"
	"go.uber.org/zap"
)

// AddPDFPagesToRAG stores PDF pages in RAG for retrieval
// Each page is stored as a separate document with metadata
func (r *RAG) AddPDFPagesToRAG(ctx context.Context, sessionID, filename string, pages []pdf.Page) error {
	if len(pages) == 0 {
		return nil
	}

	pagesAdded := 0
	chunksCreated := 0

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
			"role":        "document",
			"type":        "pdf",
			"filename":    filename,
			"page_number": fmt.Sprintf("%d", page.PageNumber),
		}

		// Content for embedding - just the text without prefix
		// The metadata already contains type, filename, and page info
		fullContent := page.Text

		// Filter metadata for JSONB storage
		structuralMetadata := filterStructuralMetadata(metadata)

		// Check if we need to chunk this page based on token count
		// Use document chunk size (3500 tokens by default)
		chunkSize := r.cfg.DocumentChunkSize
		if chunkSize <= 0 {
			chunkSize = 3500
		}

		pageTokens, err := r.countTokensForEmbedding(ctx, fullContent)
		if err != nil {
			r.logger.Warn("Failed to count tokens for PDF page, using character estimate",
				zap.Error(err),
				zap.String("filename", filename),
				zap.Int("page", page.PageNumber))
			pageTokens = len(fullContent) / 4 // Conservative estimate
		}

		if pageTokens > chunkSize {
			r.logger.Info("Chunking large PDF page",
				zap.String("filename", filename),
				zap.Int("page", page.PageNumber),
				zap.Int("tokens", pageTokens),
				zap.Int("chunk_size", chunkSize))
			r.persistDocumentChunks(ctx, structuralMetadata, fullContent)
			chunksCreated++
		} else {
			// Page fits in single chunk - use multi-vector approach
			// Store document first
			docID, err := r.store.UpsertDocument(ctx, docID, fullContent, structuralMetadata, contentHash)
			if err != nil {
				r.logger.Warn("Failed to store PDF page document",
					zap.Error(err),
					zap.String("filename", filename),
					zap.Int("page", page.PageNumber))
				continue
			}

			// Create embedding windows (may be 1 or more depending on page length)
			windows, err := r.createEmbeddingWindows(ctx, fullContent)
			if err != nil {
				r.logger.Warn("Failed to create embedding windows for PDF page",
					zap.Error(err),
					zap.String("filename", filename),
					zap.Int("page", page.PageNumber))
				continue
			}

			// Store all embedding windows
			for _, window := range windows {
				if err := r.store.CreateEmbedding(ctx, docID, window.WindowIndex, window.WindowStart, window.WindowEnd, window.WindowText, window.Embedding); err != nil {
					r.logger.Warn("Failed to store embedding window for PDF page",
						zap.Error(err),
						zap.String("filename", filename),
						zap.Int("page", page.PageNumber),
						zap.Int("window_index", window.WindowIndex))
					// Continue with other windows
				}
			}

			r.logger.Debug("Stored PDF page with multiple embedding windows",
				zap.String("filename", filename),
				zap.Int("page", page.PageNumber),
				zap.Int("windows", len(windows)))
			pagesAdded++
		}
	}

	if pagesAdded == 0 && chunksCreated == 0 {
		r.logger.Warn("No PDF pages could be embedded", zap.String("filename", filename))
		return nil
	}

	r.logger.Info("Added PDF pages to RAG",
		zap.String("filename", filename),
		zap.Int("pages", pagesAdded),
		zap.Int("chunked_pages", chunksCreated))

	return nil
}
