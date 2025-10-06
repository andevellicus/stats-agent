package rag

import (
	"context"
	"strconv"
	"strings"

	"github.com/google/uuid"
	"github.com/philippgille/chromem-go"
	"go.uber.org/zap"
)

// prepareChunks splits oversized content into chunks and prepares them for persistence.
// It uses sentence-aware chunking with overlap for better context preservation.
// Returns chromem documents and database documents for batch persistence.
func (r *RAG) prepareChunks(ctx context.Context, baseMetadata map[string]string, content string) ([]chromem.Document, []dbDocument) {
	if len(content) == 0 {
		return nil, nil
	}

	parentDocumentID := baseMetadata["document_id"]
	role := baseMetadata["role"]
	var chromemDocs []chromem.Document
	var dbDocs []dbDocument
	chunkIndex := 0

	// Split content into sentences
	sentences := r.sentenceSplitter.Split(content)
	if len(sentences) == 0 {
		sentences = []string{content}
	}

	// Determine chunk size and token limits
	chunkSize := r.maxEmbeddingChars
	if chunkSize <= 0 {
		chunkSize = r.cfg.MaxEmbeddingChars
	}
	if chunkSize <= 0 {
		chunkSize = len(content)
	}
	tokenLimit := r.maxEmbeddingTokens
	if tokenLimit <= 0 {
		tokenLimit = r.cfg.MaxEmbeddingTokens
	}
	if tokenLimit <= 0 {
		tokenLimit = len(content)
	}

	// Build chunks from sentences
	var chunks []string
	var current strings.Builder
	var currentTokens int

	for _, sentence := range sentences {
		sentence = strings.TrimSpace(sentence)
		if sentence == "" {
			continue
		}
		sentenceLength := len(sentence)
		estimatedTokens := int(float64(sentenceLength) / 3.5)

		// Handle oversized sentences
		if estimatedTokens > tokenLimit || sentenceLength > chunkSize {
			if current.Len() > 0 {
				chunks = append(chunks, current.String())
				current.Reset()
				currentTokens = 0
			}
			// Split oversized sentence into character-based segments
			for start := 0; start < sentenceLength; start += chunkSize {
				end := min(start+chunkSize, sentenceLength)
				segment := strings.TrimSpace(sentence[start:end])
				if segment != "" {
					chunks = append(chunks, segment)
				}
			}
			continue
		}

		// Calculate prospective chunk size
		prospectiveLen := current.Len()
		if prospectiveLen > 0 {
			prospectiveLen++ // Account for space separator
		}
		prospectiveLen += sentenceLength
		prospectiveTokens := currentTokens + estimatedTokens

		// Flush current chunk if adding sentence would exceed limits
		if prospectiveLen > chunkSize || prospectiveTokens > tokenLimit {
			chunks = append(chunks, current.String())
			current.Reset()
			currentTokens = 0
		}

		// Add sentence to current chunk
		if current.Len() > 0 {
			current.WriteString(" ")
		}
		current.WriteString(sentence)
		currentTokens += estimatedTokens
	}

	// Flush remaining content
	if current.Len() > 0 {
		chunks = append(chunks, current.String())
	}

	// Apply overlap between chunks for context preservation
	overlapRatio := 0.17
	processedChunks := make([]string, 0, len(chunks))
	var previousOverlap string

	for _, chunkContent := range chunks {
		chunkContent = strings.TrimSpace(chunkContent)
		if chunkContent == "" {
			continue
		}

		// Prepend overlap from previous chunk
		chunkWithOverlap := chunkContent
		if previousOverlap != "" {
			chunkWithOverlap = strings.TrimSpace(previousOverlap + " " + chunkContent)
		}
		processedChunks = append(processedChunks, chunkWithOverlap)

		// Calculate overlap for next chunk
		if overlapRatio > 0 {
			runes := []rune(chunkContent)
			if len(runes) == 0 {
				previousOverlap = ""
				continue
			}
			overlapLen := int(float64(len(runes)) * overlapRatio)
			if overlapLen < 1 {
				overlapLen = 1
			}
			if overlapLen > len(runes) {
				overlapLen = len(runes)
			}
			previousOverlap = string(runes[len(runes)-overlapLen:])
		} else {
			previousOverlap = ""
		}
	}

	// Fallback to full content if chunking produced no results
	if len(processedChunks) == 0 {
		processedChunks = append(processedChunks, content)
	}

	// Create and persist chunk documents
	for _, chunkContent := range processedChunks {
		chunkContent = strings.TrimSpace(chunkContent)
		if chunkContent == "" {
			continue
		}

		chunkDocID := uuid.New()
		chunkMetadata := cloneStringMap(baseMetadata)
		chunkMetadata["type"] = "chunk"
		chunkMetadata["chunk_index"] = strconv.Itoa(chunkIndex)
		chunkMetadata["parent_document_id"] = parentDocumentID
		chunkMetadata["parent_document_role"] = role
		chunkMetadata["document_id"] = chunkDocID.String()

		chunkHash := hashContent(normalizeForHash(chunkContent))
		if chunkHash != "" {
			chunkMetadata["content_hash"] = chunkHash
		}

		// Filter chunk metadata to structural fields only
		structuralChunkMetadata := filterStructuralMetadata(chunkMetadata)

		// Embed content directly (no augmentation)
		chunkEmbeddingContent := r.ensureEmbeddingTokenLimit(ctx, chunkContent)
		chunkEmbedding, chunkEmbedErr := r.embedder(ctx, chunkEmbeddingContent)
		if chunkEmbedErr != nil {
			r.logger.Warn("Failed to create embedding for chunk",
				zap.Error(chunkEmbedErr),
				zap.String("document_id", chunkDocID.String()),
				zap.Int("chunk_index", chunkIndex))
		}

		// Prepare database document for batch persistence
		dbDocs = append(dbDocs, dbDocument{
			DocumentID:       chunkDocID,
			Content:          chunkContent,
			EmbeddingContent: chunkEmbeddingContent,
			Metadata:         cloneStringMap(structuralChunkMetadata),
			ContentHash:      chunkHash,
			Embedding:        chunkEmbedding,
		})

		// Create in-memory document for chromem
		chunkDoc := chromem.Document{
			ID:       uuid.New().String(),
			Content:  chunkEmbeddingContent,
			Metadata: cloneStringMap(structuralChunkMetadata),
		}
		if chunkEmbedErr == nil && len(chunkEmbedding) > 0 {
			chunkDoc.Embedding = chunkEmbedding
		}
		chromemDocs = append(chromemDocs, chunkDoc)
		chunkIndex++
	}

	return chromemDocs, dbDocs
}

// min returns the smaller of two integers.
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
