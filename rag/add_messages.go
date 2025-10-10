package rag

import (
	"context"
	"encoding/json"
	"fmt"
	"regexp"
	"strconv"
	"strings"

	"stats-agent/web/format"
	"stats-agent/web/types"

	"github.com/google/uuid"
	"go.uber.org/zap"
)

func (r *RAG) AddMessagesToStore(ctx context.Context, sessionID string, messages []types.AgentMessage) error {
	processedIndices := make(map[int]bool)

	for i := range messages {
		if processedIndices[i] {
			continue
		}

		docData, skip, err := r.prepareDocumentForMessage(ctx, sessionID, messages, i, processedIndices)
		if err != nil {
			r.logger.Warn("Failed to prepare RAG document", zap.Error(err))
			continue
		}
		if skip || docData == nil {
			continue
		}

		r.persistPreparedDocument(ctx, docData)
	}

	return nil
}

func (r *RAG) ensureDatasetMetadata(sessionID string, metadata map[string]string, texts ...string) {
	if metadata == nil {
		return
	}

	if existing := strings.TrimSpace(metadata["dataset"]); existing != "" {
		metadata["dataset"] = existing
		r.rememberSessionDataset(sessionID, existing)
		return
	}

	for _, text := range texts {
		if text == "" {
			continue
		}
		if matches := datasetQueryRegex.FindStringSubmatch(text); len(matches) > 1 {
			dataset := strings.TrimSpace(matches[1])
			if dataset != "" {
				metadata["dataset"] = dataset
				r.rememberSessionDataset(sessionID, dataset)
				return
			}
		}
	}

	if sessionID == "" {
		return
	}

	if dataset := strings.TrimSpace(r.getSessionDataset(sessionID)); dataset != "" {
		metadata["dataset"] = dataset
	}
}

func (r *RAG) prepareDocumentForMessage(
	ctx context.Context,
	sessionID string,
	messages []types.AgentMessage,
	index int,
	processed map[int]bool,
) (*ragDocumentData, bool, error) {
	processed[index] = true
	message := messages[index]

	documentUUID := uuid.New()
	documentID := documentUUID.String()
	metadata := map[string]string{"document_id": documentID}
	if sessionID != "" {
		metadata["session_id"] = sessionID
	}
	r.ensureDatasetMetadata(sessionID, metadata, message.Content)

	var storedContent string
	var contentToEmbed string
	var summaryDoc *summaryDocument

	if message.Role == "assistant" && index+1 < len(messages) && messages[index+1].Role == "tool" {
		toolMessage := messages[index+1]
		processed[index+1] = true
		metadata["role"] = "fact"

		assistantContent := canonicalizeFactText(message.Content)
		toolContent := canonicalizeFactText(toolMessage.Content)

		// Extract statistical metadata FIRST (before fact generation)
		var statMeta map[string]string
		if format.HasTag(message.Content, format.PythonTag) {
			code, _ := format.ExtractTagContent(message.Content, format.PythonTag)
			statMeta = ExtractStatisticalMetadata(code, toolContent)
			r.ensureDatasetMetadata(sessionID, metadata, code, toolContent)
		}

		userContent := ""
		for prev := index - 1; prev >= 0; prev-- {
			if messages[prev].Role == "user" {
				userContent = canonicalizeFactText(messages[prev].Content)
				break
			}
		}

		factPayload := factStoredContent{
			Assistant: assistantContent,
			Tool:      toolContent,
		}
		if userContent != "" {
			factPayload.User = userContent
		}

		factJSON, marshalErr := json.Marshal(factPayload)
		if marshalErr != nil {
			r.logger.Warn("Failed to marshal fact payload, falling back to concatenated format", zap.Error(marshalErr))
			storedContent = fmt.Sprintf("%s\n\n%s", assistantContent, toolContent)
		} else {
			storedContent = string(factJSON)
		}

		re := regexp.MustCompile(`(?s)<python>(.*)</python>`)
		matches := re.FindStringSubmatch(message.Content)
		if len(matches) > 1 {
			code := strings.TrimSpace(matches[1])
			result := strings.TrimSpace(toolMessage.Content)
			// Pass statistical metadata to fact generator
			summary, err := r.generateFactSummary(ctx, code, result, statMeta)
			if err != nil {
				r.logger.Warn("LLM fact summarization failed, using fallback summary",
					zap.Error(err),
					zap.Int("code_length", len(code)),
					zap.Int("result_length", len(result)))
				contentToEmbed = "A code execution event occurred but could not be summarized."
			} else {
				contentToEmbed = strings.TrimSpace(summary)
			}
		} else {
			contentToEmbed = "An assistant action with a tool execution occurred."
		}
	} else {
		// Store all standalone messages (not part of assistant+tool pairs)
		// This includes:
		// - User questions and requests
		// - Final summaries like "Analysis Complete"
		// - Orphaned code blocks (environment failures)
		// - Any other assistant responses without tool execution

		storedContent = canonicalizeFactText(message.Content)

		// For user messages with file upload notifications, extract the user's actual question
		if message.Role == "user" && strings.Contains(message.Content, "[📎 File uploaded:") {
			// Remove file upload notification line(s) but keep the user's question
			lines := strings.Split(storedContent, "\n")
			var userLines []string
			for _, line := range lines {
				// Skip file upload notification lines
				if !strings.Contains(line, "[📎 File uploaded:") && strings.TrimSpace(line) != "" {
					userLines = append(userLines, line)
				}
			}

			// If there's no user question beyond the notification, skip storage
			// (the PDF pages are already stored via AddPDFPagesToRAG)
			if len(userLines) == 0 {
				return nil, true, nil
			}

			storedContent = strings.Join(userLines, "\n")
		}

		metadata["role"] = message.Role
		contentToEmbed = canonicalizeFactText(storedContent)

		// Check for near-duplicates using vector similarity
		// SKIP this check for user messages - every user question is contextually important
		if message.Role != "user" {
			queryEmbedding, err := r.embedder(ctx, contentToEmbed)
			if err == nil && len(queryEmbedding) > 0 {
				results, err := r.store.VectorSearchRAGDocuments(ctx, queryEmbedding, 1, sessionID)
				if err != nil {
					r.logger.Warn("Deduplication query failed, proceeding to add document anyway", zap.Error(err))
				} else if len(results) > 0 && results[0].Similarity > 0.98 && results[0].Metadata["role"] == message.Role {
					r.logger.Debug("Skipping duplicate content", zap.Float64("similarity", results[0].Similarity), zap.String("role", message.Role))
					return nil, true, nil
				}
			}
		}
	}

	if storedContent == "" {
		storedContent = contentToEmbed
	}

	role := metadata["role"]
	normalizedContent := normalizeForHash(storedContent)
	contentHash := hashContent(normalizedContent)
	if contentHash != "" {
		metadata["content_hash"] = contentHash
		existingDocID, err := r.store.FindRAGDocumentByHash(ctx, sessionID, role, contentHash)
		if err != nil {
			r.logger.Warn("Failed to check for existing RAG document",
				zap.Error(err),
				zap.String("session_id", sessionID))
			return nil, true, nil
		}
		if existingDocID != uuid.Nil {
			r.logger.Debug("Skipping duplicate RAG document",
				zap.String("existing_document_id", existingDocID.String()),
				zap.String("session_id", sessionID),
				zap.String("role", role))
			return nil, true, nil
		}
	}

	if role != "fact" && len(storedContent) > 500 {
		summary, err := r.generateSearchableSummary(ctx, storedContent)
		if err != nil {
			r.logger.Warn("Failed to create searchable summary for long message, will use full content",
				zap.Error(err),
				zap.Int("content_length", len(storedContent)))
		} else {
			summaryDoc = r.buildSummaryDocument(summary, metadata, sessionID, message.Role)
		}
	}

	r.ensureDatasetMetadata(sessionID, metadata, message.Content, storedContent, contentToEmbed)

	return &ragDocumentData{
		ID:            documentUUID,
		Metadata:      metadata,
		StoredContent: storedContent,
		EmbedContent:  contentToEmbed,
		ContentHash:   contentHash,
		SummaryDoc:    summaryDoc,
	}, false, nil
}

func (r *RAG) buildSummaryDocument(summary string, parentMetadata map[string]string, sessionID, messageRole string) *summaryDocument {
	summaryID := uuid.New()
	metadata := map[string]string{
		"role":                 messageRole,
		"document_id":          summaryID.String(),
		"type":                 "summary",
		"parent_document_id":   parentMetadata["document_id"],
		"parent_document_role": parentMetadata["role"],
	}
	if sessionID != "" {
		metadata["session_id"] = sessionID
	}
	if dataset := parentMetadata["dataset"]; dataset != "" {
		metadata["dataset"] = dataset
	}

	return &summaryDocument{
		ID:       summaryID.String(),
		Content:  summary,
		Metadata: metadata,
	}
}

func (r *RAG) persistPreparedDocument(ctx context.Context, data *ragDocumentData) {
	if data == nil {
		return
	}

	// Filter metadata to keep only structural fields for JSONB storage
	structuralMetadata := filterStructuralMetadata(data.Metadata)

	// Count tokens FIRST to decide if we should chunk
	tokenCount, err := r.countTokensForEmbedding(ctx, data.EmbedContent)
	if err != nil {
		r.logger.Warn("Failed to count tokens for content, will attempt to embed anyway",
			zap.Error(err),
			zap.String("document_id", data.Metadata["document_id"]))
		tokenCount = 0
	}

    tokenLimit := r.embeddingTokenTarget
    if tokenLimit <= 0 {
        tokenLimit = 480
    }

	if tokenCount > tokenLimit {
		// Content exceeds limit - CHUNK IT instead of truncating
		if data.Metadata["role"] == "fact" {
			previewLen := 150
			if len(data.EmbedContent) < previewLen {
				previewLen = len(data.EmbedContent)
			}
			r.logger.Warn("Fact summary exceeds token limit - should be more concise",
				zap.Int("tokens", tokenCount),
				zap.Int("limit", tokenLimit),
				zap.String("document_id", data.Metadata["document_id"]),
				zap.String("preview", data.EmbedContent[:previewLen]))
		}
		r.logger.Info("Chunking oversized content for embedding",
			zap.String("role", data.Metadata["role"]),
			zap.Int("tokens", tokenCount),
			zap.Int("limit", tokenLimit))

		// Route to appropriate chunking strategy based on role
		role := structuralMetadata["role"]
		if role == "document" {
			r.persistDocumentChunks(ctx, structuralMetadata, data.EmbedContent)
		} else {
			r.persistConversationChunks(ctx, structuralMetadata, data.EmbedContent)
		}
	} else {
		// Content is within limit - embed directly with safety check
		embedContent := r.ensureEmbeddingTokenLimit(ctx, data.EmbedContent)
		embeddingVector, embedErr := r.embedder(ctx, embedContent)
		if embedErr != nil {
			r.logger.Warn("Failed to create embedding for RAG persistence",
				zap.Error(embedErr),
				zap.String("document_id", data.Metadata["document_id"]))
		}

		if err := r.store.UpsertRAGDocument(ctx, data.ID, data.StoredContent, embedContent, structuralMetadata, data.ContentHash, embeddingVector); err != nil {
			r.logger.Warn("Failed to persist RAG document", zap.Error(err), zap.String("document_id", data.Metadata["document_id"]))
		}
	}

	if data.SummaryDoc != nil {
		r.persistSummaryDocument(ctx, data.SummaryDoc)
	}
}

// persistConversationChunks chunks conversation messages (facts, user/assistant messages)
// with overlap to maintain semantic continuity across chunks.
func (r *RAG) persistConversationChunks(ctx context.Context, baseMetadata map[string]string, content string) {
	if len(content) == 0 {
		return
	}

	parentDocumentID := baseMetadata["document_id"]
	role := baseMetadata["role"]
	chunkIndex := 0

	// Use conversation chunk size from config (1500 tokens by default)
	chunkSize := r.cfg.ConversationChunkSize
	if chunkSize <= 0 {
		chunkSize = 1500 // Default if not configured
	}

	// Get overlap ratio from config (20% by default)
	overlapRatio := r.cfg.ConversationChunkOverlap
	if overlapRatio <= 0 {
		overlapRatio = 0.20
	}

	// Calculate overlap target in tokens
	targetOverlapTokens := int(float64(chunkSize) * overlapRatio) // ~300 tokens for 1500 @ 20%

	// Chunk target is reduced to account for overlap that will be prepended
	chunkTargetTokens := chunkSize - targetOverlapTokens // ~1200 tokens

	// Word-based chunking: split by whitespace
	words := strings.Fields(content)
	if len(words) == 0 {
		return
	}

	var chunks []string
	var currentWords []string
	const tokenCheckInterval = 10 // Count tokens every 10 words to reduce API calls

	wordIdx := 0
	for wordIdx < len(words) {
		currentWords = append(currentWords, words[wordIdx])
		wordIdx++

		// Check token count every N words or at the end
		shouldCheck := (len(currentWords) % tokenCheckInterval == 0) || (wordIdx >= len(words))
		if !shouldCheck {
			continue
		}

		currentText := strings.Join(currentWords, " ")
		tokens, err := r.countTokensForEmbedding(ctx, currentText)
		if err != nil {
			r.logger.Warn("Failed to count tokens for chunk, using estimate",
				zap.Error(err),
				zap.Int("word_count", len(currentWords)))
			tokens = len(currentText) / 4 // Fallback estimate
		}

		// If we've exceeded the chunk target, backtrack
		if tokens > chunkTargetTokens {
			// Remove words until we're under the limit
			wordsToRemove := 0
			for tokens > chunkTargetTokens && len(currentWords)-wordsToRemove > 1 {
				wordsToRemove++
				trimmedWords := currentWords[:len(currentWords)-wordsToRemove]
				trimmedText := strings.Join(trimmedWords, " ")
				tokens, err = r.countTokensForEmbedding(ctx, trimmedText)
				if err != nil {
					tokens = len(trimmedText) / 4
				}
			}

			// Save this chunk (without the words that didn't fit)
			if len(currentWords)-wordsToRemove > 0 {
				chunkWords := currentWords[:len(currentWords)-wordsToRemove]
				chunks = append(chunks, strings.TrimSpace(strings.Join(chunkWords, " ")))
			}

			// Start new chunk with the word(s) that didn't fit
			currentWords = currentWords[len(currentWords)-wordsToRemove:]
		} else if wordIdx >= len(words) {
			// We're at the end and under the limit, save the final chunk
			chunks = append(chunks, strings.TrimSpace(currentText))
			currentWords = nil
		}
	}

	// Apply overlap based on TOKENS (not word count)
	processedChunks := make([]string, 0, len(chunks))
	var previousOverlapWords []string

	for _, chunkContent := range chunks {
		chunkContent = strings.TrimSpace(chunkContent)
		if chunkContent == "" {
			continue
		}

		// Prepend previous chunk's overlap
		chunkWithOverlap := chunkContent
		if len(previousOverlapWords) > 0 {
			overlapText := strings.Join(previousOverlapWords, " ")
			chunkWithOverlap = strings.TrimSpace(overlapText + " " + chunkContent)
		}

		processedChunks = append(processedChunks, chunkWithOverlap)

		// Calculate overlap for next chunk based on TOKENS (not words)
		// Find last N words from current chunk that total ~targetOverlapTokens
		chunkWords := strings.Fields(chunkContent)
		if len(chunkWords) > 0 {
			var overlapWords []string
			accumulatedTokens := 0

			// Work backwards from end of chunk, accumulating words until we reach target
			for i := len(chunkWords) - 1; i >= 0 && accumulatedTokens < targetOverlapTokens; i-- {
				overlapWords = append([]string{chunkWords[i]}, overlapWords...)
				testText := strings.Join(overlapWords, " ")
				tokens, err := r.countTokensForEmbedding(ctx, testText)
				if err == nil {
					accumulatedTokens = tokens
				} else {
					// Fallback: estimate ~4 chars per token
					accumulatedTokens = len(testText) / 4
				}
			}

			previousOverlapWords = overlapWords
		} else {
			previousOverlapWords = nil
		}
	}

	if len(processedChunks) == 0 {
		processedChunks = append(processedChunks, content)
	}

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

		// Store document first
		docID, err := r.store.UpsertDocument(ctx, chunkDocID, chunkContent, structuralChunkMetadata, chunkHash)
		if err != nil {
			r.logger.Warn("Failed to store conversation chunk",
				zap.Error(err),
				zap.String("document_id", chunkDocID.String()),
				zap.Int("chunk_index", chunkIndex))
			chunkIndex++
			continue
		}

		// Create embedding windows (conversation chunks usually fit in 1-2 windows)
		windows, err := r.createEmbeddingWindows(ctx, chunkContent)
		if err != nil {
			r.logger.Warn("Failed to create embedding windows for conversation chunk",
				zap.Error(err),
				zap.String("document_id", chunkDocID.String()),
				zap.Int("chunk_index", chunkIndex))
			chunkIndex++
			continue
		}

		// Store all embedding windows
		for _, window := range windows {
			if err := r.store.CreateEmbedding(ctx, docID, window.WindowIndex, window.WindowStart, window.WindowEnd, window.WindowText, window.Embedding); err != nil {
				r.logger.Warn("Failed to store embedding window for conversation chunk",
					zap.Error(err),
					zap.String("document_id", chunkDocID.String()),
					zap.Int("chunk_index", chunkIndex),
					zap.Int("window_index", window.WindowIndex))
			}
		}

		chunkIndex++
	}
}

// persistDocumentChunks chunks large documents (PDFs, Word docs, etc.) with NO overlap
// and stores each chunk independently. Uses larger chunk size since no overlap is needed.
func (r *RAG) persistDocumentChunks(ctx context.Context, baseMetadata map[string]string, content string) {
	if len(content) == 0 {
		return
	}

	chunkIndex := 0

	// Use document chunk size from config (3500 tokens by default)
	chunkSize := r.cfg.DocumentChunkSize
	if chunkSize <= 0 {
		chunkSize = 3500 // Default if not configured
	}

	// Word-based chunking: split by whitespace
	words := strings.Fields(content)
	if len(words) == 0 {
		return
	}

	var chunks []string
	var currentWords []string
	const tokenCheckInterval = 10 // Count tokens every 10 words to reduce API calls

	wordIdx := 0
	for wordIdx < len(words) {
		currentWords = append(currentWords, words[wordIdx])
		wordIdx++

		// Check token count every N words or at the end
		shouldCheck := (len(currentWords) % tokenCheckInterval == 0) || (wordIdx >= len(words))
		if !shouldCheck {
			continue
		}

		currentText := strings.Join(currentWords, " ")
		tokens, err := r.countTokensForEmbedding(ctx, currentText)
		if err != nil {
			r.logger.Warn("Failed to count tokens for document chunk, using estimate",
				zap.Error(err),
				zap.Int("word_count", len(currentWords)))
			tokens = len(currentText) / 4 // Fallback estimate
		}

		// If we've exceeded the chunk size, backtrack
		if tokens > chunkSize {
			// Remove words until we're under the limit
			wordsToRemove := 0
			for tokens > chunkSize && len(currentWords)-wordsToRemove > 1 {
				wordsToRemove++
				trimmedWords := currentWords[:len(currentWords)-wordsToRemove]
				trimmedText := strings.Join(trimmedWords, " ")
				tokens, err = r.countTokensForEmbedding(ctx, trimmedText)
				if err != nil {
					tokens = len(trimmedText) / 4
				}
			}

			// Save this chunk (without the words that didn't fit)
			if len(currentWords)-wordsToRemove > 0 {
				chunkWords := currentWords[:len(currentWords)-wordsToRemove]
				chunks = append(chunks, strings.TrimSpace(strings.Join(chunkWords, " ")))
			}

			// Start new chunk with the word(s) that didn't fit
			currentWords = currentWords[len(currentWords)-wordsToRemove:]
		} else if wordIdx >= len(words) {
			// We're at the end and under the limit, save the final chunk
			chunks = append(chunks, strings.TrimSpace(currentText))
			currentWords = nil
		}
	}

	// NO OVERLAP - store chunks directly
	for _, chunkContent := range chunks {
		chunkContent = strings.TrimSpace(chunkContent)
		if chunkContent == "" {
			continue
		}

		chunkDocID := uuid.New()
		chunkMetadata := cloneStringMap(baseMetadata)
		chunkMetadata["type"] = "document_chunk"
		chunkMetadata["chunk_index"] = strconv.Itoa(chunkIndex)
		// NO parent_document_id - retrieval returns chunk directly
		chunkMetadata["document_id"] = chunkDocID.String()

		chunkHash := hashContent(normalizeForHash(chunkContent))
		if chunkHash != "" {
			chunkMetadata["content_hash"] = chunkHash
		}

		// Filter chunk metadata to structural fields only
		structuralChunkMetadata := filterStructuralMetadata(chunkMetadata)

		// Store document first
		docID, err := r.store.UpsertDocument(ctx, chunkDocID, chunkContent, structuralChunkMetadata, chunkHash)
		if err != nil {
			r.logger.Warn("Failed to store document chunk",
				zap.Error(err),
				zap.String("document_id", chunkDocID.String()),
				zap.Int("chunk_index", chunkIndex))
			chunkIndex++
			continue
		}

		// Create embedding windows (document chunks may have multiple windows if large)
		windows, err := r.createEmbeddingWindows(ctx, chunkContent)
		if err != nil {
			r.logger.Warn("Failed to create embedding windows for document chunk",
				zap.Error(err),
				zap.String("document_id", chunkDocID.String()),
				zap.Int("chunk_index", chunkIndex))
			chunkIndex++
			continue
		}

		// Store all embedding windows
		for _, window := range windows {
			if err := r.store.CreateEmbedding(ctx, docID, window.WindowIndex, window.WindowStart, window.WindowEnd, window.WindowText, window.Embedding); err != nil {
				r.logger.Warn("Failed to store embedding window for document chunk",
					zap.Error(err),
					zap.String("document_id", chunkDocID.String()),
					zap.Int("chunk_index", chunkIndex),
					zap.Int("window_index", window.WindowIndex))
			}
		}

		chunkIndex++
	}
}

func (r *RAG) persistSummaryDocument(ctx context.Context, summaryDoc *summaryDocument) {
	if summaryDoc == nil {
		return
	}

	summaryMetadata := cloneStringMap(summaryDoc.Metadata)
	summaryIDStr, ok := summaryMetadata["document_id"]
	if !ok {
		r.logger.Warn("Summary document missing document_id, skipping persistence")
		return
	}

	summaryID, err := uuid.Parse(summaryIDStr)
	if err != nil {
		r.logger.Warn("Summary document has invalid document_id", zap.String("document_id", summaryIDStr), zap.Error(err))
		return
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
	summaryEmbedding, summaryErr := r.embedder(ctx, summaryEmbeddingContent)
	if summaryErr != nil {
		r.logger.Warn("Failed to create embedding for summary document",
			zap.Error(summaryErr),
			zap.String("document_id", summaryIDStr))
	}

	if err := r.store.UpsertRAGDocument(ctx, summaryID, summaryContent, summaryEmbeddingContent, structuralSummaryMetadata, summaryHash, summaryEmbedding); err != nil {
		r.logger.Warn("Failed to persist summary RAG document",
			zap.Error(err),
			zap.String("document_id", summaryIDStr))
	}
}
