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
	"github.com/philippgille/chromem-go"
	"go.uber.org/zap"
)

func (r *RAG) AddMessagesToStore(ctx context.Context, sessionID string, messages []types.AgentMessage) error {
	collection := r.db.GetCollection("long-term-memory", r.embedder)
	if collection == nil {
		return fmt.Errorf("long-term memory collection not found")
	}

	processedIndices := make(map[int]bool)
	var documentsToEmbed []chromem.Document

	var sessionFilter map[string]string
	if sessionID != "" {
		sessionFilter = map[string]string{"session_id": sessionID}
	}

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

		embeddableDocs := r.persistPreparedDocument(ctx, docData)
		documentsToEmbed = append(documentsToEmbed, embeddableDocs...)
	}

	if len(documentsToEmbed) == 0 {
		return nil
	}

	if err := collection.AddDocuments(ctx, documentsToEmbed, 4); err != nil {
		return fmt.Errorf("failed to add documents to collection: %w", err)
	}

	r.logger.Info("Added document chunks to long-term RAG memory", zap.Int("chunks_added", len(documentsToEmbed)))
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
	collection *chromem.Collection,
	sessionFilter map[string]string,
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
	var summaryDoc *chromem.Document

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
				contentToEmbed = "Fact: A code execution event occurred but could not be summarized."
			} else {
				contentToEmbed = strings.TrimSpace(summary)
			}
		} else {
			contentToEmbed = "Fact: An assistant action with a tool execution occurred."
		}
	} else {
		if message.Role == "assistant" {
			trimmed := strings.TrimSpace(message.Content)
			if strings.HasPrefix(trimmed, "Fact:") && !format.HasTag(message.Content, format.PythonTag) {
				return nil, true, nil
			}
		}

		storedContent = canonicalizeFactText(message.Content)
		metadata["role"] = message.Role
		contentToEmbed = canonicalizeFactText(storedContent)

		if collection != nil && collection.Count() > 0 {
			results, err := collection.Query(ctx, contentToEmbed, 1, sessionFilter, nil)
			if err != nil {
				r.logger.Warn("Deduplication query failed, proceeding to add document anyway", zap.Error(err))
			} else if len(results) > 0 && results[0].Similarity > 0.98 && results[0].Metadata["role"] == message.Role {
				r.logger.Debug("Skipping duplicate content", zap.Float32("similarity", results[0].Similarity), zap.String("role", message.Role))
				return nil, true, nil
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

	if role != "fact" && len(message.Content) > 500 {
		summary, err := r.generateSearchableSummary(ctx, message.Content)
		if err != nil {
			r.logger.Warn("Failed to create searchable summary for long message, will use full content",
				zap.Error(err),
				zap.Int("content_length", len(message.Content)))
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

func (r *RAG) buildSummaryDocument(summary string, parentMetadata map[string]string, sessionID, messageRole string) *chromem.Document {
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

	return &chromem.Document{
		ID:       uuid.New().String(),
		Content:  summary,
		Metadata: metadata,
	}
}

func (r *RAG) persistPreparedDocument(ctx context.Context, data *ragDocumentData) []chromem.Document {
	if data == nil {
		return nil
	}

	// Embed content directly (no augmentation - metadata is already inline in fact text)
	embedContent := r.ensureEmbeddingTokenLimit(ctx, data.EmbedContent)
	embeddingVector, embedErr := r.embedder(ctx, embedContent)
	if embedErr != nil {
		r.logger.Warn("Failed to create embedding for RAG persistence",
			zap.Error(embedErr),
			zap.String("document_id", data.Metadata["document_id"]))
	}

	// Filter metadata to keep only structural fields for JSONB storage
	structuralMetadata := filterStructuralMetadata(data.Metadata)

	if err := r.store.UpsertRAGDocument(ctx, data.ID, data.StoredContent, embedContent, structuralMetadata, data.ContentHash, embeddingVector); err != nil {
		r.logger.Warn("Failed to persist RAG document", zap.Error(err), zap.String("document_id", data.Metadata["document_id"]))
	}

	var documents []chromem.Document
	if len(data.EmbedContent) > r.maxEmbeddingChars {
		r.logger.Info("Chunking oversized message for embedding",
			zap.String("role", data.Metadata["role"]),
			zap.Int("length", len(data.EmbedContent)))
		documents = append(documents, r.persistChunks(ctx, structuralMetadata, data.EmbedContent)...)
	} else {
		doc := chromem.Document{
			ID:       uuid.New().String(),
			Content:  embedContent,
			Metadata: cloneStringMap(structuralMetadata),
		}
		if embedErr == nil && len(embeddingVector) > 0 {
			doc.Embedding = embeddingVector
		}
		documents = append(documents, doc)
	}

	if summaryDoc := r.persistSummaryDocument(ctx, data.SummaryDoc); summaryDoc != nil {
		documents = append(documents, *summaryDoc)
	}

	return documents
}

func (r *RAG) persistChunks(ctx context.Context, baseMetadata map[string]string, content string) []chromem.Document {
	if len(content) == 0 {
		return nil
	}

	parentDocumentID := baseMetadata["document_id"]
	role := baseMetadata["role"]
	var documents []chromem.Document
	chunkIndex := 0

	sentences := r.sentenceSplitter.Split(content)
	if len(sentences) == 0 {
		sentences = []string{content}
	}

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

		if estimatedTokens > tokenLimit || sentenceLength > chunkSize {
			if current.Len() > 0 {
				chunks = append(chunks, current.String())
				current.Reset()
				currentTokens = 0
			}
			for start := 0; start < sentenceLength; start += chunkSize {
				end := min(start+chunkSize, sentenceLength)
				segment := strings.TrimSpace(sentence[start:end])
				if segment != "" {
					chunks = append(chunks, segment)
				}
			}
			continue
		}

		prospectiveLen := current.Len()
		if prospectiveLen > 0 {
			prospectiveLen++
		}
		prospectiveLen += sentenceLength
		prospectiveTokens := currentTokens + estimatedTokens

		if prospectiveLen > chunkSize || prospectiveTokens > tokenLimit {
			chunks = append(chunks, current.String())
			current.Reset()
			currentTokens = 0
		}

		if current.Len() > 0 {
			current.WriteString(" ")
		}
		current.WriteString(sentence)
		currentTokens += estimatedTokens
	}

	if current.Len() > 0 {
		chunks = append(chunks, current.String())
	}

	overlapRatio := 0.17
	processedChunks := make([]string, 0, len(chunks))
	var previousOverlap string

	for _, chunkContent := range chunks {
		chunkContent = strings.TrimSpace(chunkContent)
		if chunkContent == "" {
			continue
		}

		chunkWithOverlap := chunkContent
		if previousOverlap != "" {
			chunkWithOverlap = strings.TrimSpace(previousOverlap + " " + chunkContent)
		}
		processedChunks = append(processedChunks, chunkWithOverlap)

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

		// Embed content directly (no augmentation)
		chunkEmbeddingContent := r.ensureEmbeddingTokenLimit(ctx, chunkContent)
		chunkEmbedding, chunkEmbedErr := r.embedder(ctx, chunkEmbeddingContent)
		if chunkEmbedErr != nil {
			r.logger.Warn("Failed to create embedding for chunk",
				zap.Error(chunkEmbedErr),
				zap.String("document_id", chunkDocID.String()),
				zap.Int("chunk_index", chunkIndex))
		}

		if err := r.store.UpsertRAGDocument(ctx, chunkDocID, chunkContent, chunkEmbeddingContent, structuralChunkMetadata, chunkHash, chunkEmbedding); err != nil {
			r.logger.Warn("Failed to persist chunked RAG document",
				zap.Error(err),
				zap.String("document_id", chunkDocID.String()),
				zap.Int("chunk_index", chunkIndex))
		}

		chunkDoc := chromem.Document{
			ID:       uuid.New().String(),
			Content:  chunkEmbeddingContent,
			Metadata: cloneStringMap(structuralChunkMetadata),
		}
		if chunkEmbedErr == nil && len(chunkEmbedding) > 0 {
			chunkDoc.Embedding = chunkEmbedding
		}
		documents = append(documents, chunkDoc)
		chunkIndex++
	}

	return documents
}

func (r *RAG) persistSummaryDocument(ctx context.Context, summaryDoc *chromem.Document) *chromem.Document {
	if summaryDoc == nil {
		return nil
	}

	summaryMetadata := cloneStringMap(summaryDoc.Metadata)
	summaryDoc.Metadata = summaryMetadata
	summaryIDStr, ok := summaryMetadata["document_id"]
	if !ok {
		r.logger.Warn("Summary document missing document_id, skipping persistence")
		return nil
	}

	summaryID, err := uuid.Parse(summaryIDStr)
	if err != nil {
		r.logger.Warn("Summary document has invalid document_id", zap.String("document_id", summaryIDStr), zap.Error(err))
		return nil
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

	if summaryErr == nil && len(summaryEmbedding) > 0 {
		summaryDoc.Embedding = summaryEmbedding
	}
	summaryDoc.Content = summaryEmbeddingContent

	return summaryDoc
}
