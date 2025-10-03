package rag

import (
	"context"
	"crypto/sha256"
	"database/sql"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"time"

	"stats-agent/config"
	"stats-agent/database"
	"stats-agent/llmclient"
	"stats-agent/web/format"
	"stats-agent/web/types"

	"github.com/google/uuid"
	"github.com/philippgille/chromem-go"
	"go.uber.org/zap"
)

const (
	// BGE models typically handle 512 tokens max
	// ~4 chars per token, with safety margin
	maxEmbeddingChars  = 1000 // Reduced from 1500
	maxEmbeddingTokens = 250  // Safety margin under 512
)

type RAG struct {
	cfg      *config.Config
	db       *chromem.DB
	store    *database.PostgresStore
	embedder chromem.EmbeddingFunc
	logger   *zap.Logger
}

type factStoredContent struct {
	User      string `json:"user,omitempty"`
	Assistant string `json:"assistant"`
	Tool      string `json:"tool"`
}

type hybridCandidate struct {
	DocumentID    string
	Metadata      map[string]string
	Content       string
	SemanticScore float64
	BM25Score     float64
	ExactBonus    float64
	HasSemantic   bool
	HasBM25       bool
	Score         float64
}

// Embedding request/response types moved to llmclient

type ragDocumentData struct {
	ID            uuid.UUID
	Metadata      map[string]string
	StoredContent string
	EmbedContent  string
	ContentHash   string
	SummaryDoc    *chromem.Document
}

func New(cfg *config.Config, store *database.PostgresStore, logger *zap.Logger) (*RAG, error) {
	if store == nil {
		return nil, fmt.Errorf("postgres store is required for RAG persistence")
	}

	db := chromem.NewDB()
	embedder := createLlamaCppEmbedding(cfg, logger)
	_, err := db.GetOrCreateCollection("long-term-memory", nil, embedder)
	if err != nil {
		return nil, fmt.Errorf("failed to create initial collection: %w", err)
	}
	rag := &RAG{
		cfg:      cfg,
		db:       db,
		store:    store,
		embedder: embedder,
		logger:   logger,
	}
	return rag, nil
}

func canonicalizeFactText(text string) string {
	text = strings.ReplaceAll(text, "\r\n", "\n")
	lines := strings.Split(text, "\n")
	for i, line := range lines {
		lines[i] = strings.TrimRight(line, " \t")
	}
	joined := strings.Join(lines, "\n")
	return strings.TrimSpace(joined)
}

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

	var storedContent string
	var contentToEmbed string
	var summaryDoc *chromem.Document

	if message.Role == "assistant" && index+1 < len(messages) && messages[index+1].Role == "tool" {
		toolMessage := messages[index+1]
		processed[index+1] = true
		metadata["role"] = "fact"

		assistantContent := canonicalizeFactText(message.Content)
		toolContent := canonicalizeFactText(toolMessage.Content)

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
			summary, err := r.generateFactSummary(ctx, code, result)
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

	embeddingVector, embedErr := r.embedder(ctx, data.EmbedContent)
	if embedErr != nil {
		r.logger.Warn("Failed to create embedding for RAG persistence",
			zap.Error(embedErr),
			zap.String("document_id", data.Metadata["document_id"]))
	}

	if err := r.store.UpsertRAGDocument(ctx, data.ID, data.StoredContent, data.EmbedContent, data.Metadata, data.ContentHash, embeddingVector); err != nil {
		r.logger.Warn("Failed to persist RAG document", zap.Error(err), zap.String("document_id", data.Metadata["document_id"]))
	}

	var documents []chromem.Document
	if len(data.EmbedContent) > maxEmbeddingChars {
		r.logger.Info("Chunking oversized message for embedding",
			zap.String("role", data.Metadata["role"]),
			zap.Int("length", len(data.EmbedContent)))
		documents = append(documents, r.persistChunks(ctx, data.Metadata, data.EmbedContent)...)
	} else {
		doc := chromem.Document{
			ID:       uuid.New().String(),
			Content:  data.EmbedContent,
			Metadata: cloneMetadata(data.Metadata),
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

	for j := 0; j < len(content); j += maxEmbeddingChars {
		end := min(j+maxEmbeddingChars, len(content))
		chunkContent := content[j:end]
		if int(float64(len(chunkContent))/3.5) > maxEmbeddingTokens {
			end = j + (maxEmbeddingChars * 3 / 4)
			chunkContent = content[j:end]
		}

		chunkDocID := uuid.New()
		chunkMetadata := cloneMetadata(baseMetadata)
		chunkMetadata["type"] = "chunk"
		chunkMetadata["chunk_index"] = strconv.Itoa(chunkIndex)
		chunkMetadata["parent_document_id"] = parentDocumentID
		chunkMetadata["parent_document_role"] = role
		chunkMetadata["document_id"] = chunkDocID.String()

		chunkHash := hashContent(normalizeForHash(chunkContent))
		if chunkHash != "" {
			chunkMetadata["content_hash"] = chunkHash
		}

		chunkEmbedding, chunkEmbedErr := r.embedder(ctx, chunkContent)
		if chunkEmbedErr != nil {
			r.logger.Warn("Failed to create embedding for chunk",
				zap.Error(chunkEmbedErr),
				zap.String("document_id", chunkDocID.String()),
				zap.Int("chunk_index", chunkIndex))
		}

		if err := r.store.UpsertRAGDocument(ctx, chunkDocID, chunkContent, chunkContent, chunkMetadata, chunkHash, chunkEmbedding); err != nil {
			r.logger.Warn("Failed to persist chunked RAG document",
				zap.Error(err),
				zap.String("document_id", chunkDocID.String()),
				zap.Int("chunk_index", chunkIndex))
		}

		chunkDoc := chromem.Document{
			ID:       uuid.New().String(),
			Content:  chunkContent,
			Metadata: cloneMetadata(chunkMetadata),
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

	summaryMetadata := cloneMetadata(summaryDoc.Metadata)
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

	summaryEmbedding, summaryErr := r.embedder(ctx, summaryContent)
	if summaryErr != nil {
		r.logger.Warn("Failed to create embedding for summary document",
			zap.Error(summaryErr),
			zap.String("document_id", summaryIDStr))
	}

	if err := r.store.UpsertRAGDocument(ctx, summaryID, summaryContent, summaryContent, summaryMetadata, summaryHash, summaryEmbedding); err != nil {
		r.logger.Warn("Failed to persist summary RAG document",
			zap.Error(err),
			zap.String("document_id", summaryIDStr))
	}

	if summaryErr == nil && len(summaryEmbedding) > 0 {
		summaryDoc.Embedding = summaryEmbedding
	}

	return summaryDoc
}

func cloneMetadata(src map[string]string) map[string]string {
	if src == nil {
		return nil
	}
	return cloneStringMap(src)
}

// LoadPersistedDocuments rebuilds the in-memory vector store using documents stored in Postgres.
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

func (r *RAG) Query(ctx context.Context, sessionID string, query string, nResults int) (string, error) {
	if nResults <= 0 {
		return "", nil
	}

	collection := r.db.GetCollection("long-term-memory", r.embedder)
	const maxHybridCandidates = 100
	candidateLimit := max(nResults*4, 20)
	if candidateLimit > maxHybridCandidates {
		candidateLimit = maxHybridCandidates
	}

	lowerQuery := strings.ToLower(query)
	isQueryForError := strings.Contains(lowerQuery, "error")
	candidates := make(map[string]*hybridCandidate)

	if collection == nil {
		r.logger.Warn("Vector collection not found, using BM25 fallback only")
	} else {
		total := collection.Count()
		if total > 0 {
			limit := candidateLimit
			if limit > total {
				limit = total
			}
			if limit > 0 {
				var where map[string]string
				if sessionID != "" {
					where = map[string]string{"session_id": sessionID}
				}
				semanticResults, err := collection.Query(ctx, query, limit, where, nil)
				if err != nil {
					return "", fmt.Errorf("failed to query collection: %w", err)
				}
				for _, res := range semanticResults {
					docID := res.Metadata["document_id"]
					if docID == "" {
						r.logger.Warn("Vector result missing document_id, skipping")
						continue
					}
					cand := ensureCandidate(candidates, docID, res.Metadata)
					if float64(res.Similarity) > cand.SemanticScore {
						cand.SemanticScore = float64(res.Similarity)
						cand.Content = res.Content
					}
					cand.HasSemantic = true
				}
			}
		}
	}

	bm25Results, err := r.store.SearchRAGDocumentsBM25(ctx, query, candidateLimit, sessionID)
	if err != nil {
		return "", fmt.Errorf("failed to run BM25 search: %w", err)
	}
	for _, bm := range bm25Results {
		docID := bm.Metadata["document_id"]
		if docID == "" {
			docID = bm.DocumentID.String()
			if bm.Metadata == nil {
				bm.Metadata = make(map[string]string)
			}
			bm.Metadata["document_id"] = docID
		}
		cand := ensureCandidate(candidates, docID, bm.Metadata)
		combined := bm.BM25Score + bm.ExactMatchBonus
		existingCombined := cand.BM25Score + cand.ExactBonus

		embContent := bm.EmbeddingContent
		if embContent == "" {
			embContent = bm.Content
		}

		if combined > existingCombined {
			cand.BM25Score = bm.BM25Score
			cand.ExactBonus = bm.ExactMatchBonus
			if embContent != "" {
				cand.Content = embContent
			}
		} else if cand.Content == "" && embContent != "" {
			cand.Content = embContent
		}
		cand.HasBM25 = true
	}

	if len(candidates) == 0 {
		return "", nil
	}

	docContents := make(map[string]string)

	for _, cand := range candidates {
		lookupID := resolveLookupID(cand.Metadata)
		if lookupID == "" || lookupID == cand.DocumentID {
			continue
		}

		if content, ok := docContents[lookupID]; ok {
			cand.Content = content
			continue
		}

		lookupUUID, err := uuid.Parse(lookupID)
		if err != nil {
			r.logger.Warn("Invalid lookup identifier for scoring",
				zap.String("lookup_id", lookupID),
				zap.String("document_id", cand.DocumentID))
			continue
		}

		content, err := r.store.GetRAGDocumentContent(ctx, lookupUUID)
		if err != nil {
			if !errors.Is(err, sql.ErrNoRows) {
				r.logger.Warn("Failed to load parent content for scoring",
					zap.Error(err),
					zap.String("lookup_id", lookupID),
					zap.String("document_id", cand.DocumentID))
			}
			continue
		}

		docContents[lookupID] = content
		cand.Content = content
	}

	var maxSemantic float64
	var maxBM float64
	for _, cand := range candidates {
		if cand.SemanticScore > maxSemantic {
			maxSemantic = cand.SemanticScore
		}
		bmCombined := cand.BM25Score + cand.ExactBonus
		if bmCombined > maxBM {
			maxBM = bmCombined
		}
	}

	candidateList := make([]*hybridCandidate, 0, len(candidates))
	for _, cand := range candidates {
		weighted := 0.0
		weightSum := 0.0
		if cand.HasSemantic && maxSemantic > 0 {
			weighted += 0.7 * (cand.SemanticScore / maxSemantic)
			weightSum += 0.7
		}
		if cand.HasBM25 && maxBM > 0 {
			weighted += 0.3 * ((cand.BM25Score + cand.ExactBonus) / maxBM)
			weightSum += 0.3
		}
		combined := 0.0
		if weightSum > 0 {
			combined = weighted / weightSum
		}

		role := cand.Metadata["role"]
		docType := cand.Metadata["type"]
		if role == "fact" && docType != "chunk" {
			combined *= 1.3
		}
		if cand.Metadata["type"] == "summary" {
			combined *= 1.2
		}
		if cand.Content != "" && strings.Contains(cand.Content, "Error:") && !isQueryForError {
			combined *= 0.8
		}

		cand.Score = combined
		candidateList = append(candidateList, cand)
	}

	sort.Slice(candidateList, func(i, j int) bool {
		if candidateList[i].Score == candidateList[j].Score {
			return candidateList[i].DocumentID < candidateList[j].DocumentID
		}
		return candidateList[i].Score > candidateList[j].Score
	})

	var contextBuilder strings.Builder
	contextBuilder.WriteString("<memory>\n")

	processedDocIDs := make(map[string]bool)
	lastEmittedUser := ""
	addedDocs := 0

	for _, cand := range candidateList {
		if addedDocs >= nResults {
			break
		}

		docID := cand.Metadata["document_id"]
		if docID == "" {
			r.logger.Warn("Document is missing a document_id, skipping")
			continue
		}

		lookupID := resolveLookupID(cand.Metadata)
		if lookupID == "" {
			r.logger.Warn("Unable to resolve lookup identifier for document", zap.String("document_id", docID))
			continue
		}

		if processedDocIDs[lookupID] {
			continue
		}

		content, cached := docContents[lookupID]
		if !cached {
			docUUID, err := uuid.Parse(lookupID)
			if err != nil {
				r.logger.Warn("Invalid document identifier stored in metadata",
					zap.String("document_id", docID),
					zap.String("lookup_id", lookupID),
					zap.Error(err))
				continue
			}

			content, err = r.store.GetRAGDocumentContent(ctx, docUUID)
			if err != nil {
				if errors.Is(err, sql.ErrNoRows) {
					r.logger.Warn("No stored content found for document",
						zap.String("document_id", docID),
						zap.String("lookup_id", lookupID))
				} else {
					r.logger.Warn("Failed to load RAG document content",
						zap.String("document_id", docID),
						zap.String("lookup_id", lookupID),
						zap.Error(err))
				}
				continue
			}
			docContents[lookupID] = content
		}

		role := resolveRole(cand.Metadata)

		if role == "fact" {
			var fact factStoredContent
			if err := json.Unmarshal([]byte(content), &fact); err == nil && (fact.User != "" || fact.Assistant != "" || fact.Tool != "") {
				userTrimmed := canonicalizeFactText(fact.User)
				if userTrimmed != "" && userTrimmed != lastEmittedUser {
					contextBuilder.WriteString(fmt.Sprintf("- user: %s\n", userTrimmed))
					lastEmittedUser = userTrimmed
				}
				if fact.Assistant != "" {
					contextBuilder.WriteString(fmt.Sprintf("- assistant: %s\n", canonicalizeFactText(fact.Assistant)))
				}
				if fact.Tool != "" {
					contextBuilder.WriteString(fmt.Sprintf("- tool: %s\n", canonicalizeFactText(fact.Tool)))
				}

				processedDocIDs[lookupID] = true
				addedDocs++
				continue
			}

			assistantContent := canonicalizeFactText(content)
			if assistantContent != "" {
				contextBuilder.WriteString(fmt.Sprintf("- assistant: %s\n", assistantContent))
			}
		} else {
			contextBuilder.WriteString(fmt.Sprintf("- %s: %s\n", role, content))
		}

		processedDocIDs[lookupID] = true
		addedDocs++
	}

	contextBuilder.WriteString("</memory>\n")
	return contextBuilder.String(), nil
}

func ensureCandidate(candidates map[string]*hybridCandidate, docID string, metadata map[string]string) *hybridCandidate {
	if cand, ok := candidates[docID]; ok {
		if cand.Metadata == nil {
			cand.Metadata = make(map[string]string)
		}
		for k, v := range metadata {
			if v == "" {
				continue
			}
			if existing, exists := cand.Metadata[k]; !exists || existing == "" {
				cand.Metadata[k] = v
			}
		}
		return cand
	}

	metaCopy := cloneStringMap(metadata)
	cand := &hybridCandidate{
		DocumentID: docID,
		Metadata:   metaCopy,
	}
	candidates[docID] = cand
	return cand
}

func cloneStringMap(src map[string]string) map[string]string {
	if len(src) == 0 {
		return make(map[string]string)
	}
	dst := make(map[string]string, len(src))
	for k, v := range src {
		dst[k] = v
	}
	return dst
}

func resolveLookupID(metadata map[string]string) string {
	if metadata == nil {
		return ""
	}
	docID := metadata["document_id"]
	lookupID := docID
	if docType, ok := metadata["type"]; ok && (docType == "summary" || docType == "chunk") {
		if parentID := metadata["parent_document_id"]; parentID != "" {
			lookupID = parentID
		}
	}
	return lookupID
}

func resolveRole(metadata map[string]string) string {
	if metadata == nil {
		return ""
	}
	if role := metadata["role"]; role != "" {
		return role
	}
	return metadata["parent_document_role"]
}

// DeleteSessionDocuments removes in-memory documents associated with a session from the vector store.
func (r *RAG) DeleteSessionDocuments(sessionID string) error {
	collection := r.db.GetCollection("long-term-memory", r.embedder)
	if collection == nil {
		return fmt.Errorf("long-term memory collection not found")
	}

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	if err := collection.Delete(ctx, map[string]string{"session_id": sessionID}, nil); err != nil {
		return fmt.Errorf("failed to delete session documents from collection: %w", err)
	}

	r.logger.Debug("Removed session documents from RAG collection", zap.String("session_id", sessionID))
	return nil
}

// SummarizeLongTermMemory takes a large context string and condenses it.
func (r *RAG) SummarizeLongTermMemory(ctx context.Context, context, latestUserMessage string) (string, error) {
	latestUserMessage = strings.TrimSpace(latestUserMessage)

	systemPrompt := `You are a technical summarization expert specializing in data analysis and statistics.

Your task: Extract key facts from conversation history that are relevant to the user's current question.

CRITICAL RULES:
1. Focus on DATA FINDINGS, not process descriptions
2. ALWAYS preserve: numbers, statistical measures, column names, file names, variable names
3. Extract MULTIPLE facts if relevant (1-3 facts maximum)
4. Each fact should be ONE concise sentence
5. Start each fact with "Fact:" on its own line
6. Ignore: instructions, system messages, casual chat, failed attempts
7. If no relevant facts exist, output: "Fact: No relevant prior analysis found."

WHAT TO EXTRACT:
✅ Statistical results (correlations, p-values, test results, effect sizes)
✅ Data characteristics (sample size, distributions, outliers)
✅ Analysis decisions (which test used, which variables, transformations applied)
✅ Key findings (patterns discovered, significant differences, trends)
✅ File/dataset information (which data was analyzed)

❌ DON'T EXTRACT:
- How-to instructions or explanations
- Error messages or debugging steps
- Casual conversation or greetings
- Process descriptions ("I used pandas to...")

EXAMPLES:

Example 1:
Memory: "I ran a correlation analysis on customer_data.csv using Pearson method. The correlation between age and purchase_frequency was r=0.67 with p<0.001, indicating a strong positive relationship."
User question: "What was the correlation in my customer analysis?"
Output:
Fact: Correlation analysis on customer_data.csv found strong positive relationship between age and purchase_frequency (Pearson r=0.67, p<0.001).

Example 2:
Memory: "Let me help you analyze that. First, I loaded the data. Then I checked for missing values - found 23 missing values in the income column. I dropped those rows and proceeded with the t-test. Independent samples t-test comparing male vs female salaries showed t=2.34, df=198, p=0.021, Cohen's d=0.33."
User question: "What were the results of the gender salary comparison?"
Output:
Fact: Dataset had 23 missing values in income column which were removed.
Fact: Independent t-test found statistically significant salary difference between genders (t=2.34, df=198, p=0.021) with small effect size (d=0.33).

Example 3:
Memory: "Hi! How can I help you today? What kind of analysis would you like to perform?"
User question: "What did we discuss about regression?"
Output:
Fact: No relevant prior analysis found.

Example 4:
Memory: "The dataset Q3_2024_sales.csv contains 1,247 records across 8 columns. Applied log transformation to sales_amount due to right skew. Linear regression predicting sales from advertising_spend yielded R²=0.42, β=1.23 (SE=0.15), p<0.001."
User question: "Tell me about the regression model"
Output:
Fact: Q3_2024_sales.csv dataset contains 1,247 records with sales_amount log-transformed due to skewness.
Fact: Linear regression of sales on advertising_spend shows moderate fit (R²=0.42) with significant positive effect (β=1.23, SE=0.15, p<0.001).

Remember: Be SPECIFIC. Include actual numbers, variable names, and technical details. Vague summaries are useless.`

	if latestUserMessage == "" {
		latestUserMessage = "(no specific question provided)"
	}

	userPrompt := fmt.Sprintf(`User's current question:
"%s"

Conversation history to extract from:
%s

Extract relevant facts following the rules and examples above:`, latestUserMessage, context)

	messages := []types.AgentMessage{
		{Role: "system", Content: systemPrompt},
		{Role: "user", Content: userPrompt},
	}

	// Non-streaming summarization
	summary, err := llmclient.New(r.cfg, r.logger).Chat(ctx, r.cfg.SummarizationLLMHost, messages)
	if err != nil {
		return "", fmt.Errorf("llm chat call failed for memory summary: %w", err)
	}

	summary = strings.TrimSpace(summary)
	if summary == "" {
		return "", fmt.Errorf("llm returned an empty summary for memory")
	}

	// Post-process: ensure it starts with "Fact:"
	if !strings.HasPrefix(summary, "Fact:") {
		r.logger.Warn("Summary didn't start with 'Fact:', attempting to fix",
			zap.String("summary", summary))

		// Try to salvage it by prepending "Fact:"
		summary = "Fact: " + summary
	}

	// Wrap the summary in memory tags
	return fmt.Sprintf("<memory>\n%s\n</memory>", summary), nil
}

func (r *RAG) generateFactSummary(ctx context.Context, code, result string) (string, error) {
	finalResult := result
	if strings.Contains(result, "Error:") {
		finalResult = compressMiddle(result, 800, 200, 200)
	}

	// System prompt defines the expert persona and the detailed summarization guidelines.
	systemPrompt := `You are an expert at extracting statistical facts from code execution results. Your task is to create searchable, information-dense summaries that preserve methodological details and numerical results. Focus on what was done, what was found, and what it means statistically.

Extract a statistical fact from the following code and output. Follow these rules:

RULES:
1. Start with "Fact:"
2. Maximum 200 words (be concise but complete)
3. Include specific names (test names, variable names, column names)
4. Preserve key numbers (p-values, effect sizes, R², coefficients, sample sizes)
5. State statistical conclusions when present (e.g., "significant at α=0.05", "violates normality assumption")
6. If multiple steps, use 2-3 sentences maximum
7. For errors, state what failed and why

WHAT TO CAPTURE:
- Statistical test names (Shapiro-Wilk, t-test, ANOVA, etc.)
- Variables/columns analyzed
- Key parameters (significance levels, degrees of freedom)
- Numerical results with context
- Data characteristics (sample size, distributions, missing values)
- Transformations or preprocessing applied
- Assumption check results
- Model performance metrics

EXAMPLES:

Example 1 - Normality Test:
Code: from scipy import stats; stat, p = stats.shapiro(df['residuals']); print(f"Shapiro-Wilk: W={stat:.4f}, p={p:.4f}")
Output: Shapiro-Wilk: W=0.9234, p=0.0156
Good Fact: Fact: Shapiro-Wilk normality test on residuals yielded W=0.9234, p=0.0156, indicating violation of normality assumption at α=0.05.
Bad Fact: Fact: A normality test was performed on the data.

Example 2 - Descriptive Statistics:
Code: print(df[['age', 'income', 'score']].describe())
Output: age: count=150, mean=34.23, std=8.91, min=18, max=65; income: mean=52340.12, std=12450.67; score: mean=78.45, std=12.34
Good Fact: Fact: Dataset contains 150 observations with variables age (M=34.23, SD=8.91, range 18-65), income (M=52340.12, SD=12450.67), and score (M=78.45, SD=12.34).
Bad Fact: Fact: Descriptive statistics were calculated for the dataframe.

Example 3 - Regression Model:
Code: model = LinearRegression(); model.fit(X_train, y_train); r2 = model.score(X_test, y_test); print(f"R²={r2:.3f}, Coefficients: {model.coef_}")
Output: R²=0.734, Coefficients: [2.34, -1.56, 0.89]
Good Fact: Fact: Linear regression model trained with R²=0.734 on test set, yielding coefficients [2.34, -1.56, 0.89] for predictor variables.
Bad Fact: Fact: A regression model was fitted to the training data.

Example 4 - Data Transformation:
Code: df['log_income'] = np.log(df['income'])
Output: Success: Code executed with no output.
Good Fact: Fact: Created log-transformed variable log_income from income column for normalization.
Bad Fact: Fact: A transformation was applied to the income variable.

Now extract the fact from this code and output:

Code:
{code}

Output:
{output}

Respond with only the fact, starting with "Fact:"`

	// User prompt injects the specific code and output into the template.
	userPrompt := fmt.Sprintf(`Code:
%s

Output:
%s
`, code, finalResult)

	messages := []types.AgentMessage{
		{Role: "system", Content: systemPrompt},
		{Role: "user", Content: userPrompt},
	}

	summary, err := llmclient.New(r.cfg, r.logger).Chat(ctx, r.cfg.SummarizationLLMHost, messages)
	if err != nil {
		return "", fmt.Errorf("llm chat call failed for summary: %w", err)
	}
	if summary == "" {
		return "", fmt.Errorf("llm returned an empty summary")
	}
	return strings.TrimSpace(summary), nil
}

// generateSearchableSummary distills a long message into a concise, searchable sentence.
func (r *RAG) generateSearchableSummary(ctx context.Context, content string) (string, error) {
	// This prompt is specifically designed to create summaries that are good for retrieval.
	// It focuses on intent, entities, and actions rather than just summarizing the text.
	systemPrompt := `You are an expert at creating concise, searchable summaries of user messages. Your task is to distill the user's message into a single sentence that captures the core question, action, or intent.`

	userPrompt := fmt.Sprintf(`Create a single-sentence summary of the following user message. Focus on key entities, variable names, and statistical concepts.

**User Message:**
"%s"

**Summary:**
`, content)

	messages := []types.AgentMessage{
		{Role: "system", Content: systemPrompt},
		{Role: "user", Content: userPrompt},
	}

	summary, err := llmclient.New(r.cfg, r.logger).Chat(ctx, r.cfg.SummarizationLLMHost, messages)
	if err != nil {
		return "", fmt.Errorf("llm chat call failed for searchable summary: %w", err)
	}

	if summary == "" {
		return "", fmt.Errorf("llm returned an empty searchable summary")
	}

	return strings.TrimSpace(summary), nil
}

func normalizeForHash(content string) string {
	return strings.TrimSpace(content)
}

func hashContent(content string) string {
	if content == "" {
		return ""
	}
	sum := sha256.Sum256([]byte(content))
	return hex.EncodeToString(sum[:])
}

func compressMiddle(s string, maxLength int, preserveStart int, preserveEnd int) string {
	if len(s) <= maxLength {
		return s
	}
	if preserveStart+preserveEnd >= len(s) {
		return s
	}
	return s[:preserveStart] + "\n\n[... content compressed ...]\n\n" + s[len(s)-preserveEnd:]
}

func createLlamaCppEmbedding(cfg *config.Config, logger *zap.Logger) chromem.EmbeddingFunc {
	client := llmclient.New(cfg, logger)
	return func(ctx context.Context, doc string) ([]float32, error) {
		return client.Embed(ctx, cfg.EmbeddingLLMHost, doc)
	}
}
