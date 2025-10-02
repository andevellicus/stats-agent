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

// Embedding request/response types moved to llmclient

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

	var documentsToEmbed []chromem.Document
	processedIndices := make(map[int]bool)
	var sessionFilter map[string]string
	if sessionID != "" {
		sessionFilter = map[string]string{"session_id": sessionID}
	}
	copyMetadata := func(src map[string]string) map[string]string {
		if src == nil {
			return nil
		}
		cp := make(map[string]string, len(src))
		for k, v := range src {
			cp[k] = v
		}
		return cp
	}

	for i := range messages {
		if processedIndices[i] {
			continue
		}

		message := messages[i]
		documentUUID := uuid.New()
		documentID := documentUUID.String()
		metadata := map[string]string{
			"document_id": documentID,
		}
		if sessionID != "" {
			metadata["session_id"] = sessionID
		}

		var contentToEmbed string
		var storedContent string
		var summaryDoc *chromem.Document

		if message.Role == "assistant" && i+1 < len(messages) && messages[i+1].Role == "tool" {
			toolMessage := messages[i+1]
			processedIndices[i] = true
			processedIndices[i+1] = true
			metadata["role"] = "fact"

			assistantContent := canonicalizeFactText(message.Content)
			toolContent := canonicalizeFactText(stripExecutionResultsWrapper(toolMessage.Content))

			userContent := ""
			for prev := i - 1; prev >= 0; prev-- {
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
					processedIndices[i] = true
					continue
				}
			}

			storedContent = canonicalizeFactText(message.Content)
			metadata["role"] = message.Role
			contentToEmbed = storedContent

			contentToEmbed = canonicalizeFactText(contentToEmbed)
			if collection.Count() > 0 {
				results, err := collection.Query(ctx, contentToEmbed, 1, sessionFilter, nil)
				if err != nil {
					r.logger.Warn("Deduplication query failed, proceeding to add document anyway", zap.Error(err))
				} else if len(results) > 0 && results[0].Similarity > 0.98 && results[0].Metadata["role"] == message.Role {
					r.logger.Debug("Skipping duplicate content", zap.Float32("similarity", results[0].Similarity), zap.String("role", message.Role))
					continue
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
				continue
			}
			if existingDocID != uuid.Nil {
				r.logger.Debug("Skipping duplicate RAG document",
					zap.String("existing_document_id", existingDocID.String()),
					zap.String("session_id", sessionID),
					zap.String("role", role))
				continue
			}
		}

		if role != "fact" && len(message.Content) > 500 {
			summary, err := r.generateSearchableSummary(ctx, message.Content)
			if err != nil {
				r.logger.Warn("Failed to create searchable summary for long message, will use full content",
					zap.Error(err),
					zap.Int("content_length", len(message.Content)))
			} else {
				summaryID := uuid.New()
				summaryMetadata := map[string]string{
					"role":                 message.Role,
					"document_id":          summaryID.String(),
					"type":                 "summary",
					"parent_document_id":   documentID,
					"parent_document_role": role,
				}
				if sessionID != "" {
					summaryMetadata["session_id"] = sessionID
				}
				summaryDoc = &chromem.Document{
					ID:       uuid.New().String(),
					Content:  summary,
					Metadata: summaryMetadata,
				}
			}
		}

		embeddingVector, embedErr := r.embedder(ctx, contentToEmbed)
		if embedErr != nil {
			r.logger.Warn("Failed to create embedding for RAG persistence",
				zap.Error(embedErr),
				zap.String("document_id", documentID))
		}

		if err := r.store.UpsertRAGDocument(ctx, documentUUID, storedContent, contentToEmbed, metadata, contentHash, embeddingVector); err != nil {
			r.logger.Warn("Failed to persist RAG document", zap.Error(err), zap.String("document_id", documentID))
		}

		if len(contentToEmbed) > maxEmbeddingChars {
			r.logger.Info("Chunking oversized message for embedding",
				zap.String("role", metadata["role"]),
				zap.Int("length", len(contentToEmbed)))
			parentDocumentID := metadata["document_id"]
			chunkIndex := 0
			for j := 0; j < len(contentToEmbed); j += maxEmbeddingChars {
				end := min(j+maxEmbeddingChars, len(contentToEmbed))
				chunkContent := contentToEmbed[j:end]
				if int(float64(len(chunkContent))/3.5) > maxEmbeddingTokens {
					end = j + (maxEmbeddingChars * 3 / 4)
					chunkContent = contentToEmbed[j:end]
				}
				chunkDocID := uuid.New()
				chunkMetadata := copyMetadata(metadata)
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
					Metadata: copyMetadata(chunkMetadata),
				}
				if chunkEmbedErr == nil && len(chunkEmbedding) > 0 {
					chunkDoc.Embedding = chunkEmbedding
				}
				documentsToEmbed = append(documentsToEmbed, chunkDoc)
				chunkIndex++
			}
		} else {
			doc := chromem.Document{
				ID:       uuid.New().String(),
				Content:  contentToEmbed,
				Metadata: copyMetadata(metadata),
			}
			if embedErr == nil && len(embeddingVector) > 0 {
				doc.Embedding = embeddingVector
			}
			documentsToEmbed = append(documentsToEmbed, doc)
		}

		if summaryDoc != nil {
			summaryMetadata := copyMetadata(summaryDoc.Metadata)
			summaryDoc.Metadata = summaryMetadata
			summaryIDStr, ok := summaryMetadata["document_id"]
			if !ok {
				r.logger.Warn("Summary document missing document_id, skipping persistence")
			} else {
				summaryID, err := uuid.Parse(summaryIDStr)
				if err != nil {
					r.logger.Warn("Summary document has invalid document_id", zap.String("document_id", summaryIDStr), zap.Error(err))
				} else {
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
					documentsToEmbed = append(documentsToEmbed, *summaryDoc)
				}
			}
		}
	}

	if len(documentsToEmbed) > 0 {
		err := collection.AddDocuments(ctx, documentsToEmbed, 4)
		if err != nil {
			return fmt.Errorf("failed to add documents to collection: %w", err)
		}
		r.logger.Info("Added document chunks to long-term RAG memory", zap.Int("chunks_added", len(documentsToEmbed)))
	}
	return nil
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
	collection := r.db.GetCollection("long-term-memory", r.embedder)
	if collection == nil {
		return "", fmt.Errorf("failed to get collection: collection not found")
	}
	if collection.Count() == 0 {
		return "", nil
	}

	candidateCount := min(nResults*5, collection.Count())
	var where map[string]string
	if sessionID != "" {
		where = map[string]string{"session_id": sessionID}
	}
	results, err := collection.Query(ctx, query, candidateCount, where, nil)
	if err != nil {
		return "", fmt.Errorf("failed to query collection: %w", err)
	}

	sort.Slice(results, func(i, j int) bool {
		scoreI := results[i].Similarity
		scoreJ := results[j].Similarity

		// Boost the score if the document is a fact
		if results[i].Metadata["role"] == "fact" {
			scoreI *= 1.3 // Boost facts
		}
		if results[j].Metadata["role"] == "fact" {
			scoreJ *= 1.3
		}

		// Boost summaries even more
		if results[i].Metadata["type"] == "summary" {
			scoreI *= 1.2 // Higher boost for high-level summaries
		}
		if results[j].Metadata["type"] == "summary" {
			scoreJ *= 1.2
		}

		// Slightly penalize error messages unless the query is specifically about errors
		isQueryForError := strings.Contains(strings.ToLower(query), "error")
		if strings.Contains(results[i].Content, "Error:") && !isQueryForError {
			scoreI *= 0.8 // Penalize
		}
		if strings.Contains(results[j].Content, "Error:") && !isQueryForError {
			scoreJ *= 0.8
		}

		return scoreI > scoreJ
	})

	var contextBuilder strings.Builder
	contextBuilder.WriteString("<memory>\n")

	processedDocIDs := make(map[string]bool)
	docContents := make(map[string]string)
	lastEmittedUser := ""
	addedDocs := 0

	for _, result := range results {
		if addedDocs >= nResults {
			break
		}

		docID, ok := result.Metadata["document_id"]
		if !ok {
			r.logger.Warn("Document is missing a document_id, skipping")
			continue
		}

		lookupID := docID
		if docType, hasType := result.Metadata["type"]; hasType && (docType == "summary" || docType == "chunk") {
			if parentID, hasParent := result.Metadata["parent_document_id"]; hasParent && parentID != "" {
				lookupID = parentID
			}
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

		role := result.Metadata["role"]
		if role == "" {
			if parentRole, hasParentRole := result.Metadata["parent_document_role"]; hasParentRole {
				role = parentRole
			}
		}

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
					contextBuilder.WriteString(fmt.Sprintf("- tool: %s\n", canonicalizeFactText(stripExecutionResultsWrapper(fact.Tool))))
				}

				processedDocIDs[lookupID] = true
				addedDocs++
				continue
			}

			// Fallback for legacy fact payloads
			assistantContent := canonicalizeFactText(content)
			toolContent := ""
			if idx := strings.Index(content, "<execution_results>"); idx != -1 {
				assistantContent = canonicalizeFactText(content[:idx])
				remainder := content[idx+len("<execution_results>"):]
				if endIdx := strings.Index(remainder, "</execution_results>"); endIdx != -1 {
					toolContent = canonicalizeFactText(remainder[:endIdx])
				} else {
					toolContent = canonicalizeFactText(remainder)
				}
			}

			if assistantContent != "" {
				contextBuilder.WriteString(fmt.Sprintf("- assistant: %s\n", canonicalizeFactText(assistantContent)))
			}
			if toolContent != "" {
				contextBuilder.WriteString(fmt.Sprintf("- tool: %s\n", canonicalizeFactText(toolContent)))
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

func stripExecutionResultsWrapper(text string) string {
	cleaned := strings.TrimSpace(text)
	const openTag = "<execution_results>"
	const closeTag = "</execution_results>"

	if strings.HasPrefix(cleaned, openTag) && strings.Contains(cleaned, closeTag) {
		cleaned = strings.TrimPrefix(cleaned, openTag)
		cleaned = strings.TrimSpace(cleaned)
		if idx := strings.LastIndex(cleaned, closeTag); idx != -1 {
			cleaned = cleaned[:idx]
		}
		return strings.TrimSpace(cleaned)
	}

	cleaned = strings.ReplaceAll(cleaned, openTag, "")
	cleaned = strings.ReplaceAll(cleaned, closeTag, "")
	return strings.TrimSpace(cleaned)
}

// SummarizeLongTermMemory takes a large context string and condenses it.
func (r *RAG) SummarizeLongTermMemory(ctx context.Context, context, latestUserMessage string) (string, error) {
	latestUserMessage = strings.TrimSpace(latestUserMessage)
	systemPrompt := `You are an expert at creating concise, searchable facts from code and its output. Your task is to generate a single, descriptive sentence that captures the key finding, action, or error.

	Follow these rules:
	1. The summary MUST be a single sentence.
	2. The summary MUST start with "Fact:".
	3. The summary MUST be less than 100 words.

	Here is an example:

	---
	**Input:**
	Code:
	df.head(3)

	Output:
	age gender  side
	0   55      M  left
	1   60      F  right
	2   65      M  left
	---
	**Your Output:**
	Fact: The dataframe contains columns for age, gender, and side.
	---`
	if latestUserMessage == "" {
		latestUserMessage = "(no new user question provided)"
	}

	userPrompt := fmt.Sprintf(`The user's latest question is:
"%s"

Summarize the following memory so that it highlights information that helps answer that question.

History to summarize:
%s`, latestUserMessage, context)

	messages := []types.AgentMessage{
		{Role: "system", Content: systemPrompt},
		{Role: "user", Content: userPrompt},
	}

	// Non-streaming summarization
	summary, err := llmclient.New(r.cfg, r.logger).Chat(ctx, r.cfg.SummarizationLLMHost, messages)
	if err != nil {
		return "", fmt.Errorf("llm chat call failed for memory summary: %w", err)
	}

	if summary == "" {
		return "", fmt.Errorf("llm returned an empty summary for memory")
	}

	// Wrap the summary in the same tags for consistency
	return fmt.Sprintf("<memory>\n- %s\n</memory>", strings.TrimSpace(summary)), nil
}

func (r *RAG) generateFactSummary(ctx context.Context, code, result string) (string, error) {
	finalResult := result
	if strings.Contains(result, "Error:") {
		finalResult = compressMiddle(result, 800, 200, 200)
	}

	// System prompt defines the expert persona and the core task.
	systemPrompt := `You are an expert at creating concise, searchable facts from code and its output. Your task is to generate a single, descriptive sentence that captures the key finding, action, or error.`

	// User prompt provides the specific rules, an example, and the data to process.
	userPrompt := fmt.Sprintf(`Generate a summary for the following code and output, following these rules:
1. The summary MUST be a single sentence.
2. The summary MUST start with "Fact:".
3. The summary MUST be less than 100 words.

Here is an example:
---
**Input:**
Code:
df.head(3)

Output:
   age gender  side
0   55      M  left
1   60      F  right
2   65      M  left
---
**Your Output:**
Fact: The dataframe contains columns for age, gender, and side.
---

**Input:**
Code:
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
