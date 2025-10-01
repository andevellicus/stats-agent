package rag

import (
	"context"
	"crypto/sha256"
	"database/sql"
	"encoding/hex"
	"errors"
	"fmt"
	"regexp"
	"sort"
	"stats-agent/config"
	"stats-agent/database"
	"stats-agent/llmclient"
	"stats-agent/web/types"
	"strings"

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

func (r *RAG) AddMessagesToStore(ctx context.Context, sessionID string, messages []types.AgentMessage) error {
	collection := r.db.GetCollection("long-term-memory", r.embedder)
	if collection == nil {
		return fmt.Errorf("long-term memory collection not found")
	}

	var documentsToEmbed []chromem.Document
	processedIndices := make(map[int]bool)

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
			fullFactContent := fmt.Sprintf("%s\n<execution_results>\n%s\n</execution_results>", message.Content, toolMessage.Content)
			storedContent = fullFactContent
			metadata["role"] = "fact"

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
					contentToEmbed = summary
				}
			} else {
				contentToEmbed = "Fact: An assistant action with a tool execution occurred."
			}
		} else {
			storedContent = message.Content
			metadata["role"] = message.Role
			contentToEmbed = message.Content

			if collection.Count() > 0 {
				results, err := collection.Query(ctx, contentToEmbed, 1, nil, nil)
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
				summaryMetadata := map[string]string{
					"role":        message.Role,
					"document_id": documentID,
					"type":        "summary",
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

		if err := r.store.UpsertRAGDocument(ctx, documentUUID, storedContent, metadata, contentHash); err != nil {
			r.logger.Warn("Failed to persist RAG document", zap.Error(err), zap.String("document_id", documentID))
		}

		if len(contentToEmbed) > maxEmbeddingChars {
			r.logger.Info("Chunking oversized message for embedding",
				zap.String("role", metadata["role"]),
				zap.Int("length", len(contentToEmbed)))
			for j := 0; j < len(contentToEmbed); j += maxEmbeddingChars {
				end := min(j+maxEmbeddingChars, len(contentToEmbed))
				chunkContent := contentToEmbed[j:end]
				if int(float64(len(chunkContent))/3.5) > maxEmbeddingTokens {
					end = j + (maxEmbeddingChars * 3 / 4)
					chunkContent = contentToEmbed[j:end]
				}
				chunkDoc := chromem.Document{
					ID:       uuid.New().String(),
					Content:  chunkContent,
					Metadata: metadata,
				}
				documentsToEmbed = append(documentsToEmbed, chunkDoc)
			}
		} else {
			doc := chromem.Document{
				ID:       uuid.New().String(),
				Content:  contentToEmbed,
				Metadata: metadata,
			}
			documentsToEmbed = append(documentsToEmbed, doc)
		}

		if summaryDoc != nil {
			documentsToEmbed = append(documentsToEmbed, *summaryDoc)
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

func (r *RAG) Query(ctx context.Context, query string, nResults int) (string, error) {
	collection := r.db.GetCollection("long-term-memory", r.embedder)
	if collection == nil {
		return "", fmt.Errorf("failed to get collection: collection not found")
	}
	if collection.Count() == 0 {
		return "", nil
	}

	candidateCount := min(nResults*5, collection.Count())
	results, err := collection.Query(ctx, query, candidateCount, nil, nil)
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
			scoreI *= 1.5 // Higher boost for high-level summaries
		}
		if results[j].Metadata["type"] == "summary" {
			scoreJ *= 1.5
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

		if !processedDocIDs[docID] {
			content, cached := docContents[docID]
			if !cached {
				docUUID, err := uuid.Parse(docID)
				if err != nil {
					r.logger.Warn("Invalid document_id stored in metadata", zap.String("document_id", docID), zap.Error(err))
					continue
				}

				content, err = r.store.GetRAGDocumentContent(ctx, docUUID)
				if err != nil {
					if errors.Is(err, sql.ErrNoRows) {
						r.logger.Warn("No stored content found for document", zap.String("document_id", docID))
					} else {
						r.logger.Warn("Failed to load RAG document content", zap.String("document_id", docID), zap.Error(err))
					}
					continue
				}
				docContents[docID] = content
			}

			role := result.Metadata["role"]
			contextBuilder.WriteString(fmt.Sprintf("- %s: %s\n", role, content))

			processedDocIDs[docID] = true
			addedDocs++
		}
	}

	contextBuilder.WriteString("</memory>\n")
	return contextBuilder.String(), nil
}

// SummarizeLongTermMemory takes a large context string and condenses it.
func (r *RAG) SummarizeLongTermMemory(ctx context.Context, context string) (string, error) {
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
	userPrompt := fmt.Sprintf("History to summarize:\n%s", context)

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
