package rag

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"regexp"
	"sort"
	"stats-agent/config"
	"strings"
	"time"

	"io"

	"github.com/google/uuid"
	"github.com/ollama/ollama/api"
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
	cfg          *config.Config
	db           *chromem.DB
	messageStore map[string]string
	embedder     chromem.EmbeddingFunc
	logger       *zap.Logger
}
type LlamaCppEmbeddingRequest struct {
	Content string `json:"content"`
}

type LlamaCppEmbeddingResponse []struct {
	Embedding [][]float32 `json:"embedding"`
}
type LlamaCppChatRequest struct {
	Messages []api.Message `json:"messages"`
	Stream   bool          `json:"stream"`
}
type LlamaCppChatResponse struct {
	Choices []struct {
		Message api.Message `json:"message"`
	} `json:"choices"`
}

func New(cfg *config.Config, logger *zap.Logger) (*RAG, error) {
	db := chromem.NewDB()
	embedder := createLlamaCppEmbedding(cfg, logger)
	_, err := db.GetOrCreateCollection("long-term-memory", nil, embedder)
	if err != nil {
		return nil, fmt.Errorf("failed to create initial collection: %w", err)
	}
	rag := &RAG{
		cfg:          cfg,
		db:           db,
		messageStore: make(map[string]string),
		embedder:     embedder,
		logger:       logger,
	}
	return rag, nil
}

func (r *RAG) AddMessagesToStore(ctx context.Context, messages []api.Message) error {
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

		msg := messages[i]
		var contentToEmbed string
		metadata := make(map[string]string)
		documentID := uuid.New().String()

		if msg.Role == "assistant" && i+1 < len(messages) && messages[i+1].Role == "tool" {
			toolMsg := messages[i+1]
			processedIndices[i] = true
			processedIndices[i+1] = true
			fullFactContent := fmt.Sprintf("%s\n<execution_results>\n%s\n</execution_results>", msg.Content, toolMsg.Content)
			r.messageStore[documentID] = fullFactContent
			metadata["role"] = "fact"
			metadata["document_id"] = documentID
			re := regexp.MustCompile(`(?s)<python>(.*)</python>`)
			matches := re.FindStringSubmatch(msg.Content)
			if len(matches) > 1 {
				code := strings.TrimSpace(matches[1])
				result := strings.TrimSpace(toolMsg.Content)
				summary, err := r.generateFactSummary(ctx, code, result)
				if err != nil {
					r.logger.Warn("LLM fact summarization failed, using fallback", zap.Error(err))
					contentToEmbed = "Fact: A code execution event occurred but could not be summarized."
				} else {
					contentToEmbed = summary
				}
			} else {
				contentToEmbed = "Fact: An assistant action with a tool execution occurred."
			}
		} else {
			// This is where we handle regular user/assistant messages
			r.messageStore[documentID] = msg.Content
			contentToEmbed = msg.Content
			metadata["role"] = msg.Role
			metadata["document_id"] = documentID

			if collection.Count() > 0 {
				// Query for the single most similar document
				results, err := collection.Query(ctx, contentToEmbed, 1, nil, nil)
				if err != nil {
					r.logger.Warn("Deduplication query failed, proceeding to add document", zap.Error(err))
				} else if len(results) > 0 && results[0].Similarity > 0.98 {
					r.logger.Info("Skipping duplicate content", zap.Float32("similarity", results[0].Similarity))
					continue // Skip to the next message
				}
			}
			// If the message is long, create an additional summary document for embedding.
			if len(msg.Content) > 500 { // Threshold for what's considered "long"
				summary, err := r.createSearchableSummary(ctx, msg.Content)
				if err != nil {
					r.logger.Warn("Failed to create searchable summary for long message", zap.Error(err))
				} else {
					summaryDoc := chromem.Document{
						ID:      uuid.New().String(),
						Content: summary,
						Metadata: map[string]string{
							"role":        msg.Role,
							"document_id": documentID,
							"type":        "summary",
						},
					}
					documentsToEmbed = append(documentsToEmbed, summaryDoc)
				}
			}
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
			fullContent, contentOk := r.messageStore[docID]
			if !contentOk {
				r.logger.Warn("Could not find full content for document id", zap.String("document_id", docID))
				continue
			}

			role := result.Metadata["role"]
			contextBuilder.WriteString(fmt.Sprintf("- %s: %s\n", role, fullContent))

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

	messages := []api.Message{
		{Role: "system", Content: systemPrompt},
		{Role: "user", Content: userPrompt},
	}

	summary, err := getLLMResponse(ctx, r.cfg.SummarizationLLMHost, messages, r.cfg, r.logger)
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

	systemPrompt := `You are an expert at creating concise, searchable facts.
	Your task is to generate a single, descriptive sentence that captures the key finding, action, 
	or error from the provided code and output.
	CRITICAL RULE: The summary MUST be a single sentence and less than 100 words.
	Include: operation performed, data involved, key finding, and outcome.
	Format: "Fact: [operation] on [data] revealed [finding/outcome]"
	Example: "Fact: Correlation analysis on age and income columns revealed strong positive correlation (r=0.72, p<0.001)"
	Output only the single sentence summary and nothing else.`
	userPrompt := fmt.Sprintf("Code:\n%s\nOutput:\n%s", code, finalResult)

	messages := []api.Message{
		{Role: "system", Content: systemPrompt},
		{Role: "user", Content: userPrompt},
	}

	summary, err := getLLMResponse(ctx, r.cfg.SummarizationLLMHost, messages, r.cfg, r.logger)
	if err != nil {
		return "", fmt.Errorf("llm chat call failed for summary: %w", err)
	}
	if summary == "" {
		return "", fmt.Errorf("llm returned an empty summary")
	}
	return strings.TrimSpace(summary), nil
}

// createSearchableSummary distills a long message into a concise, searchable sentence.
func (r *RAG) createSearchableSummary(ctx context.Context, content string) (string, error) {
	// This prompt is specifically designed to create summaries that are good for retrieval.
	// It focuses on intent, entities, and actions rather than just summarizing the text.
	systemPrompt := `You are an expert at creating concise, searchable summaries. 
Your task is to distill the user's message into a single sentence that captures the core question, action, or intent.
The summary MUST be a single sentence.
Focus on key entities, variable names, and statistical concepts mentioned.
The goal is to create a summary that would be highly relevant if a user later searched for the main topic of the original message.

Example Input:
"Ok, I see the data has an 'age' and 'income' column. Can you first check for missing values in both, then tell me the mean for each, and finally create a scatter plot to see if there's a relationship between them?"

Example Output:
"User wants to check for missing values and calculate the mean for 'age' and 'income', then visualize the relationship with a scatter plot."`

	messages := []api.Message{
		{Role: "system", Content: systemPrompt},
		{Role: "user", Content: content},
	}

	summary, err := getLLMResponse(ctx, r.cfg.SummarizationLLMHost, messages, r.cfg, r.logger)
	if err != nil {
		return "", fmt.Errorf("llm chat call failed for searchable summary: %w", err)
	}

	if summary == "" {
		return "", fmt.Errorf("llm returned an empty searchable summary")
	}

	return strings.TrimSpace(summary), nil
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
	return func(ctx context.Context, doc string) ([]float32, error) {
		reqBody := LlamaCppEmbeddingRequest{
			Content: doc,
		}
		jsonBody, err := json.Marshal(reqBody)
		if err != nil {
			return nil, fmt.Errorf("failed to marshal embedding request body: %w", err)
		}

		url := fmt.Sprintf("%s/v1/embeddings", cfg.EmbeddingLLMHost)
		var resp *http.Response

		for i := 0; i < cfg.MaxRetries; i++ {
			req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(jsonBody))
			if err != nil {
				return nil, fmt.Errorf("failed to create embedding request: %w", err)
			}
			req.Header.Set("Content-Type", "application/json")

			client := &http.Client{}
			resp, err = client.Do(req)
			if err != nil {
				return nil, fmt.Errorf("failed to send embedding request: %w", err)
			}

			if resp.StatusCode != http.StatusServiceUnavailable {
				break
			}

			resp.Body.Close()
			logger.Warn("Embedding model is loading, retrying", zap.Duration("retry_delay", cfg.RetryDelaySeconds))
			time.Sleep(cfg.RetryDelaySeconds)
		}
		defer resp.Body.Close()

		bodyBytes, err := io.ReadAll(resp.Body)
		if err != nil {
			return nil, fmt.Errorf("failed to read embedding response body: %w", err)
		}

		if resp.StatusCode != http.StatusOK {
			return nil, fmt.Errorf("llama.cpp server returned non-200 status for embedding: %s, body: %s", resp.Status, string(bodyBytes))
		}

		var embeddingResponse LlamaCppEmbeddingResponse
		if err := json.Unmarshal(bodyBytes, &embeddingResponse); err != nil {
			logger.Error("Failed to decode embedding response body", zap.String("raw_response", string(bodyBytes)), zap.Error(err))
			return nil, fmt.Errorf("failed to decode embedding response body: %w", err)
		}

		if len(embeddingResponse) > 0 && len(embeddingResponse[0].Embedding) > 0 {
			return embeddingResponse[0].Embedding[0], nil
		}

		return nil, fmt.Errorf("embedding response was empty")
	}
}

func getLLMResponse(ctx context.Context, llamaCppHost string, messages []api.Message, cfg *config.Config, logger *zap.Logger) (string, error) {
	reqBody := LlamaCppChatRequest{
		Messages: messages,
		Stream:   false,
	}

	jsonBody, err := json.Marshal(reqBody)
	if err != nil {
		return "", fmt.Errorf("failed to marshal request body: %w", err)
	}

	url := fmt.Sprintf("%s/v1/chat/completions", llamaCppHost)
	var resp *http.Response

	for i := 0; i < cfg.MaxRetries; i++ {
		req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(jsonBody))
		if err != nil {
			return "", fmt.Errorf("failed to create request: %w", err)
		}
		req.Header.Set("Content-Type", "application/json")

		client := &http.Client{}
		resp, err = client.Do(req)
		if err != nil {
			return "", fmt.Errorf("failed to send request to llama.cpp server: %w", err)
		}

		if resp.StatusCode != http.StatusServiceUnavailable {
			break
		}

		resp.Body.Close()
		logger.Warn("LLM is loading, retrying", zap.Duration("retry_delay", cfg.RetryDelaySeconds))
		time.Sleep(cfg.RetryDelaySeconds)
	}
	defer resp.Body.Close()

	bodyBytes, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", fmt.Errorf("failed to read llm response body: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("llama.cpp server returned non-200 status: %s, body: %s", resp.Status, string(bodyBytes))
	}

	var chatResponse LlamaCppChatResponse
	if err := json.Unmarshal(bodyBytes, &chatResponse); err != nil {
		return "", fmt.Errorf("failed to decode response body: %w", err)
	}

	if len(chatResponse.Choices) > 0 {
		return chatResponse.Choices[0].Message.Content, nil
	}

	return "", fmt.Errorf("no response choices from llama.cpp server")
}
