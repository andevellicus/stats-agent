package rag

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"log"
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
)

const maxEmbeddingChars = 1500 // A safe character limit based on the model's 512 token limit.

type RAG struct {
	cfg          *config.Config
	db           *chromem.DB
	messageStore map[string]string
	embedder     chromem.EmbeddingFunc
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

func New(cfg *config.Config) (*RAG, error) {
	db := chromem.NewDB()
	embedder := createLlamaCppEmbedding(cfg)
	_, err := db.GetOrCreateCollection("long-term-memory", nil, embedder)
	if err != nil {
		return nil, fmt.Errorf("failed to create initial collection: %w", err)
	}
	rag := &RAG{
		cfg:          cfg,
		db:           db,
		messageStore: make(map[string]string),
		embedder:     embedder,
	}
	return rag, nil
}

func (r *RAG) generateFactSummary(ctx context.Context, code, result string) (string, error) {
	finalResult := result
	if strings.Contains(result, "Error:") {
		finalResult = compressMiddle(result, 4000, 1000, 2000)
	}

	systemPrompt := `You are an expert at creating concise, searchable facts.
	Your task is to generate a single, descriptive sentence that captures the key finding, action, 
	or error from the provided code and output.
	CRITICAL RULE: The summary MUST be a single sentence and less than 100 words.
	Start the sentence with "Fact:". Output only the single sentence summary and nothing else.`
	userPrompt := fmt.Sprintf("Code:\n%s\nOutput:\n%s", code, finalResult)

	messages := []api.Message{
		{Role: "system", Content: systemPrompt},
		{Role: "user", Content: userPrompt},
	}

	summary, err := getLLMResponse(ctx, r.cfg.SummarizationLLMHost, messages, r.cfg)
	if err != nil {
		return "", fmt.Errorf("llm chat call failed for summary: %w", err)
	}
	if summary == "" {
		return "", fmt.Errorf("llm returned an empty summary")
	}
	return strings.TrimSpace(summary), nil
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
		documentID := uuid.New().String() // Each message gets a unique parent ID

		if msg.Role == "assistant" && i+1 < len(messages) && messages[i+1].Role == "tool" {
			toolMsg := messages[i+1]
			processedIndices[i] = true
			processedIndices[i+1] = true

			fullFactContent := fmt.Sprintf("%s\n<execution_results>\n%s\n</execution_results>", msg.Content, toolMsg.Content)
			r.messageStore[documentID] = fullFactContent // Store the full fact
			metadata["role"] = "fact"
			metadata["document_id"] = documentID

			re := regexp.MustCompile(`(?s)<python>(.*)</python>`)
			matches := re.FindStringSubmatch(msg.Content)
			if len(matches) > 1 {
				code := strings.TrimSpace(matches[1])
				result := strings.TrimSpace(toolMsg.Content)
				summary, err := r.generateFactSummary(ctx, code, result)
				if err != nil {
					log.Printf("Warning: LLM fact summarization failed, using fallback. Error: %v", err)
					contentToEmbed = "Fact: A code execution event occurred but could not be summarized."
				} else {
					contentToEmbed = summary
				}
			} else {
				contentToEmbed = "Fact: An assistant action with a tool execution occurred."
			}
		} else {
			r.messageStore[documentID] = msg.Content // Store the full user/assistant message
			contentToEmbed = msg.Content
			metadata["role"] = msg.Role
			metadata["document_id"] = documentID
		}

		// Chunking is based on the final content we intend to embed.
		if len(contentToEmbed) > maxEmbeddingChars {
			log.Printf("--- Chunking oversized '%s' message for embedding. ---", metadata["role"])
			for j := 0; j < len(contentToEmbed); j += maxEmbeddingChars {
				end := min(j+maxEmbeddingChars, len(contentToEmbed))
				chunkContent := contentToEmbed[j:end]

				chunkDoc := chromem.Document{
					ID:       uuid.New().String(),
					Content:  chunkContent,
					Metadata: metadata,
				}
				documentsToEmbed = append(documentsToEmbed, chunkDoc)
			}
		} else {
			// If it's not oversized (which includes all summarized facts), add it as a single document.
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
		log.Printf("--- Added %d document chunks to long-term RAG memory. ---", len(documentsToEmbed))
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
			scoreI *= 1.4
		}
		if results[j].Metadata["role"] == "fact" {
			scoreJ *= 1.4
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
			log.Printf("Warning: document is missing a document_id, skipping.")
			continue
		}

		if !processedDocIDs[docID] {
			fullContent, contentOk := r.messageStore[docID]
			if !contentOk {
				log.Printf("Warning: could not find full content for document id %s.", docID)
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
	systemPrompt := `You are an expert at summarizing conversational and analytical history.
	The following text contains a series of facts and conversational turns.
	Condense this information into a concise summary that captures the key findings, actions, and unanswered questions.
	Focus on retaining the most critical information that would be needed to continue the analysis.`
	userPrompt := fmt.Sprintf("History to summarize:\n%s", context)

	messages := []api.Message{
		{Role: "system", Content: systemPrompt},
		{Role: "user", Content: userPrompt},
	}

	summary, err := getLLMResponse(ctx, r.cfg.SummarizationLLMHost, messages, r.cfg)
	if err != nil {
		return "", fmt.Errorf("llm chat call failed for memory summary: %w", err)
	}

	if summary == "" {
		return "", fmt.Errorf("llm returned an empty summary for memory")
	}

	// Wrap the summary in the same tags for consistency
	return fmt.Sprintf("<memory>\n- %s\n</memory>", strings.TrimSpace(summary)), nil
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

func createLlamaCppEmbedding(cfg *config.Config) chromem.EmbeddingFunc {
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
			log.Printf("Embedding model is loading, retrying in %v...", cfg.RetryDelaySeconds)
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
			log.Printf("Failed to decode embedding response body. Raw response: %s", string(bodyBytes))
			return nil, fmt.Errorf("failed to decode embedding response body: %w", err)
		}

		if len(embeddingResponse) > 0 && len(embeddingResponse[0].Embedding) > 0 {
			return embeddingResponse[0].Embedding[0], nil
		}

		return nil, fmt.Errorf("embedding response was empty")
	}
}

func getLLMResponse(ctx context.Context, llamaCppHost string, messages []api.Message, cfg *config.Config) (string, error) {
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
		log.Printf("LLM is loading, retrying in %v...", cfg.RetryDelaySeconds)
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
