package rag

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io/ioutil"
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

type RAG struct {
	cfg          *config.Config
	db           *chromem.DB
	messageStore map[string]string
	embedder     chromem.EmbeddingFunc
}
type LlamaCppEmbeddingRequest struct {
	Content string `json:"content"`
}

// Corrected struct to handle the nested array response from the server
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

// createLlamaCppEmbedding now correctly handles the nested array
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
func (r *RAG) generateSummary(ctx context.Context, code, result string) (string, error) {
	systemPrompt := `You are an expert at creating concise, searchable facts. Based on the Python code and its output,
	generate a single, descriptive sentence that captures the key finding or action.
	Start the sentence with "Fact:". Output only the single sentence summary and nothing else.
	Focus on statistical results, data characteristics, and analytical findings. Include specific numbers when relevant.`
	userPrompt := fmt.Sprintf(`
Code:
%s
Output:
%s
`, code, result)

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

	bodyBytes, err := ioutil.ReadAll(resp.Body)
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
func (r *RAG) AddMessagesToStore(ctx context.Context, messages []api.Message) error {
	collection := r.db.GetCollection("long-term-memory", r.embedder)
	if collection == nil {
		return fmt.Errorf("long-term memory collection not found")
	}

	var documentsToEmbed []chromem.Document
	processedIndices := make(map[int]bool) // Keep track of messages already handled

	for i := range messages {
		if processedIndices[i] {
			continue // Skip messages that have already been processed as part of a pair
		}

		msg := messages[i]
		doc := chromem.Document{
			ID:       uuid.New().String(),
			Metadata: make(map[string]string),
		}

		// Look ahead to see if the current message is an assistant message followed by a tool message
		if msg.Role == "assistant" && i+1 < len(messages) && messages[i+1].Role == "tool" {
			toolMsg := messages[i+1]

			// Mark both messages as processed so they are not handled individually
			processedIndices[i] = true
			processedIndices[i+1] = true

			// Combine them into a single "fact"
			fullFactContent := fmt.Sprintf("%s\n<execution_results>\n%s\n</execution_results>", msg.Content, toolMsg.Content)
			messageID := uuid.New().String()

			// **OPTIMIZATION**: Only add the full fact to the messageStore.
			r.messageStore[messageID] = fullFactContent
			doc.Metadata["role"] = "fact"
			doc.Metadata["message_id"] = messageID

			// Generate a summary for the fact if it's not an error
			if !strings.Contains(toolMsg.Content, "Error:") {
				re := regexp.MustCompile(`(?s)<python>(.*)</python>`)
				matches := re.FindStringSubmatch(msg.Content)
				if len(matches) > 1 {
					code := strings.TrimSpace(matches[1])
					result := strings.TrimSpace(toolMsg.Content)
					summary, err := r.generateSummary(ctx, code, result)
					if err != nil {
						log.Printf("Warning: could not generate summary for fact, embedding full content. Error: %v", err)
						doc.Content = fullFactContent
					} else {
						doc.Content = summary
					}
				}
			} else {
				// For errors, embed the full content directly
				doc.Content = fullFactContent
			}
		} else {
			// This handles user messages and standalone assistant messages (that don't call tools)
			doc.Content = msg.Content
			doc.Metadata["role"] = msg.Role
			// No message_id is needed as we'll use the content directly.
		}

		documentsToEmbed = append(documentsToEmbed, doc)
	}

	if len(documentsToEmbed) > 0 {
		err := collection.AddDocuments(ctx, documentsToEmbed, 4)
		if err != nil {
			return fmt.Errorf("failed to add documents to collection: %w", err)
		}
		log.Printf("--- Added %d messages/facts to long-term RAG memory. ---", len(documentsToEmbed))
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

	candidateCount := min(nResults*2, collection.Count())
	results, err := collection.Query(ctx, query, candidateCount, nil, nil)
	if err != nil {
		return "", fmt.Errorf("failed to query collection: %w", err)
	}

	sort.Slice(results, func(i, j int) bool {
		scoreI := results[i].Similarity
		scoreJ := results[j].Similarity
		if results[i].Metadata["role"] == "fact" {
			scoreI *= 1.3 // Using a 1.3 boost
		}
		if results[j].Metadata["role"] == "fact" {
			scoreJ *= 1.3
		}
		return scoreI > scoreJ
	})

	if len(results) > nResults {
		results = results[:nResults]
	}

	var contextBuilder strings.Builder
	contextBuilder.WriteString("<memory>\n") // Opening tag

	for _, result := range results {
		role := result.Metadata["role"]

		if role == "fact" {
			messageID, ok := result.Metadata["message_id"]
			if !ok {
				log.Printf("Warning: fact document is missing a message_id.")
				continue
			}
			fullContent, ok := r.messageStore[messageID]
			if !ok {
				log.Printf("Warning: could not find full fact content for id %s.", messageID)
				continue
			}
			contextBuilder.WriteString(fmt.Sprintf("- %s: %s\n", role, fullContent))
		} else {
			contextBuilder.WriteString(fmt.Sprintf("- %s: %s\n", role, result.Content))
		}
	}

	contextBuilder.WriteString("</memory>\n") // Closing tag
	return contextBuilder.String(), nil
}
