package rag

import (
	"context"
	"fmt"
	"log"
	"regexp"
	"sort"
	"strings"

	"github.com/google/uuid"
	"github.com/ollama/ollama/api"
	"github.com/philippgille/chromem-go"
)

// RAG struct now includes the messageStore and summarizationModel.
type RAG struct {
	db                 *chromem.DB
	messageStore       map[string]string // The "Document Store" for full FACT content
	ollamaClient       *api.Client
	embedder           chromem.EmbeddingFunc
	embeddingModel     string
	summarizationModel string
}

// createOllamaEmbedding creates an embedding function using the Ollama API.
func createOllamaEmbedding(ollamaClient *api.Client, modelName string) chromem.EmbeddingFunc {
	return func(ctx context.Context, doc string) ([]float32, error) {
		req := &api.EmbeddingRequest{
			Model:  modelName,
			Prompt: doc,
		}
		resp, err := ollamaClient.Embeddings(ctx, req)
		if err != nil {
			return nil, err
		}
		float32Embedding := make([]float32, len(resp.Embedding))
		for i, v := range resp.Embedding {
			float32Embedding[i] = float32(v)
		}
		return float32Embedding, nil
	}
}

// New creates a new RAG instance.
func New(ctx context.Context, ollamaClient *api.Client, embeddingModel string, summarizationModel string) (*RAG, error) {
	db := chromem.NewDB()
	embedder := createOllamaEmbedding(ollamaClient, embeddingModel)
	_, err := db.GetOrCreateCollection("long-term-memory", nil, embedder)
	if err != nil {
		return nil, fmt.Errorf("failed to create initial collection: %w", err)
	}
	rag := &RAG{
		db:                 db,
		messageStore:       make(map[string]string),
		ollamaClient:       ollamaClient,
		embedder:           embedder,
		embeddingModel:     embeddingModel,
		summarizationModel: summarizationModel,
	}
	return rag, nil
}

// generateSummary now correctly uses the Chat method to create a concise fact.
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

	req := &api.ChatRequest{
		Model:    r.summarizationModel,
		Messages: messages,
		Stream:   &[]bool{false}[0],
	}

	var chatResponse api.ChatResponse
	err := r.ollamaClient.Chat(ctx, req, func(resp api.ChatResponse) error {
		chatResponse = resp
		return nil
	})
	if err != nil {
		return "", fmt.Errorf("llm chat call failed for summary: %w", err)
	}

	if chatResponse.Message.Content == "" {
		return "", fmt.Errorf("llm returned an empty summary")
	}

	return strings.TrimSpace(chatResponse.Message.Content), nil
}

// AddMessagesToStore is now fully optimized to avoid redundancy.
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
			// **OPTIMIZATION**: No message_id is needed as we'll use the content directly.
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

// Query now uses the optimized retrieval logic.
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

	var context strings.Builder
	context.WriteString("Relevant information from long-term memory:\n")

	// **OPTIMIZATION**: Use the more efficient retrieval logic.
	for _, result := range results {
		role := result.Metadata["role"]

		if role == "fact" {
			// Facts need the messageStore lookup to get the full content.
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
			context.WriteString(fmt.Sprintf("- %s: %s\n", role, fullContent))
		} else {
			// Non-facts can use the content from the vector DB directly.
			context.WriteString(fmt.Sprintf("- %s: %s\n", role, result.Content))
		}
	}

	return context.String(), nil
}
