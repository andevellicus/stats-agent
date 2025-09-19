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
	messageStore       map[string]string // The "Document Store" for full content
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

// AddMessagesToStore now implements the pointer-based system.
func (r *RAG) AddMessagesToStore(ctx context.Context, messages []api.Message) error {
	collection := r.db.GetCollection("long-term-memory", r.embedder)
	if collection == nil {
		return fmt.Errorf("long-term memory collection not found")
	}

	var documentsToEmbed []chromem.Document
	for i := range messages {
		msg := messages[i]
		messageID := uuid.New().String()
		fullContent := ""
		roleForMeta := msg.Role

		// Store the full content of the message exchange for tool calls
		if i > 0 && msg.Role == "tool" && messages[i-1].Role == "assistant" {
			prevMsg := messages[i-1]
			fullContent = fmt.Sprintf("%s\n<execution_results>\n%s\n</execution_results>", prevMsg.Content, msg.Content)
			roleForMeta = "fact" // Mark this exchange as a "fact"
		} else {
			fullContent = msg.Content
		}

		// Save the full, original content to our simple document store.
		r.messageStore[messageID] = fullContent

		var contentToEmbed string
		// If it's a successful tool execution, generate a summary for embedding.
		if roleForMeta == "fact" && !strings.Contains(msg.Content, "Error:") {
			prevMsg := messages[i-1]
			re := regexp.MustCompile(`(?s)<python>(.*)</python>`)
			matches := re.FindStringSubmatch(prevMsg.Content)
			if len(matches) > 1 {
				code := strings.TrimSpace(matches[1])
				result := strings.TrimSpace(msg.Content)
				summary, err := r.generateSummary(ctx, code, result)
				if err != nil {
					log.Printf("Warning: could not generate summary for fact, embedding full content instead. Error: %v", err)
					contentToEmbed = fullContent // Fallback to full content on error
				} else {
					contentToEmbed = summary
				}
			}
		} else {
			// For regular messages or errors, embed the full content directly.
			contentToEmbed = fullContent
		}

		documentsToEmbed = append(documentsToEmbed, chromem.Document{
			ID:      uuid.New().String(),
			Content: contentToEmbed,
			Metadata: map[string]string{
				"role":       roleForMeta,
				"message_id": messageID, // The "pointer" to the full content
			},
		})
	}

	if len(documentsToEmbed) == 0 {
		return nil
	}
	err := collection.AddDocuments(ctx, documentsToEmbed, 4)
	if err != nil {
		return fmt.Errorf("failed to add documents to collection: %w", err)
	}
	log.Printf("--- Added %d pointers to long-term RAG memory. ---", len(documentsToEmbed))
	return nil
}

// Query now performs two-stage retrieval with nuanced re-ranking.
func (r *RAG) Query(ctx context.Context, query string, nResults int) (string, error) {
	collection := r.db.GetCollection("long-term-memory", r.embedder)
	if collection == nil {
		return "", fmt.Errorf("failed to get collection: collection not found")
	}
	if collection.Count() == 0 {
		return "", nil
	}

	// Stage 1: Query the vector DB to get the most relevant concise documents.
	// We fetch more candidates than we need to give the re-ranking step more to work with.
	candidateCount := min(nResults*2, collection.Count())
	results, err := collection.Query(ctx, query, candidateCount, nil, nil)
	if err != nil {
		return "", fmt.Errorf("failed to query collection: %w", err)
	}

	// --- NEW: Nuanced Re-ranking Logic ---
	sort.Slice(results, func(i, j int) bool {
		scoreI := results[i].Similarity
		scoreJ := results[j].Similarity

		// Apply a weighted boost for "facts" to favor them in cases of similar scores.
		if results[i].Metadata["role"] == "fact" {
			scoreI *= 1.3 // A 30% boost
		}
		if results[j].Metadata["role"] == "fact" {
			scoreJ *= 1.3
		}

		return scoreI > scoreJ
	})

	// Trim the results down to the desired number after re-ranking.
	if len(results) > nResults {
		results = results[:nResults]
	}

	var context strings.Builder
	context.WriteString("Relevant information from long-term memory:\n")

	// Stage 2: Use the pointers from the re-ranked docs to fetch the full content.
	for _, result := range results {
		messageID, ok := result.Metadata["message_id"]
		if !ok {
			log.Printf("Warning: found a document in RAG without a message_id pointer.")
			continue
		}

		fullContent, ok := r.messageStore[messageID]
		if !ok {
			log.Printf("Warning: could not find full message content for id %s.", messageID)
			continue
		}

		role := result.Metadata["role"]
		context.WriteString(fmt.Sprintf("- %s: %s\n", role, fullContent))
	}

	return context.String(), nil
}
