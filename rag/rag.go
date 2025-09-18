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

// RAG struct holds the state for our RAG implementation
type RAG struct {
	db           *chromem.DB
	ollamaClient *api.Client
	embedder     chromem.EmbeddingFunc
	model        string
}

// createOllamaEmbedding is a custom implementation of chromem.EmbeddingFunc
// that uses the official ollama go client.
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
		// The ollama API returns float64, but chromem-go expects float32, so we convert.
		float32Embedding := make([]float32, len(resp.Embedding))
		for i, v := range resp.Embedding {
			float32Embedding[i] = float32(v)
		}
		return float32Embedding, nil
	}
}

// New creates a new RAG instance.
func New(ctx context.Context, ollamaClient *api.Client, model string) (*RAG, error) {
	db := chromem.NewDB()
	embedder := createOllamaEmbedding(ollamaClient, model)
	_, err := db.GetOrCreateCollection("long-term-memory", nil, embedder)
	if err != nil {
		return nil, fmt.Errorf("failed to create initial collection: %w", err)
	}
	rag := &RAG{
		db:           db,
		ollamaClient: ollamaClient,
		embedder:     embedder,
		model:        model,
	}
	return rag, nil
}

// extractKeywords pulls out potential keywords from a text chunk.
func extractKeywords(content string) []string {
	re := regexp.MustCompile(`"([^"]+)"|(\b[A-Z][a-zA-Z]+\b)|(\b\d+\.\d+\b)|(\b\d+\b)`)
	matches := re.FindAllString(content, -1)
	var keywords []string
	for _, match := range matches {
		keywords = append(keywords, strings.ToLower(strings.Trim(match, `"`)))
	}
	return keywords
}

// smartChunk splits content based on logical breaks.
func smartChunk(content string) []string {
	chunks := strings.Split(content, "\n\n")
	var nonEmptyChunks []string
	for _, chunk := range chunks {
		if strings.TrimSpace(chunk) != "" {
			nonEmptyChunks = append(nonEmptyChunks, chunk)
		}
	}
	return nonEmptyChunks
}

// AddMessagesToStore adds conversational turns to the long-term vector store.
func (r *RAG) AddMessagesToStore(ctx context.Context, messages []api.Message) error {
	collection := r.db.GetCollection("long-term-memory", r.embedder)
	if collection == nil {
		return fmt.Errorf("long-term memory collection not found")
	}
	var documents []chromem.Document
	for _, message := range messages {
		chunks := smartChunk(message.Content)
		for _, chunk := range chunks {
			docID := fmt.Sprintf("msg_%s", uuid.New().String())
			documents = append(documents, chromem.Document{
				ID:      docID,
				Content: chunk,
				Metadata: map[string]string{
					"role": message.Role,
				},
			})
		}
	}
	if len(documents) == 0 {
		return nil
	}
	err := collection.AddDocuments(ctx, documents, 4)
	if err != nil {
		return fmt.Errorf("failed to add documents to collection: %w", err)
	}
	log.Printf("--- Added %d document chunks to long-term RAG memory. ---", len(documents))
	return nil
}

// Query performs a semantic search followed by a keyword-based re-ranking.
func (r *RAG) Query(ctx context.Context, query string, nResults int) (string, error) {
	collection := r.db.GetCollection("long-term-memory", r.embedder)
	if collection == nil {
		return "", fmt.Errorf("failed to get collection: collection not found")
	}
	docCount := collection.Count()
	if docCount == 0 {
		return "", nil
	}

	// 1. Retrieve a larger set of candidates for re-ranking.
	candidateCount := min(nResults*3, docCount)

	results, err := collection.Query(ctx, query, candidateCount, nil, nil)
	if err != nil {
		return "", fmt.Errorf("failed to query collection: %w", err)
	}

	// 2. Re-rank the results based on keyword matching.
	queryKeywords := extractKeywords(query)
	for i := range results {
		contentKeywords := extractKeywords(results[i].Content)
		scoreBoost := 0.0
		for _, qk := range queryKeywords {
			for _, ck := range contentKeywords {
				if qk == ck {
					scoreBoost += 0.1 // Add a boost for each keyword match
				}
			}
		}
		results[i].Similarity += float32(scoreBoost)
	}

	// Sort the results by their new, boosted similarity score.
	sort.Slice(results, func(i, j int) bool {
		return results[i].Similarity > results[j].Similarity
	})

	// 3. Take the top N results from the re-ranked list.
	finalResults := results
	if len(results) > nResults {
		finalResults = results[:nResults]
	}

	var context strings.Builder
	context.WriteString("Relevant information from long-term memory:\n")
	for _, result := range finalResults {
		role := result.Metadata["role"]
		context.WriteString(fmt.Sprintf("- %s: %s\n", role, result.Content))
	}

	return context.String(), nil
}
