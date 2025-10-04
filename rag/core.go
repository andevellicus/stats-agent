package rag

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"strings"
	"sync"

	"stats-agent/config"
	"stats-agent/database"
	"stats-agent/llmclient"

	"github.com/google/uuid"
	"github.com/philippgille/chromem-go"
	"go.uber.org/zap"
)

const (
	// BGE models typically handle 512 tokens max
	// ~4 chars per token, with safety margin
	defaultMaxEmbeddingChars  = 1000
	defaultMaxEmbeddingTokens = 250
)

type RAG struct {
	cfg                        *config.Config
	db                         *chromem.DB
	store                      *database.PostgresStore
	embedder                   chromem.EmbeddingFunc
	logger                     *zap.Logger
	maxEmbeddingChars          int
	maxEmbeddingTokens         int
	embeddingTokenSoftLimit    int
	embeddingTokenTarget       int
	minTokenCheckCharThreshold int
	maxHybridCandidates        int
	datasetMu                  sync.RWMutex
	sessionDatasets            map[string]string
	sentenceSplitter           SentenceSplitter
	contextMaxChars            int
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
	if _, err := db.GetOrCreateCollection("long-term-memory", nil, embedder); err != nil {
		return nil, fmt.Errorf("failed to create initial collection: %w", err)
	}

	maxEmbeddingChars := cfg.MaxEmbeddingChars
	if maxEmbeddingChars <= 0 {
		maxEmbeddingChars = defaultMaxEmbeddingChars
	}
	maxEmbeddingTokens := cfg.MaxEmbeddingTokens
	if maxEmbeddingTokens <= 0 {
		maxEmbeddingTokens = defaultMaxEmbeddingTokens
	}
	embeddingSoftLimit := cfg.EmbeddingTokenSoftLimit
	embeddingTarget := cfg.EmbeddingTokenTarget
	minTokenThreshold := cfg.MinTokenCheckCharThreshold
	hybridCandidates := cfg.MaxHybridCandidates

	contextMax := cfg.RAGContextMaxChars
	if contextMax <= 0 {
		contextMax = cfg.ContextLength * 4
		if contextMax <= 0 {
			contextMax = 12000
		}
	}

	r := &RAG{
		cfg:                        cfg,
		db:                         db,
		store:                      store,
		embedder:                   embedder,
		logger:                     logger,
		maxEmbeddingChars:          maxEmbeddingChars,
		maxEmbeddingTokens:         maxEmbeddingTokens,
		embeddingTokenSoftLimit:    embeddingSoftLimit,
		embeddingTokenTarget:       embeddingTarget,
		minTokenCheckCharThreshold: minTokenThreshold,
		maxHybridCandidates:        hybridCandidates,
		sessionDatasets:            make(map[string]string),
		sentenceSplitter:           NewRegexSentenceSplitter(),
		contextMaxChars:            contextMax,
	}

	return r, nil
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

// filterStructuralMetadata keeps only structural fields for JSONB storage.
// Statistical metadata is now embedded in the fact text itself.
// Exception: dataset is kept for query boosting and session tracking.
func filterStructuralMetadata(metadata map[string]string) map[string]string {
	if len(metadata) == 0 {
		return metadata
	}

	structural := make(map[string]string)

	// Keep only these structural fields
	structuralKeys := map[string]bool{
		"session_id":           true,
		"role":                 true,
		"document_id":          true,
		"type":                 true,
		"content_hash":         true,
		"parent_document_id":   true,
		"parent_document_role": true,
		"chunk_index":          true,
		"dataset":              true, // Keep for query boosting and metadata filtering
	}

	for key, value := range metadata {
		if structuralKeys[key] {
			structural[key] = value
		}
	}

	return structural
}

// Note: augmentEmbeddingContent has been removed.
// Statistical metadata is now embedded inline in fact text during generation.
// Only structural metadata (session_id, role, document_id, etc.) is stored in JSONB.

func (r *RAG) rememberSessionDataset(sessionID, dataset string) {
	if sessionID == "" {
		return
	}
	dataset = strings.TrimSpace(dataset)
	if dataset == "" {
		return
	}
	r.datasetMu.Lock()
	r.sessionDatasets[sessionID] = dataset
	r.datasetMu.Unlock()
}

func (r *RAG) getSessionDataset(sessionID string) string {
	if sessionID == "" {
		return ""
	}
	r.datasetMu.RLock()
	dataset := r.sessionDatasets[sessionID]
	r.datasetMu.RUnlock()
	return dataset
}

func (r *RAG) clearSessionDataset(sessionID string) {
	if sessionID == "" {
		return
	}
	r.datasetMu.Lock()
	delete(r.sessionDatasets, sessionID)
	r.datasetMu.Unlock()
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

func createLlamaCppEmbedding(cfg *config.Config, logger *zap.Logger) chromem.EmbeddingFunc {
	client := llmclient.New(cfg, logger)
	return func(ctx context.Context, doc string) ([]float32, error) {
		return client.Embed(ctx, cfg.EmbeddingLLMHost, doc)
	}
}
