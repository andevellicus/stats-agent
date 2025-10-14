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
	"go.uber.org/zap"
)

// BGE-large-en-v1.5 has a 512 token hard limit; target and soft limits in config handle sizing.

// EmbeddingFunc is a function that generates embeddings for text.
type EmbeddingFunc func(ctx context.Context, text string) ([]float32, error)

type RAG struct {
	cfg                        *config.Config
	store                      *database.PostgresStore
	embedder                   EmbeddingFunc
	logger                     *zap.Logger
    embeddingTokenSoftLimit    int
    embeddingTokenTarget       int
    minTokenCheckCharThreshold int
	maxHybridCandidates        int
	datasetMu                  sync.RWMutex
	sessionDatasets            map[string]string
	sentenceSplitter           SentenceSplitter
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
	WindowIndex   int // Which embedding window matched (for multi-vector documents)
}

// Embedding request/response types moved to llmclient

type ragDocumentData struct {
	ID            uuid.UUID
	Metadata      map[string]string
	StoredContent string
	EmbedContent  string
	ContentHash   string
	SummaryDoc    *summaryDocument
}

type summaryDocument struct {
	ID       string
	Content  string
	Metadata map[string]string
}

func New(cfg *config.Config, store *database.PostgresStore, logger *zap.Logger) (*RAG, error) {
	if store == nil {
		return nil, fmt.Errorf("postgres store is required for RAG persistence")
	}

	embedder := createLlamaCppEmbedding(cfg, logger)

    embeddingSoftLimit := cfg.EmbeddingTokenSoftLimit
    embeddingTarget := cfg.EmbeddingTokenTarget
    minTokenThreshold := cfg.MinTokenCheckCharThreshold
    hybridCandidates := cfg.MaxHybridCandidates

	r := &RAG{
		cfg:                        cfg,
		store:                      store,
		embedder:                   embedder,
		logger:                     logger,
        embeddingTokenSoftLimit:    embeddingSoftLimit,
        embeddingTokenTarget:       embeddingTarget,
        minTokenCheckCharThreshold: minTokenThreshold,
		maxHybridCandidates:        hybridCandidates,
		sessionDatasets:            make(map[string]string),
		sentenceSplitter:           NewRegexSentenceSplitter(),
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

// NormalizeForHash prepares content for hashing by normalizing whitespace.
// Exported for use in deduplication logic.
func NormalizeForHash(content string) string {
	return strings.TrimSpace(content)
}

// HashContent creates a SHA-256 hash of the given content.
// Exported for use in deduplication logic.
func HashContent(content string) string {
	if content == "" {
		return ""
	}
	sum := sha256.Sum256([]byte(content))
	return hex.EncodeToString(sum[:])
}

// ContentHashesMatch checks if two pieces of content have the same hash.
// Uses RAG's standard normalization (trim whitespace) before hashing.
// This is the same logic used when storing documents to RAG.
func ContentHashesMatch(content1, content2 string) bool {
	hash1 := HashContent(NormalizeForHash(content1))
	hash2 := HashContent(NormalizeForHash(content2))
	return hash1 != "" && hash1 == hash2
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
		"filename":             true, // Original filename
		"page_number":          true, // Page number for PDFs
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

func resolveLookupID(documentID string, metadata map[string]string) string {
	if documentID == "" {
		return ""
	}
	lookupID := documentID

	if metadata == nil {
		return lookupID
	}

	docType, hasType := metadata["type"]
	if !hasType {
		return lookupID
	}

	// For conversation chunks and summaries, fetch parent document
	if docType == "summary" || docType == "chunk" {
		if parentID := metadata["parent_document_id"]; parentID != "" {
			lookupID = parentID
		}
	}

	// For document chunks, return the chunk itself (no parent lookup)
	if docType == "document_chunk" {
		return documentID
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

func createLlamaCppEmbedding(cfg *config.Config, logger *zap.Logger) EmbeddingFunc {
	client := llmclient.New(cfg, logger)
	return func(ctx context.Context, doc string) ([]float32, error) {
		return client.Embed(ctx, cfg.EmbeddingLLMHost, doc)
	}
}
