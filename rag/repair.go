package rag

import (
	"context"
	"fmt"
	"regexp"
	"strings"

	"go.uber.org/zap"
)

// RepairInconsistentFacts finds and fixes RAG documents where the embedding_content
// contradicts the metadata (e.g., says "no age variable" when metadata has "variables: age").
// This repairs issues caused by LLM hallucinations in fact generation.
func (r *RAG) RepairInconsistentFacts(ctx context.Context) error {
	r.logger.Info("Starting RAG document consistency repair")

	// Get all fact documents from database
	storedDocs, err := r.store.ListRAGDocuments(ctx)
	if err != nil {
		return fmt.Errorf("failed to list RAG documents: %w", err)
	}

	repairedCount := 0
	skippedCount := 0

	// Patterns that indicate contradictory/problematic facts
	negationPatterns := []*regexp.Regexp{
		regexp.MustCompile(`(?i)no\s+\w+\s+variable`),
		regexp.MustCompile(`(?i)does not (include|contain|have)`),
		regexp.MustCompile(`(?i)missing\s+\w+`),
		regexp.MustCompile(`(?i)cannot\s+be\s+generated`),
		regexp.MustCompile(`(?i)not\s+found`),
	}

	for _, doc := range storedDocs {
		// Only check fact documents
		if doc.Metadata["role"] != "fact" {
			continue
		}

		embeddingContent := strings.TrimSpace(doc.EmbeddingContent)
		if embeddingContent == "" {
			// Use content as embedding_content if missing
			embeddingContent = doc.Content
		}

		// Check if embedding_content contains negations
		hasNegation := false
		for _, pattern := range negationPatterns {
			if pattern.MatchString(embeddingContent) {
				hasNegation = true
				break
			}
		}

		if !hasNegation {
			continue
		}

		// Check if metadata has positive values (contradicting the negation)
		hasPositiveMetadata := false
		positiveFields := []string{"variables", "primary_test", "dataset", "p_value"}
		for _, field := range positiveFields {
			if val, ok := doc.Metadata[field]; ok && strings.TrimSpace(val) != "" {
				hasPositiveMetadata = true
				break
			}
		}

		if !hasPositiveMetadata {
			// No contradiction - the negation might be legitimate
			skippedCount++
			continue
		}

		// Found an inconsistent fact - repair it
		r.logger.Info("Repairing inconsistent fact",
			zap.String("document_id", doc.DocumentID.String()),
			zap.String("bad_embedding", embeddingContent),
			zap.Any("metadata", doc.Metadata))

		// Generate new fact using deterministic template
		newFact := r.generateDeterministicFact(doc.Metadata)

		// Update the document in database
		if err := r.store.UpsertRAGDocument(ctx, doc.DocumentID, doc.Content, newFact, doc.Metadata, doc.ContentHash, doc.Embedding); err != nil {
			r.logger.Warn("Failed to repair RAG document",
				zap.Error(err),
				zap.String("document_id", doc.DocumentID.String()))
			continue
		}

		repairedCount++
	}

	r.logger.Info("RAG document repair complete",
		zap.Int("repaired", repairedCount),
		zap.Int("skipped", skippedCount),
		zap.Int("total_checked", len(storedDocs)))

	return nil
}

// ValidateAllFacts checks all existing facts for consistency with their metadata.
// Returns a report of issues found without modifying the database.
func (r *RAG) ValidateAllFacts(ctx context.Context) (map[string]string, error) {
	r.logger.Info("Validating all RAG facts")

	storedDocs, err := r.store.ListRAGDocuments(ctx)
	if err != nil {
		return nil, fmt.Errorf("failed to list RAG documents: %w", err)
	}

	issues := make(map[string]string)

	for _, doc := range storedDocs {
		if doc.Metadata["role"] != "fact" {
			continue
		}

		embeddingContent := strings.TrimSpace(doc.EmbeddingContent)
		if embeddingContent == "" {
			embeddingContent = doc.Content
		}

		// Validate using LLM
		if !r.validateFactCoherence(ctx, embeddingContent, doc.Metadata) {
			issues[doc.DocumentID.String()] = fmt.Sprintf("Fact: '%s' | Metadata: %v", embeddingContent, doc.Metadata)
		}
	}

	r.logger.Info("Fact validation complete",
		zap.Int("issues_found", len(issues)),
		zap.Int("total_facts", len(storedDocs)))

	return issues, nil
}
