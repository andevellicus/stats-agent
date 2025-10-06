package rag

import (
	"context"
	"encoding/json"
	"fmt"
	"regexp"
	"strings"

	"stats-agent/web/format"
	"stats-agent/web/types"

	"github.com/google/uuid"
	"github.com/philippgille/chromem-go"
	"go.uber.org/zap"
)

// prepareDocumentForMessage prepares a single message for RAG storage.
// It handles fact generation for assistant+tool pairs, deduplication,
// metadata extraction, and summary document creation for long messages.
func (r *RAG) prepareDocumentForMessage(
	ctx context.Context,
	sessionID string,
	messages []types.AgentMessage,
	index int,
	collection *chromem.Collection,
	sessionFilter map[string]string,
	processed map[int]bool,
) (*ragDocumentData, bool, error) {
	processed[index] = true
	message := messages[index]

	documentUUID := uuid.New()
	documentID := documentUUID.String()
	metadata := map[string]string{"document_id": documentID}
	if sessionID != "" {
		metadata["session_id"] = sessionID
	}
	r.ensureDatasetMetadata(sessionID, metadata, message.Content)

	var storedContent string
	var contentToEmbed string
	var summaryDoc *chromem.Document

	// Handle assistant+tool message pairs as "facts"
	if message.Role == "assistant" && index+1 < len(messages) && messages[index+1].Role == "tool" {
		toolMessage := messages[index+1]
		processed[index+1] = true
		metadata["role"] = "fact"

		assistantContent := canonicalizeFactText(message.Content)
		toolContent := canonicalizeFactText(toolMessage.Content)

		// Extract statistical metadata FIRST (before fact generation)
		var statMeta map[string]string
		if format.HasTag(message.Content, format.PythonTag) {
			code, _ := format.ExtractTagContent(message.Content, format.PythonTag)
			statMeta = ExtractStatisticalMetadata(code, toolContent)
			r.ensureDatasetMetadata(sessionID, metadata, code, toolContent)
		}

		// Find preceding user message for context
		userContent := ""
		for prev := index - 1; prev >= 0; prev-- {
			if messages[prev].Role == "user" {
				userContent = canonicalizeFactText(messages[prev].Content)
				break
			}
		}

		// Store fact as JSON for structured retrieval
		factPayload := factStoredContent{
			Assistant: assistantContent,
			Tool:      toolContent,
		}
		if userContent != "" {
			factPayload.User = userContent
		}

		factJSON, marshalErr := json.Marshal(factPayload)
		if marshalErr != nil {
			r.logger.Warn("Failed to marshal fact payload, falling back to concatenated format", zap.Error(marshalErr))
			storedContent = fmt.Sprintf("%s\n\n%s", assistantContent, toolContent)
		} else {
			storedContent = string(factJSON)
		}

		// Generate searchable summary for the fact
		re := regexp.MustCompile(`(?s)<python>(.*)</python>`)
		matches := re.FindStringSubmatch(message.Content)
		if len(matches) > 1 {
			code := strings.TrimSpace(matches[1])
			result := strings.TrimSpace(toolMessage.Content)
			// Pass statistical metadata to fact generator
			summary, err := r.generateFactSummary(ctx, code, result, statMeta)
			if err != nil {
				r.logger.Warn("LLM fact summarization failed, using fallback summary",
					zap.Error(err),
					zap.Int("code_length", len(code)),
					zap.Int("result_length", len(result)))
				contentToEmbed = "Fact: A code execution event occurred but could not be summarized."
			} else {
				contentToEmbed = strings.TrimSpace(summary)
			}
		} else {
			contentToEmbed = "Fact: An assistant action with a tool execution occurred."
		}
	} else {
		// Handle regular messages (non-fact)

		// Skip assistant messages that already look like facts
		if message.Role == "assistant" {
			trimmed := strings.TrimSpace(message.Content)
			if strings.HasPrefix(trimmed, "Fact:") && !format.HasTag(message.Content, format.PythonTag) {
				return nil, true, nil
			}
		}

		storedContent = canonicalizeFactText(message.Content)
		metadata["role"] = message.Role
		contentToEmbed = canonicalizeFactText(storedContent)

		// Check for semantic duplicates
		if collection != nil && collection.Count() > 0 {
			results, err := collection.Query(ctx, contentToEmbed, 1, sessionFilter, nil)
			if err != nil {
				r.logger.Warn("Deduplication query failed, proceeding to add document anyway", zap.Error(err))
			} else if len(results) > 0 && results[0].Similarity > 0.98 && results[0].Metadata["role"] == message.Role {
				r.logger.Debug("Skipping duplicate content", zap.Float32("similarity", results[0].Similarity), zap.String("role", message.Role))
				return nil, true, nil
			}
		}
	}

	if storedContent == "" {
		storedContent = contentToEmbed
	}

	// Hash-based deduplication check
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
			return nil, true, nil
		}
		if existingDocID != uuid.Nil {
			r.logger.Debug("Skipping duplicate RAG document",
				zap.String("existing_document_id", existingDocID.String()),
				zap.String("session_id", sessionID),
				zap.String("role", role))
			return nil, true, nil
		}
	}

	// Generate searchable summary for long messages (non-facts)
	if role != "fact" && len(message.Content) > 500 {
		summary, err := r.generateSearchableSummary(ctx, message.Content)
		if err != nil {
			r.logger.Warn("Failed to create searchable summary for long message, will use full content",
				zap.Error(err),
				zap.Int("content_length", len(message.Content)))
		} else {
			summaryDoc = r.buildSummaryDocument(summary, metadata, sessionID, message.Role)
		}
	}

	// Final metadata enrichment
	r.ensureDatasetMetadata(sessionID, metadata, message.Content, storedContent, contentToEmbed)

	return &ragDocumentData{
		ID:            documentUUID,
		Metadata:      metadata,
		StoredContent: storedContent,
		EmbedContent:  contentToEmbed,
		ContentHash:   contentHash,
		SummaryDoc:    summaryDoc,
	}, false, nil
}

// buildSummaryDocument creates a summary document for long messages to improve search.
func (r *RAG) buildSummaryDocument(summary string, parentMetadata map[string]string, sessionID, messageRole string) *chromem.Document {
	summaryID := uuid.New()
	metadata := map[string]string{
		"role":                 messageRole,
		"document_id":          summaryID.String(),
		"type":                 "summary",
		"parent_document_id":   parentMetadata["document_id"],
		"parent_document_role": parentMetadata["role"],
	}
	if sessionID != "" {
		metadata["session_id"] = sessionID
	}
	if dataset := parentMetadata["dataset"]; dataset != "" {
		metadata["dataset"] = dataset
	}

	return &chromem.Document{
		ID:       uuid.New().String(),
		Content:  summary,
		Metadata: metadata,
	}
}

// ensureDatasetMetadata extracts and ensures dataset metadata is present.
// It checks existing metadata, scans text for dataset references, and uses
// session-level dataset memory as fallback.
func (r *RAG) ensureDatasetMetadata(sessionID string, metadata map[string]string, texts ...string) {
	if metadata == nil {
		return
	}

	// Use existing dataset if already set
	if existing := strings.TrimSpace(metadata["dataset"]); existing != "" {
		metadata["dataset"] = existing
		r.rememberSessionDataset(sessionID, existing)
		return
	}

	// Scan texts for dataset references
	for _, text := range texts {
		if text == "" {
			continue
		}
		if matches := datasetQueryRegex.FindStringSubmatch(text); len(matches) > 1 {
			dataset := strings.TrimSpace(matches[1])
			if dataset != "" {
				metadata["dataset"] = dataset
				r.rememberSessionDataset(sessionID, dataset)
				return
			}
		}
	}

	if sessionID == "" {
		return
	}

	// Use session-level dataset memory as fallback
	if dataset := strings.TrimSpace(r.getSessionDataset(sessionID)); dataset != "" {
		metadata["dataset"] = dataset
	}
}
