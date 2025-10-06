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
	var contentHash string

	// Handle assistant+tool message pairs as "facts"
	if message.Role == "assistant" && index+1 < len(messages) && messages[index+1].Role == "tool" {
		toolMessage := messages[index+1]
		processed[index+1] = true
		metadata["role"] = "fact"

		assistantContent := canonicalizeFactText(message.Content)
		toolContent := canonicalizeFactText(toolMessage.Content)

		// Extract statistical metadata FIRST (before fact generation)
		var statMeta map[string]string
		var extractedCode string
		if format.HasTag(message.Content, format.PythonTag) {
			code, _ := format.ExtractTagContent(message.Content, format.PythonTag)
			extractedCode = code
			statMeta = ExtractStatisticalMetadata(code, toolContent)
			r.ensureDatasetMetadata(sessionID, metadata, code, toolContent)
		}

		// Compute content hash on STABLE parts only (code + tool output)
		// This prevents assistant message variations from creating false duplicates
		if extractedCode != "" {
			hashInput := canonicalizeFactText(extractedCode) + "\n###TOOL_OUTPUT###\n" + toolContent
			contentHash = hashContent(normalizeForHash(hashInput))
			metadata["content_hash"] = contentHash

			// Check for duplicates BEFORE expensive LLM fact generation
			if contentHash != "" {
				existingDocID, err := r.store.FindRAGDocumentByHash(ctx, sessionID, "fact", contentHash)
				if err != nil {
					r.logger.Warn("Failed to check for existing RAG document during fact preparation",
						zap.Error(err),
						zap.String("session_id", sessionID))
				} else if existingDocID != uuid.Nil {
					r.logger.Debug("Skipping duplicate fact (early deduplication on code+tool)",
						zap.String("existing_document_id", existingDocID.String()),
						zap.String("session_id", sessionID))
					return nil, true, nil
				}
			}
		}

		// Generate structured content for BM25 keyword search
		// This enables exact-match queries like "find test where W=0.923"
		if r.cfg.FactUseStructuredContentForBM25 && len(statMeta) > 0 {
			storedContent = r.generateStructuredContentForBM25(statMeta)
		} else {
			// Fallback to JSON format if no metadata extracted
			userContent := ""
			for prev := index - 1; prev >= 0; prev-- {
				if messages[prev].Role == "user" {
					userContent = canonicalizeFactText(messages[prev].Content)
					break
				}
			}

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
		}

		// Generate searchable summary for the fact
		re := regexp.MustCompile(`(?s)<python>(.*)</python>`)
		matches := re.FindStringSubmatch(message.Content)
		if len(matches) > 1 {
			code := strings.TrimSpace(matches[1])
			result := strings.TrimSpace(toolMessage.Content)

			// Stage-aware fact generation strategy
			analysisStage := statMeta["analysis_stage"]
			useLLM := true

			// Determine if we should skip LLM based on stage and config
			switch analysisStage {
			case "assumption_check":
				useLLM = r.cfg.FactUseLLMForAssumptions
			case "descriptive":
				useLLM = r.cfg.FactUseLLMForDescriptive
			case "hypothesis_test", "modeling", "post_hoc":
				// Always use LLM for important stages
				useLLM = true
			default:
				// Unknown stage - use LLM to be safe
				useLLM = true
			}

			if !useLLM {
				// Use deterministic template for routine operations
				contentToEmbed = r.generateDeterministicFact(statMeta)
				r.logger.Debug("Used deterministic fact template",
					zap.String("stage", analysisStage),
					zap.String("test", statMeta["primary_test"]))
			} else {
				// Create context with timeout for fact summarization
				timeoutCtx, cancel := context.WithTimeout(ctx, r.cfg.SummarizationTimeout)

				// Pass statistical metadata to fact generator
				summary, err := r.generateFactSummary(timeoutCtx, result, statMeta)
				cancel() // Clean up context

				if err != nil {
					r.logger.Warn("LLM fact summarization failed, using deterministic fallback",
						zap.Error(err),
						zap.Int("code_length", len(code)),
						zap.Int("result_length", len(result)))
					contentToEmbed = r.generateDeterministicFact(statMeta)
				} else {
					// Verify numeric accuracy if enabled (checks both metadata and raw tool output)
					if r.cfg.FactEnableNumericVerification && !r.verifyNumericAccuracy(summary, statMeta, result) {
						r.logger.Warn("LLM fact failed numeric verification, using deterministic template",
							zap.String("llm_fact", summary),
							zap.String("stage", analysisStage))
						contentToEmbed = r.generateDeterministicFact(statMeta)
					} else {
						contentToEmbed = strings.TrimSpace(summary)
					}
				}
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

	// Hash-based deduplication check (for non-fact messages only)
	// Facts are deduplicated earlier based on code+tool hash
	role := metadata["role"]
	if role != "fact" {
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
	}

	/*
		// Generate searchable summary for long messages (non-facts)
		if role != "fact" && len(message.Content) > 500 {
			// Create context with timeout for searchable summary
			timeoutCtx, cancel := context.WithTimeout(ctx, r.cfg.SummarizationTimeout)
			summary, err := r.generateSearchableSummary(timeoutCtx, message.Content)
			cancel() // Clean up context

			if err != nil {
				r.logger.Warn("Failed to create searchable summary for long message, will use full content",
					zap.Error(err),
					zap.Int("content_length", len(message.Content)))
			} else {
				summaryDoc = r.buildSummaryDocument(summary, metadata, sessionID, message.Role)
			}
		}
	*/

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
