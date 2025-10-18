package rag

import (
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"sort"
	"strings"
	"unicode"

	"github.com/google/uuid"
	"go.uber.org/zap"
)

// getRelevantContent retrieves context-appropriate content based on document type and matched window.
// For documents (PDFs), returns matched window + 1 surrounding window (~1440 tokens).
// For facts/summaries, returns embedding window text (the concise summary).
// For other content, returns full content.
func (r *RAG) getRelevantContent(ctx context.Context, documentID uuid.UUID, metadata map[string]string, windowIndex int) (string, error) {
	docType := metadata["type"]
	role := metadata["role"]

	// For facts and summaries: return window_text (the concise summary), not full content
	if role == "fact" || docType == "summary" {
		// Get embedding windows for this document
		windows, err := r.store.GetDocumentEmbeddings(ctx, documentID)
		if err != nil {
			return "", fmt.Errorf("failed to get embeddings for fact: %w", err)
		}
		if len(windows) == 0 {
			// Fallback: no embeddings, return full content
			r.logger.Warn("No embeddings found for fact, using full content",
				zap.String("document_id", documentID.String()),
				zap.String("role", role))
			content, err := r.store.GetRAGDocumentContent(ctx, documentID)
			if err != nil {
				return "", fmt.Errorf("failed to get document content: %w", err)
			}
			return content, nil
		}
		// Return the window_text (summary) from the first embedding window
		return windows[0].WindowText, nil
	}

	// For conversation chunks: return full content
	if docType == "chunk" || (role != "document" && docType != "pdf" && docType != "document_chunk") {
		content, err := r.store.GetRAGDocumentContent(ctx, documentID)
		if err != nil {
			return "", fmt.Errorf("failed to get document content: %w", err)
		}
		return content, nil
	}

	// For documents (PDFs, large files): extract window-based context
	// Get all embedding windows for this document
	windows, err := r.store.GetDocumentEmbeddings(ctx, documentID)
	if err != nil {
		return "", fmt.Errorf("failed to get document embeddings: %w", err)
	}

	if len(windows) == 0 {
		// Fallback: no embeddings found, return full content
		r.logger.Warn("No embeddings found for document, returning full content",
			zap.String("document_id", documentID.String()))
		return r.store.GetRAGDocumentContent(ctx, documentID)
	}

	// If document has only 1 window, return it
	if len(windows) == 1 {
		return windows[0].WindowText, nil
	}

	// Extract matched window + 1 surrounding window (before and after)
	var contextParts []string

	// Add previous window if it exists
	if windowIndex > 0 && windowIndex-1 < len(windows) {
		contextParts = append(contextParts, windows[windowIndex-1].WindowText)
	}

	// Add matched window
	if windowIndex < len(windows) {
		contextParts = append(contextParts, windows[windowIndex].WindowText)
	} else {
		// Edge case: window index out of bounds, use last window
		r.logger.Warn("Window index out of bounds, using last window",
			zap.Int("window_index", windowIndex),
			zap.Int("total_windows", len(windows)))
		contextParts = append(contextParts, windows[len(windows)-1].WindowText)
	}

	// Add next window if it exists
	if windowIndex+1 < len(windows) {
		contextParts = append(contextParts, windows[windowIndex+1].WindowText)
	}

	return strings.Join(contextParts, " "), nil
}

// getRelevantContentBatch fetches parent document contents for a set of candidates using a single query.
// It resolves lookup IDs from candidate metadata and returns a map of lookupID -> content.
// Note: This returns stored full contents; window-aware slicing is still handled downstream when emitting.
func (r *RAG) getRelevantContentBatch(ctx context.Context, candidates []*hybridCandidate) (map[string]string, error) {
	result := make(map[string]string)
	if len(candidates) == 0 {
		return result, nil
	}

	// Collect unique lookup IDs for batch fetch
	idSet := make(map[string]struct{})
	for _, cand := range candidates {
		if cand == nil {
			continue
		}
		lookupID := ResolveLookupID(cand.DocumentID, cand.Metadata)
		if lookupID == "" {
			continue
		}
		idSet[lookupID] = struct{}{}
	}

	if len(idSet) == 0 {
		return result, nil
	}

	// Convert to UUIDs (skip invalid)
	lookupIDs := make([]uuid.UUID, 0, len(idSet))
	for idStr := range idSet {
		if id, err := uuid.Parse(idStr); err == nil {
			lookupIDs = append(lookupIDs, id)
		} else {
			r.logger.Warn("Invalid lookup identifier for batch fetch", zap.String("lookup_id", idStr))
		}
	}

	if len(lookupIDs) == 0 {
		return result, nil
	}

	contents, err := r.store.GetDocumentsBatch(ctx, lookupIDs)
	if err != nil {
		return nil, err
	}

	for idStr, content := range contents {
		result[idStr] = content
	}
	return result, nil
}

func (r *RAG) queryHybrid(ctx context.Context, sessionID string, query string, nResults int, excludeHashes []string, historyDocIDs []string, doneLedger string, mode string) (string, int, error) {
	if nResults <= 0 {
		return "", 0, nil
	}

	candidateLimit := max(nResults*4, 20)
	maxHybridCandidates := r.maxHybridCandidates
	if maxHybridCandidates <= 0 {
		maxHybridCandidates = r.cfg.MaxHybridCandidates
	}
	minSemanticSimilarity := r.cfg.SemanticSimilarityThreshold
	if minSemanticSimilarity <= 0 || minSemanticSimilarity > 1 {
		minSemanticSimilarity = 0.7
	}
	minBM25Score := r.cfg.BM25ScoreThreshold
	if minBM25Score < 0 {
		minBM25Score = 0
	}
	if maxHybridCandidates > 0 && candidateLimit > maxHybridCandidates {
		candidateLimit = maxHybridCandidates
	}

	// Derive metadata hints
	lowerQuery := strings.ToLower(query)
	isQueryForError := strings.Contains(lowerQuery, "error")
	metadataFilters := extractSimpleMetadata(query, r.cfg.MetadataFallbackMaxFilters)
	metadataHints := make(map[string]string, len(metadataFilters))
	for k, v := range metadataFilters {
		metadataHints[k] = strings.TrimSpace(v)
	}
	if _, exists := metadataHints["dataset"]; !exists {
		if strings.Contains(lowerQuery, "file name") || strings.Contains(lowerQuery, "uploaded file") || strings.Contains(lowerQuery, ".csv") || strings.Contains(lowerQuery, ".xlsx") || strings.Contains(lowerQuery, "dataset") {
			metadataHints["dataset"] = ""
		}
	}
	if _, exists := metadataHints["role"]; !exists {
		switch {
		case strings.Contains(lowerQuery, "assistant message") || strings.Contains(lowerQuery, "assistant response"):
			metadataHints["role"] = "assistant"
		case strings.Contains(lowerQuery, "user message") || strings.Contains(lowerQuery, "user question"):
			metadataHints["role"] = "user"
		case strings.Contains(lowerQuery, "tool output") || strings.Contains(lowerQuery, "python result") || strings.Contains(lowerQuery, "code result"):
			metadataHints["role"] = "tool"
		}
	}

	// 1) Gather candidates (vector + bm25 + batch parent content)
	candidates, docContents, err := r.gatherCandidates(ctx, sessionID, query, candidateLimit, excludeHashes, minSemanticSimilarity, minBM25Score)
	if err != nil {
		r.logger.Warn("gatherCandidates failed", zap.Error(err))
	}
	if len(candidates) == 0 {
		return "", 0, nil
	}

    // 2) Graph pre-filter: drop superseded/blocked where applicable
    candidates = r.validateCandidatesWithGraph(ctx, candidates)

    // 3) Score and rank hybrid
    candidateList := r.scoreHybrid(query, mode, metadataHints, candidates, isQueryForError)

    // 4) Filter by history
    filtered1 := r.filterHistory(candidateList, historyDocIDs)

    // 5) Bucket summaries
    filtered2 := r.bucketSummaries(filtered1)

    // 6) Deduplicate via shingles/hash
    filtered3 := r.deduplicateShingles(filtered2, excludeHashes)

    // 7) Boost via graph valid path (supports edges to tests; prefer latest states)
    filtered3 = r.boostByValidPath(ctx, filtered3)

    // 8) Format output memory block
    return r.formatMemoryBlock(ctx, filtered3, nResults, doneLedger, docContents, excludeHashes)
}

// validateCandidatesWithGraph removes candidates that are superseded or blocked by assumptions.
func (r *RAG) validateCandidatesWithGraph(ctx context.Context, candidates map[string]*hybridCandidate) map[string]*hybridCandidate {
    if r.graph == nil || !r.graph.Enabled() || len(candidates) == 0 {
        return candidates
    }
    out := make(map[string]*hybridCandidate, len(candidates))
    for id, cand := range candidates {
        if cand == nil {
            continue
        }
        lookupID := ResolveLookupID(cand.DocumentID, cand.Metadata)
        if lookupID == "" {
            continue
        }
        // Superseded check
        if sup, err := r.graph.IsSuperseded(ctx, lookupID); err == nil && sup {
            continue
        }
        // Blocked check
        if blk, err := r.graph.IsBlocked(ctx, lookupID); err == nil && blk {
            continue
        }
        out[id] = cand
    }
    return out
}

// boostByValidPath increases the score for candidates with supportive assumption edges.
func (r *RAG) boostByValidPath(ctx context.Context, list []*hybridCandidate) []*hybridCandidate {
    if r.graph == nil || !r.graph.Enabled() || len(list) == 0 {
        return list
    }
    for _, cand := range list {
        if cand == nil {
            continue
        }
        lookupID := ResolveLookupID(cand.DocumentID, cand.Metadata)
        if lookupID == "" {
            continue
        }
        // Boost if there is an incoming 'supports' edge
        if has, err := r.graph.HasIncomingEdgeType(ctx, lookupID, "supports"); err == nil && has {
            boost := r.cfg.GraphSupportsBoost
            if boost <= 0 {
                boost = 1.0
            }
            cand.Score *= boost
        }
    }
    return list
}

// gatherCandidates performs vector and BM25 searches, merges signals into candidates,
// and primes candidate.Content using a batch document fetch for parent content.
func (r *RAG) gatherCandidates(ctx context.Context, sessionID, query string, candidateLimit int, excludeHashes []string, minSemanticSimilarity, minBM25Score float64) (map[string]*hybridCandidate, map[string]string, error) {
	candidates := make(map[string]*hybridCandidate)

	// Vector search
	queryEmbedding, err := r.embedder(ctx, query)
	if err != nil {
		r.logger.Warn("Failed to generate query embedding, using BM25 fallback only", zap.Error(err))
	} else if len(queryEmbedding) > 0 {
		semanticResults, err := r.store.VectorSearchRAGDocuments(ctx, queryEmbedding, candidateLimit, sessionID, excludeHashes)
		if err != nil {
			r.logger.Warn("Vector search failed, using BM25 fallback only", zap.Error(err))
		} else {
			for _, res := range semanticResults {
				docID := res.DocumentID.String()
				similarity := res.Similarity
				if similarity < minSemanticSimilarity {
					continue
				}
				embContent := res.EmbeddingContent
				if embContent == "" {
					embContent = res.Content
				}
				cand := ensureCandidate(candidates, docID, res.Metadata)
				if similarity > cand.SemanticScore {
					cand.SemanticScore = similarity
					cand.Content = embContent
					cand.WindowIndex = res.WindowIndex
				}
				cand.HasSemantic = true
			}
		}
	}

	// BM25 search
	bm25Results, err := r.store.SearchRAGDocumentsBM25(ctx, query, candidateLimit, sessionID, excludeHashes)
	if err != nil {
		r.logger.Warn("BM25 search failed, falling back to semantic results only", zap.Error(err), zap.Int("candidate_limit", candidateLimit), zap.String("session_id", sessionID))
		bm25Results = nil
	}
	for _, bm := range bm25Results {
		docID := bm.DocumentID.String()
		combined := bm.BM25Score + bm.ExactMatchBonus
		if combined < minBM25Score {
			continue
		}
		cand := ensureCandidate(candidates, docID, bm.Metadata)
		existingCombined := cand.BM25Score + cand.ExactBonus
		embContent := bm.EmbeddingContent
		if embContent == "" {
			embContent = bm.Content
		}
		if combined > existingCombined {
			cand.BM25Score = bm.BM25Score
			cand.ExactBonus = bm.ExactMatchBonus
			if embContent != "" {
				cand.Content = embContent
			}
		} else if cand.Content == "" && embContent != "" {
			cand.Content = embContent
		}
		cand.HasBM25 = true
	}

	// Batch fetch parent contents to prime cand.Content
	docContents := make(map[string]string)
	if len(candidates) > 0 {
		candSlice := make([]*hybridCandidate, 0, len(candidates))
		for _, c := range candidates {
			candSlice = append(candSlice, c)
		}
		if contents, err := r.getRelevantContentBatch(ctx, candSlice); err == nil {
			for id, content := range contents {
				docContents[id] = content
			}
			for _, cand := range candidates {
				lookupID := ResolveLookupID(cand.DocumentID, cand.Metadata)
				if lookupID == "" || lookupID == cand.DocumentID {
					continue
				}
				if content, ok := docContents[lookupID]; ok {
					cand.Content = content
				}
			}
		} else {
			r.logger.Warn("Batch content fetch failed; falling back to per-document retrieval", zap.Error(err))
			for _, cand := range candidates {
				lookupID := ResolveLookupID(cand.DocumentID, cand.Metadata)
				if lookupID == "" || lookupID == cand.DocumentID {
					continue
				}
				if content, ok := docContents[lookupID]; ok {
					cand.Content = content
					continue
				}
				lookupUUID, err := uuid.Parse(lookupID)
				if err != nil {
					r.logger.Warn("Invalid lookup identifier for scoring", zap.String("lookup_id", lookupID), zap.String("document_id", cand.DocumentID))
					continue
				}
				content, err := r.getRelevantContent(ctx, lookupUUID, cand.Metadata, cand.WindowIndex)
				if err != nil {
					if !errors.Is(err, sql.ErrNoRows) {
						r.logger.Warn("Failed to load parent content for scoring", zap.Error(err), zap.String("lookup_id", lookupID), zap.String("document_id", cand.DocumentID), zap.Int("window_index", cand.WindowIndex))
					}
					continue
				}
				docContents[lookupID] = content
				cand.Content = content
			}
		}
	}

	return candidates, docContents, nil
}

// scoreHybrid normalizes and combines semantic and BM25 scores, applies mode-specific boosts,
// metadata hints, and echo penalties, and returns a ranked candidate slice.
func (r *RAG) scoreHybrid(query, mode string, metadataHints map[string]string, candidates map[string]*hybridCandidate, isQueryForError bool) []*hybridCandidate {
	var maxSemantic, maxBM float64
	for _, cand := range candidates {
		if cand.SemanticScore > maxSemantic {
			maxSemantic = cand.SemanticScore
		}
		bmCombined := cand.BM25Score + cand.ExactBonus
		if bmCombined > maxBM {
			maxBM = bmCombined
		}
	}

	semanticWeight := r.cfg.HybridSemanticWeight
	if semanticWeight < 0 {
		semanticWeight = 0
	}
	bm25Weight := r.cfg.HybridBM25Weight
	if bm25Weight < 0 {
		bm25Weight = 0
	}
	if semanticWeight == 0 && bm25Weight == 0 {
		semanticWeight = 1
	}

	out := make([]*hybridCandidate, 0, len(candidates))
	for _, cand := range candidates {
		weighted := 0.0
		weightSum := 0.0
		if cand.HasSemantic && maxSemantic > 0 && semanticWeight > 0 {
			weighted += semanticWeight * (cand.SemanticScore / maxSemantic)
			weightSum += semanticWeight
		}
		if cand.HasBM25 && maxBM > 0 && bm25Weight > 0 {
			weighted += bm25Weight * ((cand.BM25Score + cand.ExactBonus) / maxBM)
			weightSum += bm25Weight
		}
		combined := 0.0
		if weightSum > 0 {
			combined = weighted / weightSum
		}

		role := cand.Metadata["role"]
		docType := cand.Metadata["type"]

		var factBoost, summaryBoost, documentBoost float64
		if mode == "document" {
			factBoost = r.cfg.HybridDocumentFactBoost
			summaryBoost = r.cfg.HybridDocumentSummaryBoost
			documentBoost = r.cfg.HybridDocumentDocumentBoost
		} else {
			factBoost = r.cfg.HybridDatasetFactBoost
			summaryBoost = r.cfg.HybridDatasetSummaryBoost
			documentBoost = r.cfg.HybridDatasetDocumentBoost
		}

		if role == "fact" && docType != "chunk" && docType != "document_chunk" {
			combined *= factBoost
		}
		if docType == "summary" {
			combined *= summaryBoost
		}
		if docType == "state" {
			combined *= r.cfg.HybridStateBoost
		}
		if role == "document" || docType == "pdf" || docType == "document_chunk" {
			combined *= documentBoost
		}
		if cand.Content != "" && strings.Contains(cand.Content, "Error:") && !isQueryForError {
			combined *= r.cfg.HybridErrorPenalty
		}

		if len(metadataHints) > 0 && cand.Metadata != nil {
			metadataBoost := 0.0
			for key, hint := range metadataHints {
				hint = strings.TrimSpace(strings.ToLower(hint))
				metaVal := strings.TrimSpace(strings.ToLower(cand.Metadata[key]))
				if metaVal == "" {
					continue
				}
				if hint == "" {
					if key == "dataset" {
						metadataBoost += 0.12
					} else {
						metadataBoost += 0.05
					}
					continue
				}
				if strings.Contains(metaVal, hint) || strings.Contains(hint, metaVal) {
					if key == "dataset" {
						metadataBoost += 0.15
					} else {
						metadataBoost += 0.08
					}
				}
			}
			if metadataBoost > 0.4 {
				metadataBoost = 0.4
			}
			if metadataBoost > 0 {
				if combined <= 0 {
					combined = metadataBoost
				} else {
					combined += metadataBoost
				}
			}
		}

		if cand.Content != "" && isQueryEcho(query, cand.Content) {
			combined *= 0.1
		}
		cand.Score = combined
		out = append(out, cand)
	}

	sort.Slice(out, func(i, j int) bool {
		if out[i].Score == out[j].Score {
			return out[i].DocumentID < out[j].DocumentID
		}
		return out[i].Score > out[j].Score
	})
	return out
}

// filterHistory removes candidates whose document_id or resolved lookupID appears in history.
func (r *RAG) filterHistory(candidateList []*hybridCandidate, historyDocIDs []string) []*hybridCandidate {
	historyDocIDSet := make(map[string]bool, len(historyDocIDs))
	for _, docID := range historyDocIDs {
		if docID != "" {
			historyDocIDSet[docID] = true
		}
	}
	filtered := make([]*hybridCandidate, 0, len(candidateList))
	for _, cand := range candidateList {
		if historyDocIDSet[cand.DocumentID] {
			continue
		}
		lookupID := ResolveLookupID(cand.DocumentID, cand.Metadata)
		if lookupID != "" && historyDocIDSet[lookupID] {
			continue
		}
		filtered = append(filtered, cand)
	}
	return filtered
}

// bucketSummaries groups candidates by parent (lookupID) and keeps best summary or best non-summary.
func (r *RAG) bucketSummaries(candidates []*hybridCandidate) []*hybridCandidate {
	buckets := make(map[string][]*hybridCandidate)
	for _, cand := range candidates {
		lookupID := ResolveLookupID(cand.DocumentID, cand.Metadata)
		if lookupID == "" {
			lookupID = cand.DocumentID
		}
		buckets[lookupID] = append(buckets[lookupID], cand)
	}
	out := make([]*hybridCandidate, 0, len(buckets))
	for _, bucket := range buckets {
		if len(bucket) == 0 {
			continue
		}
		var bestSummary, bestNonSummary *hybridCandidate
		for _, cand := range bucket {
			if cand.Metadata["type"] == "summary" {
				if bestSummary == nil || cand.Score > bestSummary.Score {
					bestSummary = cand
				}
			} else {
				if bestNonSummary == nil || cand.Score > bestNonSummary.Score {
					bestNonSummary = cand
				}
			}
		}
		if bestSummary != nil {
			out = append(out, bestSummary)
		} else if bestNonSummary != nil {
			out = append(out, bestNonSummary)
		}
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].Score == out[j].Score {
			return out[i].DocumentID < out[j].DocumentID
		}
		return out[i].Score > out[j].Score
	})
	return out
}

// deduplicateShingles applies 5-gram containment dedup and hash-based fallback.
func (r *RAG) deduplicateShingles(candidates []*hybridCandidate, excludeHashes []string) []*hybridCandidate {
	filtered := make([]*hybridCandidate, 0, len(candidates))
	seenContent := make(map[string]bool)
	excludeHashSet := make(map[string]bool, len(excludeHashes))
	for _, h := range excludeHashes {
		if h != "" {
			excludeHashSet[h] = true
		}
	}
	for _, cand := range candidates {
		content := cand.Content
		if content == "" {
			filtered = append(filtered, cand)
			continue
		}
		contentHash := HashContent(NormalizeForHash(content))
		if contentHash != "" && excludeHashSet[contentHash] {
			continue
		}
		isDup := false
		for seenHash := range seenContent {
			if containment5gram(content, seenHash) > 0.9 {
				isDup = true
				break
			}
		}
		if !isDup {
			filtered = append(filtered, cand)
			if contentHash != "" {
				seenContent[contentHash] = true
			}
		}
	}
	return filtered
}

// formatMemoryBlock builds the final <memory> block from ranked candidates and returns it with count.
func (r *RAG) formatMemoryBlock(ctx context.Context, candidateList []*hybridCandidate, nResults int, doneLedger string, docContents map[string]string, excludeHashes []string) (string, int, error) {
	if docContents == nil {
		docContents = make(map[string]string)
	}
	var contextBuilder strings.Builder
	contextBuilder.WriteString("<memory>\n")

	processedDocIDs := make(map[string]bool)
	lastEmittedUser := ""
	addedDocs := 0
	excludeHashSet := make(map[string]bool, len(excludeHashes))
	for _, h := range excludeHashes {
		if h != "" {
			excludeHashSet[h] = true
		}
	}

	for _, cand := range candidateList {
		if addedDocs >= nResults {
			break
		}
		docID := cand.DocumentID
		if docID == "" {
			r.logger.Warn("Document is missing a document ID, skipping")
			continue
		}
		lookupID := ResolveLookupID(cand.DocumentID, cand.Metadata)
		if lookupID == "" {
			r.logger.Warn("Unable to resolve lookup identifier for document", zap.String("document_id", docID))
			continue
		}
		if processedDocIDs[lookupID] {
			continue
		}

		content, cached := docContents[lookupID]
		if !cached {
			docUUID, err := uuid.Parse(lookupID)
			if err != nil {
				r.logger.Warn("Invalid document identifier stored in metadata", zap.String("document_id", docID), zap.String("lookup_id", lookupID), zap.Error(err))
				continue
			}
			var err2 error
			content, err2 = r.getRelevantContent(ctx, docUUID, cand.Metadata, cand.WindowIndex)
			if err2 != nil {
				if errors.Is(err2, sql.ErrNoRows) {
					r.logger.Warn("No stored content found for document", zap.String("document_id", docID), zap.String("lookup_id", lookupID))
				} else {
					r.logger.Warn("Failed to load relevant content", zap.String("document_id", docID), zap.String("lookup_id", lookupID), zap.Int("window_index", cand.WindowIndex), zap.Error(err2))
				}
				continue
			}
			docContents[lookupID] = content
		}

		role := resolveRole(cand.Metadata)
		var lines []string
		if role == "fact" {
			var fact factStoredContent
			if err := json.Unmarshal([]byte(content), &fact); err == nil && (fact.User != "" || fact.Assistant != "" || fact.Tool != "") {
				skipFact := false
				if !skipFact && cand.Metadata != nil {
					if storedAssistantHash := cand.Metadata["assistant_hash"]; storedAssistantHash != "" {
						if excludeHashSet[storedAssistantHash] {
							skipFact = true
						}
					}
				}
				if !skipFact && fact.Assistant != "" {
					assistantHash := ComputeMessageContentHash("assistant", fact.Assistant)
					if assistantHash != "" && excludeHashSet[assistantHash] {
						skipFact = true
					}
				}
				if skipFact {
					continue
				}
				userTrimmed := canonicalizeFactText(fact.User)
				if userTrimmed != "" && userTrimmed != lastEmittedUser {
					lines = append(lines, fmt.Sprintf("- user: %s\n", userTrimmed))
					lastEmittedUser = userTrimmed
				}
				if fact.Assistant != "" {
					lines = append(lines, fmt.Sprintf("- assistant: %s\n", canonicalizeFactText(fact.Assistant)))
				}
				if fact.Tool != "" {
					lines = append(lines, fmt.Sprintf("- tool: %s\n", canonicalizeFactText(fact.Tool)))
				}
				for _, line := range lines {
					contextBuilder.WriteString(line)
				}
				processedDocIDs[lookupID] = true
				addedDocs++
				continue
			}
			assistantContent := canonicalizeFactText(content)
			if assistantContent != "" {
				lines = append(lines, fmt.Sprintf("- assistant: %s\n", assistantContent))
			}
		} else {
			label := role
			if cand.Metadata["type"] == "state" || role == "state" {
				label = "state"
			}
			lines = append(lines, fmt.Sprintf("- %s: %s\n", label, content))
		}
		for _, line := range lines {
			contextBuilder.WriteString(line)
		}
		processedDocIDs[lookupID] = true
		addedDocs++
	}

	if addedDocs == 0 {
		return "", 0, nil
	}
	if doneLedger != "" {
		contextBuilder.WriteString("\n")
		contextBuilder.WriteString(doneLedger)
		contextBuilder.WriteString("\n")
	}
	contextBuilder.WriteString("</memory>\n")
	return contextBuilder.String(), addedDocs, nil
}

// normalizeForEcho lowercases, strips punctuation, and collapses whitespace
// to enable robust near-equality checks between short queries and candidates.
func normalizeForEcho(s string) string {
	s = strings.ToLower(strings.TrimSpace(s))
	var b strings.Builder
	b.Grow(len(s))
	prevSpace := false
	for _, r := range s {
		if unicode.IsLetter(r) || unicode.IsNumber(r) {
			b.WriteRune(r)
			prevSpace = false
		} else if unicode.IsSpace(r) {
			if !prevSpace {
				b.WriteByte(' ')
				prevSpace = true
			}
		}
		// drop all other runes (punctuation/symbols)
	}
	return strings.TrimSpace(b.String())
}

// isQueryEcho returns true when the candidate content is an exact or near-exact
// echo of the query (after simple normalization). This helps down-rank user
// messages or trivial mirrors of the question.
func isQueryEcho(query, content string) bool {
	nq := normalizeForEcho(query)
	nc := normalizeForEcho(content)
	if nq == "" || nc == "" {
		return false
	}
	if nq == nc {
		return true
	}
	// If one contains the other and lengths are very close, treat as echo
	if strings.Contains(nc, nq) || strings.Contains(nq, nc) {
		lq, lc := len(nq), len(nc)
		minL := lq
		maxL := lc
		if lc < lq {
			minL, maxL = lc, lq
		}
		if minL*100 >= maxL*85 { // >=85% length overlap
			return true
		}
	}
	return false
}

func ensureCandidate(candidates map[string]*hybridCandidate, docID string, metadata map[string]string) *hybridCandidate {
	if cand, ok := candidates[docID]; ok {
		if cand.Metadata == nil {
			cand.Metadata = make(map[string]string)
		}
		for k, v := range metadata {
			if v == "" {
				continue
			}
			if existing, exists := cand.Metadata[k]; !exists || existing == "" {
				cand.Metadata[k] = v
			}
		}
		return cand
	}

	metaCopy := cloneStringMap(metadata)
	cand := &hybridCandidate{
		DocumentID: docID,
		Metadata:   metaCopy,
	}
	candidates[docID] = cand
	return cand
}

// shingles5gram generates 5-gram character shingles from a string.
// Returns a set (map[string]bool) of all 5-character substrings.
func shingles5gram(s string) map[string]bool {
	s = strings.ToLower(strings.TrimSpace(s))
	if len(s) < 5 {
		return map[string]bool{s: true}
	}

	shingles := make(map[string]bool)
	for i := 0; i <= len(s)-5; i++ {
		shingles[s[i:i+5]] = true
	}
	return shingles
}

// containment5gram computes the containment similarity between two strings using 5-gram shingles.
// Containment = |A ∩ B| / min(|A|, |B|)
// Returns a value between 0 and 1, where 1 means one string is fully contained in the other.
func containment5gram(s1, s2 string) float64 {
	shingles1 := shingles5gram(s1)
	shingles2 := shingles5gram(s2)

	if len(shingles1) == 0 || len(shingles2) == 0 {
		return 0.0
	}

	// Count intersection
	intersection := 0
	for shingle := range shingles1 {
		if shingles2[shingle] {
			intersection++
		}
	}

	// Compute containment: intersection / min(|A|, |B|)
	minSize := min(len(shingles1), len(shingles2))

	if minSize == 0 {
		return 0.0
	}

	return float64(intersection) / float64(minSize)
}
