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

func (r *RAG) queryHybrid(ctx context.Context, sessionID string, query string, nResults int, excludeHashes []string, historyDocIDs []string) (string, int, error) {
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
	candidates := make(map[string]*hybridCandidate)

	// Perform vector similarity search using pgvector
	queryEmbedding, err := r.embedder(ctx, query)
	if err != nil {
		r.logger.Warn("Failed to generate query embedding, using BM25 fallback only",
			zap.Error(err))
	} else if len(queryEmbedding) > 0 {
		semanticResults, err := r.store.VectorSearchRAGDocuments(ctx, queryEmbedding, candidateLimit, sessionID, excludeHashes)
		if err != nil {
			r.logger.Warn("Vector search failed, using BM25 fallback only",
				zap.Error(err))
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
					cand.WindowIndex = res.WindowIndex // Track which window matched
				}
				cand.HasSemantic = true
			}
		}
	}

	bm25Results, err := r.store.SearchRAGDocumentsBM25(ctx, query, candidateLimit, sessionID, excludeHashes)
	if err != nil {
		r.logger.Warn("BM25 search failed, falling back to semantic results only",
			zap.Error(err),
			zap.Int("candidate_limit", candidateLimit),
			zap.String("session_id", sessionID))
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

	if len(candidates) == 0 {
		return "", 0, nil
	}

	docContents := make(map[string]string)

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
			r.logger.Warn("Invalid lookup identifier for scoring",
				zap.String("lookup_id", lookupID),
				zap.String("document_id", cand.DocumentID))
			continue
		}

		// Use window-aware content retrieval (passes WindowIndex for multi-vector docs)
		content, err := r.getRelevantContent(ctx, lookupUUID, cand.Metadata, cand.WindowIndex)
		if err != nil {
			if !errors.Is(err, sql.ErrNoRows) {
				r.logger.Warn("Failed to load parent content for scoring",
					zap.Error(err),
					zap.String("lookup_id", lookupID),
					zap.String("document_id", cand.DocumentID),
					zap.Int("window_index", cand.WindowIndex))
			}
			continue
		}

		docContents[lookupID] = content
		cand.Content = content
	}

	var maxSemantic float64
	var maxBM float64
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

	candidateList := make([]*hybridCandidate, 0, len(candidates))
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
		// Only boost conversation facts (not document chunks)
		if role == "fact" && docType != "chunk" && docType != "document_chunk" {
			combined *= r.cfg.HybridFactBoost
		}
		if cand.Metadata["type"] == "summary" {
			combined *= r.cfg.HybridSummaryBoost
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

		// Penalize near-exact echoes of the user query to avoid retrieving
		// the question itself instead of useful context.
		if cand.Content != "" && isQueryEcho(query, cand.Content) {
			combined *= 0.1 // heavy penalty for echoes
		}

		cand.Score = combined
		candidateList = append(candidateList, cand)
	}

	sort.Slice(candidateList, func(i, j int) bool {
		if candidateList[i].Score == candidateList[j].Score {
			return candidateList[i].DocumentID < candidateList[j].DocumentID
		}
		return candidateList[i].Score > candidateList[j].Score
	})

	// === POST-QUERY PRUNING: 3-STAGE FILTERING ===

	// Build historyDocIDSet for O(1) lookup
	historyDocIDSet := make(map[string]bool, len(historyDocIDs))
	for _, docID := range historyDocIDs {
		if docID != "" {
			historyDocIDSet[docID] = true
		}
	}

	// FILTER 1: Drop candidates if document_id or lookupID is in history
	filtered1 := make([]*hybridCandidate, 0, len(candidateList))
	for _, cand := range candidateList {
		// Check direct document ID
		if historyDocIDSet[cand.DocumentID] {
			continue
		}

		// Check resolved lookup ID (parent/chunk relationships)
		lookupID := ResolveLookupID(cand.DocumentID, cand.Metadata)
		if lookupID != "" && historyDocIDSet[lookupID] {
			continue
		}

		filtered1 = append(filtered1, cand)
	}

	// FILTER 2: Proper summary bucketing
	// Group candidates by resolved parent ID, pick best summary if present, else best parent
	buckets := make(map[string][]*hybridCandidate)
	for _, cand := range filtered1 {
		lookupID := ResolveLookupID(cand.DocumentID, cand.Metadata)
		if lookupID == "" {
			lookupID = cand.DocumentID
		}
		buckets[lookupID] = append(buckets[lookupID], cand)
	}

	filtered2 := make([]*hybridCandidate, 0, len(buckets))
	for _, bucket := range buckets {
		if len(bucket) == 0 {
			continue
		}

		// Find best summary in bucket
		var bestSummary *hybridCandidate
		var bestNonSummary *hybridCandidate

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

		// Prefer summary if available, else use best non-summary
		if bestSummary != nil {
			filtered2 = append(filtered2, bestSummary)
		} else if bestNonSummary != nil {
			filtered2 = append(filtered2, bestNonSummary)
		}
	}

	// Re-sort after bucketing
	sort.Slice(filtered2, func(i, j int) bool {
		if filtered2[i].Score == filtered2[j].Score {
			return filtered2[i].DocumentID < filtered2[j].DocumentID
		}
		return filtered2[i].Score > filtered2[j].Score
	})

	// FILTER 3: Shingled containment similarity (5-gram, >0.9 threshold)
	// Also add hash-based dedup fallback
	filtered3 := make([]*hybridCandidate, 0, len(filtered2))
	seenContent := make(map[string]bool)

	// Build set of history content hashes for fallback dedup
	excludeHashSet := make(map[string]bool, len(excludeHashes))
	for _, hash := range excludeHashes {
		if hash != "" {
			excludeHashSet[hash] = true
		}
	}

	for _, cand := range filtered2 {
		content := cand.Content
		if content == "" {
			// No content to check, include it
			filtered3 = append(filtered3, cand)
			continue
		}

		// Hash-based dedup fallback (belt-and-suspenders)
		contentHash := HashContent(NormalizeForHash(content))
		if contentHash != "" && excludeHashSet[contentHash] {
			continue
		}

		// Check shingled containment against previously seen content
		isDuplicate := false
		for seenHash := range seenContent {
			if containment5gram(content, seenHash) > 0.9 {
				isDuplicate = true
				break
			}
		}

		if !isDuplicate {
			filtered3 = append(filtered3, cand)
			// Store normalized hash for future comparisons
			if contentHash != "" {
				seenContent[contentHash] = true
			}
		}
	}

	// Replace candidateList with filtered results
	candidateList = filtered3

	var contextBuilder strings.Builder
	contextBuilder.WriteString("<memory>\n")

	processedDocIDs := make(map[string]bool)
	lastEmittedUser := ""
	addedDocs := 0

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
				r.logger.Warn("Invalid document identifier stored in metadata",
					zap.String("document_id", docID),
					zap.String("lookup_id", lookupID),
					zap.Error(err))
				continue
			}

			// Use window-aware retrieval for multi-vector documents
			content, err = r.getRelevantContent(ctx, docUUID, cand.Metadata, cand.WindowIndex)
			if err != nil {
				if errors.Is(err, sql.ErrNoRows) {
					r.logger.Warn("No stored content found for document",
						zap.String("document_id", docID),
						zap.String("lookup_id", lookupID))
				} else {
					r.logger.Warn("Failed to load relevant content",
						zap.String("document_id", docID),
						zap.String("lookup_id", lookupID),
						zap.Int("window_index", cand.WindowIndex),
						zap.Error(err))
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
				// Check if assistant component is already in history
				// Don't check tool - tool outputs can be identical across different queries
				// e.g., "Check Age missingness" → "0" and "Check Gender missingness" → "0"
				skipFact := false

				// Fast path: check stored assistant_hash in metadata
				if !skipFact && cand.Metadata != nil {
					if storedAssistantHash := cand.Metadata["assistant_hash"]; storedAssistantHash != "" {
						if excludeHashSet[storedAssistantHash] {
							skipFact = true
						}
					}
				}

				// Fallback: compute hash from assistant content (for older facts without assistant_hash)
				if !skipFact && fact.Assistant != "" {
					assistantHash := ComputeMessageContentHash("assistant", fact.Assistant)
					if assistantHash != "" && excludeHashSet[assistantHash] {
						skipFact = true
					}
				}

				// Skip this fact if assistant is already in history
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
			lines = append(lines, fmt.Sprintf("- %s: %s\n", role, content))
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
	minSize := len(shingles1)
	if len(shingles2) < minSize {
		minSize = len(shingles2)
	}

	if minSize == 0 {
		return 0.0
	}

	return float64(intersection) / float64(minSize)
}
