package rag

import (
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"sort"
	"strings"

	"github.com/google/uuid"
	"go.uber.org/zap"
)

func (r *RAG) queryHybrid(ctx context.Context, sessionID string, query string, nResults int) (string, int, error) {
	if nResults <= 0 {
		return "", 0, nil
	}

	candidateLimit := max(nResults*4, 20)
	collection := r.db.GetCollection("long-term-memory", r.embedder)
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

	if collection == nil {
		r.logger.Warn("Vector collection not found, using BM25 fallback only")
	} else {
		total := collection.Count()
		if total > 0 {
			limit := candidateLimit
			if limit > total {
				limit = total
			}
			if limit > 0 {
				var where map[string]string
				if sessionID != "" {
					where = map[string]string{"session_id": sessionID}
				}
				semanticResults, err := collection.Query(ctx, query, limit, where, nil)
				if err != nil {
					return "", 0, fmt.Errorf("failed to query collection: %w", err)
				}
				for _, res := range semanticResults {
					docID := res.Metadata["document_id"]
					if docID == "" {
						r.logger.Warn("Vector result missing document_id, skipping")
						continue
					}

					similarity := float64(res.Similarity)
					if similarity < minSemanticSimilarity {
						continue
					}

					cand := ensureCandidate(candidates, docID, res.Metadata)
					if similarity > cand.SemanticScore {
						cand.SemanticScore = similarity
						cand.Content = res.Content
					}
					cand.HasSemantic = true
				}
			}
		}
	}

	bm25Results, err := r.store.SearchRAGDocumentsBM25(ctx, query, candidateLimit, sessionID)
	if err != nil {
		r.logger.Warn("BM25 search failed, falling back to semantic results only",
			zap.Error(err),
			zap.Int("candidate_limit", candidateLimit),
			zap.String("session_id", sessionID))
		bm25Results = nil
	}
	for _, bm := range bm25Results {
		docID := bm.Metadata["document_id"]
		if docID == "" {
			docID = bm.DocumentID.String()
			if bm.Metadata == nil {
				bm.Metadata = make(map[string]string)
			}
			bm.Metadata["document_id"] = docID
		}

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
		lookupID := resolveLookupID(cand.Metadata)
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

		content, err := r.store.GetRAGDocumentContent(ctx, lookupUUID)
		if err != nil {
			if !errors.Is(err, sql.ErrNoRows) {
				r.logger.Warn("Failed to load parent content for scoring",
					zap.Error(err),
					zap.String("lookup_id", lookupID),
					zap.String("document_id", cand.DocumentID))
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

	candidateList := make([]*hybridCandidate, 0, len(candidates))
	for _, cand := range candidates {
		weighted := 0.0
		weightSum := 0.0
		if cand.HasSemantic && maxSemantic > 0 {
			weighted += 0.7 * (cand.SemanticScore / maxSemantic)
			weightSum += 0.7
		}
		if cand.HasBM25 && maxBM > 0 {
			weighted += 0.3 * ((cand.BM25Score + cand.ExactBonus) / maxBM)
			weightSum += 0.3
		}
		combined := 0.0
		if weightSum > 0 {
			combined = weighted / weightSum
		}

		role := cand.Metadata["role"]
		docType := cand.Metadata["type"]
		if role == "fact" && docType != "chunk" {
			combined *= 1.3
		}
		if cand.Metadata["type"] == "summary" {
			combined *= 1.2
		}
		if cand.Content != "" && strings.Contains(cand.Content, "Error:") && !isQueryForError {
			combined *= 0.8
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

		cand.Score = combined
		candidateList = append(candidateList, cand)
	}

	sort.Slice(candidateList, func(i, j int) bool {
		if candidateList[i].Score == candidateList[j].Score {
			return candidateList[i].DocumentID < candidateList[j].DocumentID
		}
		return candidateList[i].Score > candidateList[j].Score
	})

	var contextBuilder strings.Builder
	contextBuilder.WriteString("<memory>\n")

	processedDocIDs := make(map[string]bool)
	lastEmittedUser := ""
	addedDocs := 0

	for _, cand := range candidateList {
		if addedDocs >= nResults {
			break
		}

		docID := cand.Metadata["document_id"]
		if docID == "" {
			r.logger.Warn("Document is missing a document_id, skipping")
			continue
		}

		lookupID := resolveLookupID(cand.Metadata)
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

			content, err = r.store.GetRAGDocumentContent(ctx, docUUID)
			if err != nil {
				if errors.Is(err, sql.ErrNoRows) {
					r.logger.Warn("No stored content found for document",
						zap.String("document_id", docID),
						zap.String("lookup_id", lookupID))
				} else {
					r.logger.Warn("Failed to load RAG document content",
						zap.String("document_id", docID),
						zap.String("lookup_id", lookupID),
						zap.Error(err))
				}
				continue
			}
			docContents[lookupID] = content
		}

		role := resolveRole(cand.Metadata)

		if role == "fact" {
			var fact factStoredContent
			if err := json.Unmarshal([]byte(content), &fact); err == nil && (fact.User != "" || fact.Assistant != "" || fact.Tool != "") {
				userTrimmed := canonicalizeFactText(fact.User)
				if userTrimmed != "" && userTrimmed != lastEmittedUser {
					contextBuilder.WriteString(fmt.Sprintf("- user: %s\n", userTrimmed))
					lastEmittedUser = userTrimmed
				}
				if fact.Assistant != "" {
					contextBuilder.WriteString(fmt.Sprintf("- assistant: %s\n", canonicalizeFactText(fact.Assistant)))
				}
				if fact.Tool != "" {
					contextBuilder.WriteString(fmt.Sprintf("- tool: %s\n", canonicalizeFactText(fact.Tool)))
				}

				processedDocIDs[lookupID] = true
				addedDocs++
				continue
			}

			assistantContent := canonicalizeFactText(content)
			if assistantContent != "" {
				contextBuilder.WriteString(fmt.Sprintf("- assistant: %s\n", assistantContent))
			}
		} else {
			contextBuilder.WriteString(fmt.Sprintf("- %s: %s\n", role, content))
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
