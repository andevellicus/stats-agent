package rag

import (
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"regexp"
	"sort"
	"strings"

	"github.com/google/uuid"
	"go.uber.org/zap"
)

func (r *RAG) Query(ctx context.Context, sessionID string, query string, nResults int) (string, error) {
	context, hits, err := r.queryHybrid(ctx, sessionID, query, nResults)
	if err != nil {
		return "", err
	}

	if hits > 0 || !r.cfg.EnableMetadataFallback {
		return context, nil
	}

	filters := extractSimpleMetadata(query, r.cfg.MetadataFallbackMaxFilters)
	if len(filters) == 0 {
		return context, nil
	}

	r.logger.Debug("Hybrid retrieval returned no hits, falling back to metadata query",
		zap.String("query", query),
		zap.Any("filters", filters))

	fallbackContext, err := r.QueryByMetadata(ctx, sessionID, filters, nResults)
	if err != nil {
		return "", err
	}
	if fallbackContext == "" {
		return "", nil
	}
	return fallbackContext, nil
}

func (r *RAG) queryHybrid(ctx context.Context, sessionID string, query string, nResults int) (string, int, error) {
	if nResults <= 0 {
		return "", 0, nil
	}

	collection := r.db.GetCollection("long-term-memory", r.embedder)
	const maxHybridCandidates = 100
	minSemanticSimilarity := r.cfg.SemanticSimilarityThreshold
	if minSemanticSimilarity <= 0 || minSemanticSimilarity > 1 {
		minSemanticSimilarity = 0.7
	}
	minBM25Score := r.cfg.BM25ScoreThreshold
	if minBM25Score < 0 {
		minBM25Score = 0
	}
	candidateLimit := max(nResults*4, 20)
	if candidateLimit > maxHybridCandidates {
		candidateLimit = maxHybridCandidates
	}

	lowerQuery := strings.ToLower(query)
	isQueryForError := strings.Contains(lowerQuery, "error")
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
		return "", 0, fmt.Errorf("failed to run BM25 search: %w", err)
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

type documentRecord struct {
	documentID string
	content    string
	metadata   map[string]string
}

var metadataKeyPattern = regexp.MustCompile(`^[a-zA-Z0-9_]+$`)
var datasetQueryRegex = regexp.MustCompile(`(?i)([A-Za-z0-9_\-]+\.(?:csv|tsv|xlsx?|xls))`)

var metadataTestKeywords = []struct {
	value  string
	tokens []string
}{
	{"t-test", []string{"t-test", "ttest"}},
	{"anova", []string{"anova", "aov"}},
	{"chi-square", []string{"chi-square", "chi square", "chisq", "chi2"}},
	{"pearson-correlation", []string{"pearson", "correlation"}},
	{"logistic-regression", []string{"logistic regression", "logistic"}},
}

func (r *RAG) QueryByMetadata(ctx context.Context, sessionID string, filters map[string]string, nResults int) (string, error) {
	if nResults <= 0 {
		return "", nil
	}

	conditions := make([]string, 0)
	args := make([]any, 0)

	for key, value := range filters {
		key = strings.TrimSpace(key)
		value = strings.TrimSpace(value)
		if key == "" || value == "" {
			continue
		}
		if !metadataKeyPattern.MatchString(key) {
			r.logger.Warn("Skipping metadata filter with invalid key", zap.String("key", key))
			continue
		}
		filterJSON, err := json.Marshal(map[string]string{key: value})
		if err != nil {
			r.logger.Warn("Failed to marshal metadata filter", zap.String("key", key), zap.Error(err))
			continue
		}
		conditions = append(conditions, fmt.Sprintf("metadata @> $%d::jsonb", len(args)+1))
		args = append(args, string(filterJSON))
	}

	if sessionID != "" {
		filterJSON, err := json.Marshal(map[string]string{"session_id": sessionID})
		if err != nil {
			return "", fmt.Errorf("marshal session filter: %w", err)
		}
		conditions = append(conditions, fmt.Sprintf("metadata @> $%d::jsonb", len(args)+1))
		args = append(args, string(filterJSON))
	}

	if len(conditions) == 0 {
		return "", fmt.Errorf("at least one metadata filter or sessionID must be provided")
	}

	queryBuilder := strings.Builder{}
	queryBuilder.WriteString("SELECT document_id, content, metadata FROM rag_documents WHERE ")
	queryBuilder.WriteString(strings.Join(conditions, " AND "))
	queryBuilder.WriteString(fmt.Sprintf(" ORDER BY created_at DESC LIMIT $%d", len(args)+1))
	args = append(args, nResults)

	rows, err := r.store.DB.QueryContext(ctx, queryBuilder.String(), args...)
	if err != nil {
		return "", fmt.Errorf("query rag_documents by metadata: %w", err)
	}
	defer rows.Close()

	var records []documentRecord
	for rows.Next() {
		var docID uuid.UUID
		var content string
		var metadataBytes []byte
		if err := rows.Scan(&docID, &content, &metadataBytes); err != nil {
			return "", fmt.Errorf("scan rag_documents row: %w", err)
		}

		meta := make(map[string]string)
		if len(metadataBytes) > 0 {
			if err := json.Unmarshal(metadataBytes, &meta); err != nil {
				r.logger.Warn("Failed to unmarshal document metadata", zap.Error(err), zap.String("document_id", docID.String()))
			}
		}

		if _, ok := meta["document_id"]; !ok {
			meta["document_id"] = docID.String()
		}

		records = append(records, documentRecord{
			documentID: docID.String(),
			content:    content,
			metadata:   meta,
		})
	}

	if err := rows.Err(); err != nil {
		return "", fmt.Errorf("iterate rag_documents rows: %w", err)
	}

	if len(records) == 0 {
		return "", nil
	}

	return r.renderRecordsToMemory(ctx, records, nResults), nil
}

func (r *RAG) renderRecordsToMemory(ctx context.Context, records []documentRecord, limit int) string {
	docContents := make(map[string]string)
	processedDocIDs := make(map[string]bool)
	var contextBuilder strings.Builder
	contextBuilder.WriteString("<memory>\n")

	lastEmittedUser := ""
	addedDocs := 0

	for _, record := range records {
		if addedDocs >= limit {
			break
		}

		docID := record.metadata["document_id"]
		if docID == "" {
			continue
		}

		lookupID := resolveLookupID(record.metadata)
		if lookupID == "" {
			r.logger.Warn("Unable to resolve lookup identifier for document", zap.String("document_id", docID))
			continue
		}

		if processedDocIDs[lookupID] {
			continue
		}

		content := record.content
		if lookupID != docID {
			if cached, ok := docContents[lookupID]; ok {
				content = cached
			} else {
				parsed, err := uuid.Parse(lookupID)
				if err != nil {
					r.logger.Warn("Invalid document identifier stored in metadata",
						zap.String("document_id", docID),
						zap.String("lookup_id", lookupID),
						zap.Error(err))
					continue
				}
				fetched, err := r.store.GetRAGDocumentContent(ctx, parsed)
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
				docContents[lookupID] = fetched
				content = fetched
			}
		} else {
			docContents[lookupID] = content
		}

		role := resolveRole(record.metadata)

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

	contextBuilder.WriteString("</memory>\n")
	return contextBuilder.String()
}

func extractSimpleMetadata(query string, maxFilters int) map[string]string {
	if maxFilters <= 0 {
		maxFilters = 3
	}

	filters := make(map[string]string)
	lower := strings.ToLower(query)

	addFilter := func(key, value string) {
		if len(filters) >= maxFilters {
			return
		}
		if key == "" || value == "" {
			return
		}
		if _, exists := filters[key]; !exists {
			filters[key] = value
		}
	}

	if strings.Contains(lower, "p<0.05") || strings.Contains(lower, "p < 0.05") ||
		(strings.Contains(lower, "p-value") && strings.Contains(lower, "significant")) ||
		(strings.Contains(lower, "significant") && strings.Contains(lower, "result")) {
		addFilter("sig_at_05", "true")
	}

	for _, mapping := range metadataTestKeywords {
		for _, token := range mapping.tokens {
			if strings.Contains(lower, token) {
				addFilter("primary_test", mapping.value)
				break
			}
		}
	}

	if match := datasetQueryRegex.FindStringSubmatch(query); len(match) > 1 {
		addFilter("dataset", match[1])
	}

	return filters
}
