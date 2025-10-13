package rag

import (
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"regexp"
	"strings"

	"github.com/google/uuid"
	"go.uber.org/zap"
)

type documentRecord struct {
	documentID string
	content    string
	metadata   map[string]string
}

var (
	metadataKeyPattern   = regexp.MustCompile(`^[a-zA-Z0-9_]+$`)
	datasetQueryRegex    = regexp.MustCompile(`(?i)([A-Za-z0-9_\-]+\.(?:csv|tsv|xlsx?|xls))`)
	metadataColonPattern = regexp.MustCompile(`(?i)\b(dataset|role|primary_test|analysis_stage)\s*:\s*["']?([^'"\n;,]+)`)
	metadataTestKeywords = []struct {
		value  string
		tokens []string
	}{
		{"t-test", []string{"t-test", "ttest"}},
		{"anova", []string{"anova", "aov"}},
		{"chi-square", []string{"chi-square", "chi square", "chisq", "chi2"}},
		{"pearson-correlation", []string{"pearson", "correlation"}},
		{"logistic-regression", []string{"logistic regression", "logistic"}},
	}
)

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
	queryBuilder.WriteString("SELECT id, content, metadata FROM rag_documents WHERE ")
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
    contextBuilder.WriteString("### Memory Context\n")

	lastEmittedUser := ""
	addedDocs := 0

	for _, record := range records {
		if addedDocs >= limit {
			break
		}

		docID := record.documentID
		if docID == "" {
			continue
		}

		lookupID := resolveLookupID(record.documentID, record.metadata)
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

		var lines []string
		if role == "fact" {
			var fact factStoredContent
			if err := json.Unmarshal([]byte(content), &fact); err == nil && (fact.User != "" || fact.Assistant != "" || fact.Tool != "") {
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

    // End of memory context (no XML tags)
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

	for _, match := range metadataColonPattern.FindAllStringSubmatch(query, -1) {
		key := strings.ToLower(strings.TrimSpace(match[1]))
		value := strings.TrimSpace(match[2])
		if key != "" && value != "" {
			addFilter(key, value)
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

	if strings.Contains(lower, "assistant message") || strings.Contains(lower, "assistant response") {
		addFilter("role", "assistant")
	} else if strings.Contains(lower, "user message") || strings.Contains(lower, "user question") {
		addFilter("role", "user")
	} else if strings.Contains(lower, "tool output") || strings.Contains(lower, "python result") || strings.Contains(lower, "code result") {
		addFilter("role", "tool")
	}

	if match := datasetQueryRegex.FindStringSubmatch(query); len(match) > 1 {
		if _, exists := filters["dataset"]; !exists {
			if len(filters) >= maxFilters {
				filters["dataset"] = match[1]
			} else {
				addFilter("dataset", match[1])
			}
		}
	}

	return filters
}
