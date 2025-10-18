package rag

import (
    "context"
    "strings"

    "go.uber.org/zap"
)

func (r *RAG) Query(ctx context.Context, sessionID string, query string, nResults int, excludeHashes []string, historyDocIDs []string, doneLedger string, mode string) (string, error) {
    expandedQuery := r.expandQuery(ctx, sessionID, query)
    context, hits, err := r.queryHybrid(ctx, sessionID, expandedQuery, nResults, excludeHashes, historyDocIDs, doneLedger, mode)
    if err != nil {
        return "", err
    }

    if hits > 0 || !r.cfg.EnableMetadataFallback {
        return context, nil
    }

    filters := extractSimpleMetadata(expandedQuery, r.cfg.MetadataFallbackMaxFilters)
    // Fallback: if no inline metadata tokens were found, use remembered dataset
    if len(filters) == 0 {
        if ds := strings.TrimSpace(r.getSessionDataset(sessionID)); ds != "" {
            filters = map[string]string{"dataset": ds}
        }
    }
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
