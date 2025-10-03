package rag

import (
	"context"

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
