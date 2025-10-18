package graph

import (
	"context"
	"database/sql"
	"fmt"
	"strings"

	"github.com/google/uuid"
	"github.com/lib/pq"
	"go.uber.org/zap"
)

// VariableAlias represents a canonical variable name and its raw aliases.
type VariableAlias struct {
	ID            string
	SessionID     string
	Dataset       string
	CanonicalName string
	RawAliases    []string
	CreatedAt     string
	UpdatedAt     string
}

// CreateOrUpdateAlias creates or updates a variable alias mapping.
// If the canonical name already exists for this session+dataset, it merges the raw aliases.
func (g *Graph) CreateOrUpdateAlias(ctx context.Context, sessionID, dataset, canonicalName string, rawAliases []string) error {
	if !g.Enabled() {
		return nil
	}

	sessionUUID, err := uuid.Parse(sessionID)
	if err != nil {
		return fmt.Errorf("invalid session_id UUID: %w", err)
	}

	// Check if alias already exists
	existingAlias, err := g.GetAlias(ctx, sessionID, dataset, canonicalName)
	if err != nil && err != sql.ErrNoRows {
		return fmt.Errorf("failed to check existing alias: %w", err)
	}

	if existingAlias != nil {
		// Merge aliases
		mergedAliases := mergeStringSlices(existingAlias.RawAliases, rawAliases)
		return g.updateAlias(ctx, existingAlias.ID, mergedAliases)
	}

	// Create new alias
	query := `
		INSERT INTO variable_aliases (session_id, dataset, canonical_name, raw_aliases)
		VALUES ($1, $2, $3, $4)
	`

	_, err = g.db.ExecContext(ctx, query, sessionUUID, dataset, canonicalName, pq.Array(rawAliases))
	if err != nil {
		return fmt.Errorf("failed to insert variable alias: %w", err)
	}

    g.logger.Debug("Created variable alias",
        zap.String("session", sessionID),
        zap.String("dataset", dataset),
        zap.String("canonical", canonicalName),
        zap.Strings("aliases", rawAliases))

    // Best-effort metadata touch
    _ = g.TouchMetadata(ctx, sessionID)

    return nil
}

// updateAlias updates the raw aliases for an existing variable alias.
func (g *Graph) updateAlias(ctx context.Context, aliasID string, rawAliases []string) error {
	aliasUUID, err := uuid.Parse(aliasID)
	if err != nil {
		return fmt.Errorf("invalid alias ID: %w", err)
	}

	query := `
		UPDATE variable_aliases
		SET raw_aliases = $1, updated_at = NOW()
		WHERE id = $2
	`

	_, err = g.db.ExecContext(ctx, query, pq.Array(rawAliases), aliasUUID)
	if err != nil {
		return fmt.Errorf("failed to update variable alias: %w", err)
	}

	return nil
}

// GetAlias retrieves a variable alias by session, dataset, and canonical name.
func (g *Graph) GetAlias(ctx context.Context, sessionID, dataset, canonicalName string) (*VariableAlias, error) {
	if !g.Enabled() {
		return nil, nil
	}

	sessionUUID, err := uuid.Parse(sessionID)
	if err != nil {
		return nil, fmt.Errorf("invalid session_id UUID: %w", err)
	}

	query := `
		SELECT id, session_id, dataset, canonical_name, raw_aliases, created_at, updated_at
		FROM variable_aliases
		WHERE session_id = $1 AND dataset = $2 AND canonical_name = $3
	`

	var alias VariableAlias
	var id, sessionIDUUID uuid.UUID
	var rawAliases pq.StringArray

	err = g.db.QueryRowContext(ctx, query, sessionUUID, dataset, canonicalName).
		Scan(&id, &sessionIDUUID, &alias.Dataset, &alias.CanonicalName, &rawAliases, &alias.CreatedAt, &alias.UpdatedAt)

	if err != nil {
		if err == sql.ErrNoRows {
			return nil, nil
		}
		return nil, fmt.Errorf("failed to query variable alias: %w", err)
	}

	alias.ID = id.String()
	alias.SessionID = sessionIDUUID.String()
	alias.RawAliases = rawAliases

	return &alias, nil
}

// GetAliasesForDataset retrieves all variable aliases for a session+dataset.
func (g *Graph) GetAliasesForDataset(ctx context.Context, sessionID, dataset string) ([]VariableAlias, error) {
	if !g.Enabled() {
		return nil, nil
	}

	sessionUUID, err := uuid.Parse(sessionID)
	if err != nil {
		return nil, fmt.Errorf("invalid session_id UUID: %w", err)
	}

	query := `
		SELECT id, session_id, dataset, canonical_name, raw_aliases, created_at, updated_at
		FROM variable_aliases
		WHERE session_id = $1 AND dataset = $2
		ORDER BY canonical_name ASC
	`

	rows, err := g.db.QueryContext(ctx, query, sessionUUID, dataset)
	if err != nil {
		return nil, fmt.Errorf("failed to query variable aliases: %w", err)
	}
	defer rows.Close()

	var aliases []VariableAlias
	for rows.Next() {
		var alias VariableAlias
		var id, sessionIDUUID uuid.UUID
		var rawAliases pq.StringArray

		err := rows.Scan(&id, &sessionIDUUID, &alias.Dataset, &alias.CanonicalName, &rawAliases, &alias.CreatedAt, &alias.UpdatedAt)
		if err != nil {
			return nil, fmt.Errorf("failed to scan variable alias row: %w", err)
		}

		alias.ID = id.String()
		alias.SessionID = sessionIDUUID.String()
		alias.RawAliases = rawAliases

		aliases = append(aliases, alias)
	}

	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("error iterating variable alias rows: %w", err)
	}

	return aliases, nil
}

// ResolveVariable attempts to resolve a variable name to its canonical form.
// Returns the canonical name if found, otherwise returns the original name.
func (g *Graph) ResolveVariable(ctx context.Context, sessionID, dataset, variableName string) (string, error) {
	if !g.Enabled() {
		return variableName, nil
	}

	aliases, err := g.GetAliasesForDataset(ctx, sessionID, dataset)
	if err != nil {
		return variableName, fmt.Errorf("failed to get aliases: %w", err)
	}

	// Normalize input for comparison (lowercase, trim)
    normalized := NormalizeVariableName(variableName)

	// Check if the variable name matches any canonical name
	for _, alias := range aliases {
        if NormalizeVariableName(alias.CanonicalName) == normalized {
			return alias.CanonicalName, nil
		}

		// Check if the variable name matches any raw alias
		for _, raw := range alias.RawAliases {
            if NormalizeVariableName(raw) == normalized {
                return alias.CanonicalName, nil
            }
        }
    }

	// No match found - return original
	return variableName, nil
}

// DeleteAliasesBySession deletes all variable aliases for a given session.
func (g *Graph) DeleteAliasesBySession(ctx context.Context, sessionID string) (int64, error) {
	if !g.Enabled() {
		return 0, nil
	}

	sessionUUID, err := uuid.Parse(sessionID)
	if err != nil {
		return 0, fmt.Errorf("invalid session UUID: %w", err)
	}

	query := `DELETE FROM variable_aliases WHERE session_id = $1`
	result, err := g.db.ExecContext(ctx, query, sessionUUID)
	if err != nil {
		return 0, fmt.Errorf("failed to delete aliases: %w", err)
	}

	count, err := result.RowsAffected()
	if err != nil {
		return 0, fmt.Errorf("failed to get rows affected: %w", err)
	}

	return count, nil
}

// normalizeVariableName normalizes a variable name for comparison.
// Lowercase, trim whitespace, replace common separators.
func NormalizeVariableName(name string) string {
    name = strings.TrimSpace(name)
    name = strings.ToLower(name)
    name = strings.ReplaceAll(name, "_", "")
    name = strings.ReplaceAll(name, "-", "")
    name = strings.ReplaceAll(name, " ", "")
    return name
}

// mergeStringSlices merges two string slices, removing duplicates.
func mergeStringSlices(a, b []string) []string {
	seen := make(map[string]bool)
	result := []string{}

	for _, item := range a {
		if !seen[item] {
			seen[item] = true
			result = append(result, item)
		}
	}

	for _, item := range b {
		if !seen[item] {
			seen[item] = true
			result = append(result, item)
		}
	}

	return result
}
