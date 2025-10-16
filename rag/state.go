package rag

import (
    "context"
    "crypto/sha256"
    "fmt"
    "regexp"
    "sort"
    "strings"
    "time"

    "github.com/google/uuid"
    "go.uber.org/zap"
)

// buildDeterministicStateID creates a stable UUID for a (sessionID, dataset, stage)
func buildDeterministicStateID(sessionID, dataset, stage string) uuid.UUID {
    basis := strings.Join([]string{strings.TrimSpace(sessionID), strings.TrimSpace(dataset), strings.TrimSpace(stage)}, "|")
    return uuid.NewSHA1(uuid.NameSpaceOID, []byte(basis))
}

// parseColumnList extracts quoted column names from a string like "['A', 'B']"
func parseColumnList(input string) []string {
    re := regexp.MustCompile(`['"]([^'"]+)['"]`)
    matches := re.FindAllStringSubmatch(input, -1)
    cols := make([]string, 0, len(matches))
    for _, m := range matches {
        if len(m) > 1 {
            cols = append(cols, m[1])
        }
    }
    return cols
}

// extractSchemaFromResult tries to derive schema columns and n from a tool result output
func extractSchemaFromResult(result string) (cols []string, n int) {
    // Columns via Index([...]) or Columns: [...]
    idxPattern := regexp.MustCompile(`Index\(\[([^\]]+)\]`)
    if m := idxPattern.FindStringSubmatch(result); len(m) > 1 {
        cols = parseColumnList(m[1])
    }
    if len(cols) == 0 {
        colsPattern := regexp.MustCompile(`(?i)columns?[:\s]*\[([^\]]+)\]`)
        if m := colsPattern.FindStringSubmatch(result); len(m) > 1 {
            cols = parseColumnList(m[1])
        }
    }

    // n via Shape: (rows, cols) or n=... or observations: ...
    shapePattern := regexp.MustCompile(`(?i)shape[:\s]*\((\d+),\s*\d+\)`)
    if m := shapePattern.FindStringSubmatch(result); len(m) > 1 {
        fmt.Sscanf(m[1], "%d", &n)
    }
    if n == 0 {
        nPattern := regexp.MustCompile(`(?i)\bn\s*=\s*(\d+)`)
        if m := nPattern.FindStringSubmatch(result); len(m) > 1 {
            fmt.Sscanf(m[1], "%d", &n)
        }
    }
    if n == 0 {
        obsPattern := regexp.MustCompile(`(?i)observations?[:\s]+(\d+)`)
        if m := obsPattern.FindStringSubmatch(result); len(m) > 1 {
            fmt.Sscanf(m[1], "%d", &n)
        }
    }
    return cols, n
}

// computeSchemaHash returns short hash used across the agent (first 8 hex)
func computeSchemaHash(cols []string) string {
    if len(cols) == 0 {
        return ""
    }
    c := make([]string, len(cols))
    copy(c, cols)
    sort.Strings(c)
    joined := strings.Join(c, "|")
    sum := sha256.Sum256([]byte(joined))
    return fmt.Sprintf("%x", sum[:4])
}

type numericEvidence struct {
    P   string  // verbatim string of numeric p
    PVal float64
    W   string
    WVal float64
    V   string
    VVal float64
    R   string
    RVal float64
}

// extractNumericEvidence finds verbatim numeric snippets and parsed values
func extractNumericEvidence(result string) numericEvidence {
    var ev numericEvidence

    // p or p-value or p<
    pPatterns := []*regexp.Regexp{
        regexp.MustCompile(`(?i)\bp\s*[=:]\s*([\d.]+(?:e-?\d+)?)`),
        regexp.MustCompile(`(?i)\bp[- ]?value\s*[=:]\s*([\d.]+(?:e-?\d+)?)`),
        regexp.MustCompile(`(?i)\bp\s*<\s*([\d.]+(?:e-?\d+)?)`),
        regexp.MustCompile(`(?i)\bp\s*>\s*([\d.]+(?:e-?\d+)?)`),
    }
    for _, re := range pPatterns {
        if m := re.FindStringSubmatch(result); len(m) > 1 {
            ev.P = m[1]
            // best-effort parse
            fmt.Sscanf(ev.P, "%f", &ev.PVal)
            break
        }
    }

    // Shapiro W
    if m := regexp.MustCompile(`(?i)\bW\s*[=:]\s*([\d.]+)`).FindStringSubmatch(result); len(m) > 1 {
        ev.W = m[1]
        fmt.Sscanf(ev.W, "%f", &ev.WVal)
    }

    // Cramér's V
    if m := regexp.MustCompile(`(?i)cramer'?s?\s*V\s*[=:]\s*([\d.]+)`).FindStringSubmatch(result); len(m) > 1 {
        ev.V = m[1]
        fmt.Sscanf(ev.V, "%f", &ev.VVal)
    }

    // r (Pearson)
    if m := regexp.MustCompile(`(?i)\br\s*[=:]\s*([-\d.]+)`).FindStringSubmatch(result); len(m) > 1 {
        ev.R = m[1]
        fmt.Sscanf(ev.R, "%f", &ev.RVal)
    }

    return ev
}

func within01(x float64, allowZero bool, allowOne bool) bool {
    if allowZero && allowOne {
        return x >= 0.0 && x <= 1.0
    }
    if allowZero && !allowOne {
        return x >= 0.0 && x < 1.0
    }
    if !allowZero && allowOne {
        return x > 0.0 && x <= 1.0
    }
    return x > 0.0 && x < 1.0
}

// sumChi2Counts tries to sum obvious integer tokens as contingency counts.
// Conservative: only when we detect chi2 context.
func sumChi2Counts(result string) int {
    total := 0
    re := regexp.MustCompile(`(?m)\b(\d{1,7})\b`)
    for _, m := range re.FindAllStringSubmatch(result, -1) {
        if len(m) > 1 {
            var v int
            // ignore years like 2024 by capping typical count sizes (heuristic)
            fmt.Sscanf(m[1], "%d", &v)
            if v >= 0 && v < 1000000 {
                total += v
            }
        }
    }
    return total
}

// buildStateCardContent constructs the canonical state card text.
func buildStateCardContent(dataset string, n int, stage string, schemaCols []string, schemaHash string, result string, logger *zap.Logger) (string, bool) {
    // 1) Sanity: required fields
    if dataset == "" || n <= 0 || stage == "" || len(schemaCols) == 0 || schemaHash == "" {
        return "", false
    }

    // 2) Extract verbatim numeric evidence and validate ranges where applicable
    ev := extractNumericEvidence(result)

    // Sanity checks
    if ev.P != "" && !within01(ev.PVal, true, true) {
        if logger != nil {
            logger.Warn("Dropping state: p out of range", zap.Float64("p", ev.PVal))
        }
        return "", false
    }
    if ev.W != "" && !within01(ev.WVal, false, true) {
        if logger != nil {
            logger.Warn("Dropping state: W out of range", zap.Float64("W", ev.WVal))
        }
        return "", false
    }
    if ev.V != "" && !within01(ev.VVal, true, true) {
        if logger != nil {
            logger.Warn("Dropping state: V out of range", zap.Float64("V", ev.VVal))
        }
        return "", false
    }
    if ev.R != "" && (ev.RVal < -1.0 || ev.RVal > 1.0) {
        if logger != nil {
            logger.Warn("Dropping state: r out of range", zap.Float64("r", ev.RVal))
        }
        return "", false
    }

    // Chi-square count check (best-effort): if result mentions chi2 or crosstab
    lower := strings.ToLower(result)
    if strings.Contains(lower, "chi2") || strings.Contains(lower, "chisq") || strings.Contains(lower, "crosstab") {
        sum := sumChi2Counts(result)
        if sum > 0 && sum != n {
            if logger != nil {
                logger.Warn("Dropping state: contingency totals do not match n", zap.Int("sum", sum), zap.Int("n", n))
            }
            return "", false
        }
    }

    // Header line
    header := fmt.Sprintf("[dataset:%s | n:%d | stage:%s | schema_cols:%s | schema_hash:%s]",
        dataset,
        n,
        stage,
        strings.Join(schemaCols, ","),
        schemaHash,
    )

    // Detail lines: up to 3 sentences; only verbatim numerics
    var details []string
    if ev.W != "" && ev.P != "" {
        details = append(details, fmt.Sprintf("Shapiro-Wilk: W=%s, p=%s.", ev.W, ev.P))
    } else if ev.R != "" && ev.P != "" {
        details = append(details, fmt.Sprintf("Correlation: r=%s, p=%s.", ev.R, ev.P))
    } else if ev.P != "" {
        details = append(details, fmt.Sprintf("p=%s.", ev.P))
    }
    if ev.V != "" {
        details = append(details, fmt.Sprintf("Association: Cramér's V=%s.", ev.V))
    }

    if len(details) == 0 {
        // No reliable numeric details; store only header
        return header, true
    }
    if len(details) > 3 {
        details = details[:3]
    }
    return header + "\n" + strings.Join(details, "\n"), true
}

// ingestStateCard attempts to produce and persist a State Card based on an assistant+tool pair.
func (r *RAG) ingestStateCard(ctx context.Context, sessionID string, baseMeta map[string]string, code string, toolContent string) {
    if strings.TrimSpace(toolContent) == "" {
        return
    }

    // dataset from metadata or remembered
    dataset := strings.TrimSpace(baseMeta["dataset"])
    if dataset == "" {
        dataset = r.getSessionDataset(sessionID)
    }
    if dataset == "" {
        return
    }

    // derive schema from tool output
    schemaCols, n := extractSchemaFromResult(toolContent)
    if len(schemaCols) == 0 || n <= 0 {
        // evidence-only policy: skip if we cannot prove schema and n from tool output
        return
    }
    // enforce name coherence: columns as-is; variables inferred from code must be subset of schema cols when present
    vars := ExtractStatisticalMetadata(code, toolContent)["variables"]
    if vars != "" {
        varSet := make(map[string]struct{}, len(schemaCols))
        for _, c := range schemaCols { varSet[c] = struct{}{} }
        for _, v := range strings.Split(vars, ",") {
            v = strings.TrimSpace(v)
            if v == "" { continue }
            if _, ok := varSet[v]; !ok {
                // reject due to key drift
                if r.logger != nil {
                    r.logger.Warn("Dropping state: variable not in schema_cols", zap.String("var", v))
                }
                return
            }
        }
    }

    // stage from metadata extractor
    meta := ExtractStatisticalMetadata(code, toolContent)
    stage := strings.TrimSpace(meta["analysis_stage"])
    if stage == "" {
        // default to descriptive if we cannot identify; but only if numeric evidence exists
        stage = "descriptive"
    }

    schemaHash := computeSchemaHash(schemaCols)
    content, ok := buildStateCardContent(dataset, n, stage, schemaCols, schemaHash, toolContent, r.logger)
    if !ok || strings.TrimSpace(content) == "" {
        return
    }

    // Deterministic doc ID per (session,dataset,stage)
    docID := buildDeterministicStateID(sessionID, dataset, stage)

    // Prepare metadata
    md := map[string]string{
        "session_id": sessionID,
        "role":       "state",
        "type":       "state",
        "dataset":    dataset,
        "stage":      stage,
        "schema_hash": schemaHash,
        "source_type": "tool",
        "source_captured_at": time.Now().UTC().Format(time.RFC3339),
    }
    if h := baseMeta["tool_content_hash"]; h != "" { md["source_content_hash"] = h }

    // Conflict resolution logging: if an existing state exists and schema matches but content changes, log overwrite
    if existingID, existingContent, existingMeta, err := r.store.FindStateDocument(ctx, sessionID, dataset, stage); err == nil && existingID != uuid.Nil {
        if existingMeta["schema_hash"] == schemaHash && existingContent != content {
            if r.logger != nil {
                r.logger.Info("Overwriting state due to updated evidence (same schema)",
                    zap.String("dataset", dataset), zap.String("stage", stage))
            }
        }
    }

    // Upsert document (content as StoredContent and also as window text via createEmbeddingWindows downstream)
    if _, err := r.store.UpsertDocument(ctx, docID, content, md, ""); err != nil {
        if r.logger != nil {
            r.logger.Warn("Failed to upsert state document", zap.Error(err))
        }
        return
    }

    // Create a minimal embedding window for search (use full content)
    // Reuse createEmbeddingWindows path used by persistPreparedDocument by calling it directly here
    windows, err := r.createEmbeddingWindows(ctx, content)
    if err == nil {
        for _, w := range windows {
            if e := r.store.CreateEmbedding(ctx, docID, w.WindowIndex, w.WindowStart, w.WindowEnd, w.WindowText, w.Embedding); e != nil {
                r.logger.Warn("Failed to store embedding window for state", zap.Error(e))
            }
        }
    } else if r.logger != nil {
        r.logger.Warn("Failed to create embedding for state", zap.Error(err))
    }

    // Rolling window: keep most recent 4 state docs for this session
    docs, err := r.store.ListStateDocuments(ctx, sessionID)
    if err == nil && len(docs) > 4 {
        for i := 4; i < len(docs); i++ { // delete older beyond first 4 (newest first)
            _ = r.store.DeleteRAGDocument(ctx, docs[i].ID)
        }
    }
}
