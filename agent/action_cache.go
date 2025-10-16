package agent

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"regexp"
	"sort"
	"strings"
)

// ActionSignature uniquely identifies a statistical operation
type ActionSignature struct {
    Test       string   // e.g., "chi2", "mannwhitneyu", "shapiro"
    Dataset    string   // e.g., "unique_patients.csv"
    Variables  []string // sorted: ["Failure", "Gender"]
    Filters    []string // sorted: ["Age>50", "Side==1"]
    N          int      // sample size
    SchemaHash string   // hash of column list
    // CodeHash is a fallback fingerprint of the code when no test/variables
    // can be reliably extracted. It prevents over-coalescing distinct actions
    // into the same signature (e.g., printing columns vs ROC vs thresholds).
    CodeHash   string
    // SessionID isolates identical actions across sessions (no cross-session bleed).
    SessionID  string
}

// ComputeHash returns deterministic hash of signature
func (a *ActionSignature) ComputeHash() string {
	// Sort arrays for determinism
	vars := make([]string, len(a.Variables))
	copy(vars, a.Variables)
	sort.Strings(vars)

	filters := make([]string, len(a.Filters))
	copy(filters, a.Filters)
	sort.Strings(filters)

    normalized := ActionSignature{
        Test:       strings.ToLower(a.Test),
        Dataset:    a.Dataset,
        Variables:  vars,
        Filters:    filters,
        N:          a.N,
        SchemaHash: a.SchemaHash,
        SessionID:  a.SessionID,
        // Only include CodeHash when signature is otherwise too generic
        // (no test and no variables). This keeps known tests stable while
        // differentiating generic steps by exact code.
        CodeHash:   func() string {
            if a.Test == "" && len(vars) == 0 {
                return a.CodeHash
            }
            return ""
        }(),
    }

	jsonBytes, _ := json.Marshal(normalized)
	hash := sha256.Sum256(jsonBytes)
	return fmt.Sprintf("%x", hash[:8]) // First 16 hex chars
}

// String returns human-readable representation
func (a *ActionSignature) String() string {
    if len(a.Variables) > 0 {
        return fmt.Sprintf("%s(%s)", a.Test, strings.Join(a.Variables, ","))
    }
    if a.Test != "" {
        return a.Test
    }
    if a.CodeHash != "" {
        return fmt.Sprintf("code#%s", a.CodeHash[:6])
    }
    return ""
}

// ActionResult stores the outcome of an executed action
type ActionResult struct {
    Signature ActionSignature
    Output    string // Tool result
    Success   bool   // No error occurred
    Turn      int    // Which turn executed
    Attempt   int    // 1st attempt, 2nd (retry), etc.
    // CodeNormHash stores a whitespace-insensitive hash of the executed code
    // so we can enforce exact-phrase hysteresis before skipping repeats.
    CodeNormHash string
}

// ActionCache tracks executed actions to prevent repeats
type ActionCache struct {
	// Key: signature hash → result
	completed map[string]*ActionResult

	// Track last N actions (sliding window for repeat detection)
	recentActions []ActionSignature
	windowSize    int
}

// NewActionCache creates a new action cache with specified window size
func NewActionCache(windowSize int) *ActionCache {
	return &ActionCache{
		completed:     make(map[string]*ActionResult),
		recentActions: make([]ActionSignature, 0, windowSize),
		windowSize:    windowSize,
	}
}

// PurgeSession removes cached actions belonging to a specific session.
func (c *ActionCache) PurgeSession(sessionID string) {
    if sessionID == "" {
        return
    }
    // Remove from completed map
    for hash, res := range c.completed {
        if res != nil && res.Signature.SessionID == sessionID {
            delete(c.completed, hash)
        }
    }
    // Filter sliding window
    filtered := make([]ActionSignature, 0, len(c.recentActions))
    for _, sig := range c.recentActions {
        if sig.SessionID != sessionID {
            filtered = append(filtered, sig)
        }
    }
    c.recentActions = filtered
}

// Add records a completed action
func (c *ActionCache) Add(sig ActionSignature, result ActionResult) {
	hash := sig.ComputeHash()
	c.completed[hash] = &result

	// Add to sliding window
	c.recentActions = append(c.recentActions, sig)
	if len(c.recentActions) > c.windowSize {
		c.recentActions = c.recentActions[1:]
	}
}

// Get retrieves cached result if exists
func (c *ActionCache) Get(sig ActionSignature) (*ActionResult, bool) {
	hash := sig.ComputeHash()
	result, exists := c.completed[hash]
	return result, exists
}

// CountRecentRepeats counts how many times sig appears in last N actions
func (c *ActionCache) CountRecentRepeats(sig ActionSignature) int {
	hash := sig.ComputeHash()
	count := 0
	for _, recent := range c.recentActions {
		if recent.ComputeHash() == hash {
			count++
		}
	}
	return count
}

// ExtractActionSignature parses Python code to identify the statistical operation
func ExtractActionSignature(code, dataset string, n int, schemaHash string) *ActionSignature {
    raw := code
    code = strings.ToLower(code)

    sig := &ActionSignature{
        Dataset:    dataset,
        N:          n,
        SchemaHash: schemaHash,
        Variables:  []string{},
        Filters:    []string{},
    }

	// Detect test type (ordered by specificity)
    testPatterns := []struct {
        name    string
        pattern string
    }{
        {"chi2", `chi2_contingency|chisq\.test`},
        {"fisher", `fisher_exact|fisher\.test`},
        // Classification / diagnostics
        {"roc_auc", `roc_auc_score`},
        {"roc_curve", `roc_curve\(`},
        {"value_counts", `value_counts\(`},
        {"median_group", `\.median\(|median\(`},
		{"mannwhitneyu", `mannwhitneyu|wilcox\.test.*two\.sided|mann.whitney`},
		{"wilcoxon", `wilcoxon.*signed|signtest`},
		{"ttest_ind", `ttest_ind|t\.test.*independent`},
		{"ttest_rel", `ttest_rel|t\.test.*paired`},
		{"ttest", `ttest|t\.test`},
		{"shapiro", `shapiro|stats\.shapiro`},
		{"ks_test", `kstest|ks\.test|kolmogorov`},
		{"levene", `levene|stats\.levene`},
		{"bartlett", `bartlett`},
		{"pearsonr", `pearsonr|cor\.test.*pearson`},
		{"spearmanr", `spearmanr|cor\.test.*spearman`},
		{"kendalltau", `kendalltau|cor\.test.*kendall`},
		{"anova", `f_oneway|aov\(|anova\(`},
		{"kruskal", `kruskal|kruskal\.test`},
		{"friedman", `friedman`},
		{"linregress", `linregress|linearregression\(|lm\(`},
		{"logistic", `logisticregression|glm.*binomial`},
	}

	for _, tp := range testPatterns {
		matched, _ := regexp.MatchString(tp.pattern, code)
		if matched {
			sig.Test = tp.name
			break
		}
	}

	// If no test detected, check for descriptive operations
	if sig.Test == "" {
		if matched, _ := regexp.MatchString(`\.describe\(|\.summary\(`, code); matched {
			sig.Test = "describe"
		} else if matched, _ := regexp.MatchString(`\.corr\(|correlation.*matrix`, code); matched {
			sig.Test = "corr_matrix"
		} else if matched, _ := regexp.MatchString(`isnull\(|isna\(|missing`, code); matched {
			sig.Test = "missing_check"
		}
	}

	// Extract variables (e.g., df['Age'], df["Gender"])
    // Allow spaces and punctuation inside the brackets to capture real column names
    varPattern := regexp.MustCompile(`df\[['"]([^'"\]]+)['"]\]`)
	matches := varPattern.FindAllStringSubmatch(code, -1)
	varSet := make(map[string]bool)
	for _, match := range matches {
		if len(match) > 1 {
			varName := match[1]
			// Filter out common non-variable tokens
			if varName != "df" && varName != "data" && varName != "frame" {
				varSet[varName] = true
			}
		}
	}

	// Also check dot notation: df.Age
    dotPattern := regexp.MustCompile(`df\.([A-Za-z_][A-Za-z0-9_]*)`)
	dotMatches := dotPattern.FindAllStringSubmatch(code, -1)
	for _, match := range dotMatches {
		if len(match) > 1 {
			varName := match[1]
			// Filter out pandas methods
			if !isPandasMethod(varName) {
				varSet[varName] = true
			}
		}
	}

	// Convert to sorted slice
	for v := range varSet {
		sig.Variables = append(sig.Variables, v)
	}
	sort.Strings(sig.Variables)

    // When no specific test and no variables were extracted, attach a fallback
    // code hash to distinguish distinct generic actions by exact phrase.
    if sig.Test == "" && len(sig.Variables) == 0 {
        // Normalize code by trimming whitespace; preserve exact phrase identity
        // apart from surrounding whitespace and newlines.
        trimmed := strings.TrimSpace(raw)
        if trimmed != "" {
            sum := sha256.Sum256([]byte(trimmed))
            sig.CodeHash = fmt.Sprintf("%x", sum[:8])
        }
    }

    return sig
}

// isPandasMethod checks if a name is a common pandas method (not a column)
func isPandasMethod(name string) bool {
	methods := map[string]bool{
		"head": true, "tail": true, "describe": true, "info": true, "shape": true,
		"columns": true, "index": true, "dtypes": true, "values": true,
		"drop": true, "dropna": true, "fillna": true, "isnull": true, "isna": true,
		"groupby": true, "merge": true, "join": true, "sort_values": true,
		"reset_index": true, "set_index": true, "copy": true, "mean": true,
		"median": true, "std": true, "var": true, "sum": true, "count": true,
		"min": true, "max": true, "apply": true, "corr": true, "cov": true,
	}
	return methods[name]
}

// BuildDoneLedger creates compact "done=" string for memory/prompt
func (c *ActionCache) BuildDoneLedger(sessionID string) string {
    if len(c.completed) == 0 {
        return ""
    }

    var entries []string
    for _, result := range c.completed {
        if !result.Success {
            continue // Only show successful actions
        }

        if sessionID != "" && result.Signature.SessionID != sessionID {
            continue
        }
        s := result.Signature.String()
        if s == "" {
            continue
        }
        entries = append(entries, s)
    }

	if len(entries) == 0 {
		return ""
	}

	// Sort for consistent ordering
	sort.Strings(entries)

	return "done=[" + strings.Join(entries, ", ") + "]"
}
