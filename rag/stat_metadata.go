package rag

import (
	"regexp"
	"sort"
	"strconv"
	"strings"
)

// StatMetadata holds extracted statistical information from code and results
type StatMetadata struct {
	TestTypes     []string        // All detected statistical tests
	PrimaryTest   string          // Most recent/important test
	AnalysisStage string          // assumption_check, hypothesis_test, modeling, post_hoc
	Variables     []string        // Column/variable names
	Dataset       string          // Filename being analyzed
	PValue        string          // Extracted p-value
	TestStatistic string          // t, F, chi2, etc.
	EffectSize    string          // Cohen's d, eta^2, etc.
	SampleSize    string          // N
	Significance  map[string]bool // sig_at_05, sig_at_01, etc.
}

// testPattern represents a detectable statistical test
type testPattern struct {
	regex    *regexp.Regexp
	testType string
}

// Comprehensive test patterns covering common statistical analyses
var testPatterns = []testPattern{
	// === Normality Tests (Assumption Checks) ===
	{regexp.MustCompile(`(?i)shapiro|stats\.shapiro`), "shapiro-wilk"},
	{regexp.MustCompile(`(?i)kstest|ks\.test`), "kolmogorov-smirnov"},
	{regexp.MustCompile(`(?i)anderson\(|ad\.test`), "anderson-darling"},
	{regexp.MustCompile(`(?i)jarque_bera|jarque\.bera|normaltest`), "jarque-bera"},

	// === Variance/Homogeneity Tests ===
	{regexp.MustCompile(`(?i)levene`), "levene"},
	{regexp.MustCompile(`(?i)bartlett`), "bartlett"},
	{regexp.MustCompile(`(?i)var\.test|f_test`), "f-test-variance"},

	// === Parametric Tests (Two Groups) ===
	{regexp.MustCompile(`(?i)ttest_ind|ttest_rel|paired\s+t|t\.test`), "t-test"},

	// === ANOVA Family ===
	{regexp.MustCompile(`(?i)f_oneway|aov\(|anova\(`), "anova"},
	{regexp.MustCompile(`(?i)f_twoway|anova.*formula`), "two-way-anova"},
	{regexp.MustCompile(`(?i)repeated.*anova|rm_anova`), "repeated-measures-anova"},
	{regexp.MustCompile(`(?i)ancova`), "ancova"},
	{regexp.MustCompile(`(?i)manova`), "manova"},

	// === Nonparametric Tests ===
	{regexp.MustCompile(`(?i)mannwhitneyu|wilcox\.test|mann-whitney`), "mann-whitney"},
	{regexp.MustCompile(`(?i)kruskal|kruskal\.test`), "kruskal-wallis"},
	{regexp.MustCompile(`(?i)friedman`), "friedman"},
	{regexp.MustCompile(`(?i)wilcoxon.*signed|signtest`), "wilcoxon-signed-rank"},

	// === Correlation ===
	{regexp.MustCompile(`(?i)pearsonr|cor\.test.*pearson`), "pearson-correlation"},
	{regexp.MustCompile(`(?i)spearmanr|cor\.test.*spearman`), "spearman-correlation"},
	{regexp.MustCompile(`(?i)kendalltau|cor\.test.*kendall`), "kendall-tau"},

	// === Categorical Tests ===
	{regexp.MustCompile(`(?i)chi2_contingency|chisq\.test`), "chi-square"},
	{regexp.MustCompile(`(?i)fisher_exact|fisher\.test`), "fisher-exact"},
	{regexp.MustCompile(`(?i)mcnemar`), "mcnemar"},

	// === Regression (Linear) ===
	{regexp.MustCompile(`(?i)LinearRegression\(|lm\(|OLS\(`), "linear-regression"},
	{regexp.MustCompile(`(?i)Ridge\(|RidgeCV`), "ridge-regression"},
	{regexp.MustCompile(`(?i)Lasso\(|LassoCV`), "lasso-regression"},
	{regexp.MustCompile(`(?i)ElasticNet`), "elasticnet-regression"},

	// === Regression (Generalized) ===
	{regexp.MustCompile(`(?i)LogisticRegression|glm.*binomial`), "logistic-regression"},
	{regexp.MustCompile(`(?i)PoissonRegression|glm.*poisson`), "poisson-regression"},
	{regexp.MustCompile(`(?i)glm.*negative.*binomial`), "negative-binomial-regression"},

	// === Mixed/Multilevel Models ===
	{regexp.MustCompile(`(?i)lmer\(|lme4`), "linear-mixed-effects"},
	{regexp.MustCompile(`(?i)glmer\(`), "generalized-linear-mixed-effects"},

	// === Machine Learning Models ===
	{regexp.MustCompile(`(?i)RandomForest(?:Classifier|Regressor)`), "random-forest"},
	{regexp.MustCompile(`(?i)GradientBoosting|XGBoost|LightGBM`), "gradient-boosting"},
	{regexp.MustCompile(`(?i)SVC|SVR|SVM`), "support-vector-machine"},

	// === Time Series ===
	{regexp.MustCompile(`(?i)adfuller|adf\.test`), "augmented-dickey-fuller"},
	{regexp.MustCompile(`(?i)kpss\.test`), "kpss-test"},
	{regexp.MustCompile(`(?i)acf\(|pacf\(`), "autocorrelation"},
	{regexp.MustCompile(`(?i)arima\(|ARIMA\(`), "arima"},

	// === Post-hoc Tests ===
	{regexp.MustCompile(`(?i)tukey|TukeyHSD`), "tukey-hsd"},
	{regexp.MustCompile(`(?i)bonferroni`), "bonferroni"},
	{regexp.MustCompile(`(?i)holm\s|holm-bonferroni`), "holm"},
	{regexp.MustCompile(`(?i)scheffe`), "scheffe"},
	{regexp.MustCompile(`(?i)dunnett`), "dunnett"},

	// === Survival Analysis ===
	{regexp.MustCompile(`(?i)kaplan.*meier|survfit`), "kaplan-meier"},
	{regexp.MustCompile(`(?i)coxph|cox.*proportional`), "cox-regression"},
	{regexp.MustCompile(`(?i)logrank`), "logrank-test"},

	// === Effect Size / Power ===
	{regexp.MustCompile(`(?i)cohen.*d|cohens_d`), "cohen-d"},
	{regexp.MustCompile(`(?i)eta.*squared|eta2`), "eta-squared"},
	{regexp.MustCompile(`(?i)power.*analysis|pwr\(`), "power-analysis"},
}

// Variable extraction patterns
var variablePatterns = []*regexp.Regexp{
	// Pattern 1: df['column'] or df["column"]
	regexp.MustCompile(`(?i)(?:df|data(?:set)?|frame)\s*\[\s*['"]([A-Za-z0-9_]+)['"]\s*\]`),

	// Pattern 2: df.column (dot notation)
	regexp.MustCompile(`(?i)(?:df|data(?:set)?|frame)\.([A-Za-z_][A-Za-z0-9_]*)`),

	// Pattern 3: Multiple columns df[['col1', 'col2']]
	regexp.MustCompile(`(?i)(?:df|data(?:set)?|frame)\s*\[\s*\[([^\]]+)\]\s*\]`),

	// Pattern 4: Function parameters x='column', data='column'
	regexp.MustCompile(`(?i)(?:x|y|hue|col|row|data)\s*=\s*['"]([A-Za-z0-9_]+)['"]`),

	// Pattern 5: Seaborn/plotly style
	regexp.MustCompile(`(?i)(?:x|y|color|size|hue)\s*=\s*['"]([A-Za-z0-9_]+)['"]`),
}

// Dataset detection patterns
var datasetPatterns = []*regexp.Regexp{
	regexp.MustCompile(`(?i)(?:read_csv|read\.csv|pd\.read_csv)\s*\(\s*['"]([^'"]+\.(?:csv|tsv))['"]`),
	regexp.MustCompile(`(?i)(?:read_excel|read\.excel|pd\.read_excel)\s*\(\s*['"]([^'"]+\.(?:xlsx?|xls))['"]`),
	regexp.MustCompile(`(?i)(?:read_table|read\.table)\s*\(\s*['"]([^'"]+)['"]`),
}

// Pandas methods to exclude from variable detection
var pandasMethods = map[string]bool{
	"head": true, "tail": true, "describe": true, "info": true, "shape": true,
	"columns": true, "index": true, "dtypes": true, "values": true,
	"drop": true, "dropna": true, "fillna": true, "isnull": true, "notnull": true,
	"groupby": true, "merge": true, "join": true, "concat": true, "append": true,
	"sort_values": true, "sort_index": true, "reset_index": true, "set_index": true,
	"iloc": true, "loc": true, "at": true, "iat": true,
	"mean": true, "median": true, "std": true, "var": true, "sum": true, "count": true,
	"min": true, "max": true, "apply": true, "map": true, "applymap": true,
	"pivot": true, "pivot_table": true, "melt": true, "stack": true, "unstack": true,
	"copy": true, "astype": true, "to_csv": true, "to_excel": true,
}

// ExtractStatisticalMetadata is the main entry point for metadata extraction
func ExtractStatisticalMetadata(code, result string) map[string]string {
	meta := &StatMetadata{
		Significance: make(map[string]bool),
	}

	// Extract all components
	meta.TestTypes = extractTests(code, result)
	if len(meta.TestTypes) > 0 {
		meta.PrimaryTest = meta.TestTypes[len(meta.TestTypes)-1] // Last test is usually primary
		meta.AnalysisStage = inferAnalysisStage(meta.PrimaryTest)
	}

	meta.Variables = extractVariables(code, result)
	meta.Dataset = extractDataset(code, result)

	// Extract numerical values from result
	values := extractNumericalValues(result)
	if pval, ok := values["p_value"]; ok {
		meta.PValue = pval
		if p, err := strconv.ParseFloat(pval, 64); err == nil {
			meta.Significance["sig_at_05"] = p < 0.05
			meta.Significance["sig_at_01"] = p < 0.01
			meta.Significance["sig_at_001"] = p < 0.001
		}
	}

	meta.TestStatistic = values["test_statistic"]
	meta.EffectSize = values["effect_size"]
	meta.SampleSize = values["sample_size"]

	return meta.ToMap()
}

// ToMap converts StatMetadata to a string map for storage
func (m *StatMetadata) ToMap() map[string]string {
	meta := make(map[string]string)

	if len(m.TestTypes) > 0 {
		meta["test_types"] = strings.Join(m.TestTypes, ",")
		meta["primary_test"] = m.PrimaryTest
	}

	if m.AnalysisStage != "" {
		meta["analysis_stage"] = m.AnalysisStage
	}

	if len(m.Variables) > 0 {
		meta["variables"] = strings.Join(m.Variables, ",")
		meta["variable_count"] = strconv.Itoa(len(m.Variables))
	}

	if m.Dataset != "" {
		meta["dataset"] = m.Dataset
	}

	if m.PValue != "" {
		meta["p_value"] = m.PValue
		meta["has_p_value"] = "true"
	}

	if m.TestStatistic != "" {
		meta["test_statistic"] = m.TestStatistic
	}

	if m.EffectSize != "" {
		meta["effect_size"] = m.EffectSize
	}

	if m.SampleSize != "" {
		meta["sample_size"] = m.SampleSize
	}

	// Add significance flags
	for k, v := range m.Significance {
		meta[k] = strconv.FormatBool(v)
	}

	return meta
}

// extractTests identifies all statistical tests in code and result
func extractTests(code, result string) []string {
	var tests []string
	seen := make(map[string]bool)

	for _, pattern := range testPatterns {
		if pattern.regex.MatchString(code) || pattern.regex.MatchString(result) {
			if !seen[pattern.testType] {
				tests = append(tests, pattern.testType)
				seen[pattern.testType] = true
			}
		}
	}

	return tests
}

// inferAnalysisStage determines the stage of analysis based on test type
func inferAnalysisStage(testType string) string {
	// Assumption checks
	assumptions := map[string]bool{
		"shapiro-wilk": true, "kolmogorov-smirnov": true, "anderson-darling": true,
		"jarque-bera": true, "levene": true, "bartlett": true, "f-test-variance": true,
	}
	if assumptions[testType] {
		return "assumption_check"
	}

	// Post-hoc tests
	postHoc := map[string]bool{
		"tukey-hsd": true, "bonferroni": true, "holm": true,
		"scheffe": true, "dunnett": true,
	}
	if postHoc[testType] {
		return "post_hoc"
	}

	// Modeling
	models := map[string]bool{
		"linear-regression": true, "ridge-regression": true, "lasso-regression": true,
		"elasticnet-regression": true, "logistic-regression": true,
		"poisson-regression": true, "negative-binomial-regression": true,
		"linear-mixed-effects": true, "generalized-linear-mixed-effects": true,
		"random-forest": true, "gradient-boosting": true, "support-vector-machine": true,
		"cox-regression": true, "arima": true,
	}
	if models[testType] {
		return "modeling"
	}

	// Descriptive
	descriptive := map[string]bool{
		"pearson-correlation": true, "spearman-correlation": true, "kendall-tau": true,
		"cohen-d": true, "eta-squared": true,
	}
	if descriptive[testType] {
		return "descriptive"
	}

	// Default to hypothesis test
	return "hypothesis_test"
}

// extractVariables identifies variable/column names from code and result
func extractVariables(code, result string) []string {
	unique := make(map[string]struct{})

	// Process code with all patterns
	for _, pattern := range variablePatterns {
		matches := pattern.FindAllStringSubmatch(code, -1)
		for _, match := range matches {
			if len(match) < 2 {
				continue
			}

			// Handle multiple columns in brackets: [['col1', 'col2']]
			if strings.Contains(match[1], ",") {
				cols := strings.Split(match[1], ",")
				for _, col := range cols {
					cleaned := strings.Trim(strings.TrimSpace(col), `'"`)
					if isValidVariable(cleaned) {
						unique[strings.ToLower(cleaned)] = struct{}{}
					}
				}
			} else {
				cleaned := strings.TrimSpace(match[1])
				if isValidVariable(cleaned) {
					unique[strings.ToLower(cleaned)] = struct{}{}
				}
			}
		}
	}

	// Also check result for column names in output tables
	// Common pattern: "column_name    value" in describe() or head() output
	resultPattern := regexp.MustCompile(`(?m)^([A-Za-z_][A-Za-z0-9_]*)\s+[\d\-\.]`)
	for _, match := range resultPattern.FindAllStringSubmatch(result, -1) {
		if len(match) > 1 && isValidVariable(match[1]) {
			unique[strings.ToLower(match[1])] = struct{}{}
		}
	}

	// Convert to sorted slice
	variables := make([]string, 0, len(unique))
	for v := range unique {
		variables = append(variables, v)
	}
	sort.Strings(variables)

	return variables
}

// isValidVariable checks if a string is a valid variable name
func isValidVariable(name string) bool {
	if name == "" || len(name) > 100 {
		return false
	}

	// Skip pandas methods
	if pandasMethods[strings.ToLower(name)] {
		return false
	}

	// Skip common Python keywords
	keywords := map[string]bool{
		"if": true, "else": true, "elif": true, "for": true, "while": true,
		"def": true, "class": true, "return": true, "import": true, "from": true,
		"true": true, "false": true, "none": true, "and": true, "or": true, "not": true,
	}
	if keywords[strings.ToLower(name)] {
		return false
	}

	// Must start with letter or underscore
	if name[0] != '_' && (name[0] < 'a' || name[0] > 'z') && (name[0] < 'A' || name[0] > 'Z') {
		return false
	}

	return true
}

// extractDataset identifies the dataset filename from code
func extractDataset(code, result string) string {
	for _, pattern := range datasetPatterns {
		if matches := pattern.FindStringSubmatch(code); len(matches) > 1 {
			return strings.TrimSpace(matches[1])
		}
	}

	// Also check result for dataset mentions
	for _, pattern := range datasetPatterns {
		if matches := pattern.FindStringSubmatch(result); len(matches) > 1 {
			return strings.TrimSpace(matches[1])
		}
	}

	return ""
}

// extractNumericalValues extracts statistical values from result text
func extractNumericalValues(result string) map[string]string {
	values := make(map[string]string)

	// P-values (multiple formats)
	pPatterns := []*regexp.Regexp{
		regexp.MustCompile(`(?i)p\s*[=:]\s*([\d.]+(?:e-?\d+)?)`),
		regexp.MustCompile(`(?i)p-value\s*[=:]\s*([\d.]+(?:e-?\d+)?)`),
		regexp.MustCompile(`(?i)pvalue\s*[=:]\s*([\d.]+(?:e-?\d+)?)`),
		regexp.MustCompile(`(?i)p\s*<\s*([\d.]+)`),
		regexp.MustCompile(`(?i)p\s*>\s*([\d.]+)`),
	}

	for _, re := range pPatterns {
		if matches := re.FindStringSubmatch(result); len(matches) > 1 {
			values["p_value"] = matches[1]
			break
		}
	}

	// Test statistics
    testStatPatterns := map[string]*regexp.Regexp{
        "t":    regexp.MustCompile(`(?i)(?:^|\s)t\s*[=:]\s*([-\d.]+)`),
        "F":    regexp.MustCompile(`(?i)(?:^|\s)F\s*[=:]\s*([\d.]+)`),
        "chi2": regexp.MustCompile(`(?i)chi2?\s*[=:]\s*([\d.]+)`),
        "z":    regexp.MustCompile(`(?i)(?:^|\s)z\s*[=:]\s*([-\d.]+)`),
        "r":    regexp.MustCompile(`(?i)(?:^|\s)r\s*[=:]\s*([-\d.]+)`),
        "U":    regexp.MustCompile(`(?i)(?:^|\s)U\s*[=:]\s*([\d.]+)`),
        "H":    regexp.MustCompile(`(?i)(?:^|\s)H\s*[=:]\s*([\d.]+)`),
        "W":    regexp.MustCompile(`(?i)(?:^|\s)W\s*[=:]\s*([\d.]+)`),
    }

	for key, re := range testStatPatterns {
		if matches := re.FindStringSubmatch(result); len(matches) > 1 {
			values["test_statistic"] = key + "=" + matches[1]
			break
		}
	}

	// R-squared (special handling)
	r2Patterns := []*regexp.Regexp{
		regexp.MustCompile(`(?i)R\^?2\s*[=:]\s*([\d.]+)`),
		regexp.MustCompile(`(?i)R-squared\s*[=:]\s*([\d.]+)`),
		regexp.MustCompile(`(?i)r_squared\s*[=:]\s*([\d.]+)`),
	}
	for _, re := range r2Patterns {
		if matches := re.FindStringSubmatch(result); len(matches) > 1 {
			values["test_statistic"] = "R²=" + matches[1]
			break
		}
	}

	// Effect sizes
	effectPatterns := map[string]*regexp.Regexp{
		"Cohen's d":  regexp.MustCompile(`(?i)Cohen'?s?\s*d\s*[=:]\s*([-\d.]+)`),
		"η²":         regexp.MustCompile(`(?i)eta\^?2\s*[=:]\s*([\d.]+)`),
		"Hedges' g":  regexp.MustCompile(`(?i)Hedges'?s?\s*g\s*[=:]\s*([-\d.]+)`),
		"Cramér's V": regexp.MustCompile(`(?i)Cramer'?s?\s*V\s*[=:]\s*([\d.]+)`),
		"ω²":         regexp.MustCompile(`(?i)omega\^?2\s*[=:]\s*([\d.]+)`),
	}

	for key, re := range effectPatterns {
		if matches := re.FindStringSubmatch(result); len(matches) > 1 {
			values["effect_size"] = key + "=" + matches[1]
			break
		}
	}

	// Sample size (multiple patterns)
	samplePatterns := []*regexp.Regexp{
		regexp.MustCompile(`(?i)(?:^|\s)n\s*=\s*(\d+)`),
		regexp.MustCompile(`(?i)sample\s+size\s*[=:]\s*(\d+)`),
		regexp.MustCompile(`(?i)observations?\s*[=:]\s*(\d+)`),
		regexp.MustCompile(`(?i)df\s*=\s*(\d+)`), // degrees of freedom as proxy
	}

	for _, re := range samplePatterns {
		if matches := re.FindStringSubmatch(result); len(matches) > 1 {
			values["sample_size"] = matches[1]
			break
		}
	}

	return values
}
