package rag

import "strings"

// querySynonyms maps statistical and domain-specific terms to their synonyms and variations.
// Organized by category for maintainability. Phrases should be lowercase.
var querySynonyms = map[string][]string{
	// ============================================================================
	// STATISTICAL SIGNIFICANCE & P-VALUES
	// ============================================================================
	"significant":     {"statistically significant", "sig", "p<0.05", "p<0.01", "p < 0.05", "alpha=0.05", "rejected null"},
	"not significant": {"non-significant", "ns", "n.s.", "p>0.05", "p > 0.05", "failed to reject", "no significant difference"},
	"p-value":         {"p value", "pvalue", "p=", "p <", "probability value", "significance level"},
	"alpha":           {"significance level", "α", "alpha level", "threshold", "type i error", "false positive rate"},

	// ============================================================================
	// CORRELATION & ASSOCIATION
	// ============================================================================
	"correlation":        {"relationship", "association", "r=", "r value", "correlated", "covariance", "linear relationship"},
	"correlated":         {"related", "associated", "linked", "connected"},
	"pearson":            {"pearson correlation", "pearson r", "product-moment correlation", "parametric correlation"},
	"spearman":           {"spearman rho", "spearman correlation", "rank correlation", "nonparametric correlation"},
	"kendall":            {"kendall tau", "kendall correlation"},
	"strong correlation": {"high correlation", "r>0.7", "r > 0.7", "strong relationship", "high r"},
	"weak correlation":   {"low correlation", "r<0.3", "r < 0.3", "weak relationship", "low r"},

	// ============================================================================
	// CENTRAL TENDENCY
	// ============================================================================
	"mean":   {"average", "arithmetic mean", "μ", "mean value", "avg"},
	"median": {"middle value", "50th percentile", "Q2", "second quartile"},
	"mode":   {"most frequent", "most common", "modal value"},

	// ============================================================================
	// DISPERSION & VARIABILITY
	// ============================================================================
	"variance":            {"var", "σ²", "sigma squared", "variability", "spread"},
	"standard deviation":  {"sd", "std", "stdev", "σ", "sigma", "standard error"},
	"standard error":      {"se", "SEM", "standard error of mean"},
	"range":               {"min max", "minimum maximum", "data range"},
	"interquartile range": {"iqr", "IQR", "Q3-Q1", "middle 50%"},
	"quartile":            {"percentile", "quantile"},

	// ============================================================================
	// NORMALITY & DISTRIBUTION
	// ============================================================================
	"normality":           {"normal distribution", "Gaussian", "bell curve", "normality test", "normality assumption"},
	"normal distribution": {"Gaussian distribution", "bell curve", "normal"},
	"shapiro-wilk":        {"shapiro wilk", "shapiro test", "SW test"},
	"kolmogorov-smirnov":  {"kolmogorov smirnov", "KS test", "K-S test"},
	"anderson-darling":    {"anderson darling", "AD test"},
	"jarque-bera":         {"jarque bera", "JB test"},
	"qq plot":             {"q-q plot", "quantile quantile plot", "normal probability plot"},
	"histogram":           {"distribution plot", "frequency distribution", "frequency plot"},
	"skewness":            {"skew", "asymmetry", "skewed"},
	"kurtosis":            {"tailedness", "peak"},

	// ============================================================================
	// PARAMETRIC TESTS - TWO GROUPS
	// ============================================================================
	"t-test":             {"t test", "ttest", "Student's t", "Student t", "t statistic"},
	"independent t-test": {"independent samples t-test", "two-sample t-test", "unpaired t-test", "between-subjects t-test"},
	"paired t-test":      {"paired samples t-test", "dependent t-test", "matched pairs", "within-subjects t-test", "repeated measures t-test"},

	// ============================================================================
	// PARAMETRIC TESTS - MULTIPLE GROUPS
	// ============================================================================
	"anova":                   {"analysis of variance", "F-test", "F test", "variance analysis"},
	"one-way anova":           {"oneway anova", "single factor anova"},
	"two-way anova":           {"factorial anova", "two factor anova"},
	"repeated measures anova": {"rm anova", "within-subjects anova"},
	"ancova":                  {"analysis of covariance", "covariance analysis"},
	"manova":                  {"multivariate anova", "multivariate analysis of variance"},
	"post-hoc":                {"post hoc", "pairwise comparison", "multiple comparisons"},
	"tukey":                   {"tukey hsd", "tukey test", "honest significant difference"},
	"bonferroni":              {"bonferroni correction", "bonferroni adjustment"},

	// ============================================================================
	// NONPARAMETRIC TESTS
	// ============================================================================
	"mann-whitney":   {"mann whitney", "mann whitney u", "wilcoxon rank sum", "rank sum test", "nonparametric t-test"},
	"wilcoxon":       {"wilcoxon signed rank", "signed rank test", "paired nonparametric"},
	"kruskal-wallis": {"kruskal wallis", "kruskal wallis h", "nonparametric anova"},
	"friedman":       {"friedman test", "nonparametric repeated measures"},
	"sign test":      {"median test"},

	// ============================================================================
	// CHI-SQUARE & CATEGORICAL TESTS
	// ============================================================================
	"chi-square":        {"chi square", "χ²", "chi2", "chi squared", "chisq"},
	"fisher's exact":    {"fisher exact", "fisher test", "exact test"},
	"mcnemar":           {"mcnemar test", "paired chi-square"},
	"contingency table": {"crosstab", "cross tabulation", "frequency table"},

	// ============================================================================
	// REGRESSION & MODELING
	// ============================================================================
	"regression":            {"linear model", "linear regression", "least squares", "predictive model"},
	"linear regression":     {"ols", "ordinary least squares", "simple regression"},
	"multiple regression":   {"multivariate regression", "multivariable regression"},
	"logistic regression":   {"logit", "binary regression", "classification model"},
	"polynomial regression": {"curvilinear regression", "nonlinear regression"},
	"r-squared":             {"r squared", "r²", "coefficient of determination", "r2", "rsquared"},
	"adjusted r-squared":    {"adjusted r squared", "adjusted r²"},
	"coefficient":           {"beta", "β", "slope", "regression coefficient", "parameter estimate"},
	"intercept":             {"constant", "y-intercept", "baseline"},
	"residual":              {"error", "prediction error", "residual error"},
	"fitted value":          {"predicted value", "estimate", "prediction"},

	// ============================================================================
	// ASSUMPTIONS & DIAGNOSTICS
	// ============================================================================
	"assumption":          {"prerequisite", "requirement", "condition", "test assumption"},
	"violated assumption": {"failed assumption", "assumption not met", "violated"},
	"homogeneity":         {"equal variance", "homoscedasticity", "variance homogeneity"},
	"levene":              {"levene test", "levene's test"},
	"bartlett":            {"bartlett test", "bartlett's test"},
	"independence":        {"independent observations", "no autocorrelation", "independent samples"},
	"multicollinearity":   {"collinearity", "correlated predictors", "vif", "variance inflation"},
	"outlier":             {"extreme value", "anomaly", "unusual observation", "influential point"},
	"influential point":   {"leverage point", "cook's distance"},

	// ============================================================================
	// EFFECT SIZES
	// ============================================================================
	"effect size":         {"magnitude", "practical significance", "effect measure"},
	"cohen's d":           {"cohen d", "standardized difference", "standardized mean difference"},
	"hedges' g":           {"hedges g", "corrected cohen's d"},
	"eta squared":         {"eta squared", "η²", "eta2", "proportion of variance"},
	"omega squared":       {"omega squared", "ω²", "omega2"},
	"partial eta squared": {"partial eta squared", "partial η²"},
	"cramer's v":          {"cramer v", "cramér's v", "phi coefficient"},
	"odds ratio":          {"or", "odds", "relative odds"},
	"relative risk":       {"rr", "risk ratio"},
	"confidence interval": {"ci", "confidence limits", "confidence bounds", "95% ci"},

	// ============================================================================
	// POWER ANALYSIS & SAMPLE SIZE
	// ============================================================================
	"power":          {"statistical power", "1-β", "probability of detection"},
	"power analysis": {"sample size calculation", "power calculation"},
	"sample size":    {"n", "number of observations", "number of subjects"},
	"beta":           {"type ii error", "false negative rate", "β"},

	// ============================================================================
	// DESCRIPTIVE STATISTICS
	// ============================================================================
	"descriptive statistics": {"summary statistics", "descriptive stats", "describe"},
	"frequency":              {"count", "n", "number of cases"},
	"percentage":             {"proportion", "%", "percent"},
	"distribution":           {"data distribution", "spread", "shape"},
	"summary":                {"overview", "synopsis", "descriptive summary"},

	// ============================================================================
	// DATA OPERATIONS & STRUCTURE
	// ============================================================================
	"dataset":       {"data file", "dataframe", "csv", "data", "file"},
	"variable":      {"column", "feature", "field", "attribute"},
	"variables":     {"columns", "features", "fields"},
	"column":        {"variable", "feature", "field"},
	"row":           {"observation", "case", "record", "sample"},
	"missing":       {"na", "null", "nan", "missing value", "missing data"},
	"missing data":  {"missing values", "missingness", "na values"},
	"file name":     {"filename", "dataset name", "file"},
	"uploaded file": {"data file", "loaded file"},
	"load data":     {"read data", "import data", "load file", "read file"},

	// ============================================================================
	// DATA PREPROCESSING
	// ============================================================================
	"transformation":     {"transform", "convert", "scale"},
	"log transformation": {"logarithmic transformation", "log transform", "ln"},
	"standardize":        {"z-score", "normalize", "scale", "standardization"},
	"normalize":          {"min-max scaling", "normalization"},
	"imputation":         {"impute", "fill missing", "missing value treatment"},
	"outlier removal":    {"outlier treatment", "remove outliers", "trim outliers"},

	// ============================================================================
	// VISUALIZATION
	// ============================================================================
	"plot":        {"graph", "chart", "visualization", "figure"},
	"scatterplot": {"scatter plot", "scatter chart", "xy plot"},
	"boxplot":     {"box plot", "box and whisker", "box-and-whisker plot"},
	"barplot":     {"bar plot", "bar chart", "bar graph"},
	"lineplot":    {"line plot", "line chart", "line graph"},

	// ============================================================================
	// COMPARISONS & DIFFERENCES
	// ============================================================================
	"difference":       {"comparison", "contrast", "differ", "change"},
	"compare":          {"comparison", "contrast", "compare groups", "between-group"},
	"compare groups":   {"group comparison", "between groups", "group difference"},
	"group":            {"category", "level", "factor level", "condition"},
	"before and after": {"pre post", "pre-post", "time 1 time 2", "baseline follow-up"},

	// ============================================================================
	// RESEARCH DESIGN
	// ============================================================================
	"experimental":  {"experiment", "randomized", "controlled"},
	"control group": {"control condition", "comparison group"},
	"treatment":     {"intervention", "experimental condition"},
	"randomization": {"random assignment", "randomized"},
	"baseline":      {"pre-treatment", "initial", "time 1"},

	// ============================================================================
	// TIME SERIES & LONGITUDINAL
	// ============================================================================
	"time series":     {"temporal data", "longitudinal", "repeated measures"},
	"trend":           {"pattern", "trajectory", "time trend"},
	"autocorrelation": {"serial correlation", "temporal correlation"},
	"seasonality":     {"seasonal pattern", "periodic pattern"},

	// ============================================================================
	// SURVIVAL ANALYSIS
	// ============================================================================
	"survival":       {"time-to-event", "duration", "failure time"},
	"kaplan-meier":   {"kaplan meier", "km curve", "survival curve"},
	"cox regression": {"cox proportional hazards", "cox model"},
	"hazard ratio":   {"hr", "relative hazard"},
	"censored":       {"censoring", "right-censored"},
}

// expandQuery augments the user query with statistical synonyms and related terms.
// This improves recall by ensuring searches match documents using different terminology.
func (r *RAG) expandQuery(query string) string {
	lower := strings.ToLower(query)

	// Track additions to avoid duplicates
	additionSet := make(map[string]struct{})

	// Sort phrases by length (longest first) to match more specific phrases first
	// This prevents "t-test" from matching before "independent t-test"
	var phrases []string
	for phrase := range querySynonyms {
		phrases = append(phrases, phrase)
	}

	// Sort by length descending
	for i := 0; i < len(phrases); i++ {
		for j := i + 1; j < len(phrases); j++ {
			if len(phrases[j]) > len(phrases[i]) {
				phrases[i], phrases[j] = phrases[j], phrases[i]
			}
		}
	}

	// Find matching phrases and collect synonyms
	for _, phrase := range phrases {
		if containsPhrase(lower, phrase) {
			synonyms := querySynonyms[phrase]
			for _, syn := range synonyms {
				syn = strings.TrimSpace(syn)
				if syn == "" {
					continue
				}
				additionSet[syn] = struct{}{}
			}
		}
	}

	// If no expansions found, return original query
	if len(additionSet) == 0 {
		return query
	}

	// Build expanded query, avoiding duplicates
	builder := strings.Builder{}
	builder.WriteString(query)

	// Pad existing query for boundary detection
	existingLower := " " + lower + " "

	for syn := range additionSet {
		synLower := strings.ToLower(syn)
		// Skip if synonym is already in the query
		if containsPhrase(existingLower, synLower) {
			continue
		}
		builder.WriteString(" ")
		builder.WriteString(syn)
	}

	return builder.String()
}

// containsPhrase checks if phrase exists as a word/phrase in text (not substring).
// Example: "test" won't match "testing", but will match "run test" or "test data"
func containsPhrase(text, phrase string) bool {
	phrase = strings.TrimSpace(phrase)
	if phrase == "" {
		return false
	}

	// Pad text for boundary checking
	if !strings.HasPrefix(text, " ") {
		text = " " + text
	}
	if !strings.HasSuffix(text, " ") {
		text = text + " "
	}

	// Check for phrase with word boundaries
	searchPatterns := []string{
		" " + phrase + " ", // Word boundaries on both sides
		" " + phrase + ".", // Phrase at end of sentence
		" " + phrase + ",", // Phrase before comma
		" " + phrase + "?", // Phrase at end of question
		" " + phrase + "!", // Phrase at end of exclamation
		" " + phrase + ":", // Phrase before colon
		" " + phrase + ";", // Phrase before semicolon
	}

	for _, pattern := range searchPatterns {
		if strings.Contains(text, pattern) {
			return true
		}
	}

	return false
}
