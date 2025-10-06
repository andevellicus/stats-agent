package rag

import (
	"testing"

	"stats-agent/config"

	"go.uber.org/zap"
)

func TestGenerateDeterministicFact(t *testing.T) {
	logger, _ := zap.NewDevelopment()
	cfg := &config.Config{}
	rag := &RAG{cfg: cfg, logger: logger}

	tests := []struct {
		name     string
		metadata map[string]string
		want     string
	}{
		{
			name: "assumption_check_with_all_fields",
			metadata: map[string]string{
				"primary_test":   "shapiro-wilk",
				"variables":      "residuals",
				"test_statistic": "W=0.923",
				"p_value":        "0.016",
				"sig_at_05":      "true",
				"analysis_stage": "assumption_check",
			},
			want: "Shapiro Wilk on residuals resulted in W=0.923 p=0.016 (significant at α=0.05).",
		},
		{
			name: "hypothesis_test_with_effect_size",
			metadata: map[string]string{
				"primary_test":   "t-test",
				"variables":      "score,group",
				"test_statistic": "t=2.34",
				"p_value":        "0.023",
				"sig_at_05":      "true",
				"effect_size":    "Cohen's d=0.42",
				"analysis_stage": "hypothesis_test",
			},
			want: "T Test on score,group resulted in t=2.34 p=0.023 (significant at α=0.05) with Cohen's d=0.42.",
		},
		{
			name: "descriptive_not_significant",
			metadata: map[string]string{
				"primary_test":   "pearson-correlation",
				"variables":      "age,income",
				"test_statistic": "r=0.12",
				"p_value":        "0.342",
				"sig_at_05":      "false",
				"analysis_stage": "descriptive",
			},
			want: "Pearson Correlation on age,income resulted in r=0.12 p=0.342 (not significant).",
		},
		{
			name: "minimal_metadata",
			metadata: map[string]string{
				"primary_test": "levene",
				"dataset":      "data.csv",
			},
			want: "Levene.",
		},
		{
			name: "no_metadata",
			metadata: map[string]string{
				"dataset": "experiment.csv",
			},
			want: "Analysis completed on experiment.csv.",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := rag.generateDeterministicFact(tt.metadata)
			if got != tt.want {
				t.Errorf("generateDeterministicFact() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestVerifyNumericAccuracy(t *testing.T) {
	logger, _ := zap.NewDevelopment()
	cfg := &config.Config{}
	rag := &RAG{cfg: cfg, logger: logger}

	tests := []struct {
		name       string
		fact       string
		metadata   map[string]string
		toolOutput string
		want       bool
	}{
		{
			name: "all_numbers_valid_in_metadata",
			fact: "Shapiro-Wilk test showed W=0.923, p=0.016",
			metadata: map[string]string{
				"test_statistic": "W=0.923",
				"p_value":        "0.016",
			},
			toolOutput: "",
			want:       true,
		},
		{
			name: "numbers_valid_in_tool_output",
			fact: "Mean age is 71.62 years with std=14.39",
			metadata: map[string]string{
				"variables": "age",
			},
			toolOutput: "Mean Age: N=56, mean=71.62, std=14.39, min=21 max=91",
			want:       true,
		},
		{
			name: "hallucinated_number",
			fact: "t-test resulted in t=2.55, p=0.023",
			metadata: map[string]string{
				"test_statistic": "t=2.34", // LLM changed 2.34 to 2.55
				"p_value":        "0.023",
			},
			toolOutput: "t-statistic: 2.34, p-value: 0.023",
			want:       false, // 2.55 not in metadata or tool output
		},
		{
			name: "scientific_notation_valid",
			fact: "p-value was 1.23e-05",
			metadata: map[string]string{
				"p_value": "1.23e-05",
			},
			toolOutput: "",
			want:       true,
		},
		{
			name: "no_numbers",
			fact: "Analysis completed successfully",
			metadata: map[string]string{
				"test": "chi-square",
			},
			toolOutput: "",
			want:       true,
		},
		{
			name: "sample_size_from_tool_output",
			fact: "Dataset contains 56 individuals",
			metadata: map[string]string{
				"dataset": "data.csv",
			},
			toolOutput: "Gender\nMALE      39\nFEMALE    17\nName: count, dtype: int64\nTotal: 56",
			want:       true, // 56 is in tool output
		},
		{
			name: "numbers_from_both_sources",
			fact: "t-test on 100 samples showed t=2.34, p=0.023",
			metadata: map[string]string{
				"test_statistic": "t=2.34",
				"p_value":        "0.023",
			},
			toolOutput: "Sample size: N=100",
			want:       true, // 100 from toolOutput, 2.34 and 0.023 from metadata
		},
		{
			name: "sentence_ending_period_not_part_of_number",
			fact: "The dataset contained 56 patients. The median GCS was 15.",
			metadata: map[string]string{
				"sample_size": "n=56",
			},
			toolOutput: "Median GCS: 15.00",
			want:       true, // Should match "56" and "15" despite sentence-ending periods
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := rag.verifyNumericAccuracy(tt.fact, tt.metadata, tt.toolOutput)
			if got != tt.want {
				t.Errorf("verifyNumericAccuracy() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestGenerateStructuredContentForBM25(t *testing.T) {
	logger, _ := zap.NewDevelopment()
	cfg := &config.Config{}
	rag := &RAG{cfg: cfg, logger: logger}

	tests := []struct {
		name     string
		metadata map[string]string
		want     string
	}{
		{
			name: "full_statistical_test",
			metadata: map[string]string{
				"primary_test":   "shapiro-wilk",
				"analysis_stage": "assumption_check",
				"p_value":        "0.016",
				"test_statistic": "W=0.923",
				"sig_at_05":      "true",
				"variables":      "residuals",
				"dataset":        "data.csv",
			},
			want: "test:shapiro-wilk stage:assumption_check p:0.016 stat:W=0.923 variables:residuals dataset:data.csv sig:true",
		},
		{
			name: "hypothesis_test_with_effect_size",
			metadata: map[string]string{
				"primary_test":   "t-test",
				"analysis_stage": "hypothesis_test",
				"p_value":        "0.023",
				"test_statistic": "t=2.34",
				"effect_size":    "Cohen's d=0.42",
				"variables":      "score,group",
				"sig_at_05":      "true",
			},
			want: "test:t-test stage:hypothesis_test p:0.023 stat:t=2.34 effect:Cohens-d=0.42 variables:score,group sig:true",
		},
		{
			name: "correlation_test",
			metadata: map[string]string{
				"primary_test":   "pearson-correlation",
				"analysis_stage": "descriptive",
				"test_statistic": "r=0.85",
				"p_value":        "0.001",
				"variables":      "age,income",
				"sample_size":    "100",
			},
			want: "test:pearson-correlation stage:descriptive p:0.001 stat:r=0.85 variables:age,income n:100",
		},
		{
			name: "minimal_metadata",
			metadata: map[string]string{
				"primary_test": "levene",
				"p_value":      "0.342",
			},
			want: "test:levene p:0.342",
		},
		{
			name: "empty_metadata",
			metadata: map[string]string{},
			want:     "",
		},
		{
			name: "metadata_with_whitespace",
			metadata: map[string]string{
				"primary_test": "  t-test  ",
				"p_value":      " 0.023 ",
				"variables":    " score, group ",
			},
			want: "test:t-test p:0.023 variables:score,-group",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := rag.generateStructuredContentForBM25(tt.metadata)
			if got != tt.want {
				t.Errorf("generateStructuredContentForBM25() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestNormalizeFieldKeyForBM25(t *testing.T) {
	tests := []struct {
		name string
		key  string
		want string
	}{
		{"primary_test", "primary_test", "test"},
		{"analysis_stage", "analysis_stage", "stage"},
		{"test_statistic", "test_statistic", "stat"},
		{"effect_size", "effect_size", "effect"},
		{"p_value", "p_value", "p"},
		{"sample_size", "sample_size", "n"},
		{"sig_at_05", "sig_at_05", "sig"},
		{"custom_field", "custom_field", "customfield"},
		{"another_key", "another_key", "anotherkey"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := normalizeFieldKeyForBM25(tt.key)
			if got != tt.want {
				t.Errorf("normalizeFieldKeyForBM25(%v) = %v, want %v", tt.key, got, tt.want)
			}
		})
	}
}

func TestNormalizeFieldValueForBM25(t *testing.T) {
	tests := []struct {
		name  string
		value string
		want  string
	}{
		{"simple", "shapiro-wilk", "shapiro-wilk"},
		{"with_spaces", "Cohen's d", "Cohens-d"},
		{"with_quotes", "\"value\"", "value"},
		{"with_single_quotes", "'value'", "value"},
		{"mixed", "  Test Value  ", "Test-Value"},
		{"comma_list", "age,gender,score", "age,gender,score"},
		{"equals_sign", "W=0.923", "W=0.923"},
		{"scientific_notation", "1.23e-05", "1.23e-05"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := normalizeFieldValueForBM25(tt.value)
			if got != tt.want {
				t.Errorf("normalizeFieldValueForBM25(%v) = %v, want %v", tt.value, got, tt.want)
			}
		})
	}
}
