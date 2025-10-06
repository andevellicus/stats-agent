package config

import (
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/spf13/viper"
	"go.uber.org/zap"
)

const (
	defaultContextSoftLimitRatio            = 0.75
	defaultPythonExecutorCooldownSeconds    = 5 * time.Second
	defaultPythonExecutorDialTimeoutSeconds = 3 * time.Second
	defaultPythonExecutorIOTimeoutSeconds   = 60 * time.Second
	defaultPythonExecutorMaxConnections     = 4
	defaultMaxEmbeddingChars                = 1000
	defaultMaxEmbeddingTokens               = 250
	defaultEmbeddingTokenSoftLimit          = 450
	defaultEmbeddingTokenTarget             = 400
	defaultMinTokenCheckCharThreshold       = 100
	defaultMaxHybridCandidates              = 100
	defaultHybridSemanticWeight             = 0.7
	defaultHybridBM25Weight                 = 0.3
	defaultRAGContextMaxChars               = 12000
	defaultHybridFactBoost                  = 1.3
	defaultHybridSummaryBoost               = 1.2
	defaultHybridErrorPenalty               = 0.8
	defaultPDFTokenThreshold                = 0.75
	defaultPDFFirstPagesPriority            = 3
	defaultPDFEnableTableDetection          = true
	defaultPDFSentenceBoundaryTruncate      = true
)

// Config holds the application's configuration
type Config struct {
	WebPort                          int           `mapstructure:"WEB_PORT"`
	DatabaseURL                      string        `mapstructure:"DATABASE_URL"`
	PythonExecutorAddress            string        `mapstructure:"PYTHON_EXECUTOR_ADDRESS"`
	PythonExecutorAddresses          []string      `mapstructure:"PYTHON_EXECUTOR_ADDRESSES"`
	PythonExecutorPool               []string      `mapstructure:"PYTHON_EXECUTOR_POOL"`
	MainLLMHost                      string        `mapstructure:"MAIN_LLM_HOST"`
	EmbeddingLLMHost                 string        `mapstructure:"EMBEDDING_LLM_HOST"`
	SummarizationLLMHost             string        `mapstructure:"SUMMARIZATION_LLM_HOST"`
	MaxTurns                         int           `mapstructure:"MAX_TURNS"`
	RAGResults                       int           `mapstructure:"RAG_RESULTS"`
	ContextLength                    int           `mapstructure:"CONTEXT_LENGTH"`
	ContextSoftLimitRatio            float64       `mapstructure:"CONTEXT_SOFT_LIMIT_RATIO"`
	MaxRetries                       int           `mapstructure:"MAX_RETRIES"`
	RetryDelaySeconds                time.Duration `mapstructure:"RETRY_DELAY_SECONDS"`
	ConsecutiveErrors                int           `mapstructure:"CONSECUTIVE_ERRORS"`
	LLMRequestTimeout                time.Duration `mapstructure:"LLM_REQUEST_TIMEOUT"`
	CleanupEnabled                   bool          `mapstructure:"CLEANUP_ENABLED"`
	CleanupInterval                  time.Duration `mapstructure:"CLEANUP_INTERVAL"`
	SessionRetentionAge              time.Duration `mapstructure:"SESSION_RETENTION_AGE"`
	RateLimitMessagesPerMin          int           `mapstructure:"RATE_LIMIT_MESSAGES_PER_MIN"`
	RateLimitFilesPerHour            int           `mapstructure:"RATE_LIMIT_FILES_PER_HOUR"`
	RateLimitBurstSize               int           `mapstructure:"RATE_LIMIT_BURST_SIZE"`
	SemanticSimilarityThreshold      float64       `mapstructure:"SEMANTIC_SIMILARITY_THRESHOLD"`
	BM25ScoreThreshold               float64       `mapstructure:"BM25_SCORE_THRESHOLD"`
	EnableMetadataFallback           bool          `mapstructure:"ENABLE_METADATA_FALLBACK"`
	MetadataFallbackMaxFilters       int           `mapstructure:"METADATA_FALLBACK_MAX_FILTERS"`
	PythonExecutorCooldownSeconds    time.Duration `mapstructure:"PYTHON_EXECUTOR_COOLDOWN_SECONDS"`
	PythonExecutorDialTimeoutSeconds time.Duration `mapstructure:"PYTHON_EXECUTOR_DIAL_TIMEOUT_SECONDS"`
	PythonExecutorIOTimeoutSeconds   time.Duration `mapstructure:"PYTHON_EXECUTOR_IO_TIMEOUT_SECONDS"`
	PythonExecutorMaxConnections     int           `mapstructure:"PYTHON_EXECUTOR_MAX_CONNECTIONS"`
	MaxEmbeddingChars                int           `mapstructure:"MAX_EMBEDDING_CHARS"`
	MaxEmbeddingTokens               int           `mapstructure:"MAX_EMBEDDING_TOKENS"`
	EmbeddingTokenSoftLimit          int           `mapstructure:"EMBEDDING_TOKEN_SOFT_LIMIT"`
	EmbeddingTokenTarget             int           `mapstructure:"EMBEDDING_TOKEN_TARGET"`
	MinTokenCheckCharThreshold       int           `mapstructure:"MIN_TOKEN_CHECK_CHAR_THRESHOLD"`
	MaxHybridCandidates              int           `mapstructure:"MAX_HYBRID_CANDIDATES"`
	HybridSemanticWeight             float64       `mapstructure:"HYBRID_SEMANTIC_WEIGHT"`
	HybridBM25Weight                 float64       `mapstructure:"HYBRID_BM25_WEIGHT"`
	RAGContextMaxChars               int           `mapstructure:"RAG_CONTEXT_MAX_CHARS"`
	HybridFactBoost                  float64       `mapstructure:"HYBRID_FACT_BOOST"`
	HybridSummaryBoost               float64       `mapstructure:"HYBRID_SUMMARY_BOOST"`
	HybridErrorPenalty               float64       `mapstructure:"HYBRID_ERROR_PENALTY"`
	PDFTokenThreshold                float64       `mapstructure:"PDF_TOKEN_THRESHOLD"`
	PDFFirstPagesPriority            int           `mapstructure:"PDF_FIRST_PAGES_PRIORITY"`
	PDFEnableTableDetection          bool          `mapstructure:"PDF_ENABLE_TABLE_DETECTION"`
	PDFSentenceBoundaryTruncate      bool          `mapstructure:"PDF_SENTENCE_BOUNDARY_TRUNCATE"`
}

func Load(logger *zap.Logger) *Config {
	var config Config
	viper.SetConfigName("config")
	viper.SetConfigType("yaml")
	viper.AddConfigPath(".")        // For running locally
	viper.AddConfigPath("../")      // For running from docker subdir
	viper.AddConfigPath("./config") // Common config folder
	viper.AutomaticEnv()

	// Set default values
	viper.SetDefault("WEB_PORT", 8080)
	viper.SetDefault("DATABASE_URL", "postgres://postgres:changeme@localhost:5432/stats_agent?sslmode=disable")
	viper.SetDefault("PYTHON_EXECUTOR_ADDRESSES", []string{})
	viper.SetDefault("PYTHON_EXECUTOR_POOL", []string{})
	viper.SetDefault("MAIN_LLM_HOST", "http://localhost:8080")
	viper.SetDefault("EMBEDDING_LLM_HOST", "http://localhost:8081")
	viper.SetDefault("SUMMARIZATION_LLM_HOST", "http://localhost:8082")
	viper.SetDefault("CONTEXT_LENGTH", 4096)
	viper.SetDefault("CONTEXT_SOFT_LIMIT_RATIO", defaultContextSoftLimitRatio)
	viper.SetDefault("MAX_RETRIES", 5)
	viper.SetDefault("RETRY_DELAY_SECONDS", 2)
	viper.SetDefault("CONSECUTIVE_ERRORS", 3)
	viper.SetDefault("LLM_REQUEST_TIMEOUT", 300)
	viper.SetDefault("CLEANUP_ENABLED", true)
	viper.SetDefault("CLEANUP_INTERVAL", 24)
	viper.SetDefault("SESSION_RETENTION_AGE", 168)
	viper.SetDefault("RATE_LIMIT_MESSAGES_PER_MIN", 20)
	viper.SetDefault("RATE_LIMIT_FILES_PER_HOUR", 10)
	viper.SetDefault("RATE_LIMIT_BURST_SIZE", 5)
	viper.SetDefault("SEMANTIC_SIMILARITY_THRESHOLD", 0.7)
	viper.SetDefault("BM25_SCORE_THRESHOLD", 0.15)
	viper.SetDefault("ENABLE_METADATA_FALLBACK", false)
	viper.SetDefault("METADATA_FALLBACK_MAX_FILTERS", 3)
	viper.SetDefault("PYTHON_EXECUTOR_COOLDOWN_SECONDS", 5)
	viper.SetDefault("PYTHON_EXECUTOR_DIAL_TIMEOUT_SECONDS", 3)
	viper.SetDefault("PYTHON_EXECUTOR_IO_TIMEOUT_SECONDS", 60)
	viper.SetDefault("PYTHON_EXECUTOR_MAX_CONNECTIONS", 4)
	viper.SetDefault("MAX_EMBEDDING_CHARS", 1000)
	viper.SetDefault("MAX_EMBEDDING_TOKENS", 250)
	viper.SetDefault("EMBEDDING_TOKEN_SOFT_LIMIT", 450)
	viper.SetDefault("EMBEDDING_TOKEN_TARGET", 400)
	viper.SetDefault("MIN_TOKEN_CHECK_CHAR_THRESHOLD", 100)
	viper.SetDefault("MAX_HYBRID_CANDIDATES", 100)
	viper.SetDefault("HYBRID_SEMANTIC_WEIGHT", defaultHybridSemanticWeight)
	viper.SetDefault("HYBRID_BM25_WEIGHT", defaultHybridBM25Weight)
	viper.SetDefault("RAG_CONTEXT_MAX_CHARS", defaultRAGContextMaxChars)
	viper.SetDefault("HYBRID_FACT_BOOST", defaultHybridFactBoost)
	viper.SetDefault("HYBRID_SUMMARY_BOOST", defaultHybridSummaryBoost)
	viper.SetDefault("HYBRID_ERROR_PENALTY", defaultHybridErrorPenalty)
	viper.SetDefault("PDF_TOKEN_THRESHOLD", defaultPDFTokenThreshold)
	viper.SetDefault("PDF_FIRST_PAGES_PRIORITY", defaultPDFFirstPagesPriority)
	viper.SetDefault("PDF_ENABLE_TABLE_DETECTION", defaultPDFEnableTableDetection)
	viper.SetDefault("PDF_SENTENCE_BOUNDARY_TRUNCATE", defaultPDFSentenceBoundaryTruncate)

	if err := viper.ReadInConfig(); err != nil {
		if logger != nil {
			logger.Warn("Could not read config file, using defaults/env vars", zap.Error(err))
		}
	}

	if err := viper.Unmarshal(&config); err != nil {
		// Config unmarshaling is critical - fail fast during bootstrap
		if logger != nil {
			logger.Fatal("Unable to decode config into struct", zap.Error(err))
		} else {
			// Fallback if logger not available (should not happen in practice)
			fmt.Fprintf(os.Stderr, "FATAL: Unable to decode config into struct: %v\n", err)
			os.Exit(1)
		}
	}

	// Normalize executor address configuration.
	if len(config.PythonExecutorAddresses) == 0 && len(config.PythonExecutorPool) > 0 {
		config.PythonExecutorAddresses = config.PythonExecutorPool
	}
	if len(config.PythonExecutorAddresses) == 0 && config.PythonExecutorAddress != "" {
		config.PythonExecutorAddresses = []string{config.PythonExecutorAddress}
	}
	if len(config.PythonExecutorAddresses) > 0 {
		cleaned := make([]string, 0, len(config.PythonExecutorAddresses))
		for _, addr := range config.PythonExecutorAddresses {
			addr = strings.TrimSpace(addr)
			if addr != "" {
				cleaned = append(cleaned, addr)
			}
		}
		config.PythonExecutorAddresses = cleaned
	}
	if len(config.PythonExecutorAddresses) == 0 {
		config.PythonExecutorAddresses = []string{"localhost:9999"}
	}
	config.PythonExecutorPool = config.PythonExecutorAddresses

	if config.ContextSoftLimitRatio <= 0 || config.ContextSoftLimitRatio >= 1 {
		if logger != nil {
			logger.Warn("Invalid context soft limit ratio; using default",
				zap.Float64("ratio", config.ContextSoftLimitRatio),
				zap.Float64("default", defaultContextSoftLimitRatio))
		}
		config.ContextSoftLimitRatio = defaultContextSoftLimitRatio
	}

	// Convert seconds/hours to proper time.Duration
	config.RetryDelaySeconds = config.RetryDelaySeconds * time.Second
	config.LLMRequestTimeout = config.LLMRequestTimeout * time.Second
	config.CleanupInterval = config.CleanupInterval * time.Hour
	config.SessionRetentionAge = config.SessionRetentionAge * time.Hour
	config.PythonExecutorCooldownSeconds = config.PythonExecutorCooldownSeconds * time.Second
	config.PythonExecutorDialTimeoutSeconds = config.PythonExecutorDialTimeoutSeconds * time.Second
	config.PythonExecutorIOTimeoutSeconds = config.PythonExecutorIOTimeoutSeconds * time.Second

	if config.PythonExecutorCooldownSeconds <= 0 {
		config.PythonExecutorCooldownSeconds = defaultPythonExecutorCooldownSeconds
	}
	if config.PythonExecutorDialTimeoutSeconds <= 0 {
		config.PythonExecutorDialTimeoutSeconds = defaultPythonExecutorDialTimeoutSeconds
	}
	if config.PythonExecutorIOTimeoutSeconds <= 0 {
		config.PythonExecutorIOTimeoutSeconds = defaultPythonExecutorIOTimeoutSeconds
	}
	if config.PythonExecutorMaxConnections <= 0 {
		config.PythonExecutorMaxConnections = defaultPythonExecutorMaxConnections
	}
	if config.MaxEmbeddingChars <= 0 {
		config.MaxEmbeddingChars = defaultMaxEmbeddingChars
	}
	if config.MaxEmbeddingTokens <= 0 {
		config.MaxEmbeddingTokens = defaultMaxEmbeddingTokens
	}
	if config.EmbeddingTokenSoftLimit <= 0 {
		config.EmbeddingTokenSoftLimit = defaultEmbeddingTokenSoftLimit
	}
	if config.EmbeddingTokenTarget <= 0 {
		config.EmbeddingTokenTarget = defaultEmbeddingTokenTarget
	}
	if config.MinTokenCheckCharThreshold <= 0 {
		config.MinTokenCheckCharThreshold = defaultMinTokenCheckCharThreshold
	}
	if config.MaxHybridCandidates <= 0 {
		config.MaxHybridCandidates = defaultMaxHybridCandidates
	}
	if config.HybridSemanticWeight <= 0 {
		config.HybridSemanticWeight = defaultHybridSemanticWeight
	}
	if config.HybridBM25Weight < 0 {
		config.HybridBM25Weight = defaultHybridBM25Weight
	}
	if config.HybridSemanticWeight == 0 && config.HybridBM25Weight == 0 {
		config.HybridSemanticWeight = defaultHybridSemanticWeight
		config.HybridBM25Weight = defaultHybridBM25Weight
	}
	if config.RAGContextMaxChars <= 0 {
		approx := config.ContextLength * 4
		if approx <= 0 {
			approx = defaultRAGContextMaxChars
		}
		config.RAGContextMaxChars = approx
	}
	if config.HybridFactBoost <= 0 {
		config.HybridFactBoost = defaultHybridFactBoost
	}
	if config.HybridSummaryBoost <= 0 {
		config.HybridSummaryBoost = defaultHybridSummaryBoost
	}
	if config.HybridErrorPenalty <= 0 || config.HybridErrorPenalty >= 1 {
		config.HybridErrorPenalty = defaultHybridErrorPenalty
	}
	if config.PDFTokenThreshold <= 0 || config.PDFTokenThreshold > 1 {
		if logger != nil {
			logger.Warn("Invalid PDF token threshold; using default",
				zap.Float64("threshold", config.PDFTokenThreshold),
				zap.Float64("default", defaultPDFTokenThreshold))
		}
		config.PDFTokenThreshold = defaultPDFTokenThreshold
	}
	if config.PDFFirstPagesPriority < 0 {
		config.PDFFirstPagesPriority = defaultPDFFirstPagesPriority
	}
	if config.WebPort <= 0 || config.WebPort > 65535 {
		if logger != nil {
			logger.Warn("Invalid web port; using default",
				zap.Int("port", config.WebPort),
				zap.Int("default", 8080))
		}
		config.WebPort = 8080
	}
	if config.DatabaseURL == "" {
		if logger != nil {
			logger.Fatal("DATABASE_URL is required")
		} else {
			fmt.Fprintf(os.Stderr, "FATAL: DATABASE_URL is required\n")
			os.Exit(1)
		}
	}

	return &config
}

// ContextSoftLimitTokens returns the token count threshold that triggers memory compression.
func (c *Config) ContextSoftLimitTokens() int {
	ratio := c.ContextSoftLimitRatio
	if ratio <= 0 || ratio >= 1 {
		ratio = defaultContextSoftLimitRatio
	}
	return int(float64(c.ContextLength) * ratio)
}
