package config

import (
	"fmt"
	"strings"
	"time"

	"github.com/spf13/viper"
	"go.uber.org/zap"
)

// Config holds the application's configuration
type Config struct {
	PythonExecutorAddress   string        `mapstructure:"PYTHON_EXECUTOR_ADDRESS"`
	PythonExecutorAddresses []string      `mapstructure:"PYTHON_EXECUTOR_ADDRESSES"`
	PythonExecutorPool      []string      `mapstructure:"PYTHON_EXECUTOR_POOL"`
	MainLLMHost             string        `mapstructure:"MAIN_LLM_HOST"`
	EmbeddingLLMHost        string        `mapstructure:"EMBEDDING_LLM_HOST"`
	SummarizationLLMHost    string        `mapstructure:"SUMMARIZATION_LLM_HOST"`
	MaxTurns                int           `mapstructure:"MAX_TURNS"`
	RAGResults              int           `mapstructure:"RAG_RESULTS"`
	ContextLength           int           `mapstructure:"CONTEXT_LENGTH"`
	MaxRetries              int           `mapstructure:"MAX_RETRIES"`
	RetryDelaySeconds       time.Duration `mapstructure:"RETRY_DELAY_SECONDS"`
	ConsecutiveErrors       int           `mapstructure:"CONSECUTIVE_ERRORS"`
	LLMRequestTimeout       time.Duration `mapstructure:"LLM_REQUEST_TIMEOUT"`
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
	viper.SetDefault("PYTHON_EXECUTOR_ADDRESSES", []string{})
	viper.SetDefault("PYTHON_EXECUTOR_POOL", []string{})
	viper.SetDefault("MAIN_LLM_HOST", "http://localhost:8080")
	viper.SetDefault("EMBEDDING_LLM_HOST", "http://localhost:8081")
	viper.SetDefault("SUMMARIZATION_LLM_HOST", "http://localhost:8082")
	viper.SetDefault("CONTEXT_LENGTH", 4096)
	viper.SetDefault("MAX_RETRIES", 5)
	viper.SetDefault("RETRY_DELAY_SECONDS", 2)
	viper.SetDefault("CONSECUTIVE_ERRORS", 3)
	viper.SetDefault("LLM_REQUEST_TIMEOUT", 300)

	if err := viper.ReadInConfig(); err != nil {
		if logger != nil {
			logger.Warn("Could not read config file, using defaults/env vars", zap.Error(err))
		}
	}

	if err := viper.Unmarshal(&config); err != nil {
		if logger != nil {
			logger.Fatal("Unable to decode into struct", zap.Error(err))
		} else {
			panic(fmt.Sprintf("Unable to decode config into struct: %v", err))
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

	// Convert seconds to a proper time.Duration
	config.RetryDelaySeconds = config.RetryDelaySeconds * time.Second
	config.LLMRequestTimeout = config.LLMRequestTimeout * time.Second

	return &config
}
