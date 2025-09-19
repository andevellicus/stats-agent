package config

import (
	"log"
	"time"

	"github.com/spf13/viper"
)

// Config holds the application's configuration
type Config struct {
	PythonExecutorAddress string        `mapstructure:"PYTHON_EXECUTOR_ADDRESS"`
	MainLLMHost           string        `mapstructure:"MAIN_LLM_HOST"`
	EmbeddingLLMHost      string        `mapstructure:"EMBEDDING_LLM_HOST"`
	SummarizationLLMHost  string        `mapstructure:"SUMMARIZATION_LLM_HOST"`
	MaxTurns              int           `mapstructure:"MAX_TURNS"`
	RAGResults            int           `mapstructure:"RAG_RESULTS"`
	ContextLength         int           `mapstructure:"CONTEXT_LENGTH"`
	MaxRetries            int           `mapstructure:"MAX_RETRIES"`
	RetryDelaySeconds     time.Duration `mapstructure:"RETRY_DELAY_SECONDS"`
}

func Load() *Config {
	var config Config
	viper.SetConfigName("config")
	viper.SetConfigType("yaml")
	viper.AddConfigPath(".")        // For running locally
	viper.AddConfigPath("../")      // For running from docker subdir
	viper.AddConfigPath("./config") // Common config folder
	viper.AutomaticEnv()

	// Set default values
	viper.SetDefault("MAIN_LLM_HOST", "http://localhost:8080")
	viper.SetDefault("EMBEDDING_LLM_HOST", "http://localhost:8081")
	viper.SetDefault("SUMMARIZATION_LLM_HOST", "http://localhost:8082")
	viper.SetDefault("CONTEXT_LENGTH", 4096)
	viper.SetDefault("MAX_RETRIES", 5)
	viper.SetDefault("RETRY_DELAY_SECONDS", 2)

	if err := viper.ReadInConfig(); err != nil {
		log.Printf("Warning: could not read config file, using defaults/env vars. Error: %s", err)
	}

	if err := viper.Unmarshal(&config); err != nil {
		log.Fatalf("Unable to decode into struct, %v", err)
	}

	// Convert seconds to a proper time.Duration
	config.RetryDelaySeconds = config.RetryDelaySeconds * time.Second

	return &config
}
