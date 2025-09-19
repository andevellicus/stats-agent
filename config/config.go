package config

import (
	"log"

	"github.com/spf13/viper"
)

// Config holds the application's configuration
type Config struct {
	PythonExecutorAddress string `mapstructure:"PYTHON_EXECUTOR_ADDRESS"`
	OllamaHost            string `mapstructure:"OLLAMA_HOST"`
	MaxTurns              int    `mapstructure:"MAX_TURNS"`
	Model                 string `mapstructure:"MODEL"`
	EmbeddingModel        string `mapstructure:"EMBEDDING_MODEL"`
	SummarizationModel    string `mapstructure:"SUMMARIZATION_MODEL"`
	RAGResults            int    `mapstructure:"RAG_RESULTS"`
}

func Load() *Config {
	var config Config
	viper.SetConfigName("config")
	viper.SetConfigType("yaml")
	viper.AddConfigPath(".")
	viper.AddConfigPath("./config")
	viper.AddConfigPath("/etc/stats-agent/")
	viper.AutomaticEnv()
	if err := viper.ReadInConfig(); err != nil {
		log.Fatalf("Error reading config file, %s", err)
	}
	if err := viper.Unmarshal(&config); err != nil {
		log.Fatalf("Unable to decode into struct, %v", err)
	}
	return &config
}
