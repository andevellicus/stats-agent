package main

import (
	"context"
	"fmt"
	"os"
	"os/signal"
	"stats-agent/agent"
	"stats-agent/config"
	"stats-agent/database"
	"stats-agent/rag"
	"stats-agent/tools"
	"stats-agent/web"
	"stats-agent/web/services"
	"syscall"

	"go.uber.org/zap"
)

func main() {
	ctx := context.Background()

	// Initialize logger with default level to load config
	tempLogger, err := config.InitLogger("info")
	if err != nil {
		fmt.Printf("Failed to initialize logger: %v\n", err)
		os.Exit(1)
	}

	// Load config (which includes log level setting)
	cfg := config.Load(tempLogger)

	// Re-initialize logger with configured level
	logger, err := config.InitLogger(cfg.LogLevel)
	if err != nil {
		fmt.Printf("Failed to re-initialize logger with configured level: %v\n", err)
		os.Exit(1)
	}
	defer config.Cleanup()

	// Build database connection string from environment variables (with fallback defaults for dev)
	dbHost := getEnvOrDefault("DB_HOST", "localhost")
	dbPort := getEnvOrDefault("DB_PORT", "5432")
	dbUser := getEnvOrDefault("DB_USER", "postgres")
	dbPassword := getEnvOrDefault("DB_PASSWORD", "changeme")
	dbName := getEnvOrDefault("DB_NAME", "stats_agent")
	dbSSLMode := getEnvOrDefault("DB_SSLMODE", "disable")

	connStr := fmt.Sprintf("postgres://%s:%s@%s:%s/%s?sslmode=%s",
		dbUser, dbPassword, dbHost, dbPort, dbName, dbSSLMode)

	store, err := database.NewPostgresStore(connStr)
	if err != nil {
		logger.Fatal("Failed to connect to database", zap.Error(err))
	}

	// --- Ensure Schema Exists ---
	if err := store.EnsureSchema(ctx); err != nil {
		logger.Fatal("Failed to ensure database schema", zap.Error(err))
	}

	pythonTool, err := tools.NewStatefulPythonTool(ctx, cfg, logger)
	if err != nil {
		logger.Fatal("Failed to initialize Python tool", zap.Error(err))
	}
	defer pythonTool.Close()

	// Pass the specific hosts to the RAG service
	rag, err := rag.New(cfg, store, logger)
	if err != nil {
		logger.Fatal("Failed to initialize RAG", zap.Error(err))
	}

	// Pass the main host to the Agent
	statsAgent := agent.NewAgent(cfg, pythonTool, rag, logger)

	// Initialize cleanup service and start background cleanup routine
	cleanupService := services.NewCleanupService(store, statsAgent, logger)
	go web.StartWorkspaceCleanup(cfg, cleanupService, logger)

	// Initialize web server
	webServer := web.NewServer(statsAgent, logger, cfg, store)

	// Create context that listens for interrupt signals
	ctx, cancel := signal.NotifyContext(ctx, os.Interrupt, syscall.SIGTERM)
	defer cancel()

	// Start web server
	port := fmt.Sprintf(":%d", cfg.WebPort)
	logger.Info("Starting Pocket Statistician web server", zap.String("port", port))
	if err := webServer.Start(ctx, port); err != nil {
		logger.Error("Web server error", zap.Error(err))
		os.Exit(1)
	}
}

// getEnvOrDefault returns the value of an environment variable or a default value if not set
func getEnvOrDefault(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}
