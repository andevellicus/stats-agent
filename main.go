package main

import (
	"bufio"
	"context"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"stats-agent/agent"
	"stats-agent/config"
	"stats-agent/database"
	"stats-agent/rag"
	"stats-agent/tools"
	"stats-agent/web"
	"syscall"

	"github.com/google/uuid"
	"go.uber.org/zap"
)

func main() {
	// Parse command line flags
	webMode := flag.Bool("web", false, "Run in web mode instead of CLI")
	port := flag.String("port", "8080", "Port to run web server on")
	flag.Parse()

	ctx := context.Background()

	// Initialize logger
	logger, err := config.InitLogger()
	if err != nil {
		fmt.Printf("Failed to initialize logger: %v\n", err)
		os.Exit(1)
	}
	defer config.Cleanup()
	cfg := config.Load(logger)

	connStr := "postgres://postgres:changeme@localhost:5432/stats_agent?sslmode=disable"
	store, err := database.NewPostgresStore(connStr)
	if err != nil {
		logger.Fatal("Failed to connect to database", zap.Error(err))
	}

	// --- Ensure Schema Exists ---
	if err := store.EnsureSchema(ctx); err != nil {
		logger.Fatal("Failed to ensure database schema", zap.Error(err))
	}

	pythonTool, err := tools.NewStatefulPythonTool(ctx, cfg.PythonExecutorAddresses, logger)
	if err != nil {
		logger.Fatal("Failed to initialize Python tool", zap.Error(err))
	}
	defer pythonTool.Close()

	// Pass the specific hosts to the RAG service
	rag, err := rag.New(cfg, store, logger)
	if err != nil {
		logger.Fatal("Failed to initialize RAG", zap.Error(err))
	}
	if err := rag.LoadPersistedDocuments(ctx); err != nil {
		logger.Warn("Failed to load persisted RAG documents", zap.Error(err))
	}

	// Pass the main host to the Agent
	statsAgent := agent.NewAgent(cfg, pythonTool, rag, logger)

	if *webMode {
		// Run web server
		logger.Info("Starting Pocket Statistician in web mode", zap.String("port", *port))

		// Initialize cleanup service and start background cleanup routine
		cleanupService := web.NewCleanupService(store, statsAgent, logger)
		go web.StartWorkspaceCleanup(cfg, cleanupService, logger)

		webServer := web.NewServer(statsAgent, logger, cfg, store)

		// Create context that listens for interrupt signals
		ctx, cancel := signal.NotifyContext(ctx, os.Interrupt, syscall.SIGTERM)
		defer cancel()

		addr := ":" + *port
		if err := webServer.Start(ctx, addr); err != nil {
			logger.Error("Web server error", zap.Error(err))
			os.Exit(1)
		}
	} else {
		// Run CLI mode
		logger.Info("Starting Pocket Statistician in CLI mode")
		scanner := bufio.NewScanner(os.Stdin)

		// Create a temporary session for CLI mode
		cliSessionID := "cli-session-" + uuid.NewString()
		cliWorkspace := "workspaces/" + cliSessionID
		os.MkdirAll(cliWorkspace, 0755)
		defer os.RemoveAll(cliWorkspace)

		fmt.Printf("Welcome to your Pocket Statistician. Using temporary workspace: %s\n", cliWorkspace)
		fmt.Print("> ")
		for scanner.Scan() {
			input := scanner.Text()
			if input == "exit" {
				break
			}
			//statsAgent.Run(ctx, input, cliSessionID)
			fmt.Print("> ")
		}

		if err := scanner.Err(); err != nil {
			logger.Error("Error reading from stdin", zap.Error(err))
		}
	}
}
