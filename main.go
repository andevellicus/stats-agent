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
	"stats-agent/rag"
	"stats-agent/tools"
	"stats-agent/web"
	"syscall"

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

	pythonTool, err := tools.NewStatefulPythonTool(ctx, cfg.PythonExecutorAddress, logger)
	if err != nil {
		logger.Fatal("Failed to initialize Python tool", zap.Error(err))
	}
	defer pythonTool.Close()

	// Pass the specific hosts to the RAG service
	rag, err := rag.New(cfg, logger)
	if err != nil {
		logger.Fatal("Failed to initialize RAG", zap.Error(err))
	}

	// Pass the main host to the Agent
	statsAgent := agent.NewAgent(cfg, pythonTool, rag, logger)

	if *webMode {
		// Run web server
		logger.Info("Starting Stats Agent in web mode", zap.String("port", *port))
		webServer := web.NewServer(statsAgent, logger, cfg)

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
		logger.Info("Starting Stats Agent in CLI mode")
		scanner := bufio.NewScanner(os.Stdin)
		fmt.Println("Welcome to the Stats Agent. How can I help you today?")
		fmt.Print("> ")
		for scanner.Scan() {
			input := scanner.Text()
			if input == "exit" {
				break
			}
			statsAgent.Run(ctx, input)
			fmt.Print("> ")
		}

		if err := scanner.Err(); err != nil {
			logger.Error("Error reading from stdin", zap.Error(err))
		}
	}
}
