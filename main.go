package main

import (
	"bufio"
	"context"
	"fmt"
	"os"
	"stats-agent/agent"
	"stats-agent/config"
	"stats-agent/rag"
	"stats-agent/tools"

	"go.uber.org/zap"
)

func main() {
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
