package main

import (
	"bufio"
	"context"
	"fmt"
	"log"
	"os"
	"stats-agent/agent"
	"stats-agent/config"
	"stats-agent/rag"
	"stats-agent/tools"
)

func main() {
	ctx := context.Background()
	cfg := config.Load()

	pythonTool, err := tools.NewStatefulPythonTool(ctx, cfg.PythonExecutorAddress)
	if err != nil {
		log.Fatalf("Failed to initialize Python tool: %v", err)
	}
	defer pythonTool.Close()

	// Pass the specific hosts to the RAG service
	rag, err := rag.New(cfg)
	if err != nil {
		log.Fatalf("Failed to initialize RAG: %v", err)
	}

	// Pass the main host to the Agent
	statsAgent := agent.NewAgent(cfg, pythonTool, rag)

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
		log.Println("Error reading from stdin:", err)
	}
}
