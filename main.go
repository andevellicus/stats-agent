package main

import (
	"bufio"
	"context"
	"fmt"
	"log"
	"os"

	"stats-agent/agent"

	"github.com/ollama/ollama/api"
)

func main() {
	ctx := context.Background()

	pythonTool, err := agent.NewStatefulPythonTool(ctx, "localhost:9999")
	if err != nil {
		log.Fatalf("Failed to connect to Python container. Is it running? Error: %v", err)
	}
	defer pythonTool.Close()

	ollamaClient, err := api.ClientFromEnvironment()
	if err != nil {
		log.Fatalf("Failed to create Ollama client: %v", err)
	}

	agent := agent.NewAgent(ollamaClient, pythonTool, 5)

	scanner := bufio.NewScanner(os.Stdin)
	for {
		fmt.Print("You: ")
		if !scanner.Scan() {
			break
		}
		input := scanner.Text()
		if input == "exit" {
			break
		}

		agent.Run(ctx, input)
	}

	if err := scanner.Err(); err != nil {
		log.Fatal(err)
	}
}
