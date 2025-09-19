package main

import (
	"bufio"
	"context"
	"fmt"
	"log"
	"net/http"
	"net/url"
	"os"
	"strings"

	"stats-agent/agent"
	"stats-agent/config"
	"stats-agent/rag"
	"stats-agent/tools"

	"github.com/ollama/ollama/api"
)

func main() {
	cfg := config.Load()
	ctx := context.Background()

	pythonTool, err := tools.NewStatefulPythonTool(ctx, cfg.PythonExecutorAddress)
	if err != nil {
		log.Fatalf("Failed to connect to Python container. Is it running? Error: %v", err)
	}
	defer pythonTool.Close()

	var ollamaClient *api.Client
	ollamaHost := cfg.OllamaHost
	if ollamaHost == "" {
		ollamaHost = "http://127.0.0.1:11434"
	}
	cleanOllamaHost := strings.TrimSuffix(ollamaHost, "/")
	ollamaURL, err := url.Parse(cleanOllamaHost)
	if err != nil {
		log.Fatalf("Invalid OLLAMA_HOST URL: %v", err)
	}
	ollamaClient = api.NewClient(ollamaURL, http.DefaultClient)

	// The call to rag.New is now simpler.
	rag, err := rag.New(ctx, ollamaClient, cfg.EmbeddingModel, cfg.SummarizationModel)
	if err != nil {
		log.Fatalf("Failed to create RAG instance: %v", err)
	}

	agent := agent.NewAgent(ctx, ollamaClient, pythonTool, rag, cfg.Model, cfg.MaxTurns, cfg.RAGResults)

	// ... (scanner loop is unchanged) ...
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
