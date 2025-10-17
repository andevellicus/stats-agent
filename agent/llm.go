package agent

import (
    "context"
    "stats-agent/config"
    "stats-agent/llmclient"
    "stats-agent/prompts"
    "stats-agent/web/types"

	"go.uber.org/zap"
)

func buildSystemPrompt() string { return prompts.AgentSystem() }

func buildDocumentPrompt() string { return prompts.DocumentQA() }

func getLLMResponse(ctx context.Context, llamaCppHost string, messages []types.AgentMessage, cfg *config.Config, logger *zap.Logger, temperature *float64) (<-chan string, error) {
    // Always place our analysis protocol as the first system message.
    // Keep any existing system memory/context as a separate system message after it.
    systemMessage := types.AgentMessage{Role: "system", Content: buildSystemPrompt()}
    chatMessages := append([]types.AgentMessage{systemMessage}, messages...)

    // Debug trace: summarize roles and sizes to verify payload construction
    if logger != nil {
        var users, assistants, systems, tools int
        for _, m := range chatMessages {
            switch m.Role {
            case "user":
                users++
            case "assistant":
                assistants++
            case "system":
                systems++
            case "tool":
                tools++
            }
        }
        logger.Debug("Sending messages to LLM",
            zap.Int("total", len(chatMessages)),
            zap.Int("users", users),
            zap.Int("assistants", assistants),
            zap.Int("systems", systems),
            zap.Int("tools", tools),
        )
    }

    client := llmclient.New(cfg, logger)
    return client.ChatStream(ctx, llamaCppHost, chatMessages, temperature)
}

func getLLMResponseForDocumentMode(ctx context.Context, llamaCppHost string, messages []types.AgentMessage, cfg *config.Config, logger *zap.Logger) (<-chan string, error) {
    // Use document Q&A prompt instead of dataset analysis prompt
    systemMessage := types.AgentMessage{Role: "system", Content: buildDocumentPrompt()}
    chatMessages := append([]types.AgentMessage{systemMessage}, messages...)

    // Use a slightly higher temperature for document Q&A (more natural language)
    temperature := 0.3

    if logger != nil {
        var users, assistants, systems, tools int
        for _, m := range chatMessages {
            switch m.Role {
            case "user":
                users++
            case "assistant":
                assistants++
            case "system":
                systems++
            case "tool":
                tools++
            }
        }
        logger.Debug("Doc mode: sending messages to LLM",
            zap.Int("total", len(chatMessages)),
            zap.Int("users", users),
            zap.Int("assistants", assistants),
            zap.Int("systems", systems),
            zap.Int("tools", tools),
        )
    }
    client := llmclient.New(cfg, logger)
    return client.ChatStream(ctx, llamaCppHost, chatMessages, &temperature)
}
