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

func getLLMResponse(ctx context.Context, llamaCppHost string, messages []types.AgentMessage, cfg *config.Config, logger *zap.Logger, temperature *float64) (<-chan string, error) {
	// Prepend system message enforcing the analysis protocol
	systemMessage := types.AgentMessage{Role: "system", Content: buildSystemPrompt()}
	chatMessages := append([]types.AgentMessage{systemMessage}, messages...)

	client := llmclient.New(cfg, logger)
	return client.ChatStream(ctx, llamaCppHost, chatMessages, temperature)
}
