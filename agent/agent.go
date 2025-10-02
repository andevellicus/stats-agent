package agent

import (
	"context"
	"fmt"
	"strings"

	"stats-agent/config"
	"stats-agent/llmclient"
	"stats-agent/rag"
	"stats-agent/tools"
	"stats-agent/web/types"

	"go.uber.org/zap"
)

type Agent struct {
	cfg                  *config.Config
	pythonTool           *tools.StatefulPythonTool
	rag                  *rag.RAG
	logger               *zap.Logger
	memoryManager        *MemoryManager
	executionCoordinator *ExecutionCoordinator
	responseHandler      *ResponseHandler
}

// TokenizeRequest and TokenizeResponse are used by MemoryManager
type TokenizeRequest struct {
	Content string `json:"content"`
}

type TokenizeResponse struct {
	Tokens []int `json:"tokens"`
}

func NewAgent(cfg *config.Config, pythonTool *tools.StatefulPythonTool, rag *rag.RAG, logger *zap.Logger) *Agent {
	logger.Info("Agent initialized", zap.Int("context_window_size", cfg.ContextLength))

	// Initialize specialized components
	memoryManager := NewMemoryManager(cfg, rag, logger)
	executionCoordinator := NewExecutionCoordinator(pythonTool, logger)
	responseHandler := NewResponseHandler(cfg, logger)

	return &Agent{
		cfg:                  cfg,
		pythonTool:           pythonTool,
		rag:                  rag,
		logger:               logger,
		memoryManager:        memoryManager,
		executionCoordinator: executionCoordinator,
		responseHandler:      responseHandler,
	}
}

func (a *Agent) InitializeSession(ctx context.Context, sessionID string, uploadedFiles []string) (string, error) {
	return a.pythonTool.InitializeSession(ctx, sessionID, uploadedFiles)
}

func (a *Agent) CleanupSession(sessionID string) {
	a.pythonTool.CleanupSession(sessionID)
	if a.rag != nil {
		if err := a.rag.DeleteSessionDocuments(sessionID); err != nil {
			a.logger.Warn("Failed to remove session documents from RAG",
				zap.String("session_id", sessionID),
				zap.Error(err))
		}
	}
}

// Run executes the agent's conversation loop with the given user input.
// It orchestrates memory management, LLM interaction, and Python code execution.
func (a *Agent) Run(ctx context.Context, input string, sessionID string, history []types.AgentMessage, stream *Stream) {
	// 1. Setup: Add user message and retrieve long-term context
	currentHistory := append(history, types.AgentMessage{Role: "user", Content: input})

	// Query RAG for long-term context - non-critical, log warning if fails
	longTermContext, err := a.rag.Query(ctx, input, a.cfg.RAGResults)
	if err != nil {
		a.logger.Warn("Failed to query RAG for long-term context, continuing without it",
			zap.Error(err),
			zap.String("session_id", sessionID))
	}

	// Proactively check if long-term context itself is too large
	if longTermContext != "" {
		contextTokens, err := a.memoryManager.CountTokens(ctx, longTermContext)
		if err == nil && contextTokens > int(float64(a.cfg.ContextLength)*0.75) {
			a.logger.Info("Proactive check: RAG context is too large, summarizing", zap.Int("context_tokens", contextTokens))
			_ = stream.Status("Compressing memory....")
			summarizedContext, summaryErr := a.rag.SummarizeLongTermMemory(ctx, longTermContext)
			if summaryErr == nil {
				longTermContext = summarizedContext
			}
		}
	}

	// 2. Initialize conversation loop controller
	loop := NewConversationLoop(a.cfg, a.logger)

	// 3. Main conversation loop
	for turn := 0; turn < a.cfg.MaxTurns; turn++ {
		// Manage memory before each turn - non-critical, log warning if fails
		if err := a.memoryManager.ManageHistory(ctx, sessionID, &currentHistory, stream); err != nil {
			a.logger.Warn("Failed to manage memory, continuing with current history",
				zap.Error(err),
				zap.Int("turn", turn),
				zap.String("session_id", sessionID))
		}

		// Check loop conditions (error limit, max turns)
		if shouldContinue, reason := loop.ShouldContinue(turn); !shouldContinue {
			_ = stream.Status(reason)
			break
		}

		// Build messages for LLM (combine long-term context + history)
		messagesForLLM := a.responseHandler.BuildMessagesForLLM(longTermContext, currentHistory)

		// Get LLM response - critical operation, break loop on failure
		responseChan, err := getLLMResponse(ctx, a.cfg.MainLLMHost, messagesForLLM, a.cfg, a.logger)
		if err != nil {
			a.logger.Error("Failed to get LLM response, aborting turn",
				zap.Error(err),
				zap.Int("turn", turn),
				zap.String("session_id", sessionID))
			_ = stream.Status("LLM communication error")
			break
		}

		// Collect streamed response
		llmResponse := a.responseHandler.CollectStreamedResponse(responseChan, stream)

		// Handle empty response (usually context window error)
		if a.responseHandler.IsEmpty(llmResponse) {
			longTermContext = a.handleEmptyResponse(ctx, longTermContext, stream)
			if longTermContext == "" {
				break // Recovery failed
			}
			continue
		}

		// Process response for code execution - critical operation
		execResult, err := a.executionCoordinator.ProcessResponse(ctx, llmResponse, sessionID, stream)
		if err != nil {
			a.logger.Error("Failed to process LLM response, aborting turn",
				zap.Error(err),
				zap.Int("turn", turn),
				zap.String("session_id", sessionID))
			_ = stream.Status("Response processing error")
			break
		}

		// Update history based on execution result
		if execResult.WasCodeExecuted {
			currentHistory = append(currentHistory,
				types.AgentMessage{Role: "assistant", Content: llmResponse},
				types.AgentMessage{Role: "tool", Content: execResult.Result})

			if execResult.HasError {
				_ = stream.Status("Error - attempting to self-correct")
				loop.RecordError()
			} else {
				loop.RecordSuccess()
			}
		} else {
			// No code to execute - conversation complete
			currentHistory = append(currentHistory, types.AgentMessage{Role: "assistant", Content: llmResponse})
			return
		}
	}
}

func (a *Agent) GenerateTitle(ctx context.Context, content string) (string, error) {
	systemPrompt := `You are an expert at creating concise, 5-word titles from user messages. Your task is to distill the user's message into a short, descriptive title.`

	userPrompt := fmt.Sprintf(`Create a 5-word title for the following user message.

**User Message:**
"%s"

**Title:**
`, content)

	messages := []types.AgentMessage{
		{Role: "system", Content: systemPrompt},
		{Role: "user", Content: userPrompt},
	}

    // Use non-streaming client without agent system prompt injection
    client := llmclient.New(a.cfg, a.logger)
    title, err := client.Chat(ctx, a.cfg.SummarizationLLMHost, messages)
    if err != nil {
        return "", fmt.Errorf("llm chat call failed for title generation: %w", err)
    }

	if title == "" {
		return "", fmt.Errorf("llm returned an empty title")
	}

	// Post-process: trim whitespace and strip surrounding quotes if present
	cleaned := strings.TrimSpace(title)
	cleaned = stripSurroundingQuotes(cleaned)
	return cleaned, nil
}

// stripSurroundingQuotes removes a single pair of matching leading/trailing quotes.
// Handles common ASCII and smart quote variants.
func stripSurroundingQuotes(s string) string {
	if len(s) < 2 {
		return s
	}
	pairs := map[rune]rune{
		'"':  '"',
		'\'': '\'',
		'“':  '”',
		'”':  '”', // in case only ” is used on both ends
		'‘':  '’',
		'’':  '’', // in case only ’ is used on both ends
	}
	runes := []rune(s)
	first := runes[0]
	last := runes[len(runes)-1]
	if expected, ok := pairs[first]; ok && last == expected {
		return string(runes[1 : len(runes)-1])
	}
	return s
}

// handleEmptyResponse attempts to recover from empty LLM responses by summarizing context.
func (a *Agent) handleEmptyResponse(ctx context.Context, longTermContext string, stream *Stream) string {
	a.logger.Warn("LLM response was empty, likely due to a context window error. Attempting to summarize context")
	_ = stream.Status("Compressing memory due to a context window error...")

	summarizedContext, err := a.rag.SummarizeLongTermMemory(ctx, longTermContext)
	if err != nil {
		a.logger.Error("Recovery failed: Could not summarize RAG context. Aborting turn", zap.Error(err))
		return ""
	}

	return summarizedContext
}
