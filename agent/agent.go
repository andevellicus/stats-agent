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

// Tokenize request/response types have been centralized in llmclient.

func NewAgent(cfg *config.Config, pythonTool *tools.StatefulPythonTool, rag *rag.RAG, logger *zap.Logger) *Agent {
	logger.Info("Agent initialized", zap.Int("context_window_size", cfg.ContextLength))

	// Initialize specialized components
	memoryManager := NewMemoryManager(cfg, logger)
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

// GetMemoryManager returns the agent's memory manager for token counting
func (a *Agent) GetMemoryManager() *MemoryManager {
	return a.memoryManager
}

// GetRAG returns the agent's RAG instance for document storage
func (a *Agent) GetRAG() *rag.RAG {
	return a.rag
}

// Run executes the agent's conversation loop with the given user input.
// It orchestrates memory management, LLM interaction, and Python code execution.
func (a *Agent) Run(ctx context.Context, input string, sessionID string, history []types.AgentMessage, stream *Stream) {
	// 1. Setup: Add user message to history
	userMsg := types.AgentMessage{Role: "user", Content: input}
	currentHistory := append(history, userMsg)

	// Store user message to RAG
	if a.rag != nil {
		a.rag.AddMessagesAsync(sessionID, []types.AgentMessage{userMsg})
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

		// Query RAG for long-term context before each turn
		// This ensures newly added content (PDFs, facts) is available
		queryText := input // Default: use original user question
		if turn > 0 {
			// For subsequent turns, combine user input with recent assistant focus
			// This helps retrieve context relevant to what the agent is currently working on
			for i := len(currentHistory) - 1; i >= 0; i-- {
				if currentHistory[i].Role == "assistant" {
					// Combine original question with what agent is currently doing
					queryText = input + " " + currentHistory[i].Content
					break
				}
			}
		}

        // Add timeout to RAG query to avoid hangs
        ragCtx, ragCancel := context.WithTimeout(ctx, a.cfg.LLMRequestTimeout)
        defer ragCancel()
        longTermContext, err := a.rag.Query(ragCtx, sessionID, queryText, a.cfg.RAGResults)
		if err != nil {
			a.logger.Warn("Failed to query RAG for long-term context, continuing without it",
				zap.Error(err),
				zap.String("session_id", sessionID),
				zap.Int("turn", turn))
			longTermContext = "" // Ensure empty on error
		}

		// Check if long-term context itself is too large
		if longTermContext != "" {
			contextTokens, err := a.memoryManager.CountTokens(ctx, longTermContext)
			softLimitTokens := a.cfg.ContextSoftLimitTokens()
            if err == nil && contextTokens > softLimitTokens {
                a.logger.Info("RAG context is too large, summarizing",
                    zap.Int("context_tokens", contextTokens),
                    zap.Int("turn", turn))
                _ = stream.Status("Compressing memory....")
                sumCtx, sumCancel := context.WithTimeout(ctx, a.cfg.LLMRequestTimeout)
                summarizedContext, summaryErr := a.rag.SummarizeLongTermMemory(sumCtx, longTermContext, input)
                sumCancel()
                if summaryErr == nil {
                    longTermContext = summarizedContext
                }
            }
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
			longTermContext = a.handleEmptyResponse(ctx, longTermContext, input, stream)
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
			assistantMsg := types.AgentMessage{Role: "assistant", Content: llmResponse}
			toolMsg := types.AgentMessage{Role: "tool", Content: execResult.Result}

			currentHistory = append(currentHistory, assistantMsg, toolMsg)

			// Store assistant+tool pair to RAG for fact generation
			if a.rag != nil {
				a.rag.AddMessagesAsync(sessionID, []types.AgentMessage{assistantMsg, toolMsg})
			}

			if execResult.HasError {
				_ = stream.Status("Error - attempting to self-correct")
				loop.RecordError()
			} else {
				loop.RecordSuccess()
			}
		} else {
			// No code to execute - conversation complete
			assistantMsg := types.AgentMessage{Role: "assistant", Content: llmResponse}
			currentHistory = append(currentHistory, assistantMsg)

			// Store final assistant message to RAG
			if a.rag != nil {
				a.rag.AddMessagesAsync(sessionID, []types.AgentMessage{assistantMsg})
			}

			return
		}
	}
}

func (a *Agent) GenerateTitle(ctx context.Context, content string) (string, error) {
	systemPrompt := `You create concise titles that summarize a user's message.

Guidelines:
1. Output only the title text with no labels or commentary.
2. Use at most five words.
3. Base the title entirely on the message content; never repeat the instructions or phrases like "Create a 5 word title".
4. Avoid quotation marks unless they belong in the title.`

	userPrompt := fmt.Sprintf(`User message:
%s

Respond with only the title.`, content)

	messages := []types.AgentMessage{
		{Role: "system", Content: systemPrompt},
		{Role: "user", Content: userPrompt},
	}

	client := llmclient.New(a.cfg, a.logger)
	title, err := client.Chat(ctx, a.cfg.SummarizationLLMHost, messages)
	if err != nil {
		return "", fmt.Errorf("llm chat call failed for title generation: %w", err)
	}

	cleaned := sanitizeTitle(strings.TrimSpace(title))
	if cleaned == "" {
		a.logger.Warn("LLM returned invalid title, using fallback",
			zap.String("raw_title", title),
			zap.String("content_preview", truncateString(content, 100)))
		return "Data Analysis Session", nil
	}

	// Validate word count
	wordCount := len(strings.Fields(cleaned))
	if wordCount > 6 {
		a.logger.Warn("Title exceeds word limit, truncating",
			zap.String("original_title", cleaned),
			zap.Int("word_count", wordCount))

		// Truncate to 5 words
		words := strings.Fields(cleaned)
		cleaned = strings.Join(words[:5], " ")
	}

	return cleaned, nil
}

// Helper function
func truncateString(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "..."
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

func trimQuotesAndSpaces(s string) string {
	s = strings.TrimSpace(s)
	if s == "" {
		return ""
	}
	s = stripSurroundingQuotes(s)
	return strings.TrimSpace(s)
}

func sanitizeTitle(raw string) string {
	if raw == "" {
		return ""
	}

	lines := strings.FieldsFunc(raw, func(r rune) bool {
		return r == '\n' || r == '\r'
	})

	for _, line := range lines {
		candidate := trimQuotesAndSpaces(line)
		if candidate == "" {
			continue
		}

		if strings.HasPrefix(strings.ToLower(candidate), "title:") {
			candidate = trimQuotesAndSpaces(candidate[len("title:"):])
			if candidate == "" {
				continue
			}
		}

		words := strings.Fields(candidate)
		if len(words) > 5 {
			candidate = strings.Join(words[:5], " ")
		}

		return candidate
	}

	return ""
}

// handleEmptyResponse attempts to recover from empty LLM responses by summarizing context.
func (a *Agent) handleEmptyResponse(ctx context.Context, longTermContext, latestUserMessage string, stream *Stream) string {
    a.logger.Warn("LLM response was empty, likely due to a context window error. Attempting to summarize context")
    _ = stream.Status("Compressing memory due to a context window error...")

    sumCtx, sumCancel := context.WithTimeout(ctx, a.cfg.LLMRequestTimeout)
    summarizedContext, err := a.rag.SummarizeLongTermMemory(sumCtx, longTermContext, latestUserMessage)
    sumCancel()
    if err != nil {
        a.logger.Error("Recovery failed: Could not summarize RAG context. Aborting turn", zap.Error(err))
        return ""
    }

	return summarizedContext
}
