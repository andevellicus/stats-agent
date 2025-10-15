package agent

import (
	"context"
	"fmt"
	"strings"

	"stats-agent/config"
	"stats-agent/llmclient"
	"stats-agent/prompts"
	"stats-agent/rag"
	"stats-agent/tools"
	"stats-agent/web/format"
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

// RunDocumentMode executes a simple document Q&A workflow without code execution.
// It queries RAG for document context, combines it with conversation history, and streams a single LLM response.
func (a *Agent) RunDocumentMode(ctx context.Context, input string, sessionID string, history []types.AgentMessage, stream *Stream) {
	// 1. Create user message but DON'T add to history or RAG yet
	userMsg := types.AgentMessage{
		Role:        "user",
		Content:     input,
		ContentHash: rag.ComputeMessageContentHash("user", input),
	}

	// 2. Query RAG for document context
	// Use configured document mode RAG results (default 5, more than dataset mode)
	ragResults := a.cfg.DocumentModeRAGResults
	if ragResults <= 0 {
		ragResults = 5 // Fallback if not configured
	}

	// Extract content hashes from current history to exclude from RAG results
	excludeHashes := make([]string, 0, len(history))
	for _, msg := range history {
		if msg.ContentHash != "" {
			excludeHashes = append(excludeHashes, msg.ContentHash)
		}
	}

	ragCtx, ragCancel := context.WithTimeout(ctx, a.cfg.LLMRequestTimeout)
	defer ragCancel()
	longTermContext, err := a.rag.Query(ragCtx, sessionID, input, ragResults, excludeHashes)
	if err != nil {
		a.logger.Warn("Failed to query RAG for document context, continuing without it",
			zap.Error(err),
			zap.String("session_id", sessionID))
		longTermContext = ""
	}

	// 3. Build messages for LLM (use document QA prompt)
	// Append user message to history for this request (but don't modify passed-in history yet)
	historyWithUserMsg := append(history, userMsg)

	var messagesForLLM []types.AgentMessage

	// Add long-term context if available
	if longTermContext != "" {
		messagesForLLM = append(messagesForLLM, types.AgentMessage{
			Role:    "system",
			Content: longTermContext,
		})
	}

	// Add conversation history (including current user message)
	messagesForLLM = append(messagesForLLM, historyWithUserMsg...)

	// 4. Get single LLM response with document QA prompt
	responseChan, err := getLLMResponseForDocumentMode(ctx, a.cfg.MainLLMHost, messagesForLLM, a.cfg, a.logger)
	if err != nil {
		a.logger.Error("Failed to get LLM response in document mode",
			zap.Error(err),
			zap.String("session_id", sessionID))
		_ = stream.Status("LLM communication error")
		return
	}

	// 5. Collect and stream response
	llmResponse := a.responseHandler.CollectStreamedResponse(responseChan, stream)

	if a.responseHandler.IsEmpty(llmResponse) {
		a.logger.Warn("Empty response in document mode", zap.String("session_id", sessionID))
		_ = stream.Status("Received empty response from LLM")
		return
	}

	// 6. Store assistant response to RAG (user message stored separately via chat handler)
	assistantMsg := types.AgentMessage{
		Role:        "assistant",
		Content:     llmResponse,
		ContentHash: rag.ComputeMessageContentHash("assistant", llmResponse),
	}
	if a.rag != nil {
		a.rag.AddMessagesAsync(sessionID, []types.AgentMessage{assistantMsg})
	}

	// Done - single response, no iteration
}

// Run executes the agent's conversation loop with the given user input.
// It orchestrates memory management, LLM interaction, and Python code execution.
func (a *Agent) Run(ctx context.Context, input string, sessionID string, history []types.AgentMessage, stream *Stream) {
	// 1. Create user message but DON'T add to history or RAG yet
	// It will be added at the end of the turn along with the assistant response
	userMsg := types.AgentMessage{
		Role:        "user",
		Content:     input,
		ContentHash: rag.ComputeMessageContentHash("user", input),
	}

	// 2. Initialize conversation loop controller
	loop := NewConversationLoop(a.cfg, a.logger)

	// 3. Main conversation loop
	for turn := 0; turn < a.cfg.MaxTurns; turn++ {
		// Manage memory before each turn - non-critical, log warning if fails
		if err := a.memoryManager.ManageHistory(ctx, sessionID, &history, stream); err != nil {
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
			for i := len(history) - 1; i >= 0; i-- {
				if history[i].Role == "assistant" {
					// Combine original question with what agent is currently doing
					queryText = input + " " + history[i].Content
					break
				}
			}
		}

		// Extract content hashes from current history to exclude from RAG results
		excludeHashes := make([]string, 0, len(history))
		for _, msg := range history {
			if msg.ContentHash != "" {
				excludeHashes = append(excludeHashes, msg.ContentHash)
			}
		}

		// Add timeout to RAG query to avoid hangs
		ragCtx, ragCancel := context.WithTimeout(ctx, a.cfg.LLMRequestTimeout)
		defer ragCancel()
		longTermContext, err := a.rag.Query(ragCtx, sessionID, queryText, a.cfg.RAGResults, excludeHashes)
		if err != nil {
			a.logger.Warn("Failed to query RAG for long-term context, continuing without it",
				zap.Error(err),
				zap.String("session_id", sessionID),
				zap.Int("turn", turn))
			longTermContext = "" // Ensure empty on error
		}

		// Build messages for LLM (combine long-term context + history + current user message)
		// On turn 0, append user message. On turn 1+, it's already in history
		if turn == 0 {
			history = append(history, userMsg)
		}
		messagesForLLM := a.responseHandler.BuildMessagesForLLM(longTermContext, history)

		// Ensure combined system context + history fits within context window
		{
			softLimitTokens := a.cfg.ContextSoftLimitTokens()
			// Build a temporary slice to measure combined token count (includes user message)
			combined := make([]types.AgentMessage, 0, len(history)+1)
			if longTermContext != "" {
				combined = append(combined, types.AgentMessage{Role: "system", Content: longTermContext})
			}
			combined = append(combined, history...)

			totalTokens, err := a.memoryManager.CalculateHistorySize(ctx, combined)
			if err != nil {
				a.logger.Warn("Failed to count tokens for combined context; proceeding optimistically", zap.Error(err))
			} else if totalTokens > softLimitTokens {
				_ = stream.Status("Compressing memory to fit context window....")
				// First, attempt to further summarize long-term context if present
				if longTermContext != "" {
					sumCtx, sumCancel := context.WithTimeout(ctx, a.cfg.LLMRequestTimeout)
					summarizedContext, summaryErr := a.rag.SummarizeLongTermMemory(sumCtx, longTermContext, input)
					sumCancel()
					if summaryErr == nil && summarizedContext != "" {
						longTermContext = summarizedContext
						// Recompute combined count with summarized context
						combined = combined[:0]
						combined = append(combined, types.AgentMessage{Role: "system", Content: longTermContext})
						combined = append(combined, history...)
						totalTokens, err = a.memoryManager.CalculateHistorySize(ctx, combined)
						if err != nil {
							a.logger.Warn("Token recount after summarization failed", zap.Error(err))
						}
					}
				}

				// If still over the limit, calculate how many messages to trim in one pass
				if totalTokens > softLimitTokens {
					tokensToRemove := totalTokens - softLimitTokens
					tokensAccumulated := 0
					cutoffIndex := 0

					// Find cutoff point by accumulating token counts from oldest messages
					for cutoffIndex < len(history) && tokensAccumulated < tokensToRemove {
						msg := history[cutoffIndex]

						// Ensure message has token count computed
						if !msg.TokenCountComputed {
							tokens, err := a.memoryManager.CountTokens(ctx, msg.Content)
							if err != nil {
								a.logger.Warn("Failed to count tokens for message during trimming",
									zap.Error(err),
									zap.Int("cutoff_index", cutoffIndex))
								break
							}
							history[cutoffIndex].TokenCount = tokens
							history[cutoffIndex].TokenCountComputed = true
						}

						// If this is an assistant message with code, check for tool pair
						if msg.Role == "assistant" && format.HasCodeBlock(msg.Content) &&
							cutoffIndex+1 < len(history) && history[cutoffIndex+1].Role == "tool" {
							// Remove both messages together
							toolMsg := history[cutoffIndex+1]
							if !toolMsg.TokenCountComputed {
								tokens, err := a.memoryManager.CountTokens(ctx, toolMsg.Content)
								if err != nil {
									a.logger.Warn("Failed to count tokens for tool message during trimming",
										zap.Error(err))
									break
								}
								history[cutoffIndex+1].TokenCount = tokens
								history[cutoffIndex+1].TokenCountComputed = true
							}
							tokensAccumulated += msg.TokenCount + toolMsg.TokenCount
							cutoffIndex += 2
						} else {
							tokensAccumulated += msg.TokenCount
							cutoffIndex++
						}
					}

					// Slice once at the cutoff point
					if cutoffIndex > 0 && cutoffIndex < len(history) {
						history = history[cutoffIndex:]

						messagesForLLM = a.responseHandler.BuildMessagesForLLM(longTermContext, history)

						a.logger.Info("Trimmed history to fit context window",
							zap.Int("messages_removed", cutoffIndex),
							zap.Int("tokens_removed", tokensAccumulated),
							zap.Int("remaining_messages", len(history)))
					}
				}
			}
		}

		// Get LLM response with dynamic temperature - critical operation, break loop on failure
		currentTemp := loop.GetCurrentTemperature()
		responseChan, err := getLLMResponse(ctx, a.cfg.MainLLMHost, messagesForLLM, a.cfg, a.logger, &currentTemp)
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
			assistantMsg := types.AgentMessage{
				Role:        "assistant",
				Content:     llmResponse,
				ContentHash: rag.ComputeMessageContentHash("assistant", llmResponse),
			}
			toolMsg := types.AgentMessage{
				Role:        "tool",
				Content:     execResult.Result,
				ContentHash: rag.ComputeMessageContentHash("tool", execResult.Result),
			}

			// Add assistant response and tool result to history
			history = append(history, assistantMsg, toolMsg)

			// Store assistant + tool pair to RAG (user message stored separately via chat handler)
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
			assistantMsg := types.AgentMessage{
				Role:        "assistant",
				Content:     llmResponse,
				ContentHash: rag.ComputeMessageContentHash("assistant", llmResponse),
			}

			// Add assistant response to history
			history = append(history, assistantMsg)

			// Store assistant message to RAG (user message stored separately via chat handler)
			if a.rag != nil {
				a.rag.AddMessagesAsync(sessionID, []types.AgentMessage{assistantMsg})
			}

			return
		}
	}
}

func (a *Agent) GenerateTitle(ctx context.Context, content string) (string, error) {
	systemPrompt := prompts.TitleGenerator()

	userPrompt := fmt.Sprintf(`User message:
%s

Respond with only the title.`, content)

	messages := []types.AgentMessage{
		{Role: "system", Content: systemPrompt},
		{Role: "user", Content: userPrompt},
	}

	client := llmclient.New(a.cfg, a.logger)
	title, err := client.Chat(ctx, a.cfg.SummarizationLLMHost, messages, nil) // nil = use server default temp
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
