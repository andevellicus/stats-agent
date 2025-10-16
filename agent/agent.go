package agent

import (
    "crypto/sha256"
    "context"
    "fmt"
    "regexp"
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
	queryBuilder         *QueryBuilder
	actionCache          *ActionCache
}

// Tokenize request/response types have been centralized in llmclient.

func NewAgent(cfg *config.Config, pythonTool *tools.StatefulPythonTool, rag *rag.RAG, logger *zap.Logger) *Agent {
	logger.Info("Agent initialized", zap.Int("context_window_size", cfg.ContextLength))

	// Initialize specialized components
	memoryManager := NewMemoryManager(cfg, logger)
	executionCoordinator := NewExecutionCoordinator(pythonTool, logger)
	responseHandler := NewResponseHandler(cfg, logger)
	queryBuilder := NewQueryBuilder(cfg, rag, logger)
	actionCache := NewActionCache(5) // Track last 5 actions for repeat detection

	return &Agent{
		cfg:                  cfg,
		pythonTool:           pythonTool,
		rag:                  rag,
		logger:               logger,
		memoryManager:        memoryManager,
		executionCoordinator: executionCoordinator,
		responseHandler:      responseHandler,
		queryBuilder:         queryBuilder,
		actionCache:          actionCache,
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
    if a.actionCache != nil {
        a.actionCache.PurgeSession(sessionID)
        a.logger.Info("Purged action cache for session", zap.String("session_id", sessionID))
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

	// 2. Query RAG for state (use configured DocumentModeRAGResults)
	ragResults := a.cfg.DocumentModeRAGResults

	// Extract content hashes from current history to exclude from RAG results
	excludeHashes := make([]string, 0, len(history))
	for _, msg := range history {
		if msg.ContentHash != "" {
			excludeHashes = append(excludeHashes, msg.ContentHash)
		}
	}

	ragCtx, ragCancel := context.WithTimeout(ctx, a.cfg.LLMRequestTimeout)
	defer ragCancel()
	// Document mode doesn't use post-query pruning (simpler flow)
	// Document mode doesn't use action cache (no code execution in this mode)
	state, err := a.rag.Query(ragCtx, sessionID, input, ragResults, excludeHashes, nil, "")
	if err != nil {
		a.logger.Warn("Failed to query RAG for state, continuing without it",
			zap.Error(err),
			zap.String("session_id", sessionID))
		state = ""
	}

	// 3. Build messages for LLM (use document QA prompt)
	// Append user message to history for this request (but don't modify passed-in history yet)
	historyWithUserMsg := append(history, userMsg)

	var messagesForLLM []types.AgentMessage

	// Add state if available
	if state != "" {
		messagesForLLM = append(messagesForLLM, types.AgentMessage{
			Role:    "system",
			Content: state,
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
	var ephemeralEvidence string
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

		// Query RAG for state before each turn
		// This ensures newly added content (PDFs, facts) is available
		// Build structured query using QueryBuilder (combines fact summaries, metadata, values, synonyms)
		queryText := a.queryBuilder.BuildRAGQuery(ctx, input, sessionID, history, turn)

		// Build sets for post-query pruning (O(1) lookups)
		excludeHashSet := make(map[string]bool)
		historyDocIDSet := make(map[string]bool)

		// Collect content hashes and document IDs from history
		for _, msg := range history {
			if msg.ContentHash != "" {
				excludeHashSet[msg.ContentHash] = true
			}

			// Extract document ID from metadata if available (cached from previous queries)
			if msg.Metadata != nil {
				if docID := msg.Metadata["document_id"]; docID != "" {
					// Resolve lookup ID (handles parent/chunk relationships)
					lookupID := rag.ResolveLookupID(docID, msg.Metadata)
					if lookupID != "" {
						historyDocIDSet[lookupID] = true
					}
				}
			}
		}

		// Lazy lookup: If we don't have document IDs cached in metadata, query by content hash
		if len(historyDocIDSet) == 0 && len(excludeHashSet) > 0 {
			// Convert hash set to slice for query
			hashSlice := make([]string, 0, len(excludeHashSet))
			for hash := range excludeHashSet {
				hashSlice = append(hashSlice, hash)
			}

			// Query database for document IDs by content hash
			docIDMap, err := a.rag.GetDocumentIDsByContentHash(ctx, sessionID, hashSlice)
			if err != nil {
				a.logger.Warn("Failed to lookup document IDs by content hash, continuing with hash-only dedup",
					zap.Error(err),
					zap.String("session_id", sessionID))
			} else {
				// Populate historyDocIDSet from query results
				for _, docID := range docIDMap {
					if docID != "" {
						historyDocIDSet[docID] = true
					}
				}
			}
		}

		// Convert sets to slices at call boundary
		excludeHashes := make([]string, 0, len(excludeHashSet))
		for hash := range excludeHashSet {
			excludeHashes = append(excludeHashes, hash)
		}

		historyDocIDs := make([]string, 0, len(historyDocIDSet))
		for docID := range historyDocIDSet {
			historyDocIDs = append(historyDocIDs, docID)
		}

        // Build done ledger from action cache (per session)
        doneLedger := a.actionCache.BuildDoneLedger(sessionID)

		// Add timeout to RAG query to avoid hangs
		ragCtx, ragCancel := context.WithTimeout(ctx, a.cfg.LLMRequestTimeout)
		defer ragCancel()
		state, err := a.rag.Query(ragCtx, sessionID, queryText, a.cfg.RAGResults, excludeHashes, historyDocIDs, doneLedger)
		if err != nil {
			a.logger.Warn("Failed to query RAG for state, continuing without it",
				zap.Error(err),
				zap.String("session_id", sessionID),
				zap.Int("turn", turn))
			state = "" // Ensure empty on error
		}

		// Build messages for LLM (combine state + history + current user message)
		// On turn 0, append user message. On turn 1+, it's already in history
		if turn == 0 {
			history = append(history, userMsg)
		}
		evidenceForThisTurn := ephemeralEvidence
		messagesForLLM := a.responseHandler.BuildMessagesForLLMWithEvidence(state, evidenceForThisTurn, history)
		// Evidence is ephemeral: clear after attaching once
		ephemeralEvidence = ""

		// Ensure entire payload fits within configured budgets
		{
			// Measure current messages (memory + evidence + history)
			totalTokens, err := a.memoryManager.CalculateHistorySize(ctx, messagesForLLM)
			if err != nil {
				a.logger.Warn("Failed to count tokens for combined context; proceeding optimistically", zap.Error(err))
			} else {
					// Account for fixed overhead (system prompt) and reserve a recency budget
					systemTokens, sysErr := a.memoryManager.CountTokens(ctx, prompts.AgentSystem())
					if sysErr != nil {
						a.logger.Warn("Failed to count tokens for system prompt; using 0 overhead", zap.Error(sysErr))
						systemTokens = 0
					}

					// Measure state and evidence separately to enforce an overhead cap
					stateTokens := 0
					if strings.TrimSpace(state) != "" {
						if tok, err2 := a.memoryManager.CountTokens(ctx, state); err2 == nil {
							stateTokens = tok
						}
					}
					evidenceTokens := 0
					if strings.TrimSpace(evidenceForThisTurn) != "" {
						if tok, err2 := a.memoryManager.CountTokens(ctx, evidenceForThisTurn); err2 == nil {
							evidenceTokens = tok
						}
					}

					maxPrompt := a.cfg.ContextLength - a.cfg.ResponseTokenBudget
					if maxPrompt < 0 { maxPrompt = 0 }
					recencyMin := int(float64(maxPrompt) * a.cfg.ContextSoftLimitRatio)
					if recencyMin < 0 { recencyMin = 0 }
					overheadCap := maxPrompt - recencyMin // budget for system + state + evidence
					if overheadCap < 0 { overheadCap = 0 }

					overheadTokens := systemTokens + stateTokens + evidenceTokens

					// If overhead exceeds its cap, first try to compress state
					if overheadTokens > overheadCap && strings.TrimSpace(state) != "" {
						sumCtx, sumCancel := context.WithTimeout(ctx, a.cfg.LLMRequestTimeout)
						summarized, sErr := a.rag.SummarizeState(sumCtx, state, input)
						sumCancel()
						if sErr == nil && strings.TrimSpace(summarized) != "" {
							state = summarized
							messagesForLLM = a.responseHandler.BuildMessagesForLLMWithEvidence(state, evidenceForThisTurn, history)
							// Recompute token counts
							stateTokens = 0
							if tok, err2 := a.memoryManager.CountTokens(ctx, state); err2 == nil { stateTokens = tok }
							if tok, err2 := a.memoryManager.CountTokens(ctx, messagesForLLM[0].Content); err2 == nil { _ = tok } // noop to avoid lint
							overheadTokens = systemTokens + stateTokens + evidenceTokens
						}
					}

					// If still above cap, drop ephemeral evidence (turn-only)
					if overheadTokens > overheadCap && evidenceTokens > 0 {
						evidenceForThisTurn = ""
						messagesForLLM = a.responseHandler.BuildMessagesForLLMWithEvidence(state, evidenceForThisTurn, history)
						evidenceTokens = 0
						overheadTokens = systemTokens + stateTokens
					}

					// Allowed tokens for all messages (state + evidence + history), excluding system prompt
					allowed := maxPrompt - systemTokens
					if allowed < 0 { allowed = 0 }

					if totalTokens > allowed {
						a.logger.Warn("Compressing payload to fit context window", zap.Int("totalTokens", totalTokens))
						// Recalculate after overhead adjustments
						totalTokens, err = a.memoryManager.CalculateHistorySize(ctx, messagesForLLM)
						if err != nil {
							a.logger.Warn("Token recount after overhead adjustment failed", zap.Error(err))
						}

					// If still over, trim history tokens in one pass
					if totalTokens > allowed {
						tokensToRemove := totalTokens - allowed
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

							messagesForLLM = a.responseHandler.BuildMessagesForLLMWithEvidence(state, evidenceForThisTurn, history)

							a.logger.Info("Trimmed history to fit context window",
								zap.Int("messages_removed", cutoffIndex),
								zap.Int("tokens_removed", tokensAccumulated),
								zap.Int("remaining_messages", len(history)))
						}
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
			state = a.handleEmptyResponse(ctx, state, input, stream)
			if state == "" {
				break // Recovery failed
			}
			continue
		}

        // === ACTION CACHE: Check if code about to be executed is already done ===
        var execResult *ExecutionResult
        var actionSig *ActionSignature
        var proposedCode string

        if format.HasCodeBlock(llmResponse) {
            // Extract first code block from markdown
            code, hasCode := format.ExtractCodeContent(llmResponse)
            if hasCode {
                proposedCode = code
                // Extract action signature from code
                currentDataset := getCurrentDataset(history)
                currentN := getCurrentSampleSize(history)
                schemaHash := getCurrentSchemaHash(history)

                actionSig = ExtractActionSignature(code, currentDataset, currentN, schemaHash)
                if actionSig != nil {
                    actionSig.SessionID = sessionID
                }

                // Check if action already completed successfully
                if cached, exists := a.actionCache.Get(*actionSig); exists && cached.Success {
                    // Hysteresis: require exact-phrase match before skipping
                    currentHash := a.normalizeCodeHash(code)
                    if cached.CodeNormHash != "" && cached.CodeNormHash == currentHash && !a.userRequestsRerun(input) {
                        a.logger.Info("Action already completed; skipping repeat and prompting for next step",
                            zap.String("action", actionSig.String()),
                            zap.Int("cached_turn", cached.Turn),
                            zap.Int("current_turn", turn))

                        // Inject a one-turn evidence note to steer the LLM away from repeats
                        note := fmt.Sprintf("Action %s already completed successfully in turn %d. Do not repeat; choose a different next step (e.g., effect size, post-hoc, multivariable model, or finalize).",
                            actionSig.String(), cached.Turn)
                        if ephemeralEvidence == "" {
                            ephemeralEvidence = "<evidence>\n" + note + "\n</evidence>"
                        } else {
                            ephemeralEvidence = ephemeralEvidence + "\n" + note
                        }

                        // Skip adding assistant/tool messages and proceed to next turn
                        continue
                    }
                }

				// Check for recent repeats (last N actions)
				repeatCount := a.actionCache.CountRecentRepeats(*actionSig)
				if repeatCount >= 1 {
					a.logger.Warn("Detected repeated action in recent turns",
						zap.String("action", actionSig.String()),
						zap.Int("repeat_count", repeatCount),
						zap.Int("turn", turn))
				}
			}
		}

		// Process response for code execution - critical operation
		execResult, err = a.executionCoordinator.ProcessResponse(ctx, llmResponse, sessionID, stream)
		if err != nil {
			a.logger.Error("Failed to process LLM response, aborting turn",
				zap.Error(err),
				zap.Int("turn", turn),
				zap.String("session_id", sessionID))
			_ = stream.Status("Response processing error")
			break
		}

		// Record action in cache if code was executed
        if execResult.WasCodeExecuted && actionSig != nil {
            result := ActionResult{
                Signature: *actionSig,
                Output:    execResult.Result,
                Success:   !execResult.HasError,
                Turn:      turn,
                Attempt:   1, // TODO: Track retry attempts
                CodeNormHash: a.normalizeCodeHash(proposedCode),
            }
            a.actionCache.Add(*actionSig, result)

			a.logger.Debug("Recorded action in cache",
				zap.String("action", actionSig.String()),
				zap.Bool("success", result.Success),
				zap.Int("turn", turn))
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
				// Pass action hash if available for retry budget tracking
				if actionSig != nil {
					loop.RecordError(actionSig.ComputeHash())
				} else {
					loop.RecordError()
				}
			} else {
				// Pass action hash if available to clear retry counter
				if actionSig != nil {
					loop.RecordSuccess(actionSig.ComputeHash())
				} else {
					loop.RecordSuccess()
				}
			}

			// Attach ephemeral evidence when helpful: errors or statistical identifiers
			if snippet := a.buildEvidenceSnippet(ctx, execResult.Result); snippet != "" {
				ephemeralEvidence = "<evidence>\n" + snippet + "\n</evidence>"
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
func (a *Agent) handleEmptyResponse(ctx context.Context, state, latestUserMessage string, stream *Stream) string {
	a.logger.Warn("LLM response was empty, likely due to a context window error. Attempting to summarize state")
	_ = stream.Status("Compressing memory due to a context window error...")

	sumCtx, sumCancel := context.WithTimeout(ctx, a.cfg.LLMRequestTimeout)
	summarizedContext, err := a.rag.SummarizeState(sumCtx, state, latestUserMessage)
	sumCancel()
	if err != nil {
		a.logger.Error("Recovery failed: Could not summarize state. Aborting turn", zap.Error(err))
		return ""
	}

	return summarizedContext
}

// normalizeCodeHash returns a short hash of code with whitespace collapsed.
func (a *Agent) normalizeCodeHash(code string) string {
    s := strings.TrimSpace(code)
    if s == "" {
        return ""
    }
    re := regexp.MustCompile(`\s+`)
    normalized := re.ReplaceAllString(s, " ")
    sum := sha256.Sum256([]byte(normalized))
    return fmt.Sprintf("%x", sum[:8])
}

// userRequestsRerun returns true when the user explicitly asks to rerun the same
// action and mentions explanation or rationale, allowing repeats when requested.
func (a *Agent) userRequestsRerun(input string) bool {
    lower := strings.ToLower(strings.TrimSpace(input))
    if lower == "" {
        return false
    }
    rerunPhrases := []string{"re-run", "rerun", "run again", "repeat", "redo", "try again", "recalculate", "run the same", "same code"}
    var wantsRerun bool
    for _, p := range rerunPhrases {
        if strings.Contains(lower, p) {
            wantsRerun = true
            break
        }
    }
    if !wantsRerun {
        return false
    }
    // Look for an explanation intent to avoid accidental repeats
    explain := []string{"explain", "explanation", "why", "details", "show"}
    for _, e := range explain {
        if strings.Contains(lower, e) {
            return true
        }
    }
    return false
}

// buildEvidenceSnippet constructs a 150–300 token snippet from tool output
// prioritizing identifiers, errors, and formulas. Not stored persistently.
func (a *Agent) buildEvidenceSnippet(ctx context.Context, result string) string {
	trimmed := strings.TrimSpace(result)
	if trimmed == "" {
		return ""
	}

	// Quick need check: only attach when likely useful
	lower := strings.ToLower(trimmed)
	need := strings.Contains(lower, "error") || strings.Contains(lower, "traceback") ||
		strings.Contains(lower, "p=") || strings.Contains(lower, "p<") ||
		strings.Contains(lower, " w=") || strings.Contains(lower, "cramer") || strings.Contains(lower, " r=")
	if !need {
		// Also attach if there are many digits or formulas
		digitRe := regexp.MustCompile(`\d`)
		need = digitRe.MatchString(trimmed) && (strings.Contains(trimmed, "=") || strings.Contains(trimmed, ":"))
		if !need {
			return ""
		}
	}

	lines := strings.Split(trimmed, "\n")
	var selected []string

	// 1) Capture error blocks with small context
	errIdx := -1
	for i, l := range lines {
		ll := strings.ToLower(l)
		if strings.Contains(ll, "error") || strings.Contains(ll, "traceback") {
			errIdx = i
			break
		}
	}
	if errIdx >= 0 {
		start := errIdx - 2
		if start < 0 {
			start = 0
		}
		end := errIdx + 8
		if end > len(lines) {
			end = len(lines)
		}
		selected = append(selected, lines[start:end]...)
	}

	// 2) Add lines with identifiers/formulas
	keyRe := regexp.MustCompile(`(?i)\b(p\s*[=<>]|w\s*=|r\s*=|cramer|cohen|eta|chi2|t\s*=|f\s*=|u\s*=|h\s*=)`)
	digitOrEq := regexp.MustCompile(`\d|=`)
	for _, l := range lines {
		if keyRe.MatchString(l) || digitOrEq.MatchString(l) {
			selected = append(selected, l)
		}
		if len(selected) > 400 { // guard
			break
		}
	}

	// Deduplicate and trim
	if len(selected) == 0 {
		return ""
	}
	// Remove empty lines
	compact := selected[:0]
	for _, l := range selected {
		t := strings.TrimSpace(l)
		if t != "" {
			compact = append(compact, t)
		}
	}
	snippet := strings.Join(compact, "\n")

	// Token trim to 150–300 tokens using model tokenizer if available
	minTokens := 150
	maxTokens := 300
	tokens, err := a.memoryManager.CountTokens(ctx, snippet)
	if err != nil {
		// Fallback: char-based clamp ~ 4 chars per token heuristic
		maxChars := maxTokens * 4
		if len(snippet) > maxChars {
			// preserve start and end context
			head := snippet
			if len(head) > maxChars {
				head = head[:maxChars]
			}
			return strings.TrimSpace(head)
		}
		return strings.TrimSpace(snippet)
	}

	if tokens <= maxTokens && tokens >= minTokens {
		return snippet
	}
	if tokens <= minTokens {
		return snippet // good enough
	}

	// Trim down by dropping lines from the end until within limit
	parts := strings.Split(snippet, "\n")
	for len(parts) > 1 {
		parts = parts[:len(parts)-1]
		candidate := strings.Join(parts, "\n")
		t, err := a.memoryManager.CountTokens(ctx, candidate)
		if err != nil {
			break
		}
		if t <= maxTokens {
			return candidate
		}
	}
	// Final fallback: hard truncate
	return snippet
}
