package agent

import (
    "context"
    "strings"

    "stats-agent/prompts"
    "stats-agent/rag"
    "stats-agent/web/format"
    "stats-agent/web/types"

	"go.uber.org/zap"
)

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
	state, err := a.rag.Query(ragCtx, sessionID, input, ragResults, excludeHashes, nil, "", "document")
	if err != nil {
		a.logger.Warn("Failed to query RAG for state, continuing without it",
			zap.Error(err),
			zap.String("session_id", sessionID))
		state = ""
	}

    // 3. Build messages for LLM (use document QA prompt) with optional evidence
    // Append user message to history for this request (but don't modify passed-in history yet)
    historyWithUserMsg := append(history, userMsg)

    // Build an ephemeral evidence snippet from state when the question suggests quoting/thresholds
    docEvidence := a.buildDocEvidenceSnippet(ctx, input, state)
    messagesForLLM := a.responseHandler.BuildMessagesForLLMWithEvidence(state, docEvidence, historyWithUserMsg)

    // Recency-first budgeting for document mode
    {
        totalTokens, err := a.memoryManager.CalculateHistorySize(ctx, messagesForLLM)
        if err != nil {
            a.logger.Warn("Doc: token count failed; proceeding optimistically", zap.Error(err))
        } else {
            // Fixed overhead: system prompt
            systemTokens, sysErr := a.memoryManager.CountTokens(ctx, prompts.DocumentQA())
            if sysErr != nil {
                a.logger.Warn("Doc: system prompt tokens failed", zap.Error(sysErr))
                systemTokens = 0
            }
            // Overhead parts: state + evidence
            stateTokens := 0
            if strings.TrimSpace(state) != "" {
                if tok, e := a.memoryManager.CountTokens(ctx, state); e == nil {
                    stateTokens = tok
                }
            }
            evidenceTokens := 0
            if strings.TrimSpace(docEvidence) != "" {
                if tok, e := a.memoryManager.CountTokens(ctx, docEvidence); e == nil {
                    evidenceTokens = tok
                }
            }

            maxPrompt := a.cfg.ContextLength - a.cfg.ResponseTokenBudget
            if maxPrompt < 0 {
                maxPrompt = 0
            }
            recencyMin := int(float64(maxPrompt) * a.cfg.ContextSoftLimitRatio)
            if recencyMin < 0 {
                recencyMin = 0
            }
            overheadCap := maxPrompt - recencyMin
            if overheadCap < 0 {
                overheadCap = 0
            }
            overheadTokens := systemTokens + stateTokens + evidenceTokens

            // If overhead exceeds cap, first compress state, then drop evidence
            if overheadTokens > overheadCap && strings.TrimSpace(state) != "" {
                sumCtx, sumCancel := context.WithTimeout(ctx, a.cfg.LLMRequestTimeout)
                summarized, sErr := a.rag.SummarizeState(sumCtx, state, input)
                sumCancel()
                if sErr == nil && strings.TrimSpace(summarized) != "" {
                    state = summarized
                    messagesForLLM = a.responseHandler.BuildMessagesForLLMWithEvidence(state, docEvidence, historyWithUserMsg)
                    stateTokens = 0
                    if tok, e := a.memoryManager.CountTokens(ctx, state); e == nil {
                        stateTokens = tok
                    }
                    overheadTokens = systemTokens + stateTokens + evidenceTokens
                }
            }
            if overheadTokens > overheadCap && evidenceTokens > 0 {
                // Drop ephemeral evidence for this turn
                docEvidence = ""
                messagesForLLM = a.responseHandler.BuildMessagesForLLMWithEvidence(state, docEvidence, historyWithUserMsg)
                evidenceTokens = 0
                overheadTokens = systemTokens + stateTokens
            }

            // Allowed for messages (excluding system prompt which is prepended later by getLLMResponseForDocumentMode)
            allowed := maxPrompt - systemTokens
            if allowed < 0 {
                allowed = 0
            }
            if totalTokens > allowed {
                // Trim oldest history messages until within budget
                tokensToRemove := totalTokens - allowed
                tokensAccum := 0
                cut := 0
                // Only trim from the history portion (not the first system slot in messagesForLLM)
                // messagesForLLM = [optional system(state/evidence)] + historyWithUserMsg
                // We will operate on historyWithUserMsg and rebuild
                for cut < len(historyWithUserMsg) && tokensAccum < tokensToRemove {
                    msg := historyWithUserMsg[cut]
                    if !msg.TokenCountComputed {
                        if tok, e := a.memoryManager.CountTokens(ctx, msg.Content); e == nil {
                            historyWithUserMsg[cut].TokenCount = tok
                            historyWithUserMsg[cut].TokenCountComputed = true
                        } else {
                            a.logger.Warn("Doc: failed to count tokens for trim", zap.Error(e))
                            break
                        }
                    }
                    tokensAccum += historyWithUserMsg[cut].TokenCount
                    cut++
                }
                if cut > 0 && cut < len(historyWithUserMsg) {
                    historyWithUserMsg = historyWithUserMsg[cut:]
                    messagesForLLM = a.responseHandler.BuildMessagesForLLMWithEvidence(state, docEvidence, historyWithUserMsg)
                    a.logger.Info("Doc: trimmed history to fit window",
                        zap.Int("messages_removed", cut),
                        zap.Int("remaining_messages", len(historyWithUserMsg)))
                }
            }
        }
    }

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

    // 6. If the response includes code, execute it and stream tool output
    if format.HasCodeBlock(llmResponse) {
        execResult, err := a.executionCoordinator.ProcessResponse(ctx, llmResponse, sessionID, stream)
        if err != nil {
            a.logger.Error("Failed to process LLM response in document mode",
                zap.Error(err),
                zap.String("session_id", sessionID))
            _ = stream.Status("Response processing error")
            return
        }
        // Record assistant/tool in RAG for retrieval as context
        if a.rag != nil {
            assistantMsg := types.AgentMessage{
                Role:        "assistant",
                Content:     llmResponse,
                ContentHash: rag.ComputeMessageContentHash("assistant", llmResponse),
            }
            if execResult.WasCodeExecuted {
                toolMsg := types.AgentMessage{
                    Role:        "tool",
                    Content:     execResult.Result,
                    ContentHash: rag.ComputeMessageContentHash("tool", execResult.Result),
                }
                a.rag.AddMessagesAsync(sessionID, []types.AgentMessage{assistantMsg, toolMsg})
            } else {
                a.rag.AddMessagesAsync(sessionID, []types.AgentMessage{assistantMsg})
            }
        }
        return
    }

    // 7. Store assistant response to RAG (user message stored separately via chat handler)
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

// buildDocEvidenceSnippet constructs a 150–300 token snippet from the retrieved
// document state when the question suggests quoting thresholds/definitions.
// It does not persist and is attached for one turn only.
func (a *Agent) buildDocEvidenceSnippet(ctx context.Context, input, state string) string {
    if strings.TrimSpace(state) == "" {
        return ""
    }
    lowerQ := strings.ToLower(input)
    // Heuristic: only attach when likely useful
    need := strings.Contains(lowerQ, "threshold") || strings.Contains(lowerQ, "quote") ||
        strings.Contains(lowerQ, "define") || strings.Contains(lowerQ, "definition") ||
        strings.Contains(lowerQ, "formula") || strings.Contains(lowerQ, "equation") ||
        strings.Contains(lowerQ, "figure") || strings.Contains(lowerQ, "table")
    if !need {
        return ""
    }

    // Strip <memory> tags and take the first substantive lines
    s := strings.TrimSpace(state)
    s = strings.ReplaceAll(s, "<memory>", "")
    s = strings.ReplaceAll(s, "</memory>", "")
    s = strings.TrimSpace(s)
    if s == "" {
        return ""
    }

    // Token-bound to ~150–300 tokens
    tokens, err := a.memoryManager.CountTokens(ctx, s)
    if err != nil {
        // Fallback: 1200 chars ~ 300 tokens
        if len(s) > 1200 {
            return s[:1200]
        }
        return s
    }
    if tokens <= 300 {
        return s
    }

    // Binary search down to <=300 tokens by cutting mid content
    // Simple approach: take head with incremental backoff
    low, high := 0, len(s)
    best := s
    for i := 0; i < 10; i++ {
        mid := (2*low + high) / 3 // bias toward larger slice
        if mid <= 0 { break }
        cand := strings.TrimSpace(s[:mid])
        t, e := a.memoryManager.CountTokens(ctx, cand)
        if e != nil {
            break
        }
        if t <= 300 {
            best = cand
            low = mid
        } else {
            high = mid
        }
        if high-low < 64 { // small window
            break
        }
    }
    return best
}
