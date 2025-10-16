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
		'\u201c':  '\u201d', // smart double quotes
		'\u201d':  '\u201d', // in case only closing quote is used on both ends
		'\u2018':  '\u2019', // smart single quotes
		'\u2019':  '\u2019', // in case only closing quote is used on both ends
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
