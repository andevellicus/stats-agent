package agent

import (
    "context"
    "strings"

    "stats-agent/tools"
    "stats-agent/web/format"

    "go.uber.org/zap"
)

// ExecutionCoordinator handles Python code detection, execution, and result processing.
type ExecutionCoordinator struct {
	pythonTool *tools.StatefulPythonTool
	logger     *zap.Logger
}

// ExecutionResult contains the outcome of processing an LLM response for code execution.
type ExecutionResult struct {
	WasCodeExecuted bool   // Whether code was found and executed
	Code            string // The extracted Python code
	Result          string // Execution result (or error message)
	HasError        bool   // Whether the execution resulted in an error
}

// NewExecutionCoordinator creates a new execution coordinator instance.
func NewExecutionCoordinator(pythonTool *tools.StatefulPythonTool, logger *zap.Logger) *ExecutionCoordinator {
	return &ExecutionCoordinator{
		pythonTool: pythonTool,
		logger:     logger,
	}
}

// ProcessResponse checks if the LLM response contains Python code, executes it if found,
// and returns the execution result.
// Expects LLM to output properly formatted markdown code fences as instructed in system prompt.
func (e *ExecutionCoordinator) ProcessResponse(ctx context.Context, llmResponse, sessionID string, stream *Stream) (*ExecutionResult, error) {
    // Normalize common LLM quirks (curly quotes, etc.) before any parsing
    sanitized := format.PreprocessAssistantText(llmResponse)
    // Safety: ensure any unbalanced tags are closed (for <tool> and <agent_status> tags)
    processedResponse, _ := format.CloseUnbalancedTags(sanitized)

	// Try to execute Python code if present (markdown fences only)
	code, result, wasExecuted := e.pythonTool.ExecutePythonCode(ctx, processedResponse, sessionID, nil)

	if !wasExecuted {
		return &ExecutionResult{
			WasCodeExecuted: false,
		}, nil
	}

	hasError := e.DetectError(result)

	if hasError {
		e.logger.Warn("Python execution resulted in error",
			zap.String("session_id", sessionID),
			zap.String("error_preview", result[:min(200, len(result))]))
	}

	if stream != nil {
		if err := stream.Tool(result); err != nil {
			e.logger.Warn("Failed to stream tool result",
				zap.String("session_id", sessionID),
				zap.Error(err))
		}
	}

	return &ExecutionResult{
		WasCodeExecuted: true,
		Code:            code,
		Result:          result,
		HasError:        hasError,
	}, nil
}

// DetectError checks if the execution result contains error indicators.
func (e *ExecutionCoordinator) DetectError(result string) bool {
	return strings.Contains(result, "Error:")
}
