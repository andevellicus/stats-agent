package agent

import (
	"context"
	"fmt"
	"stats-agent/tools"
	"stats-agent/web/format"
	"strings"

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
func (e *ExecutionCoordinator) ProcessResponse(ctx context.Context, llmResponse, sessionID string) (*ExecutionResult, error) {
	// Convert markdown code blocks to XML tags first
	processedResponse := e.ConvertMarkdownToXML(llmResponse)

	// Try to execute Python code if present
	code, result, wasExecuted := e.pythonTool.ExecutePythonCode(ctx, processedResponse, sessionID)

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

	// Format result with XML tags for consistency
	formattedResult := fmt.Sprintf("<execution_results>\n%s\n</execution_results>", result)

	return &ExecutionResult{
		WasCodeExecuted: true,
		Code:            code,
		Result:          formattedResult,
		HasError:        hasError,
	}, nil
}

// ConvertMarkdownToXML converts markdown code blocks (```python) to XML tags (<python>).
// This handles cases where the LLM outputs markdown format instead of the expected XML format.
// Delegates to the format package for consistent conversion logic.
func (e *ExecutionCoordinator) ConvertMarkdownToXML(text string) string {
	return format.MarkdownToXML(text)
}

// DetectError checks if the execution result contains error indicators.
func (e *ExecutionCoordinator) DetectError(result string) bool {
	return strings.Contains(result, "Error:")
}
