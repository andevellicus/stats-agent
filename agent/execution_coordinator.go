package agent

import (
    "context"
    "regexp"
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
func (e *ExecutionCoordinator) ProcessResponse(ctx context.Context, llmResponse, sessionID string, stream *Stream) (*ExecutionResult, error) {
    // Preprocess assistant text into our canonical XML tag format
    pre := format.PreprocessAssistantText(llmResponse)

    // Additional sanitation of python tag variants
    sanitizedResponse := sanitizePythonTags(pre)

    // Convert markdown code blocks to XML tags (idempotent if already converted)
    processedResponse := e.ConvertMarkdownToXML(sanitizedResponse)

    // Ensure any unbalanced tags are closed so execution can proceed
    processedResponse, _ = format.CloseUnbalancedTags(processedResponse)

	// Try to execute Python code if present
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

// sanitizePythonTags repairs minor formatting issues (extra spaces/newlines) around python tags.
func sanitizePythonTags(text string) string {
	replacements := map[string]string{
		"< python>":   "<python>",
		"<python >":   "<python>",
		"< python >":  "<python>",
		"</ python>":  "</python>",
		"</python >":  "</python>",
		"</ python >": "</python>",
		"<python\n":   "<python>\n",
		"\npython>":   "\n<python>",
		"\n/python>":  "\n</python>",
		"</python\n":  "</python>\n",
	}

    for old, new := range replacements {
        text = strings.ReplaceAll(text, old, new)
    }

    // Handle common patterns where the model writes "... something.python\n" before code
    // 1) Sentence ends with ".python" then a newline before code
    reDotPython := regexp.MustCompile(`(?m)\.python\s*\n`)
    text = reDotPython.ReplaceAllString(text, ".\n<python>\n")

    // 2) A label line like "python:" before code
    reLabelPython := regexp.MustCompile(`(?mi)^[\t ]*python:\s*\n`)
    text = reLabelPython.ReplaceAllString(text, "<python>\n")

    // 3) A standalone line "python" before code
    reLinePython := regexp.MustCompile(`(?m)^[\t ]*python\s*\n`)
    text = reLinePython.ReplaceAllString(text, "<python>\n")

    return text
}
