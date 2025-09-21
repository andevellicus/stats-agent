package handlers

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"os"
	"stats-agent/agent"
	"strings"
	"sync"

	"github.com/ollama/ollama/api"
	"go.uber.org/zap"
)

// AgentResponse represents a complete response from the agent
type AgentResponse struct {
	Content     string   `json:"content"`
	CodeBlocks  []string `json:"code_blocks"`
	OutputBlocks []string `json:"output_blocks"`
	Error       error    `json:"error,omitempty"`
}

// WebAgent wraps the original agent to capture output for web responses
type WebAgent struct {
	agent  *agent.Agent
	logger *zap.Logger
}

func NewWebAgent(agent *agent.Agent, logger *zap.Logger) *WebAgent {
	return &WebAgent{
		agent:  agent,
		logger: logger,
	}
}

// RunForWeb executes the agent and captures all output for web display
func (wa *WebAgent) RunForWeb(ctx context.Context, input string) (*AgentResponse, error) {
	wa.logger.Info("Running agent for web request", zap.String("input", input))

	// Capture stdout
	oldStdout := os.Stdout
	r, w, err := os.Pipe()
	if err != nil {
		return nil, fmt.Errorf("failed to create pipe: %w", err)
	}

	os.Stdout = w

	// Buffer to collect output
	var output bytes.Buffer
	var wg sync.WaitGroup
	wg.Add(1)

	// Read from pipe in goroutine
	go func() {
		defer wg.Done()
		io.Copy(&output, r)
	}()

	// Run the agent
	wa.agent.Run(ctx, input)

	// Restore stdout
	w.Close()
	os.Stdout = oldStdout
	wg.Wait()
	r.Close()

	// Parse the captured output
	capturedOutput := output.String()
	response := wa.parseAgentOutput(capturedOutput)

	wa.logger.Debug("Agent execution completed",
		zap.String("captured_output_preview", wa.truncateString(capturedOutput, 200)),
		zap.Int("code_blocks", len(response.CodeBlocks)),
		zap.Int("output_blocks", len(response.OutputBlocks)))

	return response, nil
}

func (wa *WebAgent) parseAgentOutput(output string) *AgentResponse {
	response := &AgentResponse{
		Content:      "",
		CodeBlocks:   []string{},
		OutputBlocks: []string{},
	}

	lines := strings.Split(output, "\n")
	var currentSection string
	var currentBlock strings.Builder

	for _, line := range lines {
		switch {
		case strings.Contains(line, "--- Executing Python Code ---"):
			if currentBlock.Len() > 0 {
				wa.addBlockToResponse(response, currentSection, currentBlock.String())
				currentBlock.Reset()
			}
			currentSection = "code_start"

		case strings.Contains(line, "--- Execution Output ---"):
			if currentSection == "code_start" && currentBlock.Len() > 0 {
				response.CodeBlocks = append(response.CodeBlocks, strings.TrimSpace(currentBlock.String()))
				currentBlock.Reset()
			}
			currentSection = "output"

		case strings.Contains(line, "--- End Execution ---"):
			if currentSection == "output" && currentBlock.Len() > 0 {
				response.OutputBlocks = append(response.OutputBlocks, strings.TrimSpace(currentBlock.String()))
				currentBlock.Reset()
			}
			currentSection = "agent_response"

		case strings.HasPrefix(line, "Agent: "):
			if currentBlock.Len() > 0 {
				wa.addBlockToResponse(response, currentSection, currentBlock.String())
				currentBlock.Reset()
			}
			currentSection = "agent_response"
			// Remove "Agent: " prefix
			line = strings.TrimPrefix(line, "Agent: ")
			currentBlock.WriteString(line)

		default:
			if currentSection == "code_start" && !strings.Contains(line, "Code to execute:") {
				// Skip the "Code to execute:" line
				if !strings.HasPrefix(line, "Code to execute:") {
					currentBlock.WriteString(line + "\n")
				}
			} else if currentSection != "" {
				currentBlock.WriteString(line + "\n")
			}
		}
	}

	// Handle any remaining content
	if currentBlock.Len() > 0 {
		wa.addBlockToResponse(response, currentSection, currentBlock.String())
	}

	// If we didn't get agent responses, use the entire output as content
	if response.Content == "" && len(response.CodeBlocks) == 0 {
		response.Content = strings.TrimSpace(output)
	}

	return response
}

func (wa *WebAgent) addBlockToResponse(response *AgentResponse, section, content string) {
	content = strings.TrimSpace(content)
	if content == "" {
		return
	}

	switch section {
	case "agent_response":
		if response.Content != "" {
			response.Content += "\n\n"
		}
		response.Content += content
	case "output":
		response.OutputBlocks = append(response.OutputBlocks, content)
	}
}

func (wa *WebAgent) truncateString(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "..."
}

// RunForWebStream executes the agent and streams output via callback
func (wa *WebAgent) RunForWebStream(ctx context.Context, input string, onChunk func(string)) (*AgentResponse, error) {
	wa.logger.Info("Running agent for web stream request", zap.String("input", input))

	// Create pipes for streaming
	r, w, err := os.Pipe()
	if err != nil {
		return nil, fmt.Errorf("failed to create pipe: %w", err)
	}

	oldStdout := os.Stdout
	os.Stdout = w

	// Buffer to collect full output for parsing
	var fullOutput bytes.Buffer
	var wg sync.WaitGroup
	wg.Add(1)

	// Stream reader goroutine
	go func() {
		defer wg.Done()
		buffer := make([]byte, 1024)
		for {
			n, err := r.Read(buffer)
			if n > 0 {
				chunk := string(buffer[:n])
				fullOutput.WriteString(chunk)
				// Stream to callback
				onChunk(chunk)
			}
			if err != nil {
				break
			}
		}
	}()

	// Run the agent
	wa.agent.Run(ctx, input)

	// Cleanup
	w.Close()
	os.Stdout = oldStdout
	wg.Wait()
	r.Close()

	// Parse the full output
	capturedOutput := fullOutput.String()
	response := wa.parseAgentOutput(capturedOutput)

	wa.logger.Debug("Agent streaming execution completed",
		zap.String("captured_output_preview", wa.truncateString(capturedOutput, 200)),
		zap.Int("code_blocks", len(response.CodeBlocks)),
		zap.Int("output_blocks", len(response.OutputBlocks)))

	return response, nil
}

// GetHistory returns the agent's conversation history
func (wa *WebAgent) GetHistory() []api.Message {
	// We'll need to add a method to the agent to expose history
	// For now, return empty slice
	return []api.Message{}
}