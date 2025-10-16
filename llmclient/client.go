package llmclient

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"stats-agent/config"
	"stats-agent/web/types"
	"strings"
	"time"

	"go.uber.org/zap"
)

// ErrContextWindowExceeded is returned when the model reports the prompt
// exceeds the available context size.
var ErrContextWindowExceeded = errors.New("context window exceeded")

type streamChoice struct {
	Delta struct {
		Content string `json:"content"`
	} `json:"delta"`
	Index int `json:"index"`
}

type streamResponse struct {
	Choices []streamChoice `json:"choices"`
}

type chatRequest struct {
	Messages    []types.AgentMessage `json:"messages"`
	Stream      bool                 `json:"stream"`
	Stop        []string             `json:"stop,omitempty"`        // Stop sequences to halt generation
	Temperature *float64             `json:"temperature,omitempty"` // Per-request temperature override
}

type chatResponse struct {
	Choices []struct {
		Message types.AgentMessage `json:"message"`
	} `json:"choices"`
}

// Embedding request/response mirror llama.cpp's expected schema
type embeddingRequest struct {
	Content string `json:"content"`
}

type embeddingResponse []struct {
	Embedding [][]float32 `json:"embedding"`
}

type Client struct {
	cfg        *config.Config
	httpClient *http.Client
	logger     *zap.Logger
}

func New(cfg *config.Config, logger *zap.Logger) *Client {
	// Use a client with the configured timeout; streaming requests rely on context
	// cancellation or server closing the stream.
	return &Client{
		cfg:        cfg,
		httpClient: &http.Client{Timeout: cfg.LLMRequestTimeout},
		logger:     logger,
	}
}

// Chat performs a non-streaming chat completion call.
// temperature is optional; pass nil to use server default.
func (c *Client) Chat(ctx context.Context, host string, messages []types.AgentMessage, temperature *float64) (string, error) {
	reqBody := chatRequest{
		Messages:    messages,
		Stream:      false,
		Temperature: temperature,
	}
	jsonBody, err := json.Marshal(reqBody)
	if err != nil {
		return "", fmt.Errorf("marshal chat request: %w", err)
	}

	url := fmt.Sprintf("%s/v1/chat/completions", strings.TrimRight(host, "/"))

	var resp *http.Response
	var lastErr error
	for attempt := 0; attempt < c.cfg.MaxRetries; attempt++ {
		req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(jsonBody))
		if err != nil {
			return "", fmt.Errorf("create chat request: %w", err)
		}
		req.Header.Set("Content-Type", "application/json")

		resp, err = c.httpClient.Do(req)
		if err != nil {
			lastErr = err
			// Do not retry on context cancellation/deadline
			if ctx.Err() != nil {
				break
			}
		} else if resp.StatusCode == http.StatusServiceUnavailable {
			// Model loading; retry with backoff
			io.Copy(io.Discard, resp.Body)
			resp.Body.Close()
			c.backoffSleep(attempt)
			continue
		} else {
			break
		}
	}
	if resp == nil {
		return "", fmt.Errorf("no response from LLM server: %w", lastErr)
	}
	defer resp.Body.Close()

	bodyBytes, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", fmt.Errorf("read chat response: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		if strings.Contains(string(bodyBytes), "exceeds the available context size") {
			return "", ErrContextWindowExceeded
		}
		return "", fmt.Errorf("llm server status %s: %s", resp.Status, string(bodyBytes))
	}

	var cr chatResponse
	if err := json.Unmarshal(bodyBytes, &cr); err != nil {
		return "", fmt.Errorf("decode chat response: %w", err)
	}
	if len(cr.Choices) == 0 {
		return "", fmt.Errorf("no response choices from llm server")
	}
	return cr.Choices[0].Message.Content, nil
}

// ChatStream performs a streaming chat completion call and returns a channel of chunks.
// temperature is optional; pass nil to use server default.
func (c *Client) ChatStream(ctx context.Context, host string, messages []types.AgentMessage, temperature *float64) (<-chan string, error) {
	// See rationale in Chat(): omit stop sequence to avoid backends removing
	// Markdown backticks from the output. The agent will still add a missing
	// closing fence if needed for robustness.
	reqBody := chatRequest{
		Messages:    messages,
		Stream:      true,
		Temperature: temperature,
	}
	jsonBody, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("marshal chat request: %w", err)
	}

	url := fmt.Sprintf("%s/v1/chat/completions", strings.TrimRight(host, "/"))
	out := make(chan string)

	go func() {
		defer close(out)

		var resp *http.Response
		// retry loop for model loading/unavailable
		for attempt := 0; attempt < c.cfg.MaxRetries; attempt++ {
			req, reqErr := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(jsonBody))
			if reqErr != nil {
				c.logger.Error("create chat stream request", zap.Error(reqErr))
				return
			}
			req.Header.Set("Content-Type", "application/json")
			req.Header.Set("Accept", "text/event-stream")

			r, err := c.httpClient.Do(req)
			if err != nil {
				if ctx.Err() != nil {
					// context canceled/deadline; just exit
					return
				}
				c.logger.Error("send chat stream request", zap.Error(err))
				return
			}

			if r.StatusCode == http.StatusServiceUnavailable {
				// backoff and retry
				io.Copy(io.Discard, r.Body)
				r.Body.Close()
				c.logger.Warn("LLM service unavailable, retrying", zap.Int("attempt", attempt+1))
				c.backoffSleep(attempt)
				continue
			}

			resp = r
			break
		}

		if resp == nil {
			c.logger.Error("no response received after retries for stream")
			return
		}
		defer resp.Body.Close()

		if resp.StatusCode != http.StatusOK {
			bodyBytes, _ := io.ReadAll(resp.Body)
			bodyString := string(bodyBytes)
			if strings.Contains(bodyString, "exceeds the available context size") {
				c.logger.Error("context window exceeded", zap.String("response", bodyString))
			} else {
				c.logger.Error("LLM server non-200 for stream", zap.String("status", resp.Status), zap.String("response", bodyString))
			}
			return
		}

		scanner := bufio.NewScanner(resp.Body)
		// Fence-aware stop: detect first complete ```python ... ``` block and stop thereafter
		var window string
		var opened bool
		openAbs := -1
		total := 0
		for scanner.Scan() {
			line := scanner.Text()
			if strings.HasPrefix(line, "data: ") {
				data := strings.TrimPrefix(line, "data: ")
				if data == "[DONE]" {
					break
				}
				var sr streamResponse
				if err := json.Unmarshal([]byte(data), &sr); err == nil {
					if len(sr.Choices) > 0 {
						chunk := sr.Choices[0].Delta.Content
						// Debug log each raw delta chunk as received from the LLM server
						//c.logger.Debug("llm stream delta", zap.String("delta", chunk))

						// Update detection window/state before emitting
						total += len(chunk)
						window += chunk
						// Cap window to a reasonable size for detection across chunk boundaries
						if len(window) > 2048 {
							window = window[len(window)-2048:]
						}
						if !opened {
							if idx := strings.Index(window, "```python"); idx != -1 {
								opened = true
								openAbs = total - (len(window) - idx)
							}
						}

						// Decide what to emit and whether to stop (on closing fence)
						toEmit := chunk
						shouldStop := false
						if opened {
							if idx := strings.LastIndex(window, "```"); idx != -1 {
								closeAbs := total - (len(window) - idx)
								if closeAbs > openAbs {
									// If closing fence falls within this chunk, trim to it
									chunkStartAbs := total - len(chunk)
									if closeAbs >= chunkStartAbs {
										cut := closeAbs - chunkStartAbs + 3 // include "```"
										if cut > 0 && cut <= len(chunk) {
											toEmit = chunk[:cut]
										}
									}
									shouldStop = true
								}
							}
						}

						if len(toEmit) > 0 {
							out <- toEmit
						}
						if shouldStop {
							break
						}
					}
				}
			}
		}
		if err := scanner.Err(); err != nil {
			c.logger.Error("read chat stream", zap.Error(err))
		}
	}()

	return out, nil
}

func (c *Client) backoffSleep(attempt int) {
    // Exponential backoff with configurable jitter and cap
    base := c.cfg.RetryDelaySeconds
    if base <= 0 {
        base = time.Second // config normalization should prevent this
    }
    d := base * time.Duration(1<<attempt)
    maxWait := c.cfg.LLMBackoffMaxSeconds
    if maxWait > 0 && d > maxWait {
        d = maxWait
    }
    jitterRatio := c.cfg.LLMBackoffJitterRatio
    if jitterRatio < 0 || jitterRatio > 1 {
        jitterRatio = 0.1
    }
    jitter := time.Duration(float64(d) * jitterRatio)
    time.Sleep(d - jitter + time.Duration(time.Now().UnixNano()%int64(2*jitter+1)))
}

// Embed generates an embedding vector for the provided document using the
// llama.cpp-compatible embeddings endpoint.
func (c *Client) Embed(ctx context.Context, host string, doc string) ([]float32, error) {
	reqBody := embeddingRequest{Content: doc}
	jsonBody, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("marshal embedding request: %w", err)
	}

	url := fmt.Sprintf("%s/v1/embeddings", strings.TrimRight(host, "/"))
	var resp *http.Response
	var lastErr error
	for attempt := 0; attempt < c.cfg.MaxRetries; attempt++ {
		req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(jsonBody))
		if err != nil {
			return nil, fmt.Errorf("create embedding request: %w", err)
		}
		req.Header.Set("Content-Type", "application/json")

		r, err := c.httpClient.Do(req)
		if err != nil {
			lastErr = err
			if ctx.Err() != nil {
				break
			}
			continue
		}

		if r.StatusCode == http.StatusServiceUnavailable {
			io.Copy(io.Discard, r.Body)
			r.Body.Close()
			c.logger.Warn("Embedding model loading, retrying")
			c.backoffSleep(attempt)
			continue
		}

		resp = r
		break
	}
	if resp == nil {
		return nil, fmt.Errorf("no response from embedding server: %w", lastErr)
	}
	defer resp.Body.Close()

	bodyBytes, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("read embedding response: %w", err)
	}
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("embedding server status %s: %s", resp.Status, string(bodyBytes))
	}

	var er embeddingResponse
	if err := json.Unmarshal(bodyBytes, &er); err != nil {
		return nil, fmt.Errorf("decode embedding response: %w", err)
	}
	if len(er) == 0 || len(er[0].Embedding) == 0 {
		return nil, fmt.Errorf("embedding response was empty")
	}
	return er[0].Embedding[0], nil
}
