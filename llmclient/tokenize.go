package llmclient

import (
    "bytes"
    "context"
    "encoding/json"
    "fmt"
    "io"
    "net/http"
    "strings"
)

// TokenizeRequest represents the payload for a /tokenize call
type TokenizeRequest struct {
    Content string `json:"content"`
}

// TokenizeResponse represents the response from a /tokenize call
type TokenizeResponse struct {
    Tokens []int `json:"tokens"`
}

// Tokenize requests tokenization for text at the given host and returns the token count.
func (c *Client) Tokenize(ctx context.Context, host string, text string) (int, error) {
    reqBody := TokenizeRequest{Content: text}
    jsonBody, err := json.Marshal(reqBody)
    if err != nil {
        return 0, fmt.Errorf("marshal tokenize request: %w", err)
    }

    url := fmt.Sprintf("%s/tokenize", strings.TrimRight(host, "/"))
    var resp *http.Response
    var lastErr error
    for attempt := 0; attempt < c.cfg.MaxRetries; attempt++ {
        req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(jsonBody))
        if err != nil {
            return 0, fmt.Errorf("create tokenize request: %w", err)
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
            c.backoffSleep(attempt)
            continue
        }

        resp = r
        break
    }

    if resp == nil {
        return 0, fmt.Errorf("no response from tokenize server: %w", lastErr)
    }
    defer resp.Body.Close()

    if resp.StatusCode != http.StatusOK {
        bodyBytes, _ := io.ReadAll(resp.Body)
        return 0, fmt.Errorf("tokenize server status %s: %s", resp.Status, strings.TrimSpace(string(bodyBytes)))
    }

    var tr TokenizeResponse
    if err := json.NewDecoder(resp.Body).Decode(&tr); err != nil {
        return 0, fmt.Errorf("decode tokenize response: %w", err)
    }
    return len(tr.Tokens), nil
}

