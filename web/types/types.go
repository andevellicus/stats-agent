package types

// ChatMessage represents a single message in the chat
type ChatMessage struct {
	Role      string `json:"role"`
	Content   string `json:"content"`
	ID        string `json:"id"`
	SessionID string `json:"session_id"`
}