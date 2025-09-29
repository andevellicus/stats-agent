package types

import (
	"time"

	"github.com/google/uuid"
)

// AgentMessage represents a message in the format expected by the agent and LLM.
type AgentMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// ChatMessage represents a single message in the chat, stored in the DB.
type ChatMessage struct {
	Role      string `json:"role"`
	Content   string `json:"content"`
	ID        string `json:"id"`
	SessionID string `json:"session_id"`
}

// Session represents a chat session.
type Session struct {
	ID            uuid.UUID
	UserID        *uuid.UUID
	CreatedAt     time.Time
	LastActive    time.Time
	WorkspacePath string
	Title         string
	IsActive      bool
}
