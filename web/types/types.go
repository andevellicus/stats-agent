package types

import (
	"time"

	"github.com/google/uuid"
)

// ChatMessage represents a single message in the chat
type ChatMessage struct {
	Role      string `json:"role"`
	Content   string `json:"content"`
	ID        string `json:"id"`
	SessionID string `json:"session_id"`
}

// Session represents a chat session
type Session struct {
	ID            uuid.UUID
	UserID        *uuid.UUID // Pointer to allow for NULL user_id
	CreatedAt     time.Time
	LastActive    time.Time
	WorkspacePath string
	Title         string
	IsActive      bool
}
