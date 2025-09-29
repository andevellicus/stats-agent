package database

import (
	"context"
	"database/sql"
	"errors" // Import the errors package
	"fmt"
	"log"
	"path/filepath"
	"time"

	"stats-agent/web/types"

	"github.com/google/uuid" // Import the pgtype package
	_ "github.com/jackc/pgx/v5/stdlib"
)

type PostgresStore struct {
	DB *sql.DB
}

func NewPostgresStore(connStr string) (*PostgresStore, error) {
	db, err := sql.Open("pgx", connStr)
	if err != nil {
		return nil, err
	}
	if err := db.Ping(); err != nil {
		return nil, err
	}
	log.Println("Successfully connected to the database")
	return &PostgresStore{DB: db}, nil
}

// EnsureSchema creates the required tables if they do not already exist.
func (s *PostgresStore) EnsureSchema(ctx context.Context) error {
	stmts := []string{
		`CREATE TABLE IF NOT EXISTS users (
            id UUID PRIMARY KEY,
            email TEXT UNIQUE,
            created_at TIMESTAMPTZ DEFAULT NOW()
        )`,
		`CREATE TABLE IF NOT EXISTS sessions (
            id UUID PRIMARY KEY,
            user_id UUID REFERENCES users(id) ON DELETE CASCADE,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            last_active TIMESTAMPTZ DEFAULT NOW(),
            workspace_path TEXT NOT NULL,
            title TEXT DEFAULT '',
            is_active BOOLEAN DEFAULT TRUE,
            rendered_files TEXT[] DEFAULT '{}'::TEXT[]
        )`,
		`CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON sessions(user_id)`,
		`CREATE INDEX IF NOT EXISTS idx_sessions_last_active ON sessions(last_active DESC)`,
		`CREATE TABLE IF NOT EXISTS messages (
            id UUID PRIMARY KEY,
            session_id UUID REFERENCES sessions(id) ON DELETE CASCADE,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            metadata JSONB DEFAULT '{}'::jsonb
        )`,
		`CREATE INDEX IF NOT EXISTS idx_messages_session_created_at ON messages(session_id, created_at)`,
		`CREATE TABLE IF NOT EXISTS session_state (
            session_id UUID PRIMARY KEY REFERENCES sessions(id) ON DELETE CASCADE,
            python_executor TEXT,
            variables JSONB,
            last_checkpoint TIMESTAMPTZ,
            code_blocks TEXT[]
        )`,
	}

	for _, stmt := range stmts {
		if _, err := s.DB.ExecContext(ctx, stmt); err != nil {
			return fmt.Errorf("failed to execute schema statement: %w", err)
		}
	}

	alterStmt := `ALTER TABLE sessions ALTER COLUMN user_id DROP NOT NULL;`
	if _, err := s.DB.ExecContext(ctx, alterStmt); err != nil {
		log.Printf("INFO: Could not drop NOT NULL constraint on sessions.user_id (it might already be altered): %v", err)
	}

	return nil
}

func (s *PostgresStore) CreateSession(ctx context.Context, userID *uuid.UUID) (uuid.UUID, error) {
	sessionID := uuid.New()
	workspacePath := filepath.Join("workspaces", sessionID.String())
	now := time.Now()
	initialTitle := fmt.Sprintf("Chat from %s", now.Format("January 2, 2006"))

	var userIDValue sql.NullString
	if userID != nil {
		userIDValue = sql.NullString{String: userID.String(), Valid: true}
	} else {
		userIDValue = sql.NullString{Valid: false}
	}

	query := `
        INSERT INTO sessions (id, user_id, created_at, last_active, workspace_path, title, is_active)
        VALUES ($1, $2, $3, $4, $5, $6, $7)
    `
	_, err := s.DB.ExecContext(ctx, query, sessionID, userIDValue, now, now, workspacePath, initialTitle, true)
	if err != nil {
		return uuid.Nil, fmt.Errorf("failed to create session: %w", err)
	}
	return sessionID, nil
}

func (s *PostgresStore) GetSessions(ctx context.Context, userID *uuid.UUID) ([]types.Session, error) {
	query := `
		SELECT id, user_id, created_at, last_active, workspace_path, title, is_active
		FROM sessions
		WHERE is_active = true
		ORDER BY last_active DESC
	`
	rows, err := s.DB.QueryContext(ctx, query)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var sessions []types.Session
	for rows.Next() {
		var sess types.Session
		var userID sql.NullString
		if err := rows.Scan(&sess.ID, &userID, &sess.CreatedAt, &sess.LastActive, &sess.WorkspacePath, &sess.Title, &sess.IsActive); err != nil {
			return nil, err
		}
		if userID.Valid {
			parsedUUID, err := uuid.Parse(userID.String)
			if err == nil {
				sess.UserID = &parsedUUID
			}
		}
		sessions = append(sessions, sess)
	}
	return sessions, nil
}

func (s *PostgresStore) CreateMessage(ctx context.Context, msg types.ChatMessage) error {
	query := `
		INSERT INTO messages (id, session_id, role, content, created_at)
		VALUES ($1, $2, $3, $4, $5)
	`
	tx, err := s.DB.BeginTx(ctx, nil)
	if err != nil {
		return err
	}
	defer tx.Rollback()

	messageUUID, err := uuid.Parse(msg.ID)
	if err != nil {
		return fmt.Errorf("invalid message ID: %w", err)
	}
	sessionUUID, err := uuid.Parse(msg.SessionID)
	if err != nil {
		return fmt.Errorf("invalid session ID in message: %w", err)
	}

	_, err = tx.ExecContext(ctx, query, messageUUID, sessionUUID, msg.Role, msg.Content, time.Now())
	if err != nil {
		return err
	}

	_, err = tx.ExecContext(ctx, `UPDATE sessions SET last_active = $1 WHERE id = $2`, time.Now(), sessionUUID)
	if err != nil {
		return err
	}

	return tx.Commit()
}

func (s *PostgresStore) GetMessagesBySession(ctx context.Context, sessionID uuid.UUID) ([]types.ChatMessage, error) {
	query := `
		SELECT id, session_id, role, content FROM messages
		WHERE session_id = $1 ORDER BY created_at ASC
	`
	rows, err := s.DB.QueryContext(ctx, query, sessionID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var messages []types.ChatMessage
	for rows.Next() {
		var msg types.ChatMessage
		var sessionUUID uuid.UUID
		if err := rows.Scan(&msg.ID, &sessionUUID, &msg.Role, &msg.Content); err != nil {
			return nil, err
		}
		msg.SessionID = sessionUUID.String()
		messages = append(messages, msg)
	}
	return messages, nil
}

func (s *PostgresStore) GetRenderedFiles(ctx context.Context, sessionID uuid.UUID) (map[string]bool, error) {
	var files []string
	query := `SELECT rendered_files FROM sessions WHERE id = $1`
	// Use pgtype.TextArray to scan the TEXT[] column
	err := s.DB.QueryRowContext(ctx, query, sessionID).Scan(&files)
	if err != nil {
		if errors.Is(err, sql.ErrNoRows) {
			return make(map[string]bool), nil // No session found, return empty map
		}
		return nil, err
	}

	rendered := make(map[string]bool)
	for _, f := range files {
		rendered[f] = true
	}
	return rendered, nil
}

func (s *PostgresStore) AddRenderedFile(ctx context.Context, sessionID uuid.UUID, filename string) error {
	query := `
		UPDATE sessions
		SET rendered_files = array_append(rendered_files, $1)
		WHERE id = $2
	`
	_, err := s.DB.ExecContext(ctx, query, filename, sessionID)
	return err
}
