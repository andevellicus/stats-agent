package database

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"path/filepath"
	"time"

	"stats-agent/web/types"

	"github.com/google/uuid"
	_ "github.com/jackc/pgx/v5/stdlib"
	"github.com/lib/pq"
)

type PostgresStore struct {
	DB *sql.DB
}

func NewPostgresStore(connStr string) (*PostgresStore, error) {
	db, err := sql.Open("pgx", connStr)
	if err != nil {
		return nil, fmt.Errorf("failed to open database connection: %w", err)
	}
	if err := db.Ping(); err != nil {
		return nil, fmt.Errorf("failed to ping database: %w", err)
	}
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
            rendered TEXT NOT NULL, 
            created_at TIMESTAMPTZ DEFAULT NOW(),
            metadata JSONB DEFAULT '{}'::jsonb
        )`,
		`CREATE INDEX IF NOT EXISTS idx_messages_session_created_at ON messages(session_id, created_at)`,
		`CREATE TABLE IF NOT EXISTS rag_documents (
            id UUID PRIMARY KEY,
            document_id UUID NOT NULL,
            content TEXT NOT NULL,
            metadata JSONB DEFAULT '{}'::jsonb,
            content_hash TEXT,
            created_at TIMESTAMPTZ DEFAULT NOW()
        )`,
	}

	for _, stmt := range stmts {
		if _, err := s.DB.ExecContext(ctx, stmt); err != nil {
			return fmt.Errorf("failed to execute schema statement: %w", err)
		}
	}

	// Attempt to drop NOT NULL constraint (may already be altered in existing databases)
	if _, err := s.DB.ExecContext(ctx, `ALTER TABLE sessions ALTER COLUMN user_id DROP NOT NULL;`); err != nil {
		// Ignore error - constraint may already be dropped or not exist
		// This is a schema migration compatibility step, not a critical operation
	}

	if _, err := s.DB.ExecContext(ctx, `ALTER TABLE rag_documents ADD COLUMN IF NOT EXISTS content_hash TEXT`); err != nil {
		return fmt.Errorf("failed to add content_hash column: %w", err)
	}

	indexStmts := []string{
		`CREATE INDEX IF NOT EXISTS idx_messages_role ON messages(role)`,
		`CREATE INDEX IF NOT EXISTS idx_sessions_user_active ON sessions(user_id, is_active, last_active DESC)`,
		`CREATE INDEX IF NOT EXISTS idx_rag_documents_created_at ON rag_documents(created_at)`,
		`CREATE UNIQUE INDEX IF NOT EXISTS idx_rag_documents_document_id ON rag_documents(document_id)`,
		`CREATE INDEX IF NOT EXISTS idx_rag_documents_content_hash ON rag_documents(content_hash)`,
		`CREATE UNIQUE INDEX IF NOT EXISTS idx_rag_documents_session_role_hash ON rag_documents (content_hash, COALESCE(metadata ->> 'session_id', ''), COALESCE(metadata ->> 'role', '')) WHERE content_hash IS NOT NULL`,
	}

	for _, stmt := range indexStmts {
		if _, err := s.DB.ExecContext(ctx, stmt); err != nil {
			return fmt.Errorf("failed to ensure rag_documents index: %w", err)
		}
	}

	return nil
}

func (s *PostgresStore) CreateUser(ctx context.Context) (uuid.UUID, error) {
	userID := uuid.New()
	query := `INSERT INTO users (id, created_at) VALUES ($1, $2)`
	_, err := s.DB.ExecContext(ctx, query, userID, time.Now())
	if err != nil {
		return uuid.Nil, fmt.Errorf("failed to create user: %w", err)
	}
	return userID, nil
}

func (s *PostgresStore) GetUserByID(ctx context.Context, userID uuid.UUID) error {
	query := `SELECT id FROM users WHERE id = $1`
	var id uuid.UUID
	err := s.DB.QueryRowContext(ctx, query, userID).Scan(&id)
	return err
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

func (s *PostgresStore) GetSessionByID(ctx context.Context, sessionID uuid.UUID) (types.Session, error) {
	query := `
		SELECT id, user_id, created_at, last_active, workspace_path, title, is_active
		FROM sessions
		WHERE id = $1
	`
	row := s.DB.QueryRowContext(ctx, query, sessionID)

	var session types.Session
	var userID sql.NullString
	if err := row.Scan(&session.ID, &userID, &session.CreatedAt, &session.LastActive, &session.WorkspacePath, &session.Title, &session.IsActive); err != nil {
		if errors.Is(err, sql.ErrNoRows) {
			return types.Session{}, fmt.Errorf("session not found: %w", err)
		}
		return types.Session{}, fmt.Errorf("failed to scan session: %w", err)
	}

	if userID.Valid {
		parsedUUID, err := uuid.Parse(userID.String)
		if err != nil {
			// UUID parsing failed - this indicates corrupted data in database
			// Log the error but continue with nil UserID rather than failing the entire query
			return types.Session{}, fmt.Errorf("failed to parse user ID from database: %w", err)
		}
		session.UserID = &parsedUUID
	}
	return session, nil
}

func (s *PostgresStore) UpdateSessionTitle(ctx context.Context, sessionID uuid.UUID, title string) error {
	query := `UPDATE sessions SET title = $1 WHERE id = $2`
	_, err := s.DB.ExecContext(ctx, query, title, sessionID)
	if err != nil {
		return fmt.Errorf("failed to update session title: %w", err)
	}
	return nil
}

func (s *PostgresStore) GetSessions(ctx context.Context, userID *uuid.UUID) ([]types.Session, error) {
	var query string
	var rows *sql.Rows
	var err error

	if userID != nil {
		query = `
			SELECT id, user_id, created_at, last_active, workspace_path, title, is_active
			FROM sessions
			WHERE is_active = true AND user_id = $1
			ORDER BY last_active DESC
		`
		rows, err = s.DB.QueryContext(ctx, query, userID)
	} else {
		query = `
			SELECT id, user_id, created_at, last_active, workspace_path, title, is_active
			FROM sessions
			WHERE is_active = true
			ORDER BY last_active DESC
		`
		rows, err = s.DB.QueryContext(ctx, query)
	}

	if err != nil {
		return nil, fmt.Errorf("failed to query sessions: %w", err)
	}
	defer rows.Close()

	var sessions []types.Session
	for rows.Next() {
		var session types.Session
		var userID sql.NullString
		if err := rows.Scan(&session.ID, &userID, &session.CreatedAt, &session.LastActive, &session.WorkspacePath, &session.Title, &session.IsActive); err != nil {
			return nil, fmt.Errorf("failed to scan session row: %w", err)
		}
		if userID.Valid {
			parsedUUID, err := uuid.Parse(userID.String)
			if err != nil {
				// UUID parsing failed - this indicates corrupted data in database
				return nil, fmt.Errorf("failed to parse user ID from session %s: %w", session.ID, err)
			}
			session.UserID = &parsedUUID
		}
		sessions = append(sessions, session)
	}

	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("error iterating session rows: %w", err)
	}

	return sessions, nil
}

func (s *PostgresStore) CreateMessage(ctx context.Context, msg types.ChatMessage) error {
	query := `
		INSERT INTO messages (id, session_id, role, content, rendered, created_at)
		VALUES ($1, $2, $3, $4, $5, $6)
	`
	tx, err := s.DB.BeginTx(ctx, nil)
	if err != nil {
		return fmt.Errorf("failed to begin transaction: %w", err)
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

	_, err = tx.ExecContext(ctx, query, messageUUID, sessionUUID, msg.Role, msg.Content, msg.Rendered, time.Now())
	if err != nil {
		return fmt.Errorf("failed to insert message: %w", err)
	}

	_, err = tx.ExecContext(ctx, `UPDATE sessions SET last_active = $1 WHERE id = $2`, time.Now(), sessionUUID)
	if err != nil {
		return fmt.Errorf("failed to update session last_active: %w", err)
	}

	if err := tx.Commit(); err != nil {
		return fmt.Errorf("failed to commit transaction: %w", err)
	}

	return nil
}

func (s *PostgresStore) GetMessagesBySession(ctx context.Context, sessionID uuid.UUID) ([]types.ChatMessage, error) {
	query := `
		SELECT id, session_id, role, content, rendered FROM messages
		WHERE session_id = $1 ORDER BY created_at ASC
	`
	rows, err := s.DB.QueryContext(ctx, query, sessionID)
	if err != nil {
		return nil, fmt.Errorf("failed to query messages: %w", err)
	}
	defer rows.Close()

	var messages []types.ChatMessage
	for rows.Next() {
		var msg types.ChatMessage
		var sessionUUID uuid.UUID
		if err := rows.Scan(&msg.ID, &sessionUUID, &msg.Role, &msg.Content, &msg.Rendered); err != nil {
			return nil, fmt.Errorf("failed to scan message row: %w", err)
		}
		msg.SessionID = sessionUUID.String()
		messages = append(messages, msg)
	}

	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("error iterating message rows: %w", err)
	}

	return messages, nil
}

func (s *PostgresStore) GetRenderedFiles(ctx context.Context, sessionID uuid.UUID) (map[string]bool, error) {
	var files pq.StringArray
	query := `SELECT rendered_files FROM sessions WHERE id = $1`

	err := s.DB.QueryRowContext(ctx, query, sessionID).Scan(&files)
	if err != nil {
		if errors.Is(err, sql.ErrNoRows) {
			return make(map[string]bool), nil // No session found, return empty map
		}
		return nil, fmt.Errorf("failed to get rendered files: %w", err)
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
	if err != nil {
		return fmt.Errorf("failed to add rendered file: %w", err)
	}
	return nil
}

func (s *PostgresStore) GetStaleSessions(ctx context.Context, lastActiveBefore time.Time) ([]uuid.UUID, error) {
	query := `
		SELECT id FROM sessions
		WHERE last_active < $1
		ORDER BY last_active ASC
	`
	rows, err := s.DB.QueryContext(ctx, query, lastActiveBefore)
	if err != nil {
		return nil, fmt.Errorf("failed to query stale sessions: %w", err)
	}
	defer rows.Close()

	var sessionIDs []uuid.UUID
	for rows.Next() {
		var id uuid.UUID
		if err := rows.Scan(&id); err != nil {
			return nil, fmt.Errorf("failed to scan session ID: %w", err)
		}
		sessionIDs = append(sessionIDs, id)
	}

	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("error iterating stale sessions: %w", err)
	}

	return sessionIDs, nil
}

func (s *PostgresStore) DeleteSession(ctx context.Context, sessionID uuid.UUID) error {
	query := `DELETE FROM sessions WHERE id = $1`
	result, err := s.DB.ExecContext(ctx, query, sessionID)
	if err != nil {
		return fmt.Errorf("failed to delete session: %w", err)
	}

	rowsAffected, err := result.RowsAffected()
	if err != nil {
		return fmt.Errorf("failed to get rows affected: %w", err)
	}

	if rowsAffected == 0 {
		return errors.New("session not found")
	}

	return nil
}
