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
	// Enable pgvector extension first
	if _, err := s.DB.ExecContext(ctx, `CREATE EXTENSION IF NOT EXISTS vector`); err != nil {
		return fmt.Errorf("failed to enable pgvector extension: %w", err)
	}

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
            mode TEXT DEFAULT 'dataset'
        )`,
		`CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON sessions(user_id)`,
		`CREATE INDEX IF NOT EXISTS idx_sessions_last_active ON sessions(last_active DESC)`,
		`CREATE TABLE IF NOT EXISTS messages (
            id UUID PRIMARY KEY,
            session_id UUID REFERENCES sessions(id) ON DELETE CASCADE,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            rendered TEXT NOT NULL,
            content_hash TEXT NOT NULL DEFAULT '',
            created_at TIMESTAMPTZ DEFAULT NOW()
        )`,
		`CREATE INDEX IF NOT EXISTS idx_messages_session_created_at ON messages(session_id, created_at)`,
		`CREATE INDEX IF NOT EXISTS idx_messages_content_hash ON messages(content_hash)`,
		`CREATE TABLE IF NOT EXISTS rag_documents (
            id UUID PRIMARY KEY,
            content TEXT NOT NULL,
            content_hash TEXT,
            metadata JSONB DEFAULT '{}'::jsonb,
            created_at TIMESTAMPTZ DEFAULT NOW()
        )`,
		`CREATE TABLE IF NOT EXISTS rag_embeddings (
            id UUID PRIMARY KEY,
            document_id UUID NOT NULL REFERENCES rag_documents(id) ON DELETE CASCADE,
            window_index INT NOT NULL,
            window_start INT NOT NULL,
            window_end INT NOT NULL,
            window_text TEXT NOT NULL,
            embedding vector(1024) NOT NULL,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            UNIQUE(document_id, window_index)
        )`,
		`CREATE TABLE IF NOT EXISTS files (
            id UUID PRIMARY KEY,
            session_id UUID NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
            filename TEXT NOT NULL,
            file_path TEXT NOT NULL,
            file_type TEXT,
            file_size BIGINT,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            message_id UUID REFERENCES messages(id) ON DELETE SET NULL,
            CONSTRAINT unique_session_filename UNIQUE(session_id, filename)
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

	// Migrate existing rag_documents to new schema
	// Check if old schema exists (has document_id column)
	var hasDocumentID bool
	err := s.DB.QueryRowContext(ctx, `
		SELECT EXISTS (
			SELECT 1 FROM information_schema.columns
			WHERE table_name = 'rag_documents' AND column_name = 'document_id'
		)
	`).Scan(&hasDocumentID)
	if err != nil {
		return fmt.Errorf("failed to check for old schema: %w", err)
	}

	if hasDocumentID {
		// Old schema exists - migrate data
		// 1. Create temporary backup
		if _, err := s.DB.ExecContext(ctx, `
			CREATE TABLE IF NOT EXISTS rag_documents_old AS
			SELECT * FROM rag_documents
		`); err != nil {
			return fmt.Errorf("failed to backup old rag_documents: %w", err)
		}

		// 2. Drop old table
		if _, err := s.DB.ExecContext(ctx, `DROP TABLE IF EXISTS rag_documents CASCADE`); err != nil {
			return fmt.Errorf("failed to drop old rag_documents: %w", err)
		}

		// 3. Recreate with new schema
		if _, err := s.DB.ExecContext(ctx, `
			CREATE TABLE rag_documents (
				id UUID PRIMARY KEY,
				content TEXT NOT NULL,
				content_hash TEXT,
				metadata JSONB DEFAULT '{}'::jsonb,
				created_at TIMESTAMPTZ DEFAULT NOW()
			)
		`); err != nil {
			return fmt.Errorf("failed to recreate rag_documents: %w", err)
		}

		// 4. Migrate data (use document_id as new id)
		if _, err := s.DB.ExecContext(ctx, `
			INSERT INTO rag_documents (id, content, content_hash, metadata, created_at)
			SELECT DISTINCT ON (document_id)
				document_id, content, content_hash, metadata, created_at
			FROM rag_documents_old
		`); err != nil {
			return fmt.Errorf("failed to migrate documents: %w", err)
		}

		// 5. Migrate embeddings if they exist
		if _, err := s.DB.ExecContext(ctx, `
			INSERT INTO rag_embeddings (id, document_id, window_index, window_start, window_end, window_text, embedding, created_at)
			SELECT
				id,
				document_id,
				0,
				0,
				COALESCE(LENGTH(embedding_content), LENGTH(content)),
				COALESCE(embedding_content, content),
				embedding,
				created_at
			FROM rag_documents_old
			WHERE embedding IS NOT NULL
		`); err != nil {
			return fmt.Errorf("failed to migrate embeddings: %w", err)
		}

		// 6. Drop backup
		if _, err := s.DB.ExecContext(ctx, `DROP TABLE IF EXISTS rag_documents_old`); err != nil {
			// Non-fatal - just log
		}
	}

	indexStmts := []string{
		`CREATE INDEX IF NOT EXISTS idx_messages_role ON messages(role)`,
		`CREATE INDEX IF NOT EXISTS idx_sessions_user_active ON sessions(user_id, is_active, last_active DESC)`,
		`CREATE INDEX IF NOT EXISTS idx_sessions_mode ON sessions(mode)`,
		`CREATE INDEX IF NOT EXISTS idx_rag_documents_created_at ON rag_documents(created_at)`,
		`CREATE INDEX IF NOT EXISTS idx_rag_documents_content_hash ON rag_documents(content_hash)`,
		`CREATE UNIQUE INDEX IF NOT EXISTS idx_rag_documents_session_role_hash ON rag_documents (content_hash, COALESCE(metadata ->> 'session_id', ''), COALESCE(metadata ->> 'role', '')) WHERE content_hash IS NOT NULL`,
		`CREATE INDEX IF NOT EXISTS idx_rag_documents_metadata_dataset ON rag_documents ((metadata ->> 'dataset'))`,
		`CREATE INDEX IF NOT EXISTS idx_rag_documents_metadata_primary_test ON rag_documents ((metadata ->> 'primary_test'))`,
		`CREATE INDEX IF NOT EXISTS idx_rag_documents_metadata_role ON rag_documents ((metadata ->> 'role'))`,
		`CREATE INDEX IF NOT EXISTS idx_rag_documents_metadata_session_id ON rag_documents ((metadata ->> 'session_id'))`,
		`CREATE INDEX IF NOT EXISTS idx_rag_embeddings_document_id ON rag_embeddings(document_id)`,
		`CREATE INDEX IF NOT EXISTS idx_rag_embeddings_vector_cosine ON rag_embeddings USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)`,
		`CREATE INDEX IF NOT EXISTS idx_files_session_id ON files(session_id)`,
		`CREATE INDEX IF NOT EXISTS idx_files_message_id ON files(message_id)`,
		`CREATE INDEX IF NOT EXISTS idx_files_created_at ON files(created_at)`,
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
	return s.CreateSessionWithMode(ctx, userID, "dataset")
}

func (s *PostgresStore) CreateSessionWithMode(ctx context.Context, userID *uuid.UUID, mode string) (uuid.UUID, error) {
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

	// Validate mode
	if mode != "dataset" && mode != "document" {
		mode = "dataset" // Default to dataset mode
	}

	query := `
        INSERT INTO sessions (id, user_id, created_at, last_active, workspace_path, title, is_active, mode)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
    `
	_, err := s.DB.ExecContext(ctx, query, sessionID, userIDValue, now, now, workspacePath, initialTitle, true, mode)
	if err != nil {
		return uuid.Nil, fmt.Errorf("failed to create session: %w", err)
	}
	return sessionID, nil
}

func (s *PostgresStore) GetSessionByID(ctx context.Context, sessionID uuid.UUID) (types.Session, error) {
	query := `
		SELECT id, user_id, created_at, last_active, workspace_path, title, is_active, COALESCE(mode, 'dataset') as mode
		FROM sessions
		WHERE id = $1
	`
	row := s.DB.QueryRowContext(ctx, query, sessionID)

	var session types.Session
	var userID sql.NullString
	if err := row.Scan(&session.ID, &userID, &session.CreatedAt, &session.LastActive, &session.WorkspacePath, &session.Title, &session.IsActive, &session.Mode); err != nil {
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

func (s *PostgresStore) UpdateSessionMode(ctx context.Context, sessionID uuid.UUID, mode string) error {
	// Validate mode
	if mode != "dataset" && mode != "document" {
		return fmt.Errorf("invalid mode: must be 'dataset' or 'document'")
	}

	query := `UPDATE sessions SET mode = $1 WHERE id = $2`
	_, err := s.DB.ExecContext(ctx, query, mode, sessionID)
	if err != nil {
		return fmt.Errorf("failed to update session mode: %w", err)
	}
	return nil
}

// UpdateSessionUser sets the user_id for a session if it is currently NULL.
// Returns nil if the update succeeds (including when it was already set by another request).
// UpdateSessionUser removed - legacy claiming disabled

func (s *PostgresStore) GetSessions(ctx context.Context, userID *uuid.UUID) ([]types.Session, error) {
	var query string
	var rows *sql.Rows
	var err error

	if userID != nil {
		query = `
			SELECT id, user_id, created_at, last_active, workspace_path, title, is_active, COALESCE(mode, 'dataset') as mode
			FROM sessions
			WHERE is_active = true AND user_id = $1
			ORDER BY last_active DESC
		`
		rows, err = s.DB.QueryContext(ctx, query, userID)
	} else {
		query = `
			SELECT id, user_id, created_at, last_active, workspace_path, title, is_active, COALESCE(mode, 'dataset') as mode
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
		if err := rows.Scan(&session.ID, &userID, &session.CreatedAt, &session.LastActive, &session.WorkspacePath, &session.Title, &session.IsActive, &session.Mode); err != nil {
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
		INSERT INTO messages (id, session_id, role, content, rendered, content_hash, created_at)
		VALUES ($1, $2, $3, $4, $5, $6, $7)
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

	_, err = tx.ExecContext(ctx, query, messageUUID, sessionUUID, msg.Role, msg.Content, msg.Rendered, msg.ContentHash, time.Now())
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

// AppendToMessageRendered appends additional HTML to an existing message's rendered field.
func (s *PostgresStore) AppendToMessageRendered(ctx context.Context, messageID string, extraHTML string) error {
	if extraHTML == "" {
		return nil
	}

	msgUUID, err := uuid.Parse(messageID)
	if err != nil {
		return fmt.Errorf("invalid message ID: %w", err)
	}

	query := `
		UPDATE messages
		SET rendered = COALESCE(rendered, '') || $1
		WHERE id = $2
	`

	if _, err := s.DB.ExecContext(ctx, query, extraHTML, msgUUID); err != nil {
		return fmt.Errorf("append to message rendered: %w", err)
	}

	return nil
}

func (s *PostgresStore) GetMessagesBySession(ctx context.Context, sessionID uuid.UUID) ([]types.ChatMessage, error) {
	query := `
		SELECT id, session_id, role, content, rendered, content_hash FROM messages
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
		if err := rows.Scan(&msg.ID, &sessionUUID, &msg.Role, &msg.Content, &msg.Rendered, &msg.ContentHash); err != nil {
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

// Note: legacy rendered_files helpers removed; feature no longer supported.

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

func (s *PostgresStore) DeleteUser(ctx context.Context, userID uuid.UUID) error {
	// Get all sessions for this user
	sessions, err := s.GetSessions(ctx, &userID)
	if err != nil {
		return fmt.Errorf("failed to get user sessions: %w", err)
	}

	// Delete RAG documents for each session
	// Continue even if some deletions fail to ensure we clean up as much as possible
	for _, session := range sessions {
		if _, err := s.DeleteRAGDocumentsBySession(ctx, session.ID); err != nil {
			// Log-worthy but don't fail the entire deletion
			// Caller can log if needed, we just skip and continue
		}
	}

	// Delete user (CASCADE deletes sessions → messages)
	query := `DELETE FROM users WHERE id = $1`
	result, err := s.DB.ExecContext(ctx, query, userID)
	if err != nil {
		return fmt.Errorf("failed to delete user: %w", err)
	}

	rowsAffected, err := result.RowsAffected()
	if err != nil {
		return fmt.Errorf("failed to get rows affected: %w", err)
	}

	if rowsAffected == 0 {
		return errors.New("user not found")
	}

	return nil
}
