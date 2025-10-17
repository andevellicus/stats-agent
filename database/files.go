package database

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"time"

	"github.com/google/uuid"
)

// FileRecord represents a file tracked in the database
type FileRecord struct {
	ID        uuid.UUID
	SessionID uuid.UUID
	Filename  string
	FilePath  string
	FileType  string
	FileSize  int64
	CreatedAt time.Time
	MessageID *uuid.UUID
}

// CreateFile inserts a new file record. If a file with the same session_id and filename
// already exists, it returns the existing file (idempotent operation).
func (s *PostgresStore) CreateFile(ctx context.Context, file FileRecord) (FileRecord, error) {
	// Use ON CONFLICT to handle race conditions - if file already exists, return it
	query := `
		INSERT INTO files (id, session_id, filename, file_path, file_type, file_size, created_at, message_id)
		VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
		ON CONFLICT (session_id, filename) DO UPDATE SET id = files.id
		RETURNING id, session_id, filename, file_path, file_type, file_size, created_at, message_id
	`

	var result FileRecord
	var messageID sql.NullString

	err := s.DB.QueryRowContext(ctx, query,
		file.ID,
		file.SessionID,
		file.Filename,
		file.FilePath,
		file.FileType,
		file.FileSize,
		file.CreatedAt,
		uuidToNullString(file.MessageID),
	).Scan(
		&result.ID,
		&result.SessionID,
		&result.Filename,
		&result.FilePath,
		&result.FileType,
		&result.FileSize,
		&result.CreatedAt,
		&messageID,
	)

	if err != nil {
		return FileRecord{}, fmt.Errorf("failed to create file record: %w", err)
	}

	result.MessageID = nullStringToUUID(messageID)
	return result, nil
}

// GetFilesBySession returns all files for a given session, ordered by creation time
func (s *PostgresStore) GetFilesBySession(ctx context.Context, sessionID uuid.UUID) ([]FileRecord, error) {
	query := `
		SELECT id, session_id, filename, file_path, file_type, file_size, created_at, message_id
		FROM files
		WHERE session_id = $1
		ORDER BY created_at ASC
	`

	rows, err := s.DB.QueryContext(ctx, query, sessionID)
	if err != nil {
		return nil, fmt.Errorf("failed to query files: %w", err)
	}
	defer rows.Close()

	var files []FileRecord
	for rows.Next() {
		var file FileRecord
		var messageID sql.NullString

		if err := rows.Scan(
			&file.ID,
			&file.SessionID,
			&file.Filename,
			&file.FilePath,
			&file.FileType,
			&file.FileSize,
			&file.CreatedAt,
			&messageID,
		); err != nil {
			return nil, fmt.Errorf("failed to scan file row: %w", err)
		}

		file.MessageID = nullStringToUUID(messageID)
		files = append(files, file)
	}

	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("error iterating file rows: %w", err)
	}

	return files, nil
}

// GetNewFilesBySession returns files created after the specified time for a session.
// This is used to detect new files since the last check.
func (s *PostgresStore) GetNewFilesBySession(ctx context.Context, sessionID uuid.UUID, after time.Time) ([]FileRecord, error) {
	query := `
		SELECT id, session_id, filename, file_path, file_type, file_size, created_at, message_id
		FROM files
		WHERE session_id = $1 AND created_at > $2
		ORDER BY created_at ASC
	`

	rows, err := s.DB.QueryContext(ctx, query, sessionID, after)
	if err != nil {
		return nil, fmt.Errorf("failed to query new files: %w", err)
	}
	defer rows.Close()

	var files []FileRecord
	for rows.Next() {
		var file FileRecord
		var messageID sql.NullString

		if err := rows.Scan(
			&file.ID,
			&file.SessionID,
			&file.Filename,
			&file.FilePath,
			&file.FileType,
			&file.FileSize,
			&file.CreatedAt,
			&messageID,
		); err != nil {
			return nil, fmt.Errorf("failed to scan file row: %w", err)
		}

		file.MessageID = nullStringToUUID(messageID)
		files = append(files, file)
	}

	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("error iterating file rows: %w", err)
	}

	return files, nil
}

// GetFileBySessionAndName retrieves a specific file by session ID and filename
func (s *PostgresStore) GetFileBySessionAndName(ctx context.Context, sessionID uuid.UUID, filename string) (FileRecord, error) {
	query := `
		SELECT id, session_id, filename, file_path, file_type, file_size, created_at, message_id
		FROM files
		WHERE session_id = $1 AND filename = $2
	`

	var file FileRecord
	var messageID sql.NullString

	err := s.DB.QueryRowContext(ctx, query, sessionID, filename).Scan(
		&file.ID,
		&file.SessionID,
		&file.Filename,
		&file.FilePath,
		&file.FileType,
		&file.FileSize,
		&file.CreatedAt,
		&messageID,
	)

	if err != nil {
		if errors.Is(err, sql.ErrNoRows) {
			return FileRecord{}, fmt.Errorf("file not found: %w", err)
		}
		return FileRecord{}, fmt.Errorf("failed to get file: %w", err)
	}

	file.MessageID = nullStringToUUID(messageID)
	return file, nil
}

// GetTrackedFilenames returns a set of all tracked filenames for a session
// This is used for efficient membership checking when scanning for new files
func (s *PostgresStore) GetTrackedFilenames(ctx context.Context, sessionID uuid.UUID) (map[string]bool, error) {
	query := `SELECT filename FROM files WHERE session_id = $1`

	rows, err := s.DB.QueryContext(ctx, query, sessionID)
	if err != nil {
		return nil, fmt.Errorf("failed to query tracked filenames: %w", err)
	}
	defer rows.Close()

	filenames := make(map[string]bool)
	for rows.Next() {
		var filename string
		if err := rows.Scan(&filename); err != nil {
			return nil, fmt.Errorf("failed to scan filename: %w", err)
		}
		filenames[filename] = true
	}

	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("error iterating filenames: %w", err)
	}

	return filenames, nil
}

// Helper functions for UUID <-> sql.NullString conversion
func uuidToNullString(u *uuid.UUID) sql.NullString {
	if u == nil {
		return sql.NullString{Valid: false}
	}
	return sql.NullString{String: u.String(), Valid: true}
}

func nullStringToUUID(ns sql.NullString) *uuid.UUID {
	if !ns.Valid {
		return nil
	}
	u, err := uuid.Parse(ns.String)
	if err != nil {
		return nil
	}
	return &u
}
