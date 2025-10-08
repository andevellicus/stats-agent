package services

import (
	"bytes"
	"context"
	"fmt"
	"os"
	"path/filepath"
	"stats-agent/database"
	"stats-agent/web/templates/components"
	"strings"
	"time"

	"github.com/a-h/templ"
	"github.com/google/uuid"
	"go.uber.org/zap"
)

type FileService struct {
	store  *database.PostgresStore
	logger *zap.Logger
}

func NewFileService(store *database.PostgresStore, logger *zap.Logger) *FileService {
	return &FileService{
		store:  store,
		logger: logger,
	}
}

// GetAndMarkNewFiles finds new files in the workspace, marks them as tracked in the DB, and returns their web paths.
// Uses the new files table for efficient tracking with proper metadata.
func (fs *FileService) GetAndMarkNewFiles(ctx context.Context, sessionID string) ([]string, error) {
	sessionUUID, err := uuid.Parse(sessionID)
	if err != nil {
		return nil, fmt.Errorf("invalid session ID: %w", err)
	}

	workspaceDir := filepath.Join("workspaces", sessionID)
	filesInDir, err := os.ReadDir(workspaceDir)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil // No workspace is not an error
		}
		return nil, fmt.Errorf("could not read workspace directory: %w", err)
	}

	// Get already tracked filenames for efficient lookup
	trackedFiles, err := fs.store.GetTrackedFilenames(ctx, sessionUUID)
	if err != nil {
		return nil, fmt.Errorf("could not get tracked files: %w", err)
	}

	var newFilePaths []string
	for _, fileEntry := range filesInDir {
		if !fileEntry.IsDir() {
			originalFileName := fileEntry.Name()

			// Skip if already tracked (check both original and sanitized names)
			if trackedFiles[originalFileName] {
				continue
			}

			// Sanitize the filename for web safety
			sanitizedFileName := sanitizeOutputFilename(originalFileName)
			if sanitizedFileName == "" {
				fs.logger.Warn("File has invalid name after sanitization, skipping",
					zap.String("original_name", originalFileName))
				continue
			}

			// If filename changed, rename the physical file
			if sanitizedFileName != originalFileName {
				oldPath := filepath.Join(workspaceDir, originalFileName)
				newPath := filepath.Join(workspaceDir, sanitizedFileName)

				if err := os.Rename(oldPath, newPath); err != nil {
					fs.logger.Warn("Failed to rename file to sanitized name",
						zap.Error(err),
						zap.String("original", originalFileName),
						zap.String("sanitized", sanitizedFileName))
					// Continue with original name if rename fails
					sanitizedFileName = originalFileName
				} else {
					fs.logger.Info("Renamed Python-generated file to sanitized name",
						zap.String("original", originalFileName),
						zap.String("sanitized", sanitizedFileName))
				}
			}

			// Get file info for metadata (use sanitized path)
			fullPath := filepath.Join(workspaceDir, sanitizedFileName)
			fileInfo, err := os.Stat(fullPath)
			if err != nil {
				fs.logger.Warn("Failed to stat file", zap.Error(err), zap.String("filename", sanitizedFileName))
				continue
			}

			// Determine file type
			ext := strings.ToLower(filepath.Ext(sanitizedFileName))
			fileType := "other"
			switch ext {
			case ".png", ".jpg", ".jpeg", ".gif":
				fileType = "image"
			case ".csv", ".xls", ".xlsx":
				fileType = "csv"
			case ".pdf":
				fileType = "pdf"
			}

			// Create file record in database with sanitized name
			webPath := filepath.ToSlash(filepath.Join("/workspaces", sessionID, sanitizedFileName))
			fileRecord := database.FileRecord{
				ID:        uuid.New(),
				SessionID: sessionUUID,
				Filename:  sanitizedFileName,
				FilePath:  webPath,
				FileType:  fileType,
				FileSize:  fileInfo.Size(),
				CreatedAt: time.Now(),
				MessageID: nil, // Will be set later if associated with a message
			}

			if _, err := fs.store.CreateFile(ctx, fileRecord); err != nil {
				fs.logger.Warn("Failed to create file record in DB",
					zap.Error(err),
					zap.String("filename", sanitizedFileName))
				// Continue - file might have been created by concurrent request
			}

			newFilePaths = append(newFilePaths, webPath)
		}
	}
	return newFilePaths, nil
}

// sanitizeOutputFilename sanitizes filenames created by Python to be web-safe.
// Replaces special characters with safe alternatives instead of URL encoding.
func sanitizeOutputFilename(filename string) string {
	// Trim leading/trailing spaces and dots
	sanitized := strings.Trim(filename, " .")

	// Remove path traversal attempts
	sanitized = strings.ReplaceAll(sanitized, "..", "")

	// Preserve extension
	ext := filepath.Ext(sanitized)
	nameWithoutExt := strings.TrimSuffix(sanitized, ext)

	// Replace special characters with safe alternatives
	nameWithoutExt = replaceSpecialCharsForOutput(nameWithoutExt)

	// Reconstruct with original extension
	sanitized = nameWithoutExt + ext

	// Limit total length to 255 characters
	if len(sanitized) > 255 {
		maxNameLen := 255 - len(ext)
		if maxNameLen > 0 {
			sanitized = nameWithoutExt[:maxNameLen] + ext
		} else {
			sanitized = sanitized[:255]
		}
	}

	return sanitized
}

func replaceSpecialCharsForOutput(s string) string {
	// Replace common special chars with readable alternatives
	s = strings.ReplaceAll(s, "%", "pct")
	s = strings.ReplaceAll(s, "&", "and")
	s = strings.ReplaceAll(s, " ", "_")

	// Replace filesystem-unsafe characters with underscore
	// These are problematic across Windows, Linux, macOS
	unsafeChars := []string{
		"<", ">", ":", "\"", "/", "\\", "|", "?", "*",
		"(", ")", "[", "]", "{", "}", "'", ",", ";", "!",
		"@", "#", "$", "^", "`", "~", "+", "=",
	}

	for _, char := range unsafeChars {
		s = strings.ReplaceAll(s, char, "_")
	}

	// Collapse multiple underscores to single
	for strings.Contains(s, "__") {
		s = strings.ReplaceAll(s, "__", "_")
	}

	// Trim leading/trailing underscores
	s = strings.Trim(s, "_")

	return s
}

// RenderFileBlocksForDB renders file blocks to a raw HTML string for database persistence.
func (fs *FileService) RenderFileBlocksForDB(ctx context.Context, filePaths []string) (string, error) {
	if len(filePaths) == 0 {
		return "", nil
	}

	var htmlBuilder strings.Builder
	for _, path := range filePaths {
		var buf bytes.Buffer
		var component templ.Component
		ext := strings.ToLower(filepath.Ext(path))
		switch ext {
		case ".png", ".jpg", ".jpeg", ".gif":
			component = components.ImageBlock(path)
		case ".csv", ".xls", ".xlsx", ".pdf":
			component = components.FileBlock(path)
		default:
			component = nil // Ignore other file types
		}

		if component != nil {
			if err := component.Render(ctx, &buf); err != nil {
				return "", fmt.Errorf("failed to render component for db: %w", err)
			}
			htmlBuilder.Write(buf.Bytes())
		}
	}
	return htmlBuilder.String(), nil
}

// RenderFileOOBWrapper renders the out-of-band file wrapper for SSE streaming.
func (fs *FileService) RenderFileOOBWrapper(ctx context.Context, fileContainerID string, filePaths []string) (string, error) {
	if len(filePaths) == 0 {
		return "", nil
	}

	var buf bytes.Buffer
	if err := components.FileOOBWrapper(fileContainerID, filePaths).Render(ctx, &buf); err != nil {
		return "", fmt.Errorf("failed to render file OOB wrapper: %w", err)
	}
	return buf.String(), nil
}
