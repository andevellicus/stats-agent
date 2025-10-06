package services

import (
	"bytes"
	"context"
	"fmt"
	"os"
	"path/filepath"
	"stats-agent/database"
	"stats-agent/utils"
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
			fileName := fileEntry.Name()

			// Skip if already tracked
			if trackedFiles[fileName] {
				continue
			}

			// Get file info for metadata
			fullPath := filepath.Join(workspaceDir, fileName)
			fileInfo, err := os.Stat(fullPath)
			if err != nil {
				fs.logger.Warn("Failed to stat file", zap.Error(err), zap.String("filename", fileName))
				continue
			}

			// Determine file type using utils
			info := utils.GetFileInfo(fileName)
			fileType := string(info.Type)

			// Create file record in database
			webPath := filepath.ToSlash(filepath.Join("/workspaces", sessionID, fileName))
			fileRecord := database.FileRecord{
				ID:        uuid.New(),
				SessionID: sessionUUID,
				Filename:  fileName,
				FilePath:  webPath,
				FileType:  fileType,
				FileSize:  fileInfo.Size(),
				CreatedAt: time.Now(),
				MessageID: nil, // Will be set later if associated with a message
			}

			if _, err := fs.store.CreateFile(ctx, fileRecord); err != nil {
				fs.logger.Warn("Failed to create file record in DB",
					zap.Error(err),
					zap.String("filename", fileName))
				// Continue - file might have been created by concurrent request
			}

			newFilePaths = append(newFilePaths, webPath)
		}
	}
	return newFilePaths, nil
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

		info := utils.GetFileInfo(path)
		if info.IsImage {
			component = components.ImageBlock(path)
		} else if info.IsRenderable && !info.IsImage {
			component = components.FileBlock(path)
		} else {
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
