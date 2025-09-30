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

// GetAndMarkNewFiles finds new files in the workspace, marks them as rendered in the DB, and returns their web paths.
func (fs *FileService) GetAndMarkNewFiles(ctx context.Context, sessionID string) ([]string, error) {
	sessionUUID, err := uuid.Parse(sessionID)
	if err != nil {
		return nil, fmt.Errorf("invalid session ID: %w", err)
	}

	workspaceDir := filepath.Join("workspaces", sessionID)
	files, err := os.ReadDir(workspaceDir)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil // No workspace is not an error
		}
		return nil, fmt.Errorf("could not read workspace directory: %w", err)
	}

	renderedFiles, err := fs.store.GetRenderedFiles(ctx, sessionUUID)
	if err != nil {
		return nil, fmt.Errorf("could not get rendered files: %w", err)
	}

	var newFilePaths []string
	for _, file := range files {
		if !file.IsDir() {
			fileName := file.Name()
			if _, rendered := renderedFiles[fileName]; !rendered {
				webPath := filepath.ToSlash(filepath.Join("/workspaces", sessionID, fileName))
				newFilePaths = append(newFilePaths, webPath)
				if err := fs.store.AddRenderedFile(ctx, sessionUUID, fileName); err != nil {
					fs.logger.Warn("Failed to mark file as rendered in DB", zap.Error(err), zap.String("filename", fileName))
				}
			}
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
		ext := strings.ToLower(filepath.Ext(path))
		switch ext {
		case ".png", ".jpg", ".jpeg", ".gif":
			component = components.ImageBlock(path)
		case ".csv", ".xls", ".xlsx":
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
