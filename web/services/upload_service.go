package services

import (
	"context"
	"fmt"
	"mime/multipart"
	"path/filepath"
	"stats-agent/agent"
	"stats-agent/database"
	"stats-agent/utils"
	"strings"
	"time"

	"github.com/google/uuid"
	"go.uber.org/zap"
)

// UploadService handles file upload operations including validation, saving, and processing.
type UploadService struct {
	store      *database.PostgresStore
	pdfService *PDFService
	logger     *zap.Logger
}

// NewUploadService creates a new UploadService instance.
func NewUploadService(store *database.PostgresStore, pdfService *PDFService, logger *zap.Logger) *UploadService {
	return &UploadService{
		store:      store,
		pdfService: pdfService,
		logger:     logger,
	}
}

// UploadResult contains the result of a file upload operation.
type UploadResult struct {
	SanitizedFilename string
	FilePath          string
	FileType          string
	PreparedMessage   string
}

// ValidateUpload checks if the uploaded file meets all requirements (type, size, safety).
// Returns the sanitized filename and error if validation fails.
func (us *UploadService) ValidateUpload(file *multipart.FileHeader) (string, error) {
	// Sanitize filename
	sanitizedFilename := utils.SanitizeFilename(file.Filename)
	if sanitizedFilename == "" {
		return "", fmt.Errorf("invalid or unsafe filename")
	}

	// Get file info for validation
	fileInfo := utils.GetFileInfo(file.Filename)
	if !fileInfo.IsAllowed {
		return "", fmt.Errorf("invalid file type. Please upload CSV, Excel, or PDF files")
	}

	// Validate file size based on type
	if !utils.ValidateFileSize(fileInfo.Type, file.Size) {
		switch fileInfo.Type {
		case utils.FileTypePDF:
			return "", fmt.Errorf("PDF file too large. Maximum size is 10MB")
		case utils.FileTypeCSV:
			return "", fmt.Errorf("CSV file too large. Maximum size is 50MB")
		default:
			return "", fmt.Errorf("file too large")
		}
	}

	return sanitizedFilename, nil
}

// TrackUploadedFile tracks a successfully uploaded file in the database.
// This should be called after the file has been saved to disk.
func (us *UploadService) TrackUploadedFile(ctx context.Context, file *multipart.FileHeader, sessionID string, sanitizedFilename string, filePath string) error {
	// Verify file exists before tracking
	workspaceDir := filepath.Join("workspaces", sessionID)
	if !utils.VerifyFileExists(workspaceDir, sanitizedFilename) {
		return fmt.Errorf("file not found after upload: %s", sanitizedFilename)
	}

	// Track file in database - non-critical operation
	sessionUUID, err := uuid.Parse(sessionID)
	if err != nil {
		us.logger.Warn("Failed to parse session ID for file tracking", zap.Error(err))
		return nil
	}

	webPath := filepath.ToSlash(filepath.Join("/workspaces", sessionID, sanitizedFilename))
	fileInfo := utils.GetFileInfo(file.Filename)

	fileRecord := database.FileRecord{
		ID:        uuid.New(),
		SessionID: sessionUUID,
		Filename:  sanitizedFilename,
		FilePath:  webPath,
		FileType:  string(fileInfo.Type),
		FileSize:  file.Size,
		CreatedAt: time.Now(),
		MessageID: nil, // Will be associated with user message later if needed
	}

	if _, err := us.store.CreateFile(ctx, fileRecord); err != nil {
		us.logger.Warn("Failed to track uploaded file in database",
			zap.Error(err),
			zap.String("filename", sanitizedFilename),
			zap.String("session_id", sessionID))
		return err
	}

	return nil
}

// ProcessPDFContent extracts text from a PDF file with smart truncation.
// Returns the extracted text or empty string if extraction fails (non-fatal).
func (us *UploadService) ProcessPDFContent(ctx context.Context, filePath string, filename string, config TruncationConfig, memManager *agent.MemoryManager) (string, error) {
	pdfText, err := us.pdfService.ExtractTextSmart(ctx, filePath, config, memManager)
	if err != nil {
		us.logger.Error("Failed to extract PDF text",
			zap.Error(err),
			zap.String("filename", filename))
		return "", err
	}

	return pdfText, nil
}

// PrepareMessageWithFile combines file content with user message.
// For PDFs, it prepends the extracted text. For other files, it adds a file attachment notice.
func (us *UploadService) PrepareMessageWithFile(message, filename, pdfContent string, isPDF bool) string {
	if isPDF && pdfContent != "" {
		// Prepend PDF content to user message
		pdfHeader := fmt.Sprintf("[PDF Content from %s]\n\n%s\n\n---\n\n", filename, pdfContent)

		if strings.TrimSpace(message) == "" {
			return pdfHeader + "Please analyze the content from this PDF and provide statistical insights."
		}
		return pdfHeader + message
	}

	// For non-PDF files or when PDF extraction failed
	if strings.TrimSpace(message) == "" {
		return fmt.Sprintf("I've uploaded %s. Please analyze this dataset and provide statistical insights.", filename)
	}
	return fmt.Sprintf("[📎 File uploaded: %s]\n\n%s", filename, message)
}
