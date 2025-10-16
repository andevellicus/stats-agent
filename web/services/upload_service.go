package services

import (
	"context"
	"fmt"
	"mime/multipart"
	"os"
	"path/filepath"
	"stats-agent/database"
	"stats-agent/rag"
	"strings"
	"time"

	"github.com/google/uuid"
	"go.uber.org/zap"
)

const (
	MaxPDFSize = 10 * 1024 * 1024 // 10MB
)

type UploadService struct {
	store      *database.PostgresStore
	pdfService *PDFService
	ragGetter  RAGGetter // Interface to get RAG instance
	logger     *zap.Logger
}

// RAGGetter interface to avoid circular dependency with agent
type RAGGetter interface {
	GetRAG() *rag.RAG
}

type UploadResult struct {
	Filename         string
	FilePath         string
	FileType         string
	DisplayMessage   string // HTML-formatted message for display
	ContentMessage   string // Plain text message for LLM/storage
	RequiresPDFIndex bool   // True if PDF needs indexing before proceeding
}

func NewUploadService(
	store *database.PostgresStore,
	pdfService *PDFService,
	ragGetter RAGGetter,
	logger *zap.Logger,
) *UploadService {
	return &UploadService{
		store:      store,
		pdfService: pdfService,
		ragGetter:  ragGetter,
		logger:     logger,
	}
}

// ValidateFile checks if the file is valid (type and size).
// Returns sanitized filename and file extension, or error if invalid.
func (us *UploadService) ValidateFile(file *multipart.FileHeader) (string, string, error) {
	// Sanitize filename
	sanitizedFilename := sanitizeFilename(file.Filename)
	if sanitizedFilename == "" {
		return "", "", fmt.Errorf("invalid or unsafe filename")
	}

	// Check file type
	ext := strings.ToLower(filepath.Ext(file.Filename))
	if ext != ".csv" && ext != ".xlsx" && ext != ".xls" && ext != ".pdf" {
		return "", "", fmt.Errorf("invalid file type. Please upload CSV, Excel, or PDF files")
	}

	// Check PDF size limit
	if ext == ".pdf" && file.Size > MaxPDFSize {
		return "", "", fmt.Errorf("PDF file too large. Maximum size is 10MB")
	}

	return sanitizedFilename, ext, nil
}

// SaveFile saves the uploaded file to the workspace directory.
// Returns the web path of the saved file.
func (us *UploadService) SaveFile(
	file *multipart.FileHeader,
	sessionID uuid.UUID,
	sanitizedFilename string,
) (string, error) {
	workspaceDir := filepath.Join("workspaces", sessionID.String())
	dst := filepath.Join(workspaceDir, sanitizedFilename)

	// Open the uploaded file
	src, err := file.Open()
	if err != nil {
		return "", fmt.Errorf("failed to open uploaded file: %w", err)
	}
	defer src.Close()

	// Create destination file
	out, err := os.Create(dst)
	if err != nil {
		return "", fmt.Errorf("failed to create destination file: %w", err)
	}
	defer out.Close()

	// Copy contents
	if _, err := out.ReadFrom(src); err != nil {
		return "", fmt.Errorf("failed to save file: %w", err)
	}

	// Verify file exists
	if !verifyFileExists(workspaceDir, sanitizedFilename) {
		return "", fmt.Errorf("file verification failed after upload")
	}

	webPath := filepath.ToSlash(filepath.Join("/workspaces", sessionID.String(), sanitizedFilename))
	return webPath, nil
}

// ProcessUpload handles the complete file upload workflow.
// Returns UploadResult with formatted messages and metadata.
func (us *UploadService) ProcessUpload(
	ctx context.Context,
	file *multipart.FileHeader,
	sessionID uuid.UUID,
	userMessage string,
) (*UploadResult, error) {
	// Validate file
	sanitizedFilename, ext, err := us.ValidateFile(file)
	if err != nil {
		return nil, err
	}

	// Save file
	webPath, err := us.SaveFile(file, sessionID, sanitizedFilename)
	if err != nil {
		us.logger.Error("Failed to save uploaded file",
			zap.Error(err),
			zap.String("filename", sanitizedFilename),
			zap.String("session_id", sessionID.String()))
		return nil, err
	}

	us.logger.Info("File uploaded successfully",
		zap.String("filename", file.Filename),
		zap.String("session_id", sessionID.String()),
		zap.Int64("size_bytes", file.Size))

	// Determine file type for database
	fileType := "csv"
	if ext == ".pdf" {
		fileType = "pdf"
	}

	// Track uploaded file in database
	fileRecord := database.FileRecord{
		ID:        uuid.New(),
		SessionID: sessionID,
		Filename:  sanitizedFilename,
		FilePath:  webPath,
		FileType:  fileType,
		FileSize:  file.Size,
		CreatedAt: time.Now(),
		MessageID: nil, // Will be associated with user message later if needed
	}
	if _, err := us.store.CreateFile(ctx, fileRecord); err != nil {
		us.logger.Warn("Failed to track uploaded file in database",
			zap.Error(err),
			zap.String("filename", sanitizedFilename),
			zap.String("session_id", sessionID.String()))
		// Continue - file is saved, just not tracked
	}

	// Handle PDF-specific processing
	if ext == ".pdf" {
		return us.processPDFUpload(ctx, sanitizedFilename, webPath, file.Filename, sessionID, userMessage)
	}

	// Handle dataset files (CSV, Excel)
	return us.processDatasetUpload(file.Filename, userMessage), nil
}

// processPDFUpload extracts pages and stores them in RAG.
func (us *UploadService) processPDFUpload(
	ctx context.Context,
	sanitizedFilename string,
	filePath string,
	originalFilename string,
	sessionID uuid.UUID,
	userMessage string,
) (*UploadResult, error) {
	// Format display message
	var displayMessage, contentMessage string
	if strings.TrimSpace(userMessage) == "" {
		contentMessage = fmt.Sprintf("[📎 File uploaded: %s]\n\nPlease analyze the content from this PDF and provide statistical insights.", originalFilename)
		displayMessage = fmt.Sprintf("[📎 PDF uploaded: %s]<br><br>Please analyze the content from this PDF and provide statistical insights.", originalFilename)
	} else {
		displayMessage = fmt.Sprintf("[📎 PDF uploaded: %s]<br><br>%s", originalFilename, userMessage)
		contentMessage = fmt.Sprintf("[📎 File uploaded: %s]\n\n%s", originalFilename, userMessage)
	}

	// Extract pages and store in RAG synchronously
	pdfCtx, pdfCancel := context.WithTimeout(ctx, 30*time.Second)
	defer pdfCancel()

	// Convert web path to filesystem path
	workspaceDir := filepath.Join("workspaces", sessionID.String())
	dst := filepath.Join(workspaceDir, sanitizedFilename)

	pages, err := us.pdfService.ExtractPages(dst)
	if err != nil {
		us.logger.Error("Failed to extract PDF pages for RAG",
			zap.Error(err),
			zap.String("filename", sanitizedFilename))
		// Continue - user can still ask questions, just without PDF content in RAG
	} else {
		ragInstance := us.ragGetter.GetRAG()
		if ragInstance == nil {
			us.logger.Warn("RAG instance not available for PDF storage")
		} else {
			if err := ragInstance.AddPDFPagesToRAG(pdfCtx, sessionID.String(), originalFilename, pages); err != nil {
				us.logger.Error("Failed to store PDF pages in RAG",
					zap.Error(err),
					zap.String("filename", sanitizedFilename),
					zap.String("session_id", sessionID.String()))
			} else {
				us.logger.Info("Successfully stored PDF pages in RAG",
					zap.String("filename", sanitizedFilename),
					zap.Int("pages", len(pages)),
					zap.String("session_id", sessionID.String()))
			}
		}
	}

	return &UploadResult{
		Filename:         sanitizedFilename,
		FilePath:         filePath,
		FileType:         "pdf",
		DisplayMessage:   displayMessage,
		ContentMessage:   contentMessage,
		RequiresPDFIndex: true,
	}, nil
}

// processDatasetUpload formats messages for CSV/Excel uploads.
func (us *UploadService) processDatasetUpload(originalFilename string, userMessage string) *UploadResult {
	var contentMessage string
	if strings.TrimSpace(userMessage) == "" {
		contentMessage = fmt.Sprintf("I've uploaded %s. Please analyze this dataset and provide statistical insights.", originalFilename)
	} else {
		contentMessage = fmt.Sprintf("[📎 File uploaded: %s]\n\n%s", originalFilename, userMessage)
	}

	return &UploadResult{
		Filename:         originalFilename,
		FilePath:         "", // Not needed for display in dataset mode
		FileType:         "csv",
		DisplayMessage:   "", // Will use ContentMessage
		ContentMessage:   contentMessage,
		RequiresPDFIndex: false,
	}
}

// sanitizeFilename sanitizes user-provided filenames for safe storage.
func sanitizeFilename(filename string) string {
	// Trim leading/trailing spaces and dots
	sanitized := strings.Trim(filename, " .")

	// Remove path traversal attempts
	sanitized = strings.ReplaceAll(sanitized, "..", "")

	// Preserve extension
	ext := filepath.Ext(sanitized)
	nameWithoutExt := strings.TrimSuffix(sanitized, ext)

	// Replace special characters with safe alternatives
	nameWithoutExt = replaceSpecialChars(nameWithoutExt)

	// Reconstruct with original extension
	sanitized = nameWithoutExt + ext

	// Limit total length to 255 characters
	if len(sanitized) > 255 {
		// Truncate name portion, preserve extension
		maxNameLen := 255 - len(ext)
		if maxNameLen > 0 {
			sanitized = nameWithoutExt[:maxNameLen] + ext
		} else {
			sanitized = sanitized[:255]
		}
	}

	return sanitized
}

func replaceSpecialChars(s string) string {
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

func verifyFileExists(workspaceDir, filename string) bool {
	safePath := filepath.Join(workspaceDir, filename)
	info, err := os.Stat(safePath)
	if os.IsNotExist(err) {
		return false
	}
	if info.IsDir() {
		return false
	}
	return true
}
