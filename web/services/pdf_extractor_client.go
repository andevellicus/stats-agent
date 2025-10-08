package services

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"os"
	"path/filepath"
	pdfTypes "stats-agent/pdf"
	"time"

	"go.uber.org/zap"
)

// PDFExtractorClient handles communication with the pdfplumber microservice
type PDFExtractorClient struct {
	baseURL    string
	httpClient *http.Client
	logger     *zap.Logger
	enabled    bool
}

// PDFExtractorResponse represents the JSON response from the extraction service
type PDFExtractorResponse struct {
	Success    bool                  `json:"success"`
	Text       string                `json:"text"`
	Pages      []PDFExtractorPage    `json:"pages"`
	TotalPages int                   `json:"total_pages"`
	Metadata   map[string]string     `json:"metadata"`
	Characters int                   `json:"characters"`
	Error      string                `json:"error,omitempty"`
}

// PDFExtractorPage represents a single page from the extraction
type PDFExtractorPage struct {
	Page int    `json:"page"`
	Text string `json:"text"`
}

// NewPDFExtractorClient creates a new PDF extractor client
func NewPDFExtractorClient(baseURL string, timeout time.Duration, enabled bool, logger *zap.Logger) *PDFExtractorClient {
	return &PDFExtractorClient{
		baseURL: baseURL,
		httpClient: &http.Client{
			Timeout: timeout,
		},
		logger:  logger,
		enabled: enabled,
	}
}

// IsEnabled returns whether the PDF extractor service is enabled
func (c *PDFExtractorClient) IsEnabled() bool {
	return c.enabled
}

// HealthCheck checks if the PDF extractor service is available
func (c *PDFExtractorClient) HealthCheck(ctx context.Context) error {
	if !c.enabled {
		return fmt.Errorf("PDF extractor is disabled")
	}

	req, err := http.NewRequestWithContext(ctx, "GET", c.baseURL+"/health", nil)
	if err != nil {
		return fmt.Errorf("failed to create health check request: %w", err)
	}

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("health check failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("health check returned status %d", resp.StatusCode)
	}

	return nil
}

// ExtractText extracts text from a PDF file using the pdfplumber service
func (c *PDFExtractorClient) ExtractText(ctx context.Context, pdfPath string) (string, error) {
	if !c.enabled {
		return "", fmt.Errorf("PDF extractor is disabled")
	}

	// Open the PDF file
	file, err := os.Open(pdfPath)
	if err != nil {
		return "", fmt.Errorf("failed to open PDF file: %w", err)
	}
	defer file.Close()

	// Create multipart form data
	body := &bytes.Buffer{}
	writer := multipart.NewWriter(body)

	part, err := writer.CreateFormFile("file", filepath.Base(pdfPath))
	if err != nil {
		return "", fmt.Errorf("failed to create form file: %w", err)
	}

	if _, err := io.Copy(part, file); err != nil {
		return "", fmt.Errorf("failed to copy file data: %w", err)
	}

	if err := writer.Close(); err != nil {
		return "", fmt.Errorf("failed to close multipart writer: %w", err)
	}

	// Create HTTP request
	req, err := http.NewRequestWithContext(ctx, "POST", c.baseURL+"/extract", body)
	if err != nil {
		return "", fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", writer.FormDataContentType())

	c.logger.Debug("Sending PDF to extraction service",
		zap.String("path", pdfPath),
		zap.String("url", c.baseURL))

	// Send request
	resp, err := c.httpClient.Do(req)
	if err != nil {
		return "", fmt.Errorf("request to PDF extractor failed: %w", err)
	}
	defer resp.Body.Close()

	// Parse response
	var result PDFExtractorResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return "", fmt.Errorf("failed to decode response: %w", err)
	}

	if !result.Success {
		return "", fmt.Errorf("extraction failed: %s", result.Error)
	}

	c.logger.Info("PDF extraction successful",
		zap.String("path", pdfPath),
		zap.Int("pages", result.TotalPages),
		zap.Int("characters", result.Characters))

	return result.Text, nil
}

// ExtractPages extracts text from each page individually using the pdfplumber service
func (c *PDFExtractorClient) ExtractPages(ctx context.Context, pdfPath string) ([]pdfTypes.Page, error) {
	if !c.enabled {
		return nil, fmt.Errorf("PDF extractor is disabled")
	}

	// Open the PDF file
	file, err := os.Open(pdfPath)
	if err != nil {
		return nil, fmt.Errorf("failed to open PDF file: %w", err)
	}
	defer file.Close()

	// Create multipart form data
	body := &bytes.Buffer{}
	writer := multipart.NewWriter(body)

	part, err := writer.CreateFormFile("file", filepath.Base(pdfPath))
	if err != nil {
		return nil, fmt.Errorf("failed to create form file: %w", err)
	}

	if _, err := io.Copy(part, file); err != nil {
		return nil, fmt.Errorf("failed to copy file data: %w", err)
	}

	if err := writer.Close(); err != nil {
		return nil, fmt.Errorf("failed to close multipart writer: %w", err)
	}

	// Create HTTP request
	req, err := http.NewRequestWithContext(ctx, "POST", c.baseURL+"/extract", body)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", writer.FormDataContentType())

	// Send request
	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("request to PDF extractor failed: %w", err)
	}
	defer resp.Body.Close()

	// Parse response
	var result PDFExtractorResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	if !result.Success {
		return nil, fmt.Errorf("extraction failed: %s", result.Error)
	}

	// Convert to pdfTypes.Page
	pages := make([]pdfTypes.Page, 0, len(result.Pages))
	for _, p := range result.Pages {
		pages = append(pages, pdfTypes.Page{
			PageNumber: p.Page,
			Text:       p.Text,
		})
	}

	c.logger.Info("PDF page extraction successful",
		zap.String("path", pdfPath),
		zap.Int("pages", len(pages)))

	return pages, nil
}
