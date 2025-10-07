package services

import (
	"context"
	"fmt"
	"regexp"
	"strings"

	"github.com/jdkato/prose/v2"
	"github.com/ledongthuc/pdf"
	"go.uber.org/zap"
)

type PDFService struct {
	logger *zap.Logger
	config *PDFConfig
}

// PDFConfig holds PDF-specific configuration
type PDFConfig struct {
	TokenThreshold           float64
	FirstPagesPriority       int
	EnableTableDetection     bool
	SentenceBoundaryTruncate bool
}

// TokenCounter interface abstracts token counting for PDF truncation
type TokenCounter interface {
	CountTokens(ctx context.Context, text string) (int, error)
}

// TruncationConfig controls how large PDFs are truncated for LLM context
type TruncationConfig struct {
	MaxTokens                int     // Token limit (from config.ContextLength)
	TokenThreshold           float64 // Percentage of context to use
	FirstPagesPrio           int     // Prioritize first N pages
	EnableTableDetection     bool    // Detect and mark tables
	SentenceBoundaryTruncate bool    // Truncate at sentence boundaries
}

func NewPDFService(logger *zap.Logger, config *PDFConfig) *PDFService {
	return &PDFService{
		logger: logger,
		config: config,
	}
}

// ExtractText extracts all text content from a PDF file
// Returns the full text with page markers for context
func (ps *PDFService) ExtractText(pdfPath string) (string, error) {
	f, r, err := pdf.Open(pdfPath)
	if err != nil {
		return "", fmt.Errorf("failed to open PDF: %w", err)
	}
	defer f.Close()

	var fullText strings.Builder
	totalPages := r.NumPage()

	ps.logger.Debug("Extracting text from PDF",
		zap.String("path", pdfPath),
		zap.Int("pages", totalPages))

	for pageNum := 1; pageNum <= totalPages; pageNum++ {
		page := r.Page(pageNum)
		if page.V.IsNull() {
			ps.logger.Warn("Skipping null page",
				zap.Int("page", pageNum))
			continue
		}

		text, err := page.GetPlainText(nil)
		if err != nil {
			ps.logger.Warn("Failed to extract text from page",
				zap.Int("page", pageNum),
				zap.Error(err))
			continue
		}

		// Add page marker for context
		fullText.WriteString(fmt.Sprintf("--- Page %d ---\n", pageNum))
		fullText.WriteString(text)
		fullText.WriteString("\n\n")
	}

	extractedText := fullText.String()
	ps.logger.Info("PDF text extraction completed",
		zap.String("path", pdfPath),
		zap.Int("pages", totalPages),
		zap.Int("characters", len(extractedText)))

	return extractedText, nil
}

// ExtractTextSmart extracts PDF text with intelligent truncation for large documents
// Uses token counting to stay within context window limits, prioritizing first pages
func (ps *PDFService) ExtractTextSmart(ctx context.Context, pdfPath string, config TruncationConfig, tokenCounter TokenCounter) (string, error) {
	f, r, err := pdf.Open(pdfPath)
	if err != nil {
		return "", fmt.Errorf("failed to open PDF: %w", err)
	}
	defer f.Close()

	totalPages := r.NumPage()
	tokenLimit := int(float64(config.MaxTokens) * config.TokenThreshold)

	ps.logger.Debug("Extracting text from PDF with token-based truncation",
		zap.String("path", pdfPath),
		zap.Int("pages", totalPages),
		zap.Int("token_limit", tokenLimit))

	// Extract pages sequentially, counting tokens as we go
	type pageContent struct {
		num    int
		text   string
		tokens int
	}

	var output strings.Builder
	var pages []pageContent
	totalTokens := 0
	totalChars := 0

	// Extract pages until we hit token limit
	for pageNum := 1; pageNum <= totalPages; pageNum++ {
		page := r.Page(pageNum)
		if page.V.IsNull() {
			ps.logger.Warn("Skipping null page", zap.Int("page", pageNum))
			continue
		}

		text, err := page.GetPlainText(nil)
		if err != nil {
			ps.logger.Warn("Failed to extract text from page",
				zap.Int("page", pageNum),
				zap.Error(err))
			continue
		}

		// Apply table detection if enabled
		if config.EnableTableDetection {
			text = ps.detectTablesInText(text)
		}

		// Format page content
		pageText := fmt.Sprintf("--- Page %d ---\n%s\n\n", pageNum, text)

		// Count tokens for this page
		tokens, err := tokenCounter.CountTokens(ctx, pageText)
		if err != nil {
			ps.logger.Warn("Failed to count tokens for page, using character estimate",
				zap.Int("page", pageNum),
				zap.Error(err))
			// Fallback: rough estimate of 4 chars per token
			tokens = len(pageText) / 4
		}

		pages = append(pages, pageContent{
			num:    pageNum,
			text:   text,
			tokens: tokens,
		})
		totalChars += len(text)

		// Check if we can add this page without exceeding limit
		if totalTokens+tokens <= tokenLimit {
			output.WriteString(pageText)
			totalTokens += tokens
		} else {
			// Check if we should truncate or stop
			// If we haven't reached FirstPagesPrio yet, try to fit it
			if pageNum <= config.FirstPagesPrio {
				// Priority pages - add even if slightly over
				output.WriteString(pageText)
				totalTokens += tokens
				ps.logger.Debug("Added priority page despite limit",
					zap.Int("page", pageNum),
					zap.Int("tokens", totalTokens),
					zap.Int("limit", tokenLimit))
			} else {
				// We've hit the limit - apply sentence boundary truncation if enabled
				if config.SentenceBoundaryTruncate {
					// Try to add partial page at sentence boundary
					remainingTokens := tokenLimit - totalTokens

					if remainingTokens > 50 { // Only try if we have reasonable room
						// Try to fit partial page at sentence boundary
						truncatedPage, err := ps.truncateAtSentenceBoundary(ctx, text, remainingTokens, tokenCounter)
						if err == nil && len(truncatedPage) > 0 && truncatedPage != text {
							output.WriteString(fmt.Sprintf("--- Page %d (partial) ---\n%s\n\n", pageNum, truncatedPage))
							ps.logger.Info("Added partial page at sentence boundary",
								zap.Int("page", pageNum),
								zap.Int("remaining_tokens", remainingTokens))
						}
					}
				}

				output.WriteString(fmt.Sprintf("\n[... Remaining %d pages truncated (token limit reached) ...]\n", totalPages-pageNum+1))
				ps.logger.Info("Truncated PDF at token limit",
					zap.Int("pages_included", pageNum-1),
					zap.Int("pages_total", totalPages),
					zap.Int("tokens_used", totalTokens),
					zap.Int("token_limit", tokenLimit))
				break
			}
		}
	}

	result := output.String()

	// Calculate metrics
	includedPages := len(pages)
	if totalTokens > tokenLimit {
		// Find how many pages were actually included
		for i, p := range pages {
			if p.num > config.FirstPagesPrio && totalTokens-p.tokens <= tokenLimit {
				includedPages = i
				break
			}
		}
	}

	ps.logger.Info("PDF token-based truncation completed",
		zap.String("path", pdfPath),
		zap.Int("total_pages", totalPages),
		zap.Int("included_pages", includedPages),
		zap.Int("tokens_used", totalTokens),
		zap.Int("token_limit", tokenLimit),
		zap.Int("characters", len(result)))

	return result, nil
}

// detectTablesInText applies heuristic table detection to mark tabular regions
func (ps *PDFService) detectTablesInText(text string) string {
	// Pattern: multiple lines with aligned columns (3+ spaces or tabs between words)
	// Look for at least 3 consecutive lines with similar spacing patterns
	lines := strings.Split(text, "\n")
	var result strings.Builder

	inTable := false
	tableLines := 0
	tablePattern := regexp.MustCompile(`\s{3,}|\t+`) // 3+ spaces or tabs indicate columns

	for _, line := range lines {
		trimmed := strings.TrimSpace(line)
		if trimmed == "" {
			if inTable {
				result.WriteString("[/TABLE]\n")
				inTable = false
				tableLines = 0
			}
			result.WriteString(line + "\n")
			continue
		}

		// Check if line has column-like structure
		matches := tablePattern.FindAllString(line, -1)
		hasColumns := len(matches) >= 2 // At least 2 column separators

		if hasColumns {
			tableLines++
			if tableLines >= 3 && !inTable {
				// Start of table detected
				result.WriteString("[TABLE DETECTED]\n")
				inTable = true
			}
			result.WriteString(line + "\n")
		} else {
			if inTable && tableLines >= 3 {
				result.WriteString("[/TABLE]\n")
				inTable = false
			}
			tableLines = 0
			result.WriteString(line + "\n")
		}
	}

	// Close table if we ended mid-table
	if inTable {
		result.WriteString("[/TABLE]\n")
	}

	return result.String()
}

// truncateAtSentenceBoundary truncates text at the last complete sentence within token limit
func (ps *PDFService) truncateAtSentenceBoundary(ctx context.Context, text string, maxTokens int, tokenCounter TokenCounter) (string, error) {
	// First check if we're already under limit
	currentTokens, err := tokenCounter.CountTokens(ctx, text)
	if err != nil {
		return text, fmt.Errorf("failed to count tokens: %w", err)
	}

	if currentTokens <= maxTokens {
		return text, nil // No truncation needed
	}

	// Use prose to segment into sentences
	doc, err := prose.NewDocument(text)
	if err != nil {
		ps.logger.Warn("Failed to create prose document for sentence detection, truncating at character boundary", zap.Error(err))
		return text, nil // Return full text if prose fails
	}

	sentences := doc.Sentences()
	if len(sentences) == 0 {
		return text, nil
	}

	// Binary search to find the last sentence that fits
	var result strings.Builder
	lastValidLength := 0

	for i, sent := range sentences {
		result.WriteString(sent.Text)
		if i < len(sentences)-1 {
			result.WriteString(" ") // Add space between sentences
		}

		candidate := result.String()
		tokens, err := tokenCounter.CountTokens(ctx, candidate)
		if err != nil {
			ps.logger.Warn("Token counting failed during sentence truncation", zap.Error(err))
			break
		}

		if tokens <= maxTokens {
			lastValidLength = len(candidate)
		} else {
			// We've exceeded the limit, return the last valid text
			if lastValidLength > 0 {
				truncated := candidate[:lastValidLength]
				ps.logger.Info("Truncated at sentence boundary",
					zap.Int("sentences_included", i),
					zap.Int("total_sentences", len(sentences)),
					zap.Int("tokens", tokens))
				return truncated + "\n\n[... Content truncated at sentence boundary ...]", nil
			}
			break
		}
	}

	// If we couldn't find a valid truncation point, return original
	return text, nil
}
