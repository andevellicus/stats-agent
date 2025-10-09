package services

import (
    "context"
    "fmt"
    "regexp"
    pdfTypes "stats-agent/pdf"
    "strings"
    "time"

	"github.com/jdkato/prose/v2"
	"github.com/ledongthuc/pdf"
	"go.uber.org/zap"
)

type PDFService struct {
    logger          *zap.Logger
    config          *PDFConfig
    extractorClient *PDFExtractorClient // Optional pdfplumber client
}

// PDFConfig holds PDF-specific configuration
type PDFConfig struct {
    TokenThreshold           float64
    FirstPagesPriority       int
    EnableTableDetection     bool
    SentenceBoundaryTruncate bool
    // Cleanup
    HeaderFooterRepeatThreshold float64
    ReferencesTrimEnabled       bool
    ReferencesCitationDensity   float64
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

func NewPDFService(logger *zap.Logger, config *PDFConfig, extractorClient *PDFExtractorClient) *PDFService {
	return &PDFService{
		logger:          logger,
		config:          config,
		extractorClient: extractorClient,
	}
}

// ExtractText extracts all text content from a PDF file
// Returns the full text with page markers for context
// Tries pdfplumber first (if enabled), falls back to ledongthuc/pdf
func (ps *PDFService) ExtractText(pdfPath string) (string, error) {
    // Build from pages so we can apply header/footer stripping uniformly
    pages, err := ps.ExtractPages(pdfPath)
    if err != nil {
        return "", err
    }
    var full strings.Builder
    for _, p := range pages {
        full.WriteString(fmt.Sprintf("--- Page %d ---\n", p.PageNumber))
        full.WriteString(p.Text)
        full.WriteString("\n\n")
    }
    text := full.String()
    ps.logger.Info("PDF text extraction completed",
        zap.String("path", pdfPath),
        zap.Int("pages", len(pages)),
        zap.Int("characters", len(text)))
    return text, nil
}

// ExtractPages extracts text from each page individually
// Returns a slice of pdf.Page structs, one per page
// Tries pdfplumber first (if enabled), falls back to ledongthuc/pdf
func (ps *PDFService) ExtractPages(pdfPath string) ([]pdfTypes.Page, error) {
    // Try pdfplumber extraction first if available
    if ps.extractorClient != nil && ps.extractorClient.IsEnabled() {
        ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
        defer cancel()

        pages, err := ps.extractorClient.ExtractPages(ctx, pdfPath)
        if err == nil {
            ps.logger.Info("PDF page extraction successful via pdfplumber",
                zap.String("path", pdfPath),
                zap.Int("pages", len(pages)))
            // Strip repeated headers/footers across pages
            pages = ps.stripRepeatedHeaderFooterWithConfig(pages)
            // Optionally trim trailing references
            if ps.config != nil && ps.config.ReferencesTrimEnabled {
                pages = ps.trimTrailingReferences(pages)
            }
            return pages, nil
        }

		ps.logger.Warn("pdfplumber page extraction failed, falling back to ledongthuc/pdf",
			zap.Error(err),
			zap.String("path", pdfPath))
	}

	// Fallback to ledongthuc/pdf extraction
	f, r, err := pdf.Open(pdfPath)
	if err != nil {
		return nil, fmt.Errorf("failed to open PDF: %w", err)
	}
	defer f.Close()

	totalPages := r.NumPage()
	pages := make([]pdfTypes.Page, 0, totalPages)

	ps.logger.Debug("Extracting pages from PDF using fallback method",
		zap.String("path", pdfPath),
		zap.Int("pages", totalPages))

    for pageNum := 1; pageNum <= totalPages; pageNum++ {
        page := r.Page(pageNum)
        if page.V.IsNull() {
            ps.logger.Warn("Skipping null page",
                zap.Int("page", pageNum))
            continue
        }

		// Use GetPlainText for text extraction
		text, err := page.GetPlainText(nil)
		if err != nil {
			ps.logger.Warn("Failed to extract text from page",
				zap.Int("page", pageNum),
				zap.Error(err))
			continue
		}

        pages = append(pages, pdfTypes.Page{
            PageNumber: pageNum,
            Text:       strings.TrimSpace(text),
        })
    }

    // Strip repeated headers/footers across pages (fallback path)
    pages = ps.stripRepeatedHeaderFooterWithConfig(pages)
    // Optionally trim trailing references
    if ps.config != nil && ps.config.ReferencesTrimEnabled {
        pages = ps.trimTrailingReferences(pages)
    }

    ps.logger.Info("PDF page extraction completed (fallback)",
        zap.String("path", pdfPath),
        zap.Int("pages_extracted", len(pages)),
        zap.Int("total_pages", totalPages))

    return pages, nil
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

// cleanPDFText cleans up PDF text extraction issues
// Handles cases where each character is separated by spaces or where words are concatenated
// stripRepeatedHeaderFooter removes lines that repeat across most pages at the top/bottom.
func stripRepeatedHeaderFooter(pages []pdfTypes.Page) []pdfTypes.Page {
    if len(pages) < 3 {
        return pages
    }
    type counter struct{ canon string; orig string; count int }
    headerCounts := map[string]*counter{}
    footerCounts := map[string]*counter{}
    // Collect first/last non-empty lines
    for _, p := range pages {
        lines := strings.Split(p.Text, "\n")
        // first non-empty
        for _, ln := range lines {
            l := strings.TrimSpace(ln)
            if l == "" { continue }
            c := canonicalLine(l)
            if c != "" {
                if headerCounts[c] == nil { headerCounts[c] = &counter{canon: c, orig: l} }
                headerCounts[c].count++
            }
            break
        }
        // last non-empty
        for i := len(lines)-1; i >= 0; i-- {
            l := strings.TrimSpace(lines[i])
            if l == "" { continue }
            c := canonicalLine(l)
            if c != "" {
                if footerCounts[c] == nil { footerCounts[c] = &counter{canon: c, orig: l} }
                footerCounts[c].count++
            }
            break
        }
    }
    // Pick header/footer candidates repeated on >=60% pages
    threshold := int(0.6*float64(len(pages)) + 0.5)
    var headerCand, footerCand *counter
    for _, v := range headerCounts {
        if v.count >= threshold && len(v.canon) >= 8 && len(v.canon) <= 200 {
            if headerCand == nil || v.count > headerCand.count { headerCand = v }
        }
    }
    for _, v := range footerCounts {
        if v.count >= threshold && len(v.canon) >= 8 && len(v.canon) <= 200 {
            if footerCand == nil || v.count > footerCand.count { footerCand = v }
        }
    }
    if headerCand == nil && footerCand == nil { return pages }
    // Remove candidates
    cleaned := make([]pdfTypes.Page, 0, len(pages))
    for _, p := range pages {
        lines := strings.Split(p.Text, "\n")
        // header
        if headerCand != nil {
            for idx, ln := range lines {
                l := strings.TrimSpace(ln)
                if l == "" { continue }
                if canonicalLine(l) == headerCand.canon {
                    lines = append(lines[:idx], lines[idx+1:]...)
                    break
                }
                break
            }
        }
        // footer
        if footerCand != nil {
            for i := len(lines)-1; i >= 0; i-- {
                l := strings.TrimSpace(lines[i])
                if l == "" { continue }
                if canonicalLine(l) == footerCand.canon {
                    lines = append(lines[:i], lines[i+1:]...)
                    break
                }
                break
            }
        }
        p.Text = strings.Join(lines, "\n")
        cleaned = append(cleaned, p)
    }
    return cleaned
}

func canonicalLine(s string) string {
    // Collapse internal whitespace and strip non-letters at ends for robust matching
    s = strings.TrimSpace(s)
    if s == "" { return s }
    reSpace := regexp.MustCompile(`\s+`)
    s = reSpace.ReplaceAllString(s, " ")
    return s
}

// stripRepeatedHeaderFooterWithConfig uses configured repeat threshold if provided
func (ps *PDFService) stripRepeatedHeaderFooterWithConfig(pages []pdfTypes.Page) []pdfTypes.Page {
    if ps.config == nil || ps.config.HeaderFooterRepeatThreshold <= 0 {
        return stripRepeatedHeaderFooter(pages)
    }
    // Temporarily override threshold by adjusting page list duplication count
    // Reuse existing implementation by prefiltering candidates using configured threshold.
    // Implement like original but swap 0.6 with configured threshold.
    if len(pages) < 3 {
        return pages
    }
    type counter struct{ canon string; orig string; count int }
    headerCounts := map[string]*counter{}
    footerCounts := map[string]*counter{}
    for _, p := range pages {
        lines := strings.Split(p.Text, "\n")
        for _, ln := range lines {
            l := strings.TrimSpace(ln)
            if l == "" { continue }
            c := canonicalLine(l)
            if c != "" {
                if headerCounts[c] == nil { headerCounts[c] = &counter{canon: c, orig: l} }
                headerCounts[c].count++
            }
            break
        }
        for i := len(lines)-1; i >= 0; i-- {
            l := strings.TrimSpace(lines[i])
            if l == "" { continue }
            c := canonicalLine(l)
            if c != "" {
                if footerCounts[c] == nil { footerCounts[c] = &counter{canon: c, orig: l} }
                footerCounts[c].count++
            }
            break
        }
    }
    thr := ps.config.HeaderFooterRepeatThreshold
    threshold := int(thr*float64(len(pages)) + 0.5)
    var headerCand, footerCand *counter
    for _, v := range headerCounts {
        if v.count >= threshold && len(v.canon) >= 8 && len(v.canon) <= 200 {
            if headerCand == nil || v.count > headerCand.count { headerCand = v }
        }
    }
    for _, v := range footerCounts {
        if v.count >= threshold && len(v.canon) >= 8 && len(v.canon) <= 200 {
            if footerCand == nil || v.count > footerCand.count { footerCand = v }
        }
    }
    if headerCand == nil && footerCand == nil { return pages }
    cleaned := make([]pdfTypes.Page, 0, len(pages))
    for _, p := range pages {
        lines := strings.Split(p.Text, "\n")
        if headerCand != nil {
            for idx, ln := range lines {
                l := strings.TrimSpace(ln)
                if l == "" { continue }
                if canonicalLine(l) == headerCand.canon {
                    lines = append(lines[:idx], lines[idx+1:]...)
                    break
                }
                break
            }
        }
        if footerCand != nil {
            for i := len(lines)-1; i >= 0; i-- {
                l := strings.TrimSpace(lines[i])
                if l == "" { continue }
                if canonicalLine(l) == footerCand.canon {
                    lines = append(lines[:i], lines[i+1:]...)
                    break
                }
                break
            }
        }
        p.Text = strings.Join(lines, "\n")
        cleaned = append(cleaned, p)
    }
    return cleaned
}

// trimTrailingReferences removes trailing pages that look like reference sections.
func (ps *PDFService) trimTrailingReferences(pages []pdfTypes.Page) []pdfTypes.Page {
    if len(pages) == 0 {
        return pages
    }
    density := 0.5
    if ps.config != nil && ps.config.ReferencesCitationDensity > 0 {
        density = ps.config.ReferencesCitationDensity
    }
    // Compile regexes
    reHeading := regexp.MustCompile(`(?i)^\s*(references|bibliography|works\s+cited|literature\s+cited)\s*$`)
    reNumbered := regexp.MustCompile(`^\s*(\[?\d+[\]\.)]|\d+\.)\s+`)
    reAuthorYear := regexp.MustCompile(`\(19\d{2}|20\d{2}\)`) // (1999) or (2021)
    reDOI := regexp.MustCompile(`(?i)\b(doi:|10\.[0-9]{4,9}/\S+)`)
    reURL := regexp.MustCompile(`(?i)https?://\S+|arxiv\.org|pmid`) 

    // Walk from end backwards, mark first non-reference page
    cutIdx := len(pages)
    for i := len(pages) - 1; i >= 0; i-- {
        txt := pages[i].Text
        lines := strings.Split(txt, "\n")
        // Check heading within first few non-empty lines
        headingFound := false
        nonEmpty := 0
        for _, ln := range lines {
            l := strings.TrimSpace(ln)
            if l == "" { continue }
            nonEmpty++
            if reHeading.MatchString(l) {
                headingFound = true
                break
            }
            if nonEmpty >= 5 { // only scan first few lines
                break
            }
        }
        // Compute citation-like line density
        nonEmpty = 0
        matches := 0
        for _, ln := range lines {
            l := strings.TrimSpace(ln)
            if l == "" { continue }
            nonEmpty++
            if reNumbered.MatchString(l) || reAuthorYear.MatchString(l) || reDOI.MatchString(l) || reURL.MatchString(l) {
                matches++
            }
        }
        var pageLooksLikeRefs bool
        if headingFound {
            pageLooksLikeRefs = true
        } else if nonEmpty > 0 && float64(matches)/float64(nonEmpty) >= density {
            pageLooksLikeRefs = true
        }
        if pageLooksLikeRefs {
            cutIdx = i // keep moving cut upwards
        } else {
            break
        }
    }
    if cutIdx < len(pages) {
        return pages[:cutIdx]
    }
    return pages
}
