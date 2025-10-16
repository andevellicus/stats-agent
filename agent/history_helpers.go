package agent

import (
	"crypto/sha256"
	"fmt"
	"regexp"
	"sort"
	"strings"

	"stats-agent/web/types"
)

// getCurrentDataset extracts the most recent dataset name from history
// Looks for patterns like pd.read_csv('file.csv') or read_excel('file.xlsx')
func getCurrentDataset(history []types.AgentMessage) string {
	// Search backwards through history for most recent dataset load
	for i := len(history) - 1; i >= 0; i-- {
		msg := history[i]
		if msg.Role != "assistant" && msg.Role != "tool" {
			continue
		}

		// Check for pd.read_csv, read_excel, etc.
		patterns := []string{
			`(?i)read_csv\s*\(\s*['"]([^'"]+\.csv)['"]`,
			`(?i)read_excel\s*\(\s*['"]([^'"]+\.xlsx?)['"]`,
			`(?i)read_table\s*\(\s*['"]([^'"]+)['"]`,
		}

		for _, pattern := range patterns {
			re := regexp.MustCompile(pattern)
			if matches := re.FindStringSubmatch(msg.Content); len(matches) > 1 {
				return matches[1]
			}
		}
	}

	return "unknown.csv"
}

// getCurrentSampleSize extracts the most recent sample size (n) from history
// Looks for patterns like "Shape: (56, 54)" or "n=56" or "observations: 56"
func getCurrentSampleSize(history []types.AgentMessage) int {
	// Search backwards through history
	for i := len(history) - 1; i >= 0; i-- {
		msg := history[i]
		if msg.Role != "tool" {
			continue // Only check tool outputs
		}

		// Pattern 1: Shape: (rows, cols)
		shapePattern := regexp.MustCompile(`(?i)shape[:\s]*\((\d+),\s*\d+\)`)
		if matches := shapePattern.FindStringSubmatch(msg.Content); len(matches) > 1 {
			var n int
			fmt.Sscanf(matches[1], "%d", &n)
			if n > 0 {
				return n
			}
		}

		// Pattern 2: n=56 or N=56
		nPattern := regexp.MustCompile(`(?i)\bn\s*=\s*(\d+)`)
		if matches := nPattern.FindStringSubmatch(msg.Content); len(matches) > 1 {
			var n int
			fmt.Sscanf(matches[1], "%d", &n)
			if n > 0 {
				return n
			}
		}

		// Pattern 3: observations: 56 or obs: 56
		obsPattern := regexp.MustCompile(`(?i)observations?[:\s]+(\d+)`)
		if matches := obsPattern.FindStringSubmatch(msg.Content); len(matches) > 1 {
			var n int
			fmt.Sscanf(matches[1], "%d", &n)
			if n > 0 {
				return n
			}
		}
	}

	return 0
}

// getCurrentSchemaHash extracts and hashes the column list from history
// Looks for df.columns output or df.head() column headers
func getCurrentSchemaHash(history []types.AgentMessage) string {
	columns := extractCurrentColumns(history)
	if len(columns) == 0 {
		return ""
	}

	// Sort columns for consistent hashing
	sorted := make([]string, len(columns))
	copy(sorted, columns)
	sort.Strings(sorted)

	// Hash the sorted column list
	combined := strings.Join(sorted, "|")
	hash := sha256.Sum256([]byte(combined))
	return fmt.Sprintf("%x", hash[:4]) // First 8 hex chars
}

// extractCurrentColumns extracts column names from recent history
func extractCurrentColumns(history []types.AgentMessage) []string {
	// Search backwards through history
	for i := len(history) - 1; i >= 0; i-- {
		msg := history[i]
		if msg.Role != "tool" {
			continue
		}

		// Pattern 1: Index(['col1', 'col2', ...], dtype='object')
		indexPattern := regexp.MustCompile(`Index\(\[([^\]]+)\]`)
		if matches := indexPattern.FindStringSubmatch(msg.Content); len(matches) > 1 {
			return parseColumnList(matches[1])
		}

		// Pattern 2: Columns: ['col1', 'col2', ...]
		colsPattern := regexp.MustCompile(`(?i)columns?[:\s]*\[([^\]]+)\]`)
		if matches := colsPattern.FindStringSubmatch(msg.Content); len(matches) > 1 {
			return parseColumnList(matches[1])
		}

		// Pattern 3: Table header in df.head() output
		// Look for lines with multiple column names separated by spaces
		lines := strings.Split(msg.Content, "\n")
		dataRowPattern := regexp.MustCompile(`^\s*\d+`)
		capitalWordPattern := regexp.MustCompile(`[A-Z_]`)
		for _, line := range lines {
			// Skip if line looks like data row (starts with number or index)
			if dataRowPattern.MatchString(line) {
				continue
			}

			// Check if line has multiple capitalized words (likely column headers)
			words := strings.Fields(line)
			if len(words) >= 3 {
				// Simple heuristic: if line has 3+ words with capitals/underscores, likely headers
				capitalCount := 0
				for _, word := range words {
					if capitalWordPattern.MatchString(word) {
						capitalCount++
					}
				}
				if capitalCount >= len(words)/2 {
					return words
				}
			}
		}
	}

	return []string{}
}

// parseColumnList parses a comma-separated list of quoted column names
func parseColumnList(input string) []string {
	// Extract strings within quotes
	pattern := regexp.MustCompile(`['"]([^'"]+)['"]`)
	matches := pattern.FindAllStringSubmatch(input, -1)

	var columns []string
	for _, match := range matches {
		if len(match) > 1 {
			columns = append(columns, match[1])
		}
	}

	return columns
}
