package utils

import (
	"os"
	"path/filepath"
	"regexp"
	"strings"

	"github.com/google/uuid"
)

// SanitizeFilename cleans filename for safe storage by removing dangerous characters
// and limiting length. It trims spaces and dots, removes parent directory references,
// and filters out non-alphanumeric characters except for safe punctuation.
func SanitizeFilename(filename string) string {
	sanitized := strings.Trim(filename, " .")
	sanitized = strings.ReplaceAll(sanitized, "..", "")
	reg := regexp.MustCompile(`[^a-zA-Z0-9._\s-]`)
	sanitized = reg.ReplaceAllString(sanitized, "")
	if len(sanitized) > 255 {
		sanitized = sanitized[:255]
	}
	return sanitized
}

// VerifyFileExists checks if file exists at the given path and is not a directory.
// Returns true if the file exists and is a regular file, false otherwise.
func VerifyFileExists(workspaceDir, filename string) bool {
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

// GenerateMessageID creates a unique message identifier using UUID v4.
func GenerateMessageID() string {
	return uuid.New().String()
}
