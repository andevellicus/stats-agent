package handlers

import (
	"net/http"
	"os"
	"path/filepath"
	"stats-agent/database"
	"strings"

	"github.com/gin-gonic/gin"
	"github.com/google/uuid"
	"go.uber.org/zap"
)

type WorkspaceHandler struct {
	store  *database.PostgresStore
	logger *zap.Logger
}

func NewWorkspaceHandler(store *database.PostgresStore, logger *zap.Logger) *WorkspaceHandler {
	return &WorkspaceHandler{
		store:  store,
		logger: logger,
	}
}

// ServeFile serves workspace files only if the requesting user owns the session
func (h *WorkspaceHandler) ServeFile(c *gin.Context) {
	// Get session ID from URL path parameter
	sessionIDStr := c.Param("sessionID")
	sessionID, err := uuid.Parse(sessionIDStr)
	if err != nil {
		h.logger.Warn("Invalid session ID in workspace request",
			zap.String("session_id", sessionIDStr),
			zap.Error(err))
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid session ID"})
		return
	}

	// Get the requesting user's ID from context (set by session middleware)
	userIDValue, exists := c.Get("userID")
	if !exists {
		h.logger.Error("User ID not found in context")
		c.JSON(http.StatusUnauthorized, gin.H{"error": "Unauthorized"})
		return
	}
	userID := userIDValue.(uuid.UUID)

    // Verify the session exists
    session, err := h.store.GetSessionByID(c.Request.Context(), sessionID)
    if err != nil {
		h.logger.Warn("Session not found for workspace access",
			zap.String("session_id", sessionID.String()),
			zap.String("user_id", userID.String()),
			zap.Error(err))
		c.JSON(http.StatusNotFound, gin.H{"error": "Session not found"})
		return
	}

    // Enforce ownership strictly for workspace files; do not auto-claim here
    if session.UserID == nil || *session.UserID != userID {
        h.logger.Warn("Unauthorized workspace access attempt",
            zap.String("session_id", sessionID.String()),
            zap.String("requesting_user_id", userID.String()),
            zap.Any("session_owner_id", session.UserID))
        c.JSON(http.StatusForbidden, gin.H{"error": "Access denied"})
        return
    }

	// Get the file path from the URL (everything after /workspaces/:sessionID/)
	filename := c.Param("filepath")
	if filename == "" || filename == "/" {
		// Prevent directory listing
		c.JSON(http.StatusForbidden, gin.H{"error": "Directory listing not allowed"})
		return
	}

	// Clean the file path to prevent path traversal
	filename = filepath.Clean(filename)

	// Ensure the filename doesn't try to escape the session directory
	if strings.Contains(filename, "..") || filepath.IsAbs(filename) {
		h.logger.Warn("Path traversal attempt detected",
			zap.String("session_id", sessionID.String()),
			zap.String("filename", filename))
		c.JSON(http.StatusForbidden, gin.H{"error": "Invalid file path"})
		return
	}

	// Build the full file path
	workspaceDir := filepath.Join("workspaces", sessionID.String())
	filePath := filepath.Join(workspaceDir, filename)

	// Verify the resolved path is still within the workspace directory
	absWorkspace, err := filepath.Abs(workspaceDir)
	if err != nil {
		h.logger.Error("Failed to resolve workspace directory",
			zap.String("workspace_dir", workspaceDir),
			zap.Error(err))
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Internal server error"})
		return
	}

	absFile, err := filepath.Abs(filePath)
	if err != nil {
		h.logger.Error("Failed to resolve file path",
			zap.String("file_path", filePath),
			zap.Error(err))
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Internal server error"})
		return
	}

	// Ensure the file is within the workspace directory
	if !strings.HasPrefix(absFile, absWorkspace) {
		h.logger.Warn("Path traversal attempt - file outside workspace",
			zap.String("session_id", sessionID.String()),
			zap.String("requested_file", absFile),
			zap.String("workspace", absWorkspace))
		c.JSON(http.StatusForbidden, gin.H{"error": "Access denied"})
		return
	}

	// Check if file exists and is not a directory
	fileInfo, err := os.Stat(filePath)
	if err != nil {
		if os.IsNotExist(err) {
			c.JSON(http.StatusNotFound, gin.H{"error": "File not found"})
		} else {
			h.logger.Error("Failed to stat file",
				zap.String("file_path", filePath),
				zap.Error(err))
			c.JSON(http.StatusInternalServerError, gin.H{"error": "Internal server error"})
		}
		return
	}

	if fileInfo.IsDir() {
		// Prevent directory access
		c.JSON(http.StatusForbidden, gin.H{"error": "Directory access not allowed"})
		return
	}

	// Serve the file
	h.logger.Debug("Serving workspace file",
		zap.String("session_id", sessionID.String()),
		zap.String("filename", filename),
		zap.String("user_id", userID.String()))

	c.File(filePath)
}
