package handlers

import (
	"github.com/gin-gonic/gin"
	"go.uber.org/zap"
)

// respondWithError logs the technical error and returns a user-friendly message
func respondWithError(c *gin.Context, statusCode int, technicalError error, userMessage string, logger *zap.Logger, fields ...zap.Field) {
	// Log technical error with context
	if logger != nil {
		fields = append(fields, zap.Error(technicalError))
		logger.Error("Request failed", fields...)
	}

	// Return user-friendly message
	c.JSON(statusCode, gin.H{"error": userMessage})
}

// respondWithClientError returns a client error (no logging needed for validation errors)
func respondWithClientError(c *gin.Context, statusCode int, userMessage string) {
	c.JSON(statusCode, gin.H{"error": userMessage})
}
