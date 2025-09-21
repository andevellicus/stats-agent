package config

import (
	"go.uber.org/zap"
)

var globalLogger *zap.Logger

// InitLogger initializes a Zap logger and returns it
func InitLogger() (*zap.Logger, error) {
	config := zap.NewDevelopmentConfig()
	config.Level = zap.NewAtomicLevelAt(zap.InfoLevel)

	logger, err := config.Build()
	if err != nil {
		return nil, err
	}

	// Store for cleanup purposes
	globalLogger = logger

	return logger, nil
}

// GetLogger returns the global logger instance (for backward compatibility during transition)
func GetLogger() *zap.Logger {
	if globalLogger == nil {
		// Fallback to a basic logger if not initialized
		globalLogger, _ = zap.NewDevelopment()
	}
	return globalLogger
}

// Cleanup flushes any buffered log entries
func Cleanup() {
	if globalLogger != nil {
		globalLogger.Sync()
	}
}