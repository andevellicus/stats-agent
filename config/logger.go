package config

import (
	"strings"

	"go.uber.org/zap"
	"go.uber.org/zap/zapcore"
)

var globalLogger *zap.Logger

// InitLogger initializes a Zap logger with the specified level and returns it
func InitLogger(logLevelStr string) (*zap.Logger, error) {
	config := zap.NewDevelopmentConfig()

	// Parse log level from string
	var level zapcore.Level
	switch strings.ToLower(logLevelStr) {
	case "debug":
		level = zap.DebugLevel
	case "info":
		level = zap.InfoLevel
	case "warn", "warning":
		level = zap.WarnLevel
	case "error":
		level = zap.ErrorLevel
	default:
		level = zap.InfoLevel
	}

	config.Level = zap.NewAtomicLevelAt(level)

	logger, err := config.Build()
	if err != nil {
		return nil, err
	}

	// Store for cleanup purposes
	globalLogger = logger

	return logger, nil
}

// Cleanup flushes any buffered log entries
func Cleanup() {
	if globalLogger != nil {
		globalLogger.Sync()
	}
}