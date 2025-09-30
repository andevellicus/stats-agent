package errors

import (
	"errors"
	"fmt"
)

// Common error types for categorization and handling

var (
	// ErrNotFound indicates a requested resource was not found
	ErrNotFound = errors.New("resource not found")

	// ErrInvalidInput indicates invalid user input
	ErrInvalidInput = errors.New("invalid input")

	// ErrUnauthorized indicates unauthorized access attempt
	ErrUnauthorized = errors.New("unauthorized")

	// ErrServiceUnavailable indicates a required service is unavailable
	ErrServiceUnavailable = errors.New("service unavailable")

	// ErrDatabaseOperation indicates a database operation failed
	ErrDatabaseOperation = errors.New("database operation failed")

	// ErrPythonExecution indicates Python code execution failed
	ErrPythonExecution = errors.New("python execution failed")

	// ErrLLMCommunication indicates LLM communication failed
	ErrLLMCommunication = errors.New("llm communication failed")
)

// WrapError wraps an error with context message and stack
func WrapError(err error, message string) error {
	if err == nil {
		return nil
	}
	return fmt.Errorf("%s: %w", message, err)
}

// WrapErrorf wraps an error with formatted context message
func WrapErrorf(err error, format string, args ...interface{}) error {
	if err == nil {
		return nil
	}
	message := fmt.Sprintf(format, args...)
	return fmt.Errorf("%s: %w", message, err)
}

// IsNotFound checks if error is a not found error
func IsNotFound(err error) bool {
	return errors.Is(err, ErrNotFound)
}

// IsInvalidInput checks if error is an invalid input error
func IsInvalidInput(err error) bool {
	return errors.Is(err, ErrInvalidInput)
}

// IsServiceUnavailable checks if error is a service unavailable error
func IsServiceUnavailable(err error) bool {
	return errors.Is(err, ErrServiceUnavailable)
}
