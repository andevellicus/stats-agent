package agent

import (
	"stats-agent/config"

	"go.uber.org/zap"
)

// ConversationLoop manages the agent's turn loop, error tracking, and breaking conditions.
type ConversationLoop struct {
	cfg               *config.Config
	consecutiveErrors int
	logger            *zap.Logger
}

// NewConversationLoop creates a new conversation loop instance.
func NewConversationLoop(cfg *config.Config, logger *zap.Logger) *ConversationLoop {
	return &ConversationLoop{
		cfg:               cfg,
		consecutiveErrors: 0,
		logger:            logger,
	}
}

// ShouldContinue checks if the loop should continue based on turn count and consecutive errors.
// Returns (shouldContinue, reason). If shouldContinue is false, reason contains the break message.
func (c *ConversationLoop) ShouldContinue(turn int) (bool, string) {
	// Check if we've hit the error limit
	if c.consecutiveErrors >= c.cfg.ConsecutiveErrors {
		c.logger.Warn("Agent produced consecutive errors, breaking loop to request user feedback",
			zap.Int("consecutive_errors", c.cfg.ConsecutiveErrors))
		return false, "Consecutive errors, user feedback needed."
	}

	// Check if we've hit max turns
	if turn >= c.cfg.MaxTurns {
		c.logger.Info("Reached maximum turns limit",
			zap.Int("max_turns", c.cfg.MaxTurns))
		return false, "Maximum turns reached."
	}

	return true, ""
}

// RecordError increments the consecutive error counter and logs it.
func (c *ConversationLoop) RecordError() {
	c.consecutiveErrors++
	c.logger.Debug("Recorded execution error",
		zap.Int("consecutive_errors", c.consecutiveErrors))
}

// RecordSuccess resets the consecutive error counter.
func (c *ConversationLoop) RecordSuccess() {
	if c.consecutiveErrors > 0 {
		c.logger.Debug("Resetting consecutive error count after successful execution")
		c.consecutiveErrors = 0
	}
}

// GetConsecutiveErrors returns the current consecutive error count.
func (c *ConversationLoop) GetConsecutiveErrors() int {
	return c.consecutiveErrors
}
