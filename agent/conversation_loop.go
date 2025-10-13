package agent

import (
	"stats-agent/config"

	"go.uber.org/zap"
)

// ConversationLoop manages the agent's turn loop, error tracking, temperature adjustment, and breaking conditions.
type ConversationLoop struct {
	cfg                *config.Config
	consecutiveErrors  int
	currentTemperature float64 // Dynamic temperature based on error count
	logger             *zap.Logger
}

// NewConversationLoop creates a new conversation loop instance.
func NewConversationLoop(cfg *config.Config, logger *zap.Logger) *ConversationLoop {
	return &ConversationLoop{
		cfg:                cfg,
		consecutiveErrors:  0,
		currentTemperature: cfg.BaseTemperature,
		logger:             logger,
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

// GetCurrentTemperature returns the current temperature based on consecutive errors.
// Temperature increases linearly with each error, capped at MaxTemperature.
func (c *ConversationLoop) GetCurrentTemperature() float64 {
	return c.currentTemperature
}

// RecordError increments the consecutive error counter, increases temperature, and logs it.
func (c *ConversationLoop) RecordError() {
	c.consecutiveErrors++

	// Calculate new temperature: base + (errors * step), capped at max
	newTemp := c.cfg.BaseTemperature + (float64(c.consecutiveErrors) * c.cfg.TemperatureStep)
	if newTemp > c.cfg.MaxTemperature {
		newTemp = c.cfg.MaxTemperature
	}
	c.currentTemperature = newTemp

	c.logger.Debug("Recorded execution error, increasing temperature",
		zap.Int("consecutive_errors", c.consecutiveErrors),
		zap.Float64("new_temperature", c.currentTemperature))
}

// RecordSuccess resets the consecutive error counter and temperature to baseline.
func (c *ConversationLoop) RecordSuccess() {
	if c.consecutiveErrors > 0 {
		c.logger.Debug("Resetting consecutive error count and temperature after successful execution",
			zap.Int("previous_errors", c.consecutiveErrors),
			zap.Float64("previous_temperature", c.currentTemperature),
			zap.Float64("reset_to", c.cfg.BaseTemperature))
		c.consecutiveErrors = 0
		c.currentTemperature = c.cfg.BaseTemperature
	}
}

// GetConsecutiveErrors returns the current consecutive error count.
func (c *ConversationLoop) GetConsecutiveErrors() int {
	return c.consecutiveErrors
}
