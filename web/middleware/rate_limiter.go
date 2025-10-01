package middleware

import (
	"net/http"
	"strconv"
	"sync"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/google/uuid"
	"go.uber.org/zap"
)

// RateLimiterConfig holds configuration for rate limiting
type RateLimiterConfig struct {
	MessagesPerMinute int           // Max messages per session per minute
	FilesPerHour      int           // Max file uploads per session per hour
	BurstSize         int           // Allow burst of N requests
	CleanupInterval   time.Duration // How often to clean up old entries
}

// TokenBucket implements a token bucket rate limiter
type TokenBucket struct {
	tokens       float64
	maxTokens    float64
	refillRate   float64 // tokens per second
	lastRefill   time.Time
	mu           sync.Mutex
}

// NewTokenBucket creates a new token bucket
func NewTokenBucket(maxTokens float64, refillRate float64) *TokenBucket {
	return &TokenBucket{
		tokens:     maxTokens,
		maxTokens:  maxTokens,
		refillRate: refillRate,
		lastRefill: time.Now(),
	}
}

// Allow checks if a request can proceed and consumes a token if so
func (tb *TokenBucket) Allow() bool {
	tb.mu.Lock()
	defer tb.mu.Unlock()

	now := time.Now()
	elapsed := now.Sub(tb.lastRefill).Seconds()

	// Refill tokens based on elapsed time
	tb.tokens = min(tb.maxTokens, tb.tokens+(elapsed*tb.refillRate))
	tb.lastRefill = now

	if tb.tokens >= 1.0 {
		tb.tokens -= 1.0
		return true
	}
	return false
}

// Remaining returns the number of tokens remaining
func (tb *TokenBucket) Remaining() int {
	tb.mu.Lock()
	defer tb.mu.Unlock()

	now := time.Now()
	elapsed := now.Sub(tb.lastRefill).Seconds()
	tokens := min(tb.maxTokens, tb.tokens+(elapsed*tb.refillRate))
	return int(tokens)
}

// SessionRateLimiter manages rate limits per session
type SessionRateLimiter struct {
	config         RateLimiterConfig
	messageLimits  map[uuid.UUID]*TokenBucket
	fileLimits     map[uuid.UUID]*TokenBucket
	mu             sync.RWMutex
	logger         *zap.Logger
	stopCleanup    chan struct{}
}

// NewSessionRateLimiter creates a new session-based rate limiter
func NewSessionRateLimiter(config RateLimiterConfig, logger *zap.Logger) *SessionRateLimiter {
	limiter := &SessionRateLimiter{
		config:        config,
		messageLimits: make(map[uuid.UUID]*TokenBucket),
		fileLimits:    make(map[uuid.UUID]*TokenBucket),
		logger:        logger,
		stopCleanup:   make(chan struct{}),
	}

	// Start cleanup goroutine
	go limiter.cleanupRoutine()

	return limiter
}

// cleanupRoutine periodically removes stale entries
func (srl *SessionRateLimiter) cleanupRoutine() {
	ticker := time.NewTicker(srl.config.CleanupInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			srl.cleanup()
		case <-srl.stopCleanup:
			return
		}
	}
}

// cleanup removes rate limiters for sessions that are likely inactive
func (srl *SessionRateLimiter) cleanup() {
	srl.mu.Lock()
	defer srl.mu.Unlock()

	// Simple cleanup: clear all. In production, you might track last access time.
	// Since sessions are stored in cookies and cleaned up separately, this is fine.
	if len(srl.messageLimits) > 1000 {
		srl.logger.Info("Cleaning up rate limiter cache", zap.Int("message_limiters", len(srl.messageLimits)))
		srl.messageLimits = make(map[uuid.UUID]*TokenBucket)
		srl.fileLimits = make(map[uuid.UUID]*TokenBucket)
	}
}

// Stop stops the cleanup routine
func (srl *SessionRateLimiter) Stop() {
	close(srl.stopCleanup)
}

// AllowMessage checks if a message can be sent for the given session
func (srl *SessionRateLimiter) AllowMessage(sessionID uuid.UUID) bool {
	srl.mu.Lock()
	bucket, exists := srl.messageLimits[sessionID]
	if !exists {
		// Create new bucket: MessagesPerMinute tokens, refill at rate/60 per second
		refillRate := float64(srl.config.MessagesPerMinute) / 60.0
		bucket = NewTokenBucket(float64(srl.config.BurstSize), refillRate)
		srl.messageLimits[sessionID] = bucket
	}
	srl.mu.Unlock()

	return bucket.Allow()
}

// AllowFile checks if a file upload can proceed for the given session
func (srl *SessionRateLimiter) AllowFile(sessionID uuid.UUID) bool {
	srl.mu.Lock()
	bucket, exists := srl.fileLimits[sessionID]
	if !exists {
		// Create new bucket: FilesPerHour tokens, refill at rate/3600 per second
		refillRate := float64(srl.config.FilesPerHour) / 3600.0
		bucket = NewTokenBucket(float64(srl.config.FilesPerHour), refillRate)
		srl.fileLimits[sessionID] = bucket
	}
	srl.mu.Unlock()

	return bucket.Allow()
}

// GetMessageLimit returns remaining message tokens for a session
func (srl *SessionRateLimiter) GetMessageLimit(sessionID uuid.UUID) (remaining int, limit int) {
	srl.mu.RLock()
	bucket, exists := srl.messageLimits[sessionID]
	srl.mu.RUnlock()

	if !exists {
		return srl.config.BurstSize, srl.config.BurstSize
	}
	return bucket.Remaining(), srl.config.BurstSize
}

// RateLimitMiddleware creates a Gin middleware for rate limiting
// For "message" type, it checks if a file is being uploaded and applies file rate limiting if so
func RateLimitMiddleware(limiter *SessionRateLimiter, limitType string) gin.HandlerFunc {
	return func(c *gin.Context) {
		sessionIDValue, exists := c.Get("sessionID")
		if !exists {
			// Session middleware should run before this
			c.AbortWithStatusJSON(http.StatusInternalServerError, gin.H{"error": "session not initialized"})
			return
		}

		sessionID := sessionIDValue.(uuid.UUID)
		var allowed bool
		var remaining, limit int
		actualLimitType := limitType

		// For message endpoints, check if a file is being uploaded
		if limitType == "message" {
			_, err := c.FormFile("file")
			if err == nil {
				// File is present, apply file rate limiting instead
				actualLimitType = "file"
			}
		}

		switch actualLimitType {
		case "message":
			allowed = limiter.AllowMessage(sessionID)
			remaining, limit = limiter.GetMessageLimit(sessionID)
		case "file":
			allowed = limiter.AllowFile(sessionID)
			// For files, we don't expose remaining (too complex with hourly buckets)
			remaining, limit = limiter.config.FilesPerHour, limiter.config.FilesPerHour
		default:
			c.AbortWithStatusJSON(http.StatusInternalServerError, gin.H{"error": "unknown limit type"})
			return
		}

		// Add rate limit headers
		c.Header("X-RateLimit-Limit", formatInt(limit))
		c.Header("X-RateLimit-Remaining", formatInt(remaining))

		if !allowed {
			// Get logger from context
			logger, _ := c.Get("logger")
			zapLogger, _ := logger.(*zap.Logger)
			if zapLogger != nil {
				zapLogger.Warn("Rate limit exceeded",
					zap.String("session_id", sessionID.String()),
					zap.String("limit_type", limitType),
					zap.Int("limit", limit))
			}

			c.Header("Retry-After", "60") // Suggest retry after 60 seconds
			c.AbortWithStatusJSON(http.StatusTooManyRequests, gin.H{
				"error":     "rate limit exceeded",
				"limit":     limit,
				"remaining": remaining,
				"retry_after": 60,
			})
			return
		}

		c.Next()
	}
}

// formatInt converts int to string for headers
func formatInt(n int) string {
	return strconv.Itoa(n)
}
