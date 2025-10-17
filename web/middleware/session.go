package middleware

import (
    "database/sql"
    "net/http"
    "stats-agent/database"
    "strings"

	"github.com/gin-gonic/gin"
	"github.com/google/uuid"
	"go.uber.org/zap"
)

const SessionCookieName = "stats_agent_session"
const UserCookieName = "stats_agent_user"
const CookieMaxAge = 30 * 24 * 60 * 60 // 30 days

// isSecureRequest checks if the request was made over HTTPS
// This handles both direct HTTPS and proxied HTTPS (via X-Forwarded-Proto header)
func isSecureRequest(c *gin.Context) bool {
    // Direct TLS
    if c.Request.TLS != nil {
        return true
    }
    // Standard proxy headers
    if v := c.GetHeader("X-Forwarded-Proto"); v == "https" {
        return true
    }
    if v := c.GetHeader("X-Forwarded-Scheme"); v == "https" {
        return true
    }
    if v := c.GetHeader("X-Forwarded-SSL"); v == "on" || v == "1" || v == "true" {
        return true
    }
    // RFC 7239 Forwarded header: e.g., Forwarded: proto=https; host=example.com
    if fwd := c.GetHeader("Forwarded"); fwd != "" {
        // Case-insensitive contains check
        if strings.Contains(strings.ToLower(fwd), "proto=https") {
            return true
        }
    }
    // Cloudflare header
    if cf := c.GetHeader("Cf-Visitor"); cf != "" {
        // cf-visitor: {"scheme":"https"}
        if strings.Contains(cf, "\"https\"") {
            return true
        }
    }
    return false
}

// setSecureCookie sets a cookie with appropriate security flags
func setSecureCookie(c *gin.Context, name, value string) {
    secure := isSecureRequest(c)
    // Use SameSite=Lax to allow normal navigation and HTMX/SSE flows across
    // common reverse-proxy setups while still mitigating CSRF.
    // Strict can prevent cookies from being sent in legitimate top-level
    // navigations or event stream requests depending on referrers.
    c.SetCookie(name, value, CookieMaxAge, "/", "", secure, true)
    c.SetSameSite(http.SameSiteLaxMode)
}

// SetSecureCookie is a public wrapper for use by handlers
func SetSecureCookie(c *gin.Context, name, value string, maxAge int) {
    secure := isSecureRequest(c)
    c.SetCookie(name, value, maxAge, "/", "", secure, true)
    c.SetSameSite(http.SameSiteLaxMode)
}

func SessionMiddleware(store *database.PostgresStore) gin.HandlerFunc {
	return func(c *gin.Context) {
		// Get logger from context (set by server)
		logger, _ := c.Get("logger")
		zapLogger, _ := logger.(*zap.Logger)

		// First, handle user authentication
		// Only validate existing users, don't create new ones
		userCookie, err := c.Cookie(UserCookieName)
		var userID *uuid.UUID
		

		if err == http.ErrNoCookie {
			// No user cookie - user will be created on first message
			
		} else if err != nil {
			c.AbortWithStatusJSON(http.StatusInternalServerError, gin.H{"error": "Failed to parse user cookie"})
			return
		} else {
			parsedUserID, parseErr := uuid.Parse(userCookie)
			if parseErr != nil {
				// Invalid UUID format in cookie - treat as corrupted, user will be recreated on first message
				if zapLogger != nil {
					zapLogger.Warn("Corrupted user UUID in cookie, will recreate on first message",
						zap.String("cookie_value", userCookie),
						zap.Error(parseErr))
				}
				
			} else {
				// Verify the user exists in the database
				dbErr := store.GetUserByID(c.Request.Context(), parsedUserID)
				if dbErr != nil {
					if dbErr == sql.ErrNoRows {
						// User doesn't exist, will be recreated on first message
						
					} else {
						// Database error during verification
						if zapLogger != nil {
							zapLogger.Error("Failed to verify user from cookie",
								zap.Error(dbErr),
								zap.String("user_id", parsedUserID.String()))
						}
						// Continue anyway - user will be recreated on first message
						
					}
				} else {
					// User exists and is valid
					userID = &parsedUserID
					
				}
			}
		}

		// Now handle session
		// Only validate existing sessions, don't create new ones
		sessionCookie, err := c.Cookie(SessionCookieName)
		var sessionID *uuid.UUID
		

		if err == http.ErrNoCookie {
			// No session cookie - session will be created on first message
			
		} else if err != nil {
			c.AbortWithStatusJSON(http.StatusInternalServerError, gin.H{"error": "Failed to parse session cookie"})
			return
		} else {
			parsedID, parseErr := uuid.Parse(sessionCookie)
			if parseErr != nil {
				// Invalid UUID format in cookie - treat as corrupted, session will be recreated on first message
				if zapLogger != nil {
					zapLogger.Warn("Corrupted session UUID in cookie, will recreate on first message",
						zap.String("cookie_value", sessionCookie),
						zap.Error(parseErr))
				}
				
			} else {
				// Check if the session from the cookie exists in the database
				session, dbErr := store.GetSessionByID(c.Request.Context(), parsedID)
				if dbErr != nil {
					if dbErr == sql.ErrNoRows {
						// Session doesn't exist, will be created on first message
						
					} else {
						// Database error during verification
						if zapLogger != nil {
							zapLogger.Warn("Failed to verify session from cookie, will recreate on first message",
								zap.Error(dbErr),
								zap.String("session_id", parsedID.String()))
						}
						// Continue anyway - session will be recreated on first message
						
					}
				} else {
					// Verify session belongs to this user if user exists
					if userID != nil && session.UserID != nil && *session.UserID == *userID {
						sessionID = &parsedID
						
					} else if userID == nil && session.UserID == nil {
						// Both unowned, session is valid
						sessionID = &parsedID
						
					} else {
						// Session exists but ownership mismatch, will be recreated on first message
						
					}
				}
			}
		}

		// Set context values as pointers - nil means not yet created
		c.Set("userID", userID)
		c.Set("sessionID", sessionID)
		c.Next()
	}
}
