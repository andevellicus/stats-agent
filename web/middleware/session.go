package middleware

import (
	"database/sql"
	"net/http"
	"stats-agent/database"

	"github.com/gin-gonic/gin"
	"github.com/google/uuid"
	"go.uber.org/zap"
)

const SessionCookieName = "stats_agent_session"
const UserCookieName = "stats_agent_user"
const CookieMaxAge = 30 * 24 * 60 * 60 // 30 days

func SessionMiddleware(store *database.PostgresStore) gin.HandlerFunc {
	return func(c *gin.Context) {
		// Get logger from context (set by server)
		logger, _ := c.Get("logger")
		zapLogger, _ := logger.(*zap.Logger)

		// First, handle user authentication
		userCookie, err := c.Cookie(UserCookieName)
		var userID uuid.UUID
		createNewUser := false

		if err == http.ErrNoCookie {
			createNewUser = true
		} else if err != nil {
			c.AbortWithStatusJSON(http.StatusInternalServerError, gin.H{"error": "Failed to parse user cookie"})
			return
		} else {
			parsedUserID, parseErr := uuid.Parse(userCookie)
			if parseErr != nil {
				// Invalid UUID format in cookie - treat as corrupted and create new user
				if zapLogger != nil {
					zapLogger.Warn("Corrupted user UUID in cookie, creating new user",
						zap.String("cookie_value", userCookie),
						zap.Error(parseErr))
				}
				createNewUser = true
			} else {
				// Verify the user exists in the database
				dbErr := store.GetUserByID(c.Request.Context(), parsedUserID)
				if dbErr != nil {
					if dbErr == sql.ErrNoRows {
						createNewUser = true
					} else {
						c.AbortWithStatusJSON(http.StatusInternalServerError, gin.H{"error": "Failed to verify user"})
						return
					}
				} else {
					userID = parsedUserID
				}
			}
		}

		if createNewUser {
			var creationErr error
			userID, creationErr = store.CreateUser(c.Request.Context())
			if creationErr != nil {
				c.AbortWithStatusJSON(http.StatusInternalServerError, gin.H{"error": "Failed to create user"})
				return
			}
			// Set the user cookie with a long expiration
			c.SetCookie(UserCookieName, userID.String(), CookieMaxAge, "/", "", false, true)
		}

		// Now handle session
		sessionCookie, err := c.Cookie(SessionCookieName)
		var sessionID uuid.UUID
		createNewSession := false

		if err == http.ErrNoCookie {
			createNewSession = true
		} else if err != nil {
			c.AbortWithStatusJSON(http.StatusInternalServerError, gin.H{"error": "Failed to parse session cookie"})
			return
		} else {
			parsedID, parseErr := uuid.Parse(sessionCookie)
			if parseErr != nil {
				// Invalid UUID format in cookie - treat as corrupted and create new session
				if zapLogger != nil {
					zapLogger.Warn("Corrupted session UUID in cookie, creating new session",
						zap.String("cookie_value", sessionCookie),
						zap.Error(parseErr))
				}
				createNewSession = true
			} else {
				// Check if the session from the cookie exists in the database
				session, dbErr := store.GetSessionByID(c.Request.Context(), parsedID)
				if dbErr != nil {
					if dbErr == sql.ErrNoRows {
						createNewSession = true
					} else {
						if zapLogger != nil {
							zapLogger.Warn("Failed to verify session, creating new session",
								zap.Error(dbErr),
								zap.String("session_id", parsedID.String()),
								zap.String("user_id", userID.String()))
						}
						createNewSession = true
					}
				} else {
					// Verify session belongs to this user
					if session.UserID != nil && *session.UserID == userID {
						sessionID = parsedID
					} else {
						// Session exists but doesn't belong to this user, create new session
						createNewSession = true
					}
				}
			}
		}

		if createNewSession {
			var creationErr error
			sessionID, creationErr = store.CreateSession(c.Request.Context(), &userID)
			if creationErr != nil {
				c.AbortWithStatusJSON(http.StatusInternalServerError, gin.H{"error": "Failed to create session"})
				return
			}
			c.SetCookie(SessionCookieName, sessionID.String(), CookieMaxAge, "/", "", false, true)
		}

		c.Set("userID", userID)
		c.Set("sessionID", sessionID)
		c.Next()
	}
}
