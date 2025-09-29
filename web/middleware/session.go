package middleware

import (
	"database/sql"
	"net/http"
	"stats-agent/database"

	"github.com/gin-gonic/gin"
	"github.com/google/uuid"
)

const SessionCookieName = "stats_agent_session"
const CookieMaxAge = 30 * 24 * 60 * 60 // 30 days

func SessionMiddleware(store *database.PostgresStore) gin.HandlerFunc {
	return func(c *gin.Context) {
		cookie, err := c.Cookie(SessionCookieName)
		var sessionID uuid.UUID
		createNewSession := false

		if err == http.ErrNoCookie {
			createNewSession = true
		} else if err != nil {
			c.AbortWithStatusJSON(http.StatusInternalServerError, gin.H{"error": "Failed to parse session cookie"})
			return
		} else {
			parsedID, parseErr := uuid.Parse(cookie)
			if parseErr != nil {
				// Invalid UUID in cookie, so create a new session
				createNewSession = true
			} else {
				// Check if the session from the cookie exists in the database
				_, dbErr := store.GetSessionByID(c.Request.Context(), parsedID)
				if dbErr != nil {
					if dbErr == sql.ErrNoRows {
						// The session is not in the DB, so we need to create a new one
						createNewSession = true
					} else {
						// A different database error occurred
						c.AbortWithStatusJSON(http.StatusInternalServerError, gin.H{"error": "Failed to verify session"})
						return
					}
				} else {
					// The session is valid
					sessionID = parsedID
				}
			}
		}

		if createNewSession {
			var creationErr error
			sessionID, creationErr = store.CreateSession(c.Request.Context(), nil)
			if creationErr != nil {
				c.AbortWithStatusJSON(http.StatusInternalServerError, gin.H{"error": "Failed to create session"})
				return
			}
			// Set the cookie for the new session
			c.SetCookie(SessionCookieName, sessionID.String(), CookieMaxAge, "/", "", false, true)
		}

		c.Set("sessionID", sessionID)
		c.Next()
	}
}
