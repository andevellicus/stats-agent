package middleware

import (
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

		if err == http.ErrNoCookie {
			// No cookie, create a new session
			sessionID, err = store.CreateSession(c.Request.Context(), nil) // Assuming CreateSession handles anonymous users
			if err != nil {
				c.AbortWithStatusJSON(http.StatusInternalServerError, gin.H{"error": "Failed to create session"})
				return
			}
			c.SetCookie(SessionCookieName, sessionID.String(), CookieMaxAge, "/", "", false, true)
		} else if err != nil {
			c.AbortWithStatusJSON(http.StatusInternalServerError, gin.H{"error": "Failed to parse session cookie"})
			return
		} else {
			sessionID, err = uuid.Parse(cookie)
			if err != nil {
				c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "Invalid session ID"})
				return
			}
		}

		c.Set("sessionID", sessionID)
		c.Next()
	}
}
