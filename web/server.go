package web

import (
	"context"
	"net/http"
	"os"
	"stats-agent/agent"
	"stats-agent/config"
	"stats-agent/web/handlers"
	"time"

	"github.com/gin-gonic/gin"
	"go.uber.org/zap"
)

type Server struct {
	router *gin.Engine
	agent  *agent.Agent
	logger *zap.Logger
	config *config.Config
}

func NewServer(agent *agent.Agent, logger *zap.Logger, config *config.Config) *Server {
	// Set Gin mode based on environment
	gin.SetMode(gin.ReleaseMode)

	// Ensure the base workspaces directory exists
	if err := os.MkdirAll("workspaces", 0755); err != nil {
		logger.Fatal("Failed to create workspaces directory", zap.Error(err))
	}

	router := gin.New()

	// Add middleware
	router.Use(gin.Recovery())
	router.Use(func(c *gin.Context) {
		// Add logger to context
		c.Set("logger", logger)
		c.Next()
	})

	server := &Server{
		router: router,
		agent:  agent,
		logger: logger,
		config: config,
	}

	server.setupRoutes()
	return server
}

func (s *Server) setupRoutes() {
	// Serve static files
	s.router.Static("/static", "./web/static")
	s.router.Static("/workspaces", "./workspaces")

	// Chat handlers
	chatHandler := handlers.NewChatHandler(s.agent, s.logger)

	// Web routes
	s.router.GET("/", chatHandler.Index)
	s.router.POST("/chat", chatHandler.SendMessage)
	s.router.GET("/chat/stream", chatHandler.StreamResponse)
}

func (s *Server) Start(ctx context.Context, addr string) error {
	s.logger.Info("Starting web server", zap.String("address", addr))

	srv := &http.Server{
		Addr:    addr,
		Handler: s.router,
	}

	// Start server in a goroutine
	go func() {
		if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			s.logger.Error("Web server failed to start", zap.Error(err))
		}
	}()

	// Wait for context cancellation
	<-ctx.Done()

	s.logger.Info("Shutting down web server")
	return srv.Shutdown(context.Background())
}

func StartWorkspaceCleanup(interval time.Duration, maxAge time.Duration, logger *zap.Logger) {
	ticker := time.NewTicker(interval)
	defer ticker.Stop()

	// Create a dummy chat handler to access the cleanup method.
	// This is a bit of a hack, a better solution would be to refactor the session management
	// into its own struct that can be shared between the server and the cleanup routine.
	chatHandler := handlers.NewChatHandler(nil, logger)

	for {
		<-ticker.C
		logger.Info("Running scheduled workspace cleanup")
		chatHandler.CleanupWorkspaces(maxAge, logger)
	}
}
