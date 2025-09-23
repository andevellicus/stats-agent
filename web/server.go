package web

import (
	"context"
	"net/http"
	"stats-agent/agent"
	"stats-agent/config"
	"stats-agent/web/handlers"

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
	s.router.Static("/workspace", "./workspace")

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
