package web

import (
    "context"
    "net/http"
    "os"
    "stats-agent/agent"
    "stats-agent/config"
    "stats-agent/database"
    "stats-agent/web/handlers"
    "stats-agent/web/middleware"
    "stats-agent/web/services"
    "time"

    "github.com/gin-gonic/gin"
    "go.uber.org/zap"
    neturl "net/url"
    "strconv"
)

type Server struct {
	router *gin.Engine
	agent  *agent.Agent
	logger *zap.Logger
	config *config.Config
	store  *database.PostgresStore
}

func NewServer(agent *agent.Agent, logger *zap.Logger, config *config.Config, store *database.PostgresStore) *Server {
	gin.SetMode(gin.ReleaseMode)

	if err := os.MkdirAll("workspaces", 0755); err != nil {
		logger.Fatal("Failed to create workspaces directory", zap.Error(err))
	}

	router := gin.New()

	router.Use(gin.Recovery())
	router.Use(func(c *gin.Context) {
		c.Set("logger", logger)
		c.Next()
	})

	// Apply the session middleware to all routes
	router.Use(middleware.SessionMiddleware(store))

	server := &Server{
		router: router,
		agent:  agent,
		logger: logger,
		config: config,
		store:  store,
	}

	server.setupRoutes()
	return server
}

func (s *Server) setupRoutes() {
	s.router.Static("/static", "./web/static")
	s.router.Static("/workspaces", "./workspaces")

	// Initialize services
	fileService := services.NewFileService(s.store, s.logger)
	messageService := services.NewMessageService(s.store, s.logger)
	streamService := services.NewStreamService(s.logger)
    pdfConfig := &services.PDFConfig{
        TokenThreshold:           s.config.PDFTokenThreshold,
        FirstPagesPriority:       s.config.PDFFirstPagesPriority,
        EnableTableDetection:     s.config.PDFEnableTableDetection,
        SentenceBoundaryTruncate: s.config.PDFSentenceBoundaryTruncate,
        HeaderFooterRepeatThreshold: s.config.PDFHeaderFooterRepeatThreshold,
        ReferencesTrimEnabled:       s.config.PDFReferencesTrimEnabled,
        ReferencesCitationDensity:   s.config.PDFReferencesCitationDensity,
    }

    // Initialize PDF extractor client (pdfplumber microservice)
    extractorURL := buildPDFExtractorURL(s.config.PDFExtractorURL, s.config)
    pdfExtractorClient := services.NewPDFExtractorClient(
        extractorURL,
        s.config.PDFExtractorTimeout*time.Second,
        s.config.PDFExtractorEnabled,
        s.logger,
    )

	// Check if PDF extractor service is available
	if pdfExtractorClient.IsEnabled() {
		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
		defer cancel()

		if err := pdfExtractorClient.HealthCheck(ctx); err != nil {
			s.logger.Warn("PDF extractor service not available, will use fallback",
				zap.Error(err),
				zap.String("url", s.config.PDFExtractorURL))
		} else {
			s.logger.Info("PDF extractor service ready",
				zap.String("url", s.config.PDFExtractorURL))
		}
	}

	pdfService := services.NewPDFService(s.logger, pdfConfig, pdfExtractorClient)
	chatService := services.NewChatService(s.agent, s.store, s.logger, fileService, messageService, streamService)

	// Initialize rate limiter
	rateLimiterConfig := middleware.RateLimiterConfig{
		MessagesPerMinute: s.config.RateLimitMessagesPerMin,
		FilesPerHour:      s.config.RateLimitFilesPerHour,
		BurstSize:         s.config.RateLimitBurstSize,
		CleanupInterval:   5 * time.Minute,
	}
	rateLimiter := middleware.NewSessionRateLimiter(rateLimiterConfig, s.logger)

	// Initialize handlers with services
	chatHandler := handlers.NewChatHandler(chatService, streamService, pdfService, s.agent, s.config, s.logger, s.store)

	s.router.GET("/", chatHandler.Index)
	s.router.POST("/chat", middleware.RateLimitMiddleware(rateLimiter, "message"), chatHandler.SendMessage)
	s.router.GET("/chat/new", chatHandler.NewChat)
	s.router.GET("/chat/stream", chatHandler.StreamResponse)
	s.router.POST("/chat/stop", chatHandler.StopAgent)
	s.router.GET("/chat/:sessionID", chatHandler.LoadSession)
	s.router.DELETE("/chat/:sessionID", chatHandler.DeleteSession)
}

// buildPDFExtractorURL appends configured tuning params as query args.
func buildPDFExtractorURL(base string, cfg *config.Config) string {
    if base == "" || cfg == nil {
        return base
    }
    u, err := neturl.Parse(base)
    if err != nil {
        return base
    }
    q := u.Query()
    // Add only when set (non-zero/true/non-empty)
    if cfg.PDFExtractorMode != "" { q.Set("mode", cfg.PDFExtractorMode) }
    if cfg.PDFExtractorWordMargin > 0 { q.Set("wm", trimFloat(cfg.PDFExtractorWordMargin)) }
    if cfg.PDFExtractorCharMargin > 0 { q.Set("cm", trimFloat(cfg.PDFExtractorCharMargin)) }
    if cfg.PDFExtractorLineMargin > 0 { q.Set("lm", trimFloat(cfg.PDFExtractorLineMargin)) }
    if cfg.PDFExtractorBoxesFlow != 0 { q.Set("bf", trimFloat(cfg.PDFExtractorBoxesFlow)) }
    if cfg.PDFExtractorUseTextFlow { q.Set("flow", "1") }
    if cfg.PDFExtractorXTolerance > 0 { q.Set("xt", trimFloat(cfg.PDFExtractorXTolerance)) }
    if cfg.PDFExtractorYTolerance > 0 { q.Set("yt", trimFloat(cfg.PDFExtractorYTolerance)) }
    u.RawQuery = q.Encode()
    return u.String()
}

func trimFloat(f float64) string { return strconv.FormatFloat(f, 'f', -1, 64) }

func (s *Server) Start(ctx context.Context, addr string) error {
	s.logger.Info("Starting web server", zap.String("address", addr))

	srv := &http.Server{
		Addr:    addr,
		Handler: s.router,
	}

	go func() {
		if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			s.logger.Error("Web server failed to start", zap.Error(err))
		}
	}()

	<-ctx.Done()

	s.logger.Info("Shutting down web server")
	return srv.Shutdown(context.Background())
}

// StartWorkspaceCleanup runs a background goroutine that periodically cleans up stale sessions
func StartWorkspaceCleanup(cfg *config.Config, cleanupService *CleanupService, logger *zap.Logger) {
	if !cfg.CleanupEnabled {
		logger.Info("Workspace cleanup disabled by configuration")
		return
	}

	logger.Info("Starting workspace cleanup routine",
		zap.Duration("interval", cfg.CleanupInterval),
		zap.Duration("retention_age", cfg.SessionRetentionAge))

	ticker := time.NewTicker(cfg.CleanupInterval)
	defer ticker.Stop()

	// Run cleanup immediately on startup
	runCleanup(cleanupService, cfg, logger)

	// Then run on schedule
	for range ticker.C {
		runCleanup(cleanupService, cfg, logger)
	}
}

// runCleanup executes a single cleanup cycle with timeout
func runCleanup(cleanupService *CleanupService, cfg *config.Config, logger *zap.Logger) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()

	deleted, err := cleanupService.CleanupStaleWorkspaces(ctx, cfg.SessionRetentionAge)
	if err != nil {
		logger.Error("Workspace cleanup failed",
			zap.Error(err),
			zap.Duration("retention_age", cfg.SessionRetentionAge))
		return
	}

	if deleted > 0 {
		logger.Info("Workspace cleanup completed",
			zap.Int("sessions_deleted", deleted),
			zap.Duration("retention_age", cfg.SessionRetentionAge))
	}
}
