# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a stats-agent application (branded as "Pocket Statistician") written in Go that implements an intelligent statistical analysis agent. The agent executes Python code in isolated sessions, maintains conversational memory through RAG (Retrieval-Augmented Generation), and provides a modern web interface. The application uses multiple specialized LLaMA.cpp servers and maintains a persistent Python execution environment with session isolation.

## Architecture

### Core Components

- **Agent** (`agent/`): Main conversation loop coordinating between LLM, Python execution, and memory management. Uses a stateless message history approach where the full conversation is passed on each turn.
- **RAG** (`rag/`): Vector database storage and retrieval using chromem-go for long-term memory. Automatically generates searchable facts from code execution pairs.
- **Tools** (`tools/`): Python code execution environment with stateful sessions and executor pooling for high availability.
- **Database** (`database/`): PostgreSQL store for sessions, messages, and file tracking. Uses UUIDs for all IDs.
- **Web** (`web/`): Modern web interface using Gin, templ, HTMX, and Tailwind CSS with SSE streaming.
- **Config** (`config/`): Centralized configuration using Viper with YAML files and environment variable overrides.

### Multi-LLM Setup

The application uses **three separate** LLaMA.cpp server instances, each optimized for different tasks:
- **Main LLM** (port 8080): Primary conversational AI with streaming support
- **Embedding LLM** (port 8081): Text embeddings for RAG functionality (uses BGE models typically)
- **Summarization LLM** (port 8082): Context summarization for memory compression

Configuration in `config.yaml`:
```yaml
MAIN_LLM_HOST: "http://localhost:8080"
EMBEDDING_LLM_HOST: "http://localhost:8081"
SUMMARIZATION_LLM_HOST: "http://localhost:8082"
```

### Database Schema

PostgreSQL with the following key tables:
- **users**: Basic user tracking (UUID id, email, created_at)
- **sessions**: Chat sessions (UUID id, user_id nullable, workspace_path, title, is_active, timestamps)
- **messages**: Chat messages (UUID id, session_id, role, content, rendered HTML, created_at, metadata JSONB)
- **files**: File tracking (UUID id, session_id, filename, file_path, file_type, file_size, message_id nullable, created_at)
- **rag_documents**: Vector embeddings for long-term memory (UUID id, document_id, content, embedding, metadata, created_at)

Note: Session state is maintained in-memory by Python executor Docker containers, not in the database.

### Python Execution Protocol

- **Session Management**: Each session gets a dedicated workspace directory under `workspaces/<session_id>/`
- **TCP Protocol**: Custom protocol using `<session_id>|<code><|EOM|>` format
- **Executor Pool**: 5 Docker containers by default (ports 9999, 9998, 9997, 9996, 9995) with automatic failover
- **Session Binding**: Once a session is assigned to an executor, it stays bound to maintain state
- **Stateful Environment**: Python sessions persist variables, imports, and dataframes across executions

The executor in `docker/executor.py` maintains a global `sessions` dict where each session ID maps to its own namespace.

### Memory Management Strategy

The agent automatically manages context windows using a two-tier memory system:

1. **Short-term**: Recent messages in the conversation history (passed to LLM on each turn)
2. **Long-term**: Older messages moved to RAG when context reaches 75% capacity

**Chunking Process** (in `agent/agent.go:manageMemory`):
- When context threshold is hit, the history is cut in half
- Special handling ensures assistant-tool message pairs are never split
- Moved messages are processed by RAG to generate searchable embeddings

**Fact Generation** (in `rag/rag.go:AddMessagesToStore`):
- Assistant + tool message pairs are combined into "facts"
- The summarization LLM creates single-sentence summaries like: "Fact: The dataframe contains columns for age, gender, and side."
- Facts get a 1.3x similarity boost during retrieval

**Query Boosting**:
- Facts: 1.3x boost
- Summaries: 1.5x boost
- Error messages: 0.8x penalty (unless query mentions "error")

## Error Handling Patterns

The codebase follows a **layered error handling strategy** where each architectural layer has consistent patterns for error management:

### 1. Bootstrap Layer (main.go, config/)
- **Pattern**: Use `logger.Fatal()` for critical startup dependencies → immediate exit
- **When**: Database connection, schema setup, essential services (Python executors, RAG)
- **Example**: Database connection failure terminates the application
- **Rationale**: No point continuing if core infrastructure is unavailable

### 2. Data Access Layer (database/, tools/)
- **Pattern**: Always return wrapped errors with context: `fmt.Errorf("operation failed: %w", err)`
- **Logging**: None at this layer - let callers decide how to handle
- **Example**: `database.GetSessionByID()` returns `fmt.Errorf("session not found: %w", sql.ErrNoRows)`
- **Rationale**: Separation of concerns - data layer should not make logging decisions

### 3. Business Logic Layer (agent/, rag/)
- **Pattern**:
  - Return errors for critical failures (LLM communication, code execution)
  - Log warnings for degraded but recoverable states (RAG unavailable, summarization failed)
- **Logging**:
  - `logger.Error()` + break loop for unrecoverable failures
  - `logger.Warn()` for degraded mode, continue execution
- **Example**: RAG query failure logs warning but agent continues without long-term context
- **Rationale**: Maximize availability - graceful degradation is better than failure

### 4. Service Layer (web/services/)
- **Pattern**: Return errors to handlers, log with contextual information
- **Logging**: `logger.Error()` with structured zap fields (session_id, file_count, etc.)
- **Example**: File service logs errors but continues processing remaining operations
- **Rationale**: Services coordinate complex operations - detailed logging aids debugging

### 5. HTTP Handler Layer (web/handlers/, web/middleware/)
- **Pattern**:
  - Middleware: `c.AbortWithStatusJSON(statusCode, gin.H{"error": "message"})` for auth/validation
  - Handlers: `c.JSON(statusCode, gin.H{"error": "message"})` for operational errors
- **Status Codes**:
  - 400: Client errors (invalid input, missing fields)
  - 401/403: Authentication/authorization failures
  - 404: Resource not found
  - 500: Server errors (database failures, internal errors)
- **Logging**: Always log errors with request context (session_id, user_id where available)
- **Rationale**: HTTP layer translates internal errors to user-friendly responses

### Error Wrapping Examples

```go
// Data layer - return wrapped error
func (s *PostgresStore) GetSession(id uuid.UUID) error {
    if err := s.DB.Query(...); err != nil {
        return fmt.Errorf("failed to query session: %w", err)
    }
}

// Business layer - log and handle gracefully
func (a *Agent) Run(ctx context.Context, input string) {
    longTermContext, err := a.rag.Query(ctx, input)
    if err != nil {
        a.logger.Warn("RAG unavailable, continuing without long-term context",
            zap.Error(err))
        // Continue execution with empty context
    }
}

// HTTP layer - return appropriate status
func (h *ChatHandler) SendMessage(c *gin.Context) {
    if err := h.store.CreateMessage(ctx, msg); err != nil {
        h.logger.Error("Failed to save message",
            zap.Error(err),
            zap.String("session_id", sessionID))
        c.JSON(http.StatusInternalServerError, gin.H{
            "error": "Could not save message",
        })
        return
    }
}
```

### Common Error Types

See `errors/errors.go` for predefined error types:
- `ErrNotFound`: Resource not found
- `ErrInvalidInput`: Invalid user input
- `ErrServiceUnavailable`: Required service unavailable
- `ErrDatabaseOperation`: Database operation failed
- `ErrPythonExecution`: Python code execution failed
- `ErrLLMCommunication`: LLM communication failed

## Development Commands

### Running the Application

**Development Mode:**
```bash
go run main.go
# Server starts on :8080 by default
```

**Production Binary:**
```bash
go build -o stats-agent
./stats-agent
```

**Custom Port:**
Set in `config.yaml`:
```yaml
WEB_PORT: 3000
```

Or use environment variable:
```bash
WEB_PORT=3000 go run main.go
```

### Docker Services

Start all backend services (LLMs, Python executors, PostgreSQL):
```bash
cd docker && docker-compose up -d
```

View logs:
```bash
cd docker && docker-compose logs -f [service-name]
# service-name: main-llm, embedding-llm, summarization-llm, python-executor-1, postgres
```

Stop services:
```bash
cd docker && docker-compose down
```

### Template Development

The web interface uses **templ** for type-safe HTML templates. Templates are located in `web/templates/` with `.templ` extensions. The compiled Go files have `_templ.go` suffixes.

Generate templ files after editing `.templ` files:
```bash
~/go/bin/templ generate web/templates
```

Or if templ is in PATH:
```bash
templ generate web/templates
```

### Tailwind CSS

Rebuild CSS after modifying Tailwind classes:
```bash
tailwindcss -i ./web/static/css/input.css -o ./web/static/css/output.css --watch
```

## Key Configuration Parameters

In `config.yaml`:

**Web Server:**
- `WEB_PORT`: Web server port (default: 8080)

**Agent Behavior:**
- `MAX_TURNS`: Maximum conversation turns before requiring user input (default: 30)
- `CONTEXT_LENGTH`: LLM context window size in tokens (default: 16384)
- `CONSECUTIVE_ERRORS`: Error limit before breaking execution loop (default: 5)
- `RAG_RESULTS`: Number of memory items to retrieve from vector DB (default: 3)
- `LLM_REQUEST_TIMEOUT`: Timeout for LLM requests in seconds (default: 300)

**Python Executors:**
- `PYTHON_EXECUTOR_ADDRESSES`: Array of executor addresses for pooling

**Session Cleanup:**
- `CLEANUP_ENABLED`: Enable/disable automatic session cleanup (default: true)
- `CLEANUP_INTERVAL`: Hours between cleanup runs (default: 24)
- `SESSION_RETENTION_AGE`: Hours before inactive sessions are deleted (default: 168 = 7 days)

**Rate Limiting:**
- `RATE_LIMIT_MESSAGES_PER_MIN`: Max messages per session per minute (default: 20)
- `RATE_LIMIT_FILES_PER_HOUR`: Max file uploads per session per hour (default: 10)
- `RATE_LIMIT_BURST_SIZE`: Allow burst of N requests (default: 5)

**PDF Processing:**
- `PDF_TOKEN_THRESHOLD`: Use N% of context window for PDF content (default: 0.75 = 75%)
- `PDF_FIRST_PAGES_PRIORITY`: Keep first N pages if possible (default: 3)
- `PDF_ENABLE_TABLE_DETECTION`: Detect and mark tables in extracted text (default: true)
- `PDF_SENTENCE_BOUNDARY_TRUNCATE`: Truncate at sentence boundaries for better context (default: true)

All config values support environment variable overrides (uppercase names).

## Session Cleanup

The application automatically cleans up inactive sessions and their associated workspaces to prevent disk space issues.

### How It Works

1. **Background Routine**: A goroutine runs in the background on server startup
2. **Scheduled Execution**: Cleanup runs immediately on startup, then on the configured interval
3. **Retention Policy**: Sessions inactive longer than `SESSION_RETENTION_AGE` are deleted
4. **Full Cleanup**: Each deletion removes:
   - Database records (sessions, messages, files via cascade)
   - RAG documents for the session
   - Python executor session bindings
   - Workspace directory and all files

### Configuration

In `config.yaml`:

```yaml
# Disable cleanup entirely
CLEANUP_ENABLED: false

# Run cleanup every hour
CLEANUP_INTERVAL: 1

# Delete sessions after 1 day (for development)
SESSION_RETENTION_AGE: 24

# Delete sessions after 30 days (for production)
SESSION_RETENTION_AGE: 720
```

### Implementation Details

- **Service**: `web/cleanup_service.go` - Reusable cleanup logic
- **Database**: `database.GetStaleSessions()` - Queries sessions by `last_active` timestamp
- **Routine**: `web/server.go:StartWorkspaceCleanup()` - Background scheduler
- **Timeout**: 5-minute timeout per cleanup cycle
- **Logging**: Detailed logs for monitoring cleanup operations

### Manual Deletion

Users can manually delete sessions via the web UI. The same cleanup service is used to ensure consistent deletion across both automatic and manual operations.

## Rate Limiting

The application implements per-session rate limiting to prevent abuse and ensure fair resource usage.

### How It Works

1. **Token Bucket Algorithm**: Uses a token bucket implementation with configurable refill rates
2. **Per-Session Tracking**: Each session has separate rate limit buckets for messages and file uploads
3. **Automatic Detection**: The middleware automatically detects file uploads and applies the appropriate limit
4. **Graceful Responses**: Returns HTTP 429 with `Retry-After` header when limits are exceeded

### Configuration

In `config.yaml`:

```yaml
RATE_LIMIT_MESSAGES_PER_MIN: 20  # Max messages per session per minute
RATE_LIMIT_FILES_PER_HOUR: 10    # Max file uploads per session per hour
RATE_LIMIT_BURST_SIZE: 5         # Allow burst of N requests
```

### Implementation Details

- **Middleware**: `web/middleware/rate_limiter.go` - Token bucket rate limiter with session tracking
- **Token Bucket**: Refills at a constant rate, allowing bursts up to `BurstSize`
- **Cleanup**: Background goroutine removes stale rate limiters every 5 minutes
- **Headers**: Sets `X-RateLimit-Limit` and `X-RateLimit-Remaining` on all responses
- **Dual Limits**: Messages and file uploads have separate limits; the middleware detects files automatically

### Response Format

When rate limited:
```json
{
  "error": "rate limit exceeded",
  "limit": 20,
  "remaining": 0,
  "retry_after": 60
}
```

HTTP headers:
- `X-RateLimit-Limit`: Maximum requests allowed
- `X-RateLimit-Remaining`: Requests remaining in current window
- `Retry-After`: Seconds to wait before retrying (when rate limited)

## Web Interface Architecture

### Stack
- **Gin**: HTTP routing and middleware
- **templ**: Type-safe HTML templating (compiled to Go)
- **HTMX**: Dynamic frontend interactions (hx-post, hx-boost, hx-swap)
- **Tailwind CSS**: Utility-first styling
- **SSE (Server-Sent Events)**: Real-time streaming of agent responses

### Request Flow

1. User submits message → POST `/chat`
2. Handler saves user message to DB and returns HTML with SSE loader component
3. Frontend automatically connects to GET `/chat/stream?session_id=X&user_message_id=Y`
4. Handler captures agent's stdout, forwards LLM chunks directly via SSE (4KB buffers)
5. Frontend receives JSON events: `{type: "chunk", content: "..."}`
6. Frontend renders markdown using marked.js (code blocks shown with syntax highlighting)
7. Custom `<agent_status>` tags are converted to styled HTML components during streaming
8. New files (images, CSVs) are detected and streamed as separate events
9. After stream ends, messages are parsed and saved to DB with pre-rendered HTML

**Important**: Agent execution is decoupled from HTTP connection lifecycle:
- Agent runs with `context.Background()` (10-minute timeout), not HTTP request context
- If user navigates away, streaming stops but agent continues running
- DB save always happens when agent completes, preserving conversation history
- Handler goroutine waits for agent completion before returning (ensures data integrity)

### Message Rendering

Messages have two forms:
- **Content**: Raw markdown text with code blocks (` ```python ... ``` `)
- **Rendered**: Pre-rendered HTML stored in DB to avoid re-rendering on page load

The `processAgentContentForDB` function in `web/handlers/chat.go` converts markdown to HTML using templ components. Legacy messages may contain XML tags (`<python>code</python>`) for backward compatibility.

## Logging

The application uses **Zap** structured logging with dependency injection:
- Logger is initialized in `main.go` via `config.InitLogger(cfg.LogLevel)`
- Logger is passed to all components via constructors (Agent, RAG, Python tool, handlers)
- No global logger state - each component receives its own logger reference
- Log level is configurable via `LOG_LEVEL` in `config.yaml` (options: debug, info, warn, error)
- Default level is `info`
- Automatic cleanup via `defer config.Cleanup()`

Example:
```go
logger.Info("Message", zap.String("key", value), zap.Int("count", n))
logger.Debug("Debug info", zap.String("response", text))
logger.Error("Failed", zap.Error(err))
```

To see debug logs (including LLM responses), set `LOG_LEVEL: debug` in `config.yaml`.

## Agent Execution Flow

1. User input is appended to message history
2. RAG query retrieves relevant long-term context (top 3 results by default)
3. Agent enters turn loop (max 30 turns):
   - Check consecutive error count (break if ≥5)
   - Prepend long-term context to current history
   - Stream LLM response chunk-by-chunk
   - If response contains markdown code blocks (` ```python ... ``` `), extract and execute code
   - Append execution results as "tool" message
   - If error detected, increment consecutive error counter
   - If no code blocks, return (conversation complete)
4. Memory management runs before each turn (moves old messages to RAG at 75% context)

## Python Tool Execution Details

The `StatefulPythonTool` in `tools/python.go` implements:
- **Executor Pool**: Round-robin with health tracking and cooldown after failures
- **Session Affinity**: Sessions stick to their assigned executor to maintain state
- **Initialization**: Pre-loads pandas, numpy, matplotlib, seaborn, scipy on first use
- **Code Block Parsing**: Extracts code from markdown (` ```python ... ``` `) with XML fallback for backward compatibility
- **Output Capture**: Redirects stdout to capture print statements and dataframe outputs
- **Timeout Handling**: 60-second I/O timeout per execution

The Go tool maintains a `sessionAddr map[string]string` to track which executor owns each session.

## Code Format Handling

The system uses a **markdown-first** approach for code blocks (migrated from XML tags in commits c05192c and 4f89869):

### Primary Format: Markdown
- **LLM Output**: System prompt instructs LLM to output ` ```python ... ``` ` code blocks natively
- **Extraction**: `extractMarkdownCode()` in `tools/python.go` parses markdown fences first
- **Streaming**: LLM client (`llmclient/client.go`) detects code fence boundaries and can stop streaming at closing ` ``` `
- **Frontend**: marked.js renders markdown code blocks with syntax highlighting

### Backward Compatibility: XML Tags
- **Legacy Support**: `extractXMLCode()` falls back to parsing `<python>...</python>` tags for old database messages
- **Format Package**: `web/format/format.go` provides utilities for handling both formats
- **Tag Definitions**: Centralized tag constants (PythonTag, ToolTag, AgentStatusTag) in format package
- **Conversion**: `CloseUnbalancedTags()` ensures any incomplete tags from legacy messages are properly closed

### Key Implementation Details
- **Code Extraction Order**: Markdown first (`extractMarkdownCode`), then XML fallback (`extractXMLCode`)
- **Fence Detection**: LLM client tracks opening ` ```python ` and closing ` ``` ` during streaming (lines 242-273 in `llmclient/client.go`)
- **No Conversion**: Markdown stays as markdown; no conversion to XML happens during processing
- **Database Storage**: Messages stored in their original format (markdown for new, XML for legacy)

## Session Management

- Sessions are created automatically on first visit (via middleware)
- Session ID stored in cookie named `stats-agent-session`
- Each session gets a workspace directory: `workspaces/<session_id>/`
- Session cleanup deletes: DB records (cascades to messages), executor binding, workspace directory
- Active sessions are listed in the sidebar (ordered by last_active DESC)

## Important Notes

- CLI mode is currently **disabled** (see line 102 in `main.go` - Run call is commented out)
- The `session_state` table in the database schema is **unused** and can be removed
- Python executor containers must share the `workspaces/` volume with the Go application
- templ components must be regenerated after editing `.templ` files
- The agent uses markdown-to-HTML conversion for assistant messages (via `gomarkdown/markdown`)
- File uploads are restricted to `.csv`, `.xlsx`, `.xls` extensions
- New files created by Python are auto-detected and streamed to the UI as image or download links