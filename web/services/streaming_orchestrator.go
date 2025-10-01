package services

import (
	"bytes"
	"context"
	"io"
	"log"
	"os"

	"go.uber.org/zap"
)

// StreamingOrchestrator encapsulates the stdout capture, tee, and streaming logic
type StreamingOrchestrator struct {
	originalStdout *os.File
	pipeReader     *os.File
	pipeWriter     *os.File
	buffer         *bytes.Buffer
	logger         *zap.Logger
}

// NewStreamingOrchestrator creates a new orchestrator with proper error handling
func NewStreamingOrchestrator(logger *zap.Logger) (*StreamingOrchestrator, error) {
	r, w, err := os.Pipe()
	if err != nil {
		return nil, err
	}

	return &StreamingOrchestrator{
		originalStdout: os.Stdout,
		pipeReader:     r,
		pipeWriter:     w,
		buffer:         &bytes.Buffer{},
		logger:         logger,
	}, nil
}

// StartCapture redirects stdout to the internal pipe
func (so *StreamingOrchestrator) StartCapture() {
	os.Stdout = so.pipeWriter
	log.SetOutput(so.pipeWriter)
}

// StopCapture restores stdout and closes the write end of the pipe
// This should always be called via defer to ensure cleanup
func (so *StreamingOrchestrator) StopCapture() {
	os.Stdout = so.originalStdout
	log.SetOutput(so.originalStdout)
	if so.pipeWriter != nil {
		so.pipeWriter.Close()
	}
}

// GetTeeReader returns a reader that both streams data and saves to buffer
func (so *StreamingOrchestrator) GetTeeReader() io.Reader {
	return io.TeeReader(so.pipeReader, so.buffer)
}

// GetCapturedContent returns all captured content from the buffer
func (so *StreamingOrchestrator) GetCapturedContent() string {
	return so.buffer.String()
}

// StreamAndWait processes the stream word-by-word and blocks until complete
// This combines the stream processing goroutine into the calling flow
func (so *StreamingOrchestrator) StreamAndWait(
	ctx context.Context,
	streamService *StreamService,
	writeFunc func(StreamData) error,
) {
	teeReader := so.GetTeeReader()
	streamService.ProcessStreamByWord(ctx, teeReader, writeFunc)
}
