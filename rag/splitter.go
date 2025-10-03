package rag

import (
	"regexp"
	"strings"
)

type SentenceSplitter interface {
	Split(text string) []string
}

type RegexSentenceSplitter struct {
	pattern *regexp.Regexp
}

func NewRegexSentenceSplitter() RegexSentenceSplitter {
	// Splits on sentence end punctuation followed by whitespace/newline
	pattern := regexp.MustCompile(`(?m)(?<=[.!?])\s+`)
	return RegexSentenceSplitter{pattern: pattern}
}

func (s RegexSentenceSplitter) Split(text string) []string {
	trimmed := strings.TrimSpace(text)
	if trimmed == "" {
		return nil
	}

	sentences := s.pattern.Split(trimmed, -1)
	result := make([]string, 0, len(sentences))
	for _, sentence := range sentences {
		sent := strings.TrimSpace(sentence)
		if sent != "" {
			result = append(result, sent)
		}
	}
	if len(result) == 0 {
		return []string{trimmed}
	}
	return result
}
