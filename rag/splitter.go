package rag

import "strings"

type SentenceSplitter interface {
	Split(text string) []string
}

type RegexSentenceSplitter struct{}

func NewRegexSentenceSplitter() RegexSentenceSplitter {
	return RegexSentenceSplitter{}
}

func (RegexSentenceSplitter) Split(text string) []string {
	trimmed := strings.TrimSpace(text)
	if trimmed == "" {
		return nil
	}

	runes := []rune(trimmed)
	var sentences []string
	var builder strings.Builder

	isBoundary := func(r rune) bool {
		switch r {
		case '.', '!', '?':
			return true
		default:
			return false
		}
	}

	flush := func() {
		if builder.Len() == 0 {
			return
		}
		sentence := strings.TrimSpace(builder.String())
		if sentence != "" {
			sentences = append(sentences, sentence)
		}
		builder.Reset()
	}

	for idx, r := range runes {
		builder.WriteRune(r)
		if !isBoundary(r) {
			continue
		}
		// Look ahead to determine if this is end of sentence
		next := idx + 1
		for next < len(runes) && (runes[next] == ' ' || runes[next] == '\n' || runes[next] == '\t') {
			next++
		}
		if next >= len(runes) || isBoundary(runes[next]) {
			continue
		}
		flush()
	}

	flush()

	if len(sentences) == 0 {
		return []string{trimmed}
	}
	return sentences
}
