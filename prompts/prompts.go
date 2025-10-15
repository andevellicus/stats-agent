package prompts

import _ "embed"

// Embedded prompt files

//go:embed agent_system.txt
var agentSystem string

//go:embed summarize_memory.txt
var summarizeMemory string

//go:embed fact_summary.txt
var factSummary string

//go:embed searchable_summary.txt
var searchableSummary string

//go:embed pdf_key_facts.txt
var pdfKeyFacts string

//go:embed title_generator.txt
var titleGenerator string

//go:embed document_qa.txt
var documentQA string

func AgentSystem() string         { return agentSystem }
func SummarizeMemory() string     { return summarizeMemory }
func FactSummary() string         { return factSummary }
func SearchableSummary() string   { return searchableSummary }
func PDFKeyFacts() string         { return pdfKeyFacts }
func TitleGenerator() string      { return titleGenerator }
func DocumentQA() string          { return documentQA }

