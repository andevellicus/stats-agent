package services

import "stats-agent/web/types"

// MessageGroupingService handles the grouping and conversion of messages for UI display.
type MessageGroupingService struct{}

// NewMessageGroupingService creates a new MessageGroupingService instance.
func NewMessageGroupingService() *MessageGroupingService {
	return &MessageGroupingService{}
}

// GroupMessages converts a flat list of messages into grouped format for UI rendering.
// User messages are grouped separately, while assistant and tool messages are grouped together
// as "agent" messages. This grouping enables better visual organization in the chat interface.
func (mgs *MessageGroupingService) GroupMessages(messages []types.ChatMessage) []types.MessageGroup {
	var groups []types.MessageGroup
	i := 0
	for i < len(messages) {
		switch messages[i].Role {
		case "user":
			groups = append(groups, types.MessageGroup{PrimaryRole: "user", Messages: []types.ChatMessage{messages[i]}})
			i++
		case "system":
			groups = append(groups, types.MessageGroup{PrimaryRole: "system", Messages: []types.ChatMessage{messages[i]}})
			i++
		case "assistant", "tool":
			var agentMessages []types.ChatMessage
			for i < len(messages) && (messages[i].Role == "assistant" || messages[i].Role == "tool") {
				agentMessages = append(agentMessages, messages[i])
				i++
			}
			if len(agentMessages) > 0 {
				groups = append(groups, types.MessageGroup{PrimaryRole: "agent", Messages: agentMessages})
			}
		default:
			i++
		}
	}
	return groups
}

// ToAgentMessages converts ChatMessages to AgentMessages by filtering out system messages
// and keeping only user, assistant, and tool roles. This is used when preparing messages
// for the agent conversation loop.
func (mgs *MessageGroupingService) ToAgentMessages(messages []types.ChatMessage) []types.AgentMessage {
	var agentMessages []types.AgentMessage
	for _, message := range messages {
		if message.Role == "user" || message.Role == "assistant" || message.Role == "tool" {
			agentMessages = append(agentMessages, types.AgentMessage{
				Role:    message.Role,
				Content: message.Content,
			})
		}
	}
	return agentMessages
}
