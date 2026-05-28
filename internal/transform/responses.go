// Package transform handles conversion between OpenAI Responses API and
// DeepSeek Chat Completions API formats for /v1/responses endpoint support.
package transform

import (
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"time"
)

// _textPartTypes is the set of content part types that carry plain text.
var _textPartTypes = map[string]struct{}{
	"output_text": {},
	"input_text":  {},
	"text":        {},
}

// generateID creates a random hex ID with the given prefix.
func generateID(prefix string) string {
	b := make([]byte, 8)
	rand.Read(b)
	return prefix + "_" + hex.EncodeToString(b)
}

// generateResponseID creates a new response_id.
func GenerateResponseID() string {
	return generateID("resp")
}

// itemID creates a new output item ID.
func itemID(prefix string) string {
	return generateID(prefix)
}

// nowUnix returns the current Unix timestamp in seconds.
func nowUnix() int64 {
	return time.Now().Unix()
}

// PlainTextFromContent normalizes Responses-style text content into a plain string.
// Supported formats:
//   - plain string
//   - list of {"type":"output_text"|"input_text"|"text", "text":"..."} objects
//   - a single dict with a text field
func PlainTextFromContent(content any) string {
	if content == nil {
		return ""
	}
	switch v := content.(type) {
	case string:
		return v
	case []any:
		var chunks []string
		for _, item := range v {
			switch it := item.(type) {
			case string:
				chunks = append(chunks, it)
			case map[string]any:
				tp, _ := it["type"].(string)
				if _, ok := _textPartTypes[tp]; ok || it["text"] != nil {
					if txt, ok := it["text"].(string); ok {
						chunks = append(chunks, txt)
					} else if raw := it["text"]; raw != nil {
						b, _ := json.Marshal(raw)
						chunks = append(chunks, string(b))
					}
				}
			}
		}
		result := ""
		for _, c := range chunks {
			result += c
		}
		return result
	case map[string]any:
		tp, _ := v["type"].(string)
		if _, ok := _textPartTypes[tp]; ok || v["text"] != nil {
			if txt, ok := v["text"].(string); ok {
				return txt
			}
			if raw := v["text"]; raw != nil {
				b, _ := json.Marshal(raw)
				return string(b)
			}
		}
		b, _ := json.Marshal(v)
		return string(b)
	default:
		return fmt.Sprintf("%v", v)
	}
}

// normalizeChatRole maps Responses/Codex roles to ChatCompletions roles.
func normalizeChatRole(role string) string {
	switch role {
	case "system", "user", "assistant", "tool", "latest_reminder":
		return role
	case "developer":
		return "system"
	default:
		return "user"
	}
}

// InputItemsToMessages converts Responses API input items to Chat Completions messages.
func InputItemsToMessages(input any) ([]map[string]any, error) {
	if input == nil {
		return nil, nil
	}
	if s, ok := input.(string); ok {
		return []map[string]any{{"role": "user", "content": s}}, nil
	}
	items, ok := input.([]any)
	if !ok {
		return nil, fmt.Errorf("input must be a string or list, got %T", input)
	}

	var messages []map[string]any
	i := 0
	for i < len(items) {
		item, ok := items[i].(map[string]any)
		if !ok {
			i++
			continue
		}
		itemType, _ := item["type"].(string)

		switch itemType {
		case "message":
			role := normalizeChatRole(getString(item, "role", "user"))
			messages = append(messages, messageFromResponseContent(role, item["content"]))
			i++

		case "input_text", "output_text", "text":
			role := normalizeChatRole(getString(item, "role", "user"))
			messages = append(messages, map[string]any{
				"role":    role,
				"content": getString(item, "text", ""),
			})
			i++

		case "function_call":
			var toolCalls []map[string]any
			var contentParts []string

			for i < len(items) {
				nextItem, ok := items[i].(map[string]any)
				if !ok {
					break
				}
				if nt, _ := nextItem["type"].(string); nt != "function_call" {
					break
				}
				callID := getString(nextItem, "call_id", "")
				if callID == "" {
					callID = getString(nextItem, "id", "")
				}
				if callID == "" {
					callID = itemID("call")
				}
				name := getString(nextItem, "name", "")
				arguments := getString(nextItem, "arguments", "")

				if name != "" {
					if c := PlainTextFromContent(nextItem["content"]); c != "" {
						contentParts = append(contentParts, c)
					}
					toolCalls = append(toolCalls, map[string]any{
						"id":   callID,
						"type": "function",
						"function": map[string]any{
							"name":      name,
							"arguments": arguments,
						},
					})
				}
				i++
			}

			if len(toolCalls) > 0 {
				content := ""
				for _, cp := range contentParts {
					if content != "" {
						content += "\n"
					}
					content += cp
				}
				messages = append(messages, map[string]any{
					"role":       "assistant",
					"content":    content,
					"tool_calls": toolCalls,
				})
			}
			// loop continues without i++ since i was already advanced

		case "function_call_output":
			messages = append(messages, map[string]any{
				"role":         "tool",
				"tool_call_id": getString(item, "call_id", ""),
				"content":      PlainTextFromContent(item["output"]),
			})
			i++

		case "reasoning", "summary_text":
			// silently skip unsupported input types
			i++

		default:
			// Codex often sends simple {role, content} objects without a "type" field.
			// Treat these as Chat messages directly.
			if role := getString(item, "role", ""); role != "" || item["content"] != nil {
				messages = append(messages, map[string]any{
					"role":    normalizeChatRole(role),
					"content": PlainTextFromContent(item["content"]),
				})
				i++
				continue
			}
			return nil, fmt.Errorf("unsupported input item type: %s", itemType)
		}
	}
	return messages, nil
}

// messageFromResponseContent converts Responses content blocks into a Chat message.
func messageFromResponseContent(role string, content any) map[string]any {
	msg := map[string]any{
		"role":    role,
		"content": PlainTextFromContent(content),
	}
	return msg
}

// ChatMessageToOutputItems converts a DeepSeek Chat Completions assistant message
// to Responses API output items (as []any for JSON compatibility).
func ChatMessageToOutputItems(message map[string]any) []any {
	var outputItems []any

	content := PlainTextFromContent(message["content"])
	if content != "" {
		outputItems = append(outputItems, map[string]any{
			"id":     itemID("msg"),
			"type":   "message",
			"role":   "assistant",
			"status": "completed",
			"content": []any{
				map[string]any{
					"type":        "output_text",
					"text":        content,
					"annotations": []any{},
				},
			},
		})
	}

	toolCalls, _ := message["tool_calls"].([]any)
	for _, raw := range toolCalls {
		tc, ok := raw.(map[string]any)
		if !ok {
			continue
		}
		fn, _ := tc["function"].(map[string]any)
		fnName := ""
		fnArgs := "{}"
		if fn != nil {
			fnName, _ = fn["name"].(string)
			if a, ok := fn["arguments"].(string); ok {
				fnArgs = a
			}
		}
		outputItems = append(outputItems, map[string]any{
			"id":        itemID("fc"),
			"type":      "function_call",
			"call_id":   getString(tc, "id", itemID("call")),
			"name":      fnName,
			"arguments": fnArgs,
		})
	}

	return outputItems
}

// OutputTextFromItems extracts the concatenated text from output items.
func OutputTextFromItems(items []any) string {
	var chunks []string
	for _, raw := range items {
		item, ok := raw.(map[string]any)
		if !ok {
			continue
		}
		if t, _ := item["type"].(string); t != "message" {
			continue
		}
		if text := PlainTextFromContent(item["content"]); text != "" {
			chunks = append(chunks, text)
		}
	}
	result := ""
	for _, c := range chunks {
		result += c
	}
	return result
}

// BuildResponsesEnvelope constructs the Responses API response format from
// Chat Completions API response data.
func BuildResponsesEnvelope(
	responseID string,
	model string,
	previousResponseID string,
	outputItems []any,
	deepseekResponse map[string]any,
) map[string]any {
	usage, _ := deepseekResponse["usage"].(map[string]any)
	inputTokens := intValue(usage, "prompt_tokens")
	outputTokens := intValue(usage, "completion_tokens")
	totalTokens := intValue(usage, "total_tokens")

	return map[string]any{
		"id":                   responseID,
		"object":               "response",
		"created_at":           nowUnix(),
		"status":               "completed",
		"model":                model,
		"previous_response_id": previousResponseID,
		"output":               outputItems,
		"output_text":          OutputTextFromItems(outputItems),
		"usage": map[string]any{
			"input_tokens":  inputTokens,
			"output_tokens": outputTokens,
			"total_tokens":  totalTokens,
		},
	}
}

// BuildChatPayload builds the Chat Completions payload from Responses API request data.
func BuildChatPayload(
	model string,
	messages []map[string]any,
	tools []map[string]any,
	reasoningEffort string,
	thinkingType string,
	requestPayload map[string]any,
) map[string]any {
	payload := map[string]any{
		"model":    model,
		"messages": messages,
		"stream":   false,
	}

	if thinkingType != "" {
		payload["thinking"] = map[string]string{"type": thinkingType}
	}

	if reasoningEffort != "" {
		payload["reasoning_effort"] = reasoningEffort
	}

	if requestPayload != nil {
		if maxTok := requestPayload["max_output_tokens"]; maxTok != nil {
			payload["max_tokens"] = maxTok
		} else if maxTok := requestPayload["max_tokens"]; maxTok != nil {
			payload["max_tokens"] = maxTok
		}

		for _, key := range []string{"temperature", "top_p", "stop", "response_format"} {
			if v, ok := requestPayload[key]; ok && v != nil {
				payload[key] = v
			}
		}
	}

	if len(tools) > 0 {
		payload["tools"] = tools
	}

	return payload
}

// NormalizeResponsesTool converts a Responses API tool definition into one or more
// Chat Completions tool definitions.
func NormalizeResponsesTool(tool map[string]any) []map[string]any {
	toolType, _ := tool["type"].(string)

	switch toolType {
	case "function":
		// Codex may send tools in either format:
		//   Chat Completions: {type, function: {name, description, parameters}}
		//   Responses API:    {type, name, description, parameters}
		// Always normalize to Chat Completions format for DeepSeek upstream.
		var fnName, fnDesc string
		var fnParams any
		if fn, ok := tool["function"].(map[string]any); ok {
			fnName, _ = fn["name"].(string)
			fnDesc, _ = fn["description"].(string)
			fnParams = fn["parameters"]
		} else {
			fnName, _ = tool["name"].(string)
			fnDesc, _ = tool["description"].(string)
			fnParams = tool["parameters"]
		}
		if fnName == "" {
			return nil
		}
		if fnParams == nil {
			fnParams = map[string]any{"type": "object", "properties": map[string]any{}}
		}
		return []map[string]any{{
			"type": "function",
			"function": map[string]any{
				"name":        fnName,
				"description": fnDesc,
				"parameters":  fnParams,
			},
		}}

	case "web_search":
		return []map[string]any{{
			"type": "function",
			"function": map[string]any{
				"name":        "proxy_web_search",
				"description": "Search the web for current information.",
				"parameters": map[string]any{
					"type": "object",
					"properties": map[string]any{
						"query": map[string]any{"type": "string"},
					},
					"required":             []string{"query"},
					"additionalProperties": false,
				},
			},
		}}

	case "image_generation":
		return []map[string]any{{
			"type": "function",
			"function": map[string]any{
				"name":        "proxy_image_generate",
				"description": "Generate an image using the configured DeepSeek proxy image provider.",
				"parameters": map[string]any{
					"type": "object",
					"properties": map[string]any{
						"prompt":  map[string]any{"type": "string"},
						"size":    map[string]any{"type": "string"},
						"n":       map[string]any{"type": "integer"},
						"quality": map[string]any{"type": "string"},
						"style":   map[string]any{"type": "string"},
					},
					"required":             []string{"prompt"},
					"additionalProperties": false,
				},
			},
		}}

	default:
		// Unsupported tool types (mcp, custom, namespace, etc.) are dropped
		return nil
	}
}

// ResponsesStreamEvent is a single SSE event in Responses API format.
type ResponsesStreamEvent struct {
	Event string
	Data  map[string]any
}

// BuildResponsesStreamEvents generates the complete SSE event sequence for a
// non-streaming response body, producing the sequence Codex expects.
func BuildResponsesStreamEvents(responseBody map[string]any) []ResponsesStreamEvent {
	responseID, _ := responseBody["id"].(string)
	outputItems, _ := responseBody["output"].([]any)
	inProgress := copyMap(responseBody)
	inProgress["status"] = "in_progress"

	var events []ResponsesStreamEvent

	// 1. response.created
	events = append(events, ResponsesStreamEvent{
		Event: "response.created",
		Data: map[string]any{
			"type":     "response.created",
			"response": responseBody,
		},
	})

	// 2. response.in_progress
	events = append(events, ResponsesStreamEvent{
		Event: "response.in_progress",
		Data: map[string]any{
			"type":     "response.in_progress",
			"response": inProgress,
		},
	})

	// 3. For each output item
	for outputIndex, raw := range outputItems {
		item, ok := raw.(map[string]any)
		if !ok {
			continue
		}

		events = append(events, ResponsesStreamEvent{
			Event: "response.output_item.added",
			Data: map[string]any{
				"type":         "response.output_item.added",
				"response_id":  responseID,
				"output_index": outputIndex,
				"item":         item,
			},
		})

		if tp, _ := item["type"].(string); tp == "message" {
			contentList, _ := item["content"].([]any)
			for contentIndex, rawContent := range contentList {
				content, ok := rawContent.(map[string]any)
				if !ok {
					continue
				}
				if ct, _ := content["type"].(string); ct != "output_text" {
					continue
				}

				part := map[string]any{
					"type":        "output_text",
					"text":        "",
					"annotations": []any{},
				}

				events = append(events, ResponsesStreamEvent{
					Event: "response.content_part.added",
					Data: map[string]any{
						"type":          "response.content_part.added",
						"response_id":   responseID,
						"item_id":       getString(item, "id", ""),
						"output_index":  outputIndex,
						"content_index": contentIndex,
						"part":          part,
					},
				})

				text := getString(content, "text", "")

				if text != "" {
					events = append(events, ResponsesStreamEvent{
						Event: "response.output_text.delta",
						Data: map[string]any{
							"type":          "response.output_text.delta",
							"response_id":   responseID,
							"item_id":       getString(item, "id", ""),
							"output_index":  outputIndex,
							"content_index": contentIndex,
							"delta":         text,
						},
					})
				}

				events = append(events, ResponsesStreamEvent{
					Event: "response.output_text.done",
					Data: map[string]any{
						"type":          "response.output_text.done",
						"response_id":   responseID,
						"item_id":       getString(item, "id", ""),
						"output_index":  outputIndex,
						"content_index": contentIndex,
						"text":          text,
					},
				})

				events = append(events, ResponsesStreamEvent{
					Event: "response.content_part.done",
					Data: map[string]any{
						"type":          "response.content_part.done",
						"response_id":   responseID,
						"item_id":       getString(item, "id", ""),
						"output_index":  outputIndex,
						"content_index": contentIndex,
						"part": map[string]any{
							"type":        "output_text",
							"text":        text,
							"annotations": content["annotations"],
						},
					},
				})
			}
		}

		events = append(events, ResponsesStreamEvent{
			Event: "response.output_item.done",
			Data: map[string]any{
				"type":         "response.output_item.done",
				"response_id":  responseID,
				"output_index": outputIndex,
				"item":         item,
			},
		})
	}

	// 4. response.completed
	events = append(events, ResponsesStreamEvent{
		Event: "response.completed",
		Data: map[string]any{
			"type":     "response.completed",
			"response": responseBody,
		},
	})

	// 5. [DONE] sentinel
	events = append(events, ResponsesStreamEvent{
		Event: "",
		Data:  nil, // signals [DONE]
	})

	return events
}

// FormatSSEEvent formats a single SSE event to bytes.
func FormatSSEEvent(event ResponsesStreamEvent) []byte {
	if event.Data == nil {
		return []byte("data: [DONE]\n\n")
	}
	payload, _ := json.Marshal(event.Data)
	return []byte(fmt.Sprintf("event: %s\ndata: %s\n\n", event.Event, string(payload)))
}

// copyMap makes a shallow copy of a map.
func copyMap(src map[string]any) map[string]any {
	dst := make(map[string]any, len(src))
	for k, v := range src {
		dst[k] = v
	}
	return dst
}

// getString safely extracts a string from a map.
func getString(m map[string]any, key, fallback string) string {
	if v, ok := m[key]; ok {
		if s, ok := v.(string); ok {
			return s
		}
	}
	return fallback
}

// intValue safely extracts an int from a map.
func intValue(m map[string]any, key string) int {
	if v, ok := m[key]; ok {
		switch n := v.(type) {
		case float64:
			return int(n)
		case int:
			return n
		case int64:
			return int(n)
		case json.Number:
			i, _ := n.Int64()
			return int(i)
		}
	}
	return 0
}

// ToResponse converts a DeepSeek Chat Completions response to an OpenAI
// Responses API format response. This is the single entry point for the
// /v1/responses endpoint's response conversion.
//
// It extracts the assistant message from choices[0].message, converts it
// to Responses output items, and wraps everything in the Responses envelope
// with id, object, created_at, status, model, output, output_text, and usage.
func ToResponse(
	deepseekResponse map[string]any,
	requestPayload map[string]any,
	responseID string,
	previousResponseID string,
	model string,
) map[string]any {
	// Extract model from request if not explicitly provided.
	if model == "" {
		if m, _ := requestPayload["model"].(string); m != "" {
			model = m
		}
	}

	// Extract assistant message from choices.
	var assistantMessage map[string]any
	if choices, ok := deepseekResponse["choices"].([]any); ok && len(choices) > 0 {
		if choice, ok := choices[0].(map[string]any); ok {
			assistantMessage, _ = choice["message"].(map[string]any)
		}
	}
	if assistantMessage == nil {
		// Fallback: try choices as []map[string]any (alternate unmarshal path).
		if choices, ok := deepseekResponse["choices"].([]map[string]any); ok && len(choices) > 0 {
			assistantMessage, _ = choices[0]["message"].(map[string]any)
		}
	}

	// Build output items from assistant message.
	outputItems := ChatMessageToOutputItems(assistantMessage)

	// Build and return the full Responses API envelope.
	return BuildResponsesEnvelope(
		responseID, model, previousResponseID,
		outputItems, deepseekResponse,
	)
}
