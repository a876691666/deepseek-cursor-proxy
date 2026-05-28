package transform

import (
	"testing"
)

func TestPlainTextFromContent(t *testing.T) {
	tests := []struct {
		name     string
		input    any
		expected string
	}{
		{"nil returns empty", nil, ""},
		{"plain string", "hello", "hello"},
		{"output_text list", []any{
			map[string]any{"type": "output_text", "text": "hello"},
			map[string]any{"type": "output_text", "text": " world"},
		}, "hello world"},
		{"input_text list", []any{
			map[string]any{"type": "input_text", "text": "test"},
		}, "test"},
		{"mixed with string items", []any{
			"hello ",
			map[string]any{"type": "output_text", "text": "world"},
		}, "hello world"},
		{"single dict with output_text", map[string]any{
			"type": "output_text", "text": "result",
		}, "result"},
		{"dict without type but with text", map[string]any{
			"text": "result2",
		}, "result2"},
		{"unknown type fallback", 42, "42"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := PlainTextFromContent(tt.input)
			if got != tt.expected {
				t.Errorf("PlainTextFromContent(%v) = %q, want %q", tt.input, got, tt.expected)
			}
		})
	}
}

func TestInputItemsToMessages(t *testing.T) {
	tests := []struct {
		name          string
		input         any
		expectedCount int
		expectedRoles []string
	}{
		{
			name:          "nil input",
			input:         nil,
			expectedCount: 0,
		},
		{
			name:          "string input creates user message",
			input:         "hello world",
			expectedCount: 1,
			expectedRoles: []string{"user"},
		},
		{
			name: "message type items",
			input: []any{
				map[string]any{
					"type":    "message",
					"role":    "user",
					"content": []any{map[string]any{"type": "input_text", "text": "hello"}},
				},
				map[string]any{
					"type":    "message",
					"role":    "assistant",
					"content": []any{map[string]any{"type": "output_text", "text": "hi there"}},
				},
			},
			expectedCount: 2,
			expectedRoles: []string{"user", "assistant"},
		},
		{
			name: "function_call and function_call_output",
			input: []any{
				map[string]any{
					"type":      "function_call",
					"call_id":   "call_1",
					"name":      "read_file",
					"arguments": `{"path":"/foo"}`,
				},
				map[string]any{
					"type":    "function_call_output",
					"call_id": "call_1",
					"output":  "file content here",
				},
			},
			expectedCount: 2,
			expectedRoles: []string{"assistant", "tool"},
		},
		{
			name: "parallel function_calls become one assistant message",
			input: []any{
				map[string]any{
					"type":      "function_call",
					"call_id":   "call_1",
					"name":      "read_file",
					"arguments": `{"path":"/a"}`,
				},
				map[string]any{
					"type":      "function_call",
					"call_id":   "call_2",
					"name":      "read_file",
					"arguments": `{"path":"/b"}`,
				},
			},
			expectedCount: 1,
			expectedRoles: []string{"assistant"},
		},
		{
			name: "developer role maps to system",
			input: []any{
				map[string]any{
					"type":    "message",
					"role":    "developer",
					"content": "system prompt",
				},
			},
			expectedCount: 1,
			expectedRoles: []string{"system"},
		},
		{
			name: "simple role+content objects without type field (Codex format)",
			input: []any{
				map[string]any{"role": "user", "content": "hello"},
				map[string]any{"role": "assistant", "content": "hi there"},
			},
			expectedCount: 2,
			expectedRoles: []string{"user", "assistant"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			msgs, err := InputItemsToMessages(tt.input)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if len(msgs) != tt.expectedCount {
				t.Errorf("got %d messages, want %d", len(msgs), tt.expectedCount)
			}
			for i, expectedRole := range tt.expectedRoles {
				if i < len(msgs) {
					actualRole, _ := msgs[i]["role"].(string)
					if actualRole != expectedRole {
						t.Errorf("message[%d] role = %q, want %q", i, actualRole, expectedRole)
					}
				}
			}
		})
	}
}

func TestChatMessageToOutputItems(t *testing.T) {
	msg := map[string]any{
		"role":    "assistant",
		"content": "Here is the result.",
		"tool_calls": []any{
			map[string]any{
				"id":   "call_abc",
				"type": "function",
				"function": map[string]any{
					"name":      "search",
					"arguments": `{"query":"test"}`,
				},
			},
		},
	}

	items := ChatMessageToOutputItems(msg)
	if len(items) != 2 {
		t.Fatalf("expected 2 output items (message + function_call), got %d", len(items))
	}

	msgItem, _ := items[0].(map[string]any)
	if msgItem["type"] != "message" {
		t.Errorf("first item type = %q, want \"message\"", msgItem["type"])
	}
	if msgItem["role"] != "assistant" {
		t.Errorf("message role = %q, want \"assistant\"", msgItem["role"])
	}

	fcItem, _ := items[1].(map[string]any)
	if fcItem["type"] != "function_call" {
		t.Errorf("second item type = %q, want \"function_call\"", fcItem["type"])
	}
	if fcItem["call_id"] != "call_abc" {
		t.Errorf("call_id = %q, want \"call_abc\"", fcItem["call_id"])
	}
}

func TestBuildResponsesEnvelope(t *testing.T) {
	outputItems := []any{
		map[string]any{
			"id":     "msg_1",
			"type":   "message",
			"role":   "assistant",
			"status": "completed",
			"content": []map[string]any{
				{"type": "output_text", "text": "Hello world"},
			},
		},
	}

	deepseekResp := map[string]any{
		"model": "deepseek-v4-pro",
		"usage": map[string]any{
			"prompt_tokens":     float64(100),
			"completion_tokens": float64(50),
			"total_tokens":      float64(150),
		},
	}

	envelope := BuildResponsesEnvelope(
		"resp_test123",
		"deepseek-v4-pro",
		"resp_prev456",
		outputItems,
		deepseekResp,
	)

	if envelope["id"] != "resp_test123" {
		t.Errorf("id = %q, want \"resp_test123\"", envelope["id"])
	}
	if envelope["object"] != "response" {
		t.Errorf("object = %q, want \"response\"", envelope["object"])
	}
	if envelope["status"] != "completed" {
		t.Errorf("status = %q, want \"completed\"", envelope["status"])
	}
	if envelope["previous_response_id"] != "resp_prev456" {
		t.Errorf("previous_response_id = %q", envelope["previous_response_id"])
	}

	usage, ok := envelope["usage"].(map[string]any)
	if !ok {
		t.Fatal("usage is not a map")
	}
	if usage["input_tokens"].(int) != 100 {
		t.Errorf("input_tokens = %v, want 100", usage["input_tokens"])
	}
	if usage["output_tokens"].(int) != 50 {
		t.Errorf("output_tokens = %v, want 50", usage["output_tokens"])
	}
	if usage["total_tokens"].(int) != 150 {
		t.Errorf("total_tokens = %v, want 150", usage["total_tokens"])
	}
}

func TestBuildResponsesStreamEvents(t *testing.T) {
	responseBody := map[string]any{
		"id":     "resp_test",
		"object": "response",
		"status": "completed",
		"model":  "deepseek-v4-pro",
		"output": []any{
			map[string]any{
				"id":     "msg_1",
				"type":   "message",
				"role":   "assistant",
				"status": "completed",
				"content": []any{
					map[string]any{
						"type":        "output_text",
						"text":        "Hello",
						"annotations": []any{},
					},
				},
			},
		},
	}

	events := BuildResponsesStreamEvents(responseBody)

	// Should have: created + in_progress + output_item.added + content_part.added +
	//             output_text.delta + output_text.done + content_part.done +
	//             output_item.done + completed + [DONE] = 10 events
	if len(events) != 10 {
		t.Errorf("expected 10 events, got %d", len(events))
	}

	if events[0].Event != "response.created" {
		t.Errorf("first event = %q, want \"response.created\"", events[0].Event)
	}
	if events[1].Event != "response.in_progress" {
		t.Errorf("second event = %q, want \"response.in_progress\"", events[1].Event)
	}
	lastEvent := events[len(events)-1]
	if lastEvent.Event != "" || lastEvent.Data != nil {
		t.Errorf("last event should be the [DONE] sentinel")
	}

	// Format some events to verify they produce valid SSE bytes.
	for _, ev := range events[:3] {
		data := FormatSSEEvent(ev)
		if len(data) == 0 {
			t.Errorf("empty SSE bytes for event %q", ev.Event)
		}
	}
}

func TestNormalizeResponsesTool(t *testing.T) {
	// Chat Completions format (function wrapper already present).
	funcTool := map[string]any{
		"type": "function",
		"function": map[string]any{
			"name":        "my_tool",
			"description": "does things",
			"parameters":  map[string]any{"type": "object", "properties": map[string]any{}},
		},
	}
	result := NormalizeResponsesTool(funcTool)
	if len(result) != 1 {
		t.Fatalf("expected 1 tool, got %d", len(result))
	}
	if result[0]["type"] != "function" {
		t.Errorf("function tool type changed to %q", result[0]["type"])
	}
	fn := result[0]["function"].(map[string]any)
	if fn["name"] != "my_tool" {
		t.Errorf("name = %q, want \"my_tool\"", fn["name"])
	}

	// Responses API format (name/description/parameters at top level, no function wrapper).
	responsesTool := map[string]any{
		"type":        "function",
		"name":        "read_file",
		"description": "Read a file from disk",
		"parameters": map[string]any{
			"type": "object",
			"properties": map[string]any{
				"path": map[string]any{"type": "string"},
			},
			"required": []any{"path"},
		},
	}
	responsesResult := NormalizeResponsesTool(responsesTool)
	if len(responsesResult) != 1 {
		t.Fatalf("expected 1 tool from Responses format, got %d", len(responsesResult))
	}
	responsesFn := responsesResult[0]["function"].(map[string]any)
	if responsesFn["name"] != "read_file" {
		t.Errorf("name = %q, want \"read_file\"", responsesFn["name"])
	}
	if responsesFn["description"] != "Read a file from disk" {
		t.Errorf("description = %q", responsesFn["description"])
	}

	// web_search maps to proxy_web_search.
	wsTool := map[string]any{"type": "web_search"}
	wsResult := NormalizeResponsesTool(wsTool)
	if len(wsResult) != 1 {
		t.Fatalf("expected 1 tool from web_search, got %d", len(wsResult))
	}
	wsFn := wsResult[0]["function"].(map[string]any)
	if wsFn["name"] != "proxy_web_search" {
		t.Errorf("web_search mapped to %q, want \"proxy_web_search\"", wsFn["name"])
	}

	// image_generation maps to proxy_image_generate.
	igTool := map[string]any{"type": "image_generation"}
	igResult := NormalizeResponsesTool(igTool)
	if len(igResult) != 1 {
		t.Fatalf("expected 1 tool from image_generation, got %d", len(igResult))
	}
	igFn := igResult[0]["function"].(map[string]any)
	if igFn["name"] != "proxy_image_generate" {
		t.Errorf("image_generation mapped to %q, want \"proxy_image_generate\"", igFn["name"])
	}

	// Function tool with missing name is dropped.
	anonTool := map[string]any{"type": "function", "description": "no name"}
	if result := NormalizeResponsesTool(anonTool); result != nil {
		t.Errorf("expected nil for function tool with missing name, got %v", result)
	}

	// Unsupported types return nil.
	unknown := NormalizeResponsesTool(map[string]any{"type": "custom"})
	if unknown != nil {
		t.Errorf("expected nil for custom tool type, got %v", unknown)
	}
}

func TestBuildChatPayload(t *testing.T) {
	msgs := []map[string]any{{"role": "user", "content": "hello"}}
	tools := []map[string]any{{
		"type":     "function",
		"function": map[string]any{"name": "test_tool"},
	}}

	payload := BuildChatPayload("deepseek-v4-pro", msgs, tools, "high", "enabled", map[string]any{
		"max_output_tokens": float64(4096),
		"temperature":       float64(0.7),
	})

	if payload["model"] != "deepseek-v4-pro" {
		t.Errorf("model = %q", payload["model"])
	}
	if payload["reasoning_effort"] != "high" {
		t.Errorf("reasoning_effort = %q, want \"high\"", payload["reasoning_effort"])
	}
	thinking, ok := payload["thinking"].(map[string]string)
	if !ok || thinking["type"] != "enabled" {
		t.Errorf("thinking config = %v, want {\"type\":\"enabled\"}", payload["thinking"])
	}
	toolsPayload, _ := payload["tools"].([]map[string]any)
	if len(toolsPayload) != 1 {
		t.Errorf("expected 1 tool, got %d", len(toolsPayload))
	}
	if maxTok, ok := payload["max_tokens"].(float64); !ok || int(maxTok) != 4096 {
		t.Errorf("max_tokens = %v, want 4096", payload["max_tokens"])
	}
}

func TestToResponse(t *testing.T) {
	deepseekResponse := map[string]any{
		"id":    "chatcmpl-test",
		"model": "deepseek-v4-pro",
		"choices": []any{
			map[string]any{
				"index": float64(0),
				"message": map[string]any{
					"role":    "assistant",
					"content": "Hello! I can help with that.",
				},
				"finish_reason": "stop",
			},
		},
		"usage": map[string]any{
			"prompt_tokens":     float64(50),
			"completion_tokens": float64(30),
			"total_tokens":      float64(80),
		},
	}

	requestPayload := map[string]any{
		"model": "deepseek-v4-pro",
		"messages": []any{
			map[string]any{"role": "user", "content": "help"},
		},
	}

	result := ToResponse(deepseekResponse, requestPayload, "resp_abc", "", "")

	// Check envelope fields.
	if result["id"] != "resp_abc" {
		t.Errorf("id = %q, want \"resp_abc\"", result["id"])
	}
	if result["object"] != "response" {
		t.Errorf("object = %q, want \"response\"", result["object"])
	}
	if result["status"] != "completed" {
		t.Errorf("status = %q, want \"completed\"", result["status"])
	}
	if result["model"] != "deepseek-v4-pro" {
		t.Errorf("model = %q, want \"deepseek-v4-pro\"", result["model"])
	}

	// Check output items.
	output, _ := result["output"].([]any)
	if len(output) != 1 {
		t.Fatalf("expected 1 output item, got %d", len(output))
	}
	msgItem, _ := output[0].(map[string]any)
	if msgItem["type"] != "message" {
		t.Errorf("output item type = %q, want \"message\"", msgItem["type"])
	}

	// Check usage.
	usage, _ := result["usage"].(map[string]any)
	if usage["input_tokens"].(int) != 50 {
		t.Errorf("input_tokens = %v, want 50", usage["input_tokens"])
	}
	if usage["output_tokens"].(int) != 30 {
		t.Errorf("output_tokens = %v, want 30", usage["output_tokens"])
	}

	// Check output_text.
	if result["output_text"] != "Hello! I can help with that." {
		t.Errorf("output_text = %q", result["output_text"])
	}
}

func TestToResponseWithToolCalls(t *testing.T) {
	deepseekResponse := map[string]any{
		"id": "chatcmpl-test2",
		"choices": []any{
			map[string]any{
				"index": float64(0),
				"message": map[string]any{
					"role":    "assistant",
					"content": "",
					"tool_calls": []any{
						map[string]any{
							"id":   "call_1",
							"type": "function",
							"function": map[string]any{
								"name":      "read_file",
								"arguments": `{"path":"/tmp/test"}`,
							},
						},
					},
				},
			},
		},
		"usage": map[string]any{
			"prompt_tokens":     float64(30),
			"completion_tokens": float64(20),
			"total_tokens":      float64(50),
		},
	}

	result := ToResponse(deepseekResponse, nil, "resp_tool", "", "test-model")

	output, _ := result["output"].([]any)
	// Should have 1 function_call item (no text message since content is empty).
	if len(output) != 1 {
		t.Fatalf("expected 1 output item (function_call), got %d", len(output))
	}
	fc, _ := output[0].(map[string]any)
	if fc["type"] != "function_call" {
		t.Errorf("type = %q, want \"function_call\"", fc["type"])
	}
	if fc["call_id"] != "call_1" {
		t.Errorf("call_id = %q, want \"call_1\"", fc["call_id"])
	}
	if fc["name"] != "read_file" {
		t.Errorf("name = %q, want \"read_file\"", fc["name"])
	}
}

func TestToResponseModelFromPayload(t *testing.T) {
	deepseekResponse := map[string]any{
		"choices": []any{
			map[string]any{
				"index": float64(0),
				"message": map[string]any{
					"role":    "assistant",
					"content": "ok",
				},
			},
		},
	}

	// Model from requestPayload when model param is empty.
	requestPayload := map[string]any{"model": "from-payload"}
	result := ToResponse(deepseekResponse, requestPayload, "resp_x", "", "")
	if result["model"] != "from-payload" {
		t.Errorf("model = %q, want \"from-payload\" (from request payload)", result["model"])
	}
}
