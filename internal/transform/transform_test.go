package transform

import (
	"encoding/json"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/a876691666/deepseek-cursor-proxy/internal/config"
	"github.com/a876691666/deepseek-cursor-proxy/internal/store"
)

func defaultConfig(t *testing.T) (config.Config, *store.Store) {
	t.Helper()
	cfg := config.Defaults()
	cfg.MissingReasoningStrategy = "recover"
	dir := t.TempDir()
	st, err := store.New(filepath.Join(dir, "cache.sqlite3"), 0, 0)
	if err != nil {
		t.Fatalf("store: %v", err)
	}
	t.Cleanup(func() { _ = st.Close() })
	return cfg, st
}

func TestStripCursorThinkingBlocks(t *testing.T) {
	in := "<think>secret reasoning</think>\n\nactual answer"
	got := StripCursorThinkingBlocks(in)
	if got != "actual answer" {
		t.Errorf("got %q", got)
	}
	in2 := "<thinking>open block without close"
	got2 := StripCursorThinkingBlocks(in2)
	if got2 != "" {
		t.Errorf("expected open thinking block to be stripped, got %q", got2)
	}
}

func TestExtractTextContentFromList(t *testing.T) {
	content := []any{
		map[string]any{"type": "text", "text": "hello"},
		map[string]any{"type": "text", "text": "world"},
	}
	got := ExtractTextContent(content)
	if got != "hello\nworld" {
		t.Errorf("got %q", got)
	}
}

func TestExtractTextContentImageOmitted(t *testing.T) {
	content := []any{
		map[string]any{"type": "image_url", "image_url": "..."},
		map[string]any{"type": "text", "text": "ok"},
	}
	got := ExtractTextContent(content)
	if !strings.Contains(got, "[image_url omitted by DeepSeek text proxy]") {
		t.Errorf("expected image omission marker, got %q", got)
	}
	if !strings.Contains(got, "ok") {
		t.Errorf("expected text content, got %q", got)
	}
}

func TestNormalizeReasoningEffort(t *testing.T) {
	cases := map[string]string{
		"low":     "high",
		"medium":  "high",
		"high":    "high",
		"max":     "max",
		"xhigh":   "max",
		"unknown": "high",
	}
	for in, want := range cases {
		if got := NormalizeReasoningEffort(in); got != want {
			t.Errorf("NormalizeReasoningEffort(%q)=%q want %q", in, got, want)
		}
	}
}

func TestPrepareUpstreamRequestBasic(t *testing.T) {
	cfg, st := defaultConfig(t)
	payload := map[string]any{
		"model":  "deepseek-v4-pro",
		"stream": true,
		"messages": []any{
			map[string]any{"role": "system", "content": "Be brief."},
			map[string]any{"role": "user", "content": "Hi"},
		},
		"max_completion_tokens": float64(123),
	}
	prepared := PrepareUpstreamRequest(payload, cfg, st, "Bearer xyz")
	if prepared.OriginalModel != "deepseek-v4-pro" {
		t.Errorf("original model: %q", prepared.OriginalModel)
	}
	if prepared.UpstreamModel != "deepseek-v4-pro" {
		t.Errorf("upstream model: %q", prepared.UpstreamModel)
	}
	if v := prepared.Payload["max_tokens"]; v != float64(123) {
		t.Errorf("max_completion_tokens should map to max_tokens, got %v", v)
	}
	if _, ok := prepared.Payload["max_completion_tokens"]; ok {
		t.Errorf("max_completion_tokens should be filtered out")
	}
	options, _ := prepared.Payload["stream_options"].(map[string]any)
	if v, _ := options["include_usage"].(bool); !v {
		t.Errorf("stream_options.include_usage must be true")
	}
	thinking, _ := prepared.Payload["thinking"].(map[string]any)
	if t2, _ := thinking["type"].(string); t2 != "enabled" {
		t.Errorf("thinking type: %v", thinking)
	}
	if effort := prepared.Payload["reasoning_effort"]; effort != "high" {
		t.Errorf("reasoning_effort: %v", effort)
	}
}

func TestPrepareUpstreamRequestUsesConfigFallbackModel(t *testing.T) {
	cfg, st := defaultConfig(t)
	payload := map[string]any{
		"messages": []any{
			map[string]any{"role": "user", "content": "Hello"},
		},
	}
	prepared := PrepareUpstreamRequest(payload, cfg, st, "Bearer x")
	if prepared.UpstreamModel != cfg.UpstreamModel {
		t.Errorf("expected fallback model, got %q", prepared.UpstreamModel)
	}
}

func TestPrepareUpstreamLegacyFunctionsConverted(t *testing.T) {
	cfg, st := defaultConfig(t)
	payload := map[string]any{
		"messages": []any{
			map[string]any{"role": "user", "content": "x"},
		},
		"functions": []any{
			map[string]any{"name": "do", "description": "do it"},
		},
		"function_call": "auto",
	}
	prepared := PrepareUpstreamRequest(payload, cfg, st, "Bearer x")
	tools, _ := prepared.Payload["tools"].([]any)
	if len(tools) != 1 {
		t.Fatalf("expected 1 tool, got %d", len(tools))
	}
	tool, _ := tools[0].(map[string]any)
	if tool["type"] != "function" {
		t.Errorf("tool type: %v", tool["type"])
	}
	if prepared.Payload["tool_choice"] != "auto" {
		t.Errorf("tool_choice: %v", prepared.Payload["tool_choice"])
	}
}

func TestPrepareUpstreamThinkingDisabledStripsReasoning(t *testing.T) {
	cfg, st := defaultConfig(t)
	cfg.Thinking = "disabled"
	payload := map[string]any{
		"messages": []any{
			map[string]any{
				"role":              "assistant",
				"content":           "ok",
				"reasoning_content": "should be stripped",
			},
			map[string]any{"role": "user", "content": "next"},
		},
	}
	prepared := PrepareUpstreamRequest(payload, cfg, st, "Bearer x")
	messages, _ := prepared.Payload["messages"].([]any)
	first, _ := messages[0].(map[string]any)
	if _, ok := first["reasoning_content"]; ok {
		t.Errorf("reasoning_content must be removed when thinking disabled")
	}
}

func TestRecoverFromMissingReasoning(t *testing.T) {
	cfg, st := defaultConfig(t)
	payload := map[string]any{
		"messages": []any{
			map[string]any{"role": "system", "content": "sys"},
			map[string]any{"role": "user", "content": "first"},
			map[string]any{
				"role":    "assistant",
				"content": "",
				"tool_calls": []any{
					map[string]any{
						"id":       "call_a",
						"type":     "function",
						"function": map[string]any{"name": "n", "arguments": "{}"},
					},
				},
			},
			map[string]any{"role": "tool", "tool_call_id": "call_a", "content": "result"},
			map[string]any{"role": "user", "content": "thanks"},
		},
	}
	prepared := PrepareUpstreamRequest(payload, cfg, st, "Bearer x")
	if prepared.MissingReasoningMessages != 0 {
		t.Errorf("expected recover to clear missing, got %d", prepared.MissingReasoningMessages)
	}
	if prepared.RecoveredReasoningMessages == 0 {
		t.Errorf("expected recovered count > 0")
	}
	if prepared.RecoveryNotice == "" {
		t.Errorf("expected recovery notice for first-time recovery")
	}
	messages, _ := prepared.Payload["messages"].([]any)
	// Should keep system, then recovery system, then last user only.
	roles := make([]string, 0, len(messages))
	for _, m := range messages {
		if mm, ok := m.(map[string]any); ok {
			role, _ := mm["role"].(string)
			roles = append(roles, role)
		}
	}
	if !reflect.DeepEqual(roles, []string{"system", "system", "user"}) {
		t.Errorf("recovered roles: %v", roles)
	}
}

func TestStrictMissingReasoningPropagates(t *testing.T) {
	cfg, st := defaultConfig(t)
	cfg.MissingReasoningStrategy = "reject"
	payload := map[string]any{
		"messages": []any{
			map[string]any{"role": "user", "content": "hi"},
			map[string]any{
				"role":    "assistant",
				"content": "",
				"tool_calls": []any{
					map[string]any{
						"id":       "c1",
						"type":     "function",
						"function": map[string]any{"name": "n", "arguments": "{}"},
					},
				},
			},
			map[string]any{"role": "tool", "tool_call_id": "c1", "content": "ok"},
		},
	}
	prepared := PrepareUpstreamRequest(payload, cfg, st, "Bearer x")
	if prepared.MissingReasoningMessages == 0 {
		t.Errorf("expected missing-reasoning detection in strict mode")
	}
	if prepared.RecoveredReasoningMessages != 0 {
		t.Errorf("strict mode must not recover")
	}
}

func TestRestoreReasoningFromCache(t *testing.T) {
	cfg, st := defaultConfig(t)
	authorization := "Bearer xyz"
	upstreamModel := cfg.UpstreamModel
	thinking := map[string]any{"type": "enabled"}
	reasoningEffort := "high"
	scope := ReasoningCacheNamespace(cfg, upstreamModel, thinking, reasoningEffort, authorization)
	priorMessages := []map[string]any{
		{"role": "user", "content": "hi"},
	}
	convScope := store.ConversationScope(priorMessages, scope)
	assistantMessage := map[string]any{
		"role":              "assistant",
		"content":           "",
		"reasoning_content": "the cached reasoning",
		"tool_calls": []any{
			map[string]any{"id": "t1", "type": "function", "function": map[string]any{"name": "n", "arguments": "{}"}},
		},
	}
	if _, err := st.StoreAssistantMessage(assistantMessage, convScope); err != nil {
		t.Fatal(err)
	}
	payload := map[string]any{
		"messages": []any{
			map[string]any{"role": "user", "content": "hi"},
			map[string]any{
				"role":    "assistant",
				"content": "",
				"tool_calls": []any{
					map[string]any{"id": "t1", "type": "function", "function": map[string]any{"name": "n", "arguments": "{}"}},
				},
			},
			map[string]any{"role": "tool", "tool_call_id": "t1", "content": "result"},
			map[string]any{"role": "user", "content": "next"},
		},
	}
	prepared := PrepareUpstreamRequest(payload, cfg, st, authorization)
	if prepared.PatchedReasoningMessages != 1 {
		t.Errorf("expected reasoning_content to be patched in, got %d", prepared.PatchedReasoningMessages)
	}
	messages, _ := prepared.Payload["messages"].([]any)
	assistant, _ := messages[1].(map[string]any)
	if assistant["reasoning_content"] != "the cached reasoning" {
		t.Errorf("reasoning_content not restored: %v", assistant["reasoning_content"])
	}
}

func TestPrefixResponseContent(t *testing.T) {
	payload := map[string]any{
		"choices": []any{
			map[string]any{
				"message": map[string]any{"content": "answer"},
			},
		},
	}
	if !PrefixResponseContent(payload, "PREFIX:") {
		t.Errorf("expected prefix to be applied")
	}
	choices, _ := payload["choices"].([]any)
	first, _ := choices[0].(map[string]any)
	message, _ := first["message"].(map[string]any)
	if message["content"] != "PREFIX:answer" {
		t.Errorf("content: %v", message["content"])
	}
}

func TestRewriteResponseBodyRewritesModel(t *testing.T) {
	cfg, st := defaultConfig(t)
	_ = cfg
	original := map[string]any{
		"model": "deepseek-v4-pro",
		"choices": []any{
			map[string]any{
				"message": map[string]any{
					"role":              "assistant",
					"content":           "ans",
					"reasoning_content": "thoughts",
				},
			},
		},
	}
	body, _ := json.Marshal(original)
	out := RewriteResponseBody(body, "rebrand-pro", st, []map[string]any{{"role": "user", "content": "x"}}, "ns", "PFX:")
	var parsed map[string]any
	if err := json.Unmarshal(out, &parsed); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if parsed["model"] != "rebrand-pro" {
		t.Errorf("model not rebranded: %v", parsed["model"])
	}
	choices, _ := parsed["choices"].([]any)
	first, _ := choices[0].(map[string]any)
	message, _ := first["message"].(map[string]any)
	if !strings.HasPrefix(message["content"].(string), "PFX:") {
		t.Errorf("expected prefix, got %v", message["content"])
	}
}
