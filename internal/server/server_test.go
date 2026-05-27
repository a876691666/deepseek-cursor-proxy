package server

import (
	"encoding/json"
	"io"
	"log"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/pocketbase/pocketbase"
	"github.com/pocketbase/pocketbase/core"

	"github.com/a876691666/deepseek-cursor-proxy/internal/config"
	"github.com/a876691666/deepseek-cursor-proxy/internal/store"
)

func newTestServer(t *testing.T, upstream *httptest.Server) *Server {
	t.Helper()
	cfg := config.Defaults()
	cfg.UpstreamBaseURL = upstream.URL
	cfg.MissingReasoningStrategy = "recover"

	dir := t.TempDir()
	pb := pocketbase.NewWithConfig(pocketbase.Config{
		DefaultDataDir:  dir,
		HideStartBanner: true,
	})
	if err := pb.Bootstrap(); err != nil {
		t.Fatalf("pb bootstrap: %v", err)
	}
	t.Cleanup(func() { pb.ResetBootstrapState() })

	// Create reasoning_cache table.
	if _, err := pb.FindCollectionByNameOrId("reasoning_cache"); err != nil {
		c := core.NewBaseCollection("reasoning_cache")
		c.Fields = core.FieldsList{
			&core.TextField{Name: "key", Required: true},
			&core.TextField{Name: "reasoning"},
			&core.TextField{Name: "message_json"},
		}
		if err := pb.Save(c); err != nil {
			t.Fatalf("create collection: %v", err)
		}
	}

	st := store.New(pb, 0, 0)
	logger := log.New(io.Discard, "", 0)
	return New(cfg, st, logger, pb)
}

func TestHealthz(t *testing.T) {
	srv := newTestServer(t, httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {})))
	r := httptest.NewRequest(http.MethodGet, "/v1/healthz", nil)
	w := httptest.NewRecorder()
	srv.ServeHTTP(w, r)
	if w.Code != http.StatusOK {
		t.Errorf("status: %d", w.Code)
	}
	var body map[string]any
	if err := json.Unmarshal(w.Body.Bytes(), &body); err != nil {
		t.Fatal(err)
	}
	if body["ok"] != true {
		t.Errorf("body: %v", body)
	}
}

func TestModels(t *testing.T) {
	srv := newTestServer(t, httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {})))
	r := httptest.NewRequest(http.MethodGet, "/v1/models", nil)
	w := httptest.NewRecorder()
	srv.ServeHTTP(w, r)
	if w.Code != http.StatusOK {
		t.Errorf("status: %d", w.Code)
	}
	var body map[string]any
	if err := json.Unmarshal(w.Body.Bytes(), &body); err != nil {
		t.Fatal(err)
	}
	data, ok := body["data"].([]any)
	if !ok || len(data) == 0 {
		t.Fatalf("expected non-empty models list")
	}
}

func TestPostMissingAuthRejected(t *testing.T) {
	srv := newTestServer(t, httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {})))
	r := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil)
	r.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	srv.ServeHTTP(w, r)
	if w.Code != http.StatusUnauthorized {
		t.Errorf("expected 401, got %d", w.Code)
	}
}

func TestPostWrongPathRejected(t *testing.T) {
	srv := newTestServer(t, httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {})))
	r := httptest.NewRequest(http.MethodPost, "/v1/embeddings", nil)
	r.Header.Set("Authorization", "Bearer sk-test")
	w := httptest.NewRecorder()
	srv.ServeHTTP(w, r)
	if w.Code != http.StatusNotFound {
		t.Errorf("expected 404, got %d", w.Code)
	}
}

func TestChatCompletionSimple(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "Bearer sk-test" {
			http.Error(w, "unauthorized", http.StatusUnauthorized)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		io.WriteString(w, `{"id":"1","object":"chat.completion","model":"test","choices":[{"index":0,"message":{"role":"assistant","content":"hello"},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}`)
	}))
	defer upstream.Close()
	srv := newTestServer(t, upstream)
	body := `{"model":"deepseek-v4-pro","messages":[{"role":"user","content":"hi"}],"stream":false}`
	r := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
	r.Header.Set("Content-Type", "application/json")
	r.Header.Set("Authorization", "Bearer sk-test")
	w := httptest.NewRecorder()
	srv.ServeHTTP(w, r)
	if w.Code != http.StatusOK {
		t.Errorf("status: %d body: %s", w.Code, w.Body.String())
	}
	var resp map[string]any
	json.Unmarshal(w.Body.Bytes(), &resp)
	if resp["model"] != "deepseek-v4-pro" {
		t.Errorf("model: %v", resp["model"])
	}
}

func TestChatCompletionStreaming(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, _ := w.(http.Flusher)
		chunks := []string{
			`data: {"id":"1","object":"chat.completion.chunk","model":"test","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}`,
			`data: {"id":"1","object":"chat.completion.chunk","model":"test","choices":[{"index":0,"delta":{"content":"hello"},"finish_reason":null}]}`,
			`data: {"id":"1","object":"chat.completion.chunk","model":"test","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}`,
			`data: [DONE]`,
		}
		for _, c := range chunks {
			io.WriteString(w, c+"\n\n")
			flusher.Flush()
		}
	}))
	defer upstream.Close()
	srv := newTestServer(t, upstream)
	body := `{"model":"deepseek-v4-pro","messages":[{"role":"user","content":"hi"}],"stream":true}`
	r := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
	r.Header.Set("Content-Type", "application/json")
	r.Header.Set("Authorization", "Bearer sk-test")
	w := httptest.NewRecorder()
	srv.ServeHTTP(w, r)
	if w.Code != http.StatusOK {
		t.Errorf("status: %d", w.Code)
	}
	resp := w.Body.String()
	if !strings.Contains(resp, "data: [DONE]") {
		t.Errorf("missing [DONE] in stream: %s", resp)
	}
}

func TestOptionsReturnsCORS(t *testing.T) {
	srv := newTestServer(t, httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {})))
	srv.Config.CORS = true
	r := httptest.NewRequest(http.MethodOptions, "/v1/chat/completions", nil)
	w := httptest.NewRecorder()
	srv.ServeHTTP(w, r)
	if w.Code != http.StatusNoContent {
		t.Errorf("status: %d", w.Code)
	}
	if w.Header().Get("Access-Control-Allow-Origin") != "*" {
		t.Errorf("missing CORS header")
	}
}

// ---- Anthropic handler tests ----

func newTestServerWithAnthropic(t *testing.T, openaiUpstream, anthropicUpstream *httptest.Server) *Server {
	t.Helper()
	cfg := config.Defaults()
	cfg.UpstreamBaseURL = openaiUpstream.URL
	cfg.AnthropicBaseURL = anthropicUpstream.URL
	cfg.AnthropicAPIPath = "/v1/messages"
	cfg.MissingReasoningStrategy = "recover"

	dir := t.TempDir()
	pb := pocketbase.NewWithConfig(pocketbase.Config{
		DefaultDataDir:  dir,
		HideStartBanner: true,
	})
	if err := pb.Bootstrap(); err != nil {
		t.Fatalf("pb bootstrap: %v", err)
	}
	t.Cleanup(func() { pb.ResetBootstrapState() })

	if _, err := pb.FindCollectionByNameOrId("reasoning_cache"); err != nil {
		c := core.NewBaseCollection("reasoning_cache")
		c.Fields = core.FieldsList{
			&core.TextField{Name: "key", Required: true},
			&core.TextField{Name: "reasoning"},
			&core.TextField{Name: "message_json"},
		}
		if err := pb.Save(c); err != nil {
			t.Fatalf("create collection: %v", err)
		}
	}

	st := store.New(pb, 0, 0)
	logger := log.New(io.Discard, "", 0)
	return New(cfg, st, logger, pb)
}

func TestAnthropicMessagesNotFoundWhenDisabled(t *testing.T) {
	srv := newTestServer(t, httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {})))
	body := `{"model":"claude-sonnet-4-6","max_tokens":100,"messages":[{"role":"user","content":"hi"}]}`
	r := httptest.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader(body))
	r.Header.Set("Content-Type", "application/json")
	r.Header.Set("Authorization", "Bearer sk-test")
	w := httptest.NewRecorder()
	srv.ServeHTTP(w, r)
	if w.Code != http.StatusNotFound {
		t.Errorf("expected 404 for anthropic endpoint when not configured, got %d", w.Code)
	}
}

func TestAnthropicMessagesSimple(t *testing.T) {
	anthropicUpstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "Bearer sk-test" {
			http.Error(w, "unauthorized", http.StatusUnauthorized)
			return
		}
		if r.URL.Path != "/v1/messages" {
			http.Error(w, "wrong path", http.StatusNotFound)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		io.WriteString(w, `{"id":"msg_1","type":"message","role":"assistant","model":"claude-sonnet-4-6","content":[{"type":"text","text":"hello"}],"stop_reason":"end_turn","usage":{"input_tokens":10,"output_tokens":5}}`)
	}))
	defer anthropicUpstream.Close()

	openaiUpstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "should not be called", http.StatusInternalServerError)
	}))
	defer openaiUpstream.Close()

	srv := newTestServerWithAnthropic(t, openaiUpstream, anthropicUpstream)
	body := `{"model":"claude-sonnet-4-6","max_tokens":100,"messages":[{"role":"user","content":"hi"}]}`
	r := httptest.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader(body))
	r.Header.Set("Content-Type", "application/json")
	r.Header.Set("Authorization", "Bearer sk-test")
	w := httptest.NewRecorder()
	srv.ServeHTTP(w, r)
	if w.Code != http.StatusOK {
		t.Errorf("status: %d body: %s", w.Code, w.Body.String())
	}
	var resp map[string]any
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatal(err)
	}
	if resp["model"] != "deepseek-v4-pro" {
		t.Errorf("expected model from config, got %v", resp["model"])
	}
}

func TestAnthropicMessagesStreaming(t *testing.T) {
	anthropicUpstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, _ := w.(http.Flusher)
		events := []string{
			"event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"claude-sonnet-4-6\",\"content\":[],\"usage\":{\"input_tokens\":10}}}\n",
			"event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n",
			"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"hello\"}}\n",
			"event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n",
			"event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":5}}\n",
			"event: message_stop\ndata: {\"type\":\"message_stop\"}\n",
		}
		for _, e := range events {
			io.WriteString(w, e+"\n")
			flusher.Flush()
		}
	}))
	defer anthropicUpstream.Close()

	openaiUpstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "should not be called", http.StatusInternalServerError)
	}))
	defer openaiUpstream.Close()

	srv := newTestServerWithAnthropic(t, openaiUpstream, anthropicUpstream)
	body := `{"model":"claude-sonnet-4-6","max_tokens":100,"messages":[{"role":"user","content":"hi"}],"stream":true}`
	r := httptest.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader(body))
	r.Header.Set("Content-Type", "application/json")
	r.Header.Set("Authorization", "Bearer sk-test")
	w := httptest.NewRecorder()
	srv.ServeHTTP(w, r)
	if w.Code != http.StatusOK {
		t.Errorf("status: %d", w.Code)
	}
	resp := w.Body.String()
	if !strings.Contains(resp, "event: message_start") {
		t.Errorf("missing message_start event: %s", resp)
	}
	if !strings.Contains(resp, "event: message_stop") {
		t.Errorf("missing message_stop event: %s", resp)
	}
	if !strings.Contains(resp, "text_delta") {
		t.Errorf("missing text_delta: %s", resp)
	}
}

func TestAnthropicMessagesModelRewrite(t *testing.T) {
	anthropicUpstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		io.WriteString(w, `{"id":"msg_1","type":"message","role":"assistant","model":"original-deepseek-model","content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn","usage":{"input_tokens":5,"output_tokens":2}}`)
	}))
	defer anthropicUpstream.Close()

	openaiUpstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "should not be called", http.StatusInternalServerError)
	}))
	defer openaiUpstream.Close()

	srv := newTestServerWithAnthropic(t, openaiUpstream, anthropicUpstream)
	body := `{"model":"claude-sonnet-4-6","max_tokens":100,"messages":[{"role":"user","content":"hi"}]}`
	r := httptest.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader(body))
	r.Header.Set("Content-Type", "application/json")
	r.Header.Set("Authorization", "Bearer sk-test")
	w := httptest.NewRecorder()
	srv.ServeHTTP(w, r)
	if w.Code != http.StatusOK {
		t.Errorf("status: %d", w.Code)
	}
	var resp map[string]any
	json.Unmarshal(w.Body.Bytes(), &resp)
	if resp["model"] != "deepseek-v4-pro" {
		t.Errorf("expected model from config, got %v", resp["model"])
	}
}

func TestAnthropicMessagesSSEModelRewrite(t *testing.T) {
	anthropicUpstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, _ := w.(http.Flusher)
		events := []string{
			"event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"upstream-model\",\"content\":[],\"usage\":{\"input_tokens\":5}}}\n",
			"event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":2}}\n",
			"event: message_stop\ndata: {\"type\":\"message_stop\"}\n",
		}
		for _, e := range events {
			io.WriteString(w, e+"\n")
			flusher.Flush()
		}
	}))
	defer anthropicUpstream.Close()

	openaiUpstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "should not be called", http.StatusInternalServerError)
	}))
	defer openaiUpstream.Close()

	srv := newTestServerWithAnthropic(t, openaiUpstream, anthropicUpstream)
	body := `{"model":"claude-sonnet-4-6","max_tokens":100,"messages":[{"role":"user","content":"hi"}],"stream":true}`
	r := httptest.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader(body))
	r.Header.Set("Content-Type", "application/json")
	r.Header.Set("Authorization", "Bearer sk-test")
	w := httptest.NewRecorder()
	srv.ServeHTTP(w, r)
	if w.Code != http.StatusOK {
		t.Errorf("status: %d", w.Code)
	}
	resp := w.Body.String()
	// The SSE data should have the model rewritten.
	if !strings.Contains(resp, `"model":"deepseek-v4-pro"`) {
		t.Errorf("expected model from config in SSE stream, got: %s", resp)
	}
}

func TestAnthropicMessagesMissingAuthRejected(t *testing.T) {
	anthropicUpstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "should not be reached", http.StatusInternalServerError)
	}))
	defer anthropicUpstream.Close()

	openaiUpstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "should not be called", http.StatusInternalServerError)
	}))
	defer openaiUpstream.Close()

	srv := newTestServerWithAnthropic(t, openaiUpstream, anthropicUpstream)
	r := httptest.NewRequest(http.MethodPost, "/v1/messages", nil)
	w := httptest.NewRecorder()
	srv.ServeHTTP(w, r)
	if w.Code != http.StatusUnauthorized {
		t.Errorf("expected 401 for missing auth, got %d", w.Code)
	}
}

func TestAnthropicMessagesUpstreamError(t *testing.T) {
	anthropicUpstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadRequest)
		io.WriteString(w, `{"type":"error","error":{"type":"invalid_request_error","message":"bad request"}}`)
	}))
	defer anthropicUpstream.Close()

	openaiUpstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "should not be called", http.StatusInternalServerError)
	}))
	defer openaiUpstream.Close()

	srv := newTestServerWithAnthropic(t, openaiUpstream, anthropicUpstream)
	body := `{"model":"claude-sonnet-4-6","max_tokens":100,"messages":[{"role":"user","content":"hi"}]}`
	r := httptest.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader(body))
	r.Header.Set("Content-Type", "application/json")
	r.Header.Set("Authorization", "Bearer sk-test")
	w := httptest.NewRecorder()
	srv.ServeHTTP(w, r)
	if w.Code != http.StatusBadRequest {
		t.Errorf("expected 400 from upstream, got %d", w.Code)
	}
}
