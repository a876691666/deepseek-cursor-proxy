package server

import (
	"bytes"
	"encoding/json"
	"io"
	"log"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"testing"

	"github.com/a876691666/deepseek-cursor-proxy/internal/config"
	"github.com/a876691666/deepseek-cursor-proxy/internal/store"
)

func newTestServer(t *testing.T, upstream *httptest.Server) *Server {
	t.Helper()
	cfg := config.Defaults()
	cfg.UpstreamBaseURL = upstream.URL
	cfg.MissingReasoningStrategy = "recover"
	dir := t.TempDir()
	st, err := store.New(filepath.Join(dir, "cache.sqlite3"), 0, 0)
	if err != nil {
		t.Fatalf("store: %v", err)
	}
	t.Cleanup(func() { _ = st.Close() })
	logger := log.New(io.Discard, "", 0)
	return New(cfg, st, logger)
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
	data, _ := body["data"].([]any)
	if len(data) == 0 {
		t.Fatalf("expected models")
	}
}

func TestPostMissingAuthRejected(t *testing.T) {
	srv := newTestServer(t, httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {})))
	r := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader([]byte(`{"messages":[]}`)))
	w := httptest.NewRecorder()
	srv.ServeHTTP(w, r)
	if w.Code != http.StatusUnauthorized {
		t.Errorf("expected 401, got %d", w.Code)
	}
}

func TestPostInvalidJSON(t *testing.T) {
	srv := newTestServer(t, httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {})))
	r := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader([]byte(`{not-json`)))
	r.Header.Set("Authorization", "Bearer abc")
	w := httptest.NewRecorder()
	srv.ServeHTTP(w, r)
	if w.Code != http.StatusBadRequest {
		t.Errorf("expected 400, got %d", w.Code)
	}
}

func TestPostUnsupportedPath(t *testing.T) {
	srv := newTestServer(t, httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {})))
	r := httptest.NewRequest(http.MethodPost, "/v1/something", bytes.NewReader([]byte(`{}`)))
	w := httptest.NewRecorder()
	srv.ServeHTTP(w, r)
	if w.Code != http.StatusNotFound {
		t.Errorf("expected 404, got %d", w.Code)
	}
}

func TestPostNonStreamingForwardsAndRewritesModel(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Echo the upstream model back via a fake response.
		body, _ := io.ReadAll(r.Body)
		var payload map[string]any
		_ = json.Unmarshal(body, &payload)
		response := map[string]any{
			"id":    "chatcmpl-1",
			"model": payload["model"],
			"choices": []any{
				map[string]any{
					"index": 0,
					"message": map[string]any{
						"role":              "assistant",
						"content":           "answer",
						"reasoning_content": "thinking",
					},
					"finish_reason": "stop",
				},
			},
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(response)
	}))
	defer upstream.Close()
	srv := newTestServer(t, upstream)
	body := []byte(`{
		"model":"deepseek-v4-pro",
		"messages":[{"role":"user","content":"hi"}]
	}`)
	r := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(body))
	r.Header.Set("Authorization", "Bearer xyz")
	w := httptest.NewRecorder()
	srv.ServeHTTP(w, r)
	if w.Code != http.StatusOK {
		t.Fatalf("status=%d body=%s", w.Code, w.Body.String())
	}
	var parsed map[string]any
	if err := json.Unmarshal(w.Body.Bytes(), &parsed); err != nil {
		t.Fatal(err)
	}
	if parsed["model"] != "deepseek-v4-pro" {
		t.Errorf("model: %v", parsed["model"])
	}
	choices, _ := parsed["choices"].([]any)
	first, _ := choices[0].(map[string]any)
	message, _ := first["message"].(map[string]any)
	if message["content"] != "answer" {
		t.Errorf("content: %v", message["content"])
	}
}

func TestPostStreamingForwardsSSE(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		flusher, _ := w.(http.Flusher)
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		chunks := []map[string]any{
			{
				"id":    "x",
				"model": "deepseek-v4-pro",
				"choices": []any{
					map[string]any{
						"index": 0,
						"delta": map[string]any{"role": "assistant", "reasoning_content": "thinking"},
					},
				},
			},
			{
				"id":    "x",
				"model": "deepseek-v4-pro",
				"choices": []any{
					map[string]any{
						"index": 0,
						"delta": map[string]any{"content": "answer"},
					},
				},
			},
			{
				"id":    "x",
				"model": "deepseek-v4-pro",
				"choices": []any{
					map[string]any{
						"index":         0,
						"delta":         map[string]any{},
						"finish_reason": "stop",
					},
				},
			},
		}
		for _, c := range chunks {
			data, _ := json.Marshal(c)
			_, _ = w.Write([]byte("data: "))
			_, _ = w.Write(data)
			_, _ = w.Write([]byte("\n\n"))
			flusher.Flush()
		}
		_, _ = w.Write([]byte("data: [DONE]\n\n"))
		flusher.Flush()
	}))
	defer upstream.Close()
	srv := newTestServer(t, upstream)
	body := []byte(`{
		"model":"deepseek-v4-pro",
		"stream":true,
		"messages":[{"role":"user","content":"hi"}]
	}`)
	r := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(body))
	r.Header.Set("Authorization", "Bearer xyz")
	w := httptest.NewRecorder()
	srv.ServeHTTP(w, r)
	if w.Code != http.StatusOK {
		t.Fatalf("status=%d body=%s", w.Code, w.Body.String())
	}
	output := w.Body.String()
	if !strings.Contains(output, "data: [DONE]") {
		t.Errorf("expected DONE marker; got %q", output)
	}
	// Cursor display adapter mirrors the reasoning into <think>.
	if !strings.Contains(output, "<think>") {
		t.Errorf("expected <think> mirror; got %q", output)
	}
	if !strings.Contains(output, "answer") {
		t.Errorf("expected answer chunk passthrough")
	}
}

func TestPostUpstreamError(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusUnauthorized)
		_, _ = w.Write([]byte(`{"error":{"message":"bad key"}}`))
	}))
	defer upstream.Close()
	srv := newTestServer(t, upstream)
	body := []byte(`{"model":"deepseek-v4-pro","messages":[{"role":"user","content":"hi"}]}`)
	r := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(body))
	r.Header.Set("Authorization", "Bearer xyz")
	w := httptest.NewRecorder()
	srv.ServeHTTP(w, r)
	if w.Code != http.StatusUnauthorized {
		t.Errorf("expected upstream 401 surfaced, got %d", w.Code)
	}
}
