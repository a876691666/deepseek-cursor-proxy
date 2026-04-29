// Package server implements the OpenAI-compatible HTTP proxy.
package server

import (
	"bufio"
	"bytes"
	"compress/gzip"
	"compress/zlib"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net"
	"net/http"
	"net/url"
	"strings"
	"sync"
	"time"

	"github.com/a876691666/deepseek-cursor-proxy/internal/config"
	"github.com/a876691666/deepseek-cursor-proxy/internal/store"
	"github.com/a876691666/deepseek-cursor-proxy/internal/streaming"
	"github.com/a876691666/deepseek-cursor-proxy/internal/transform"
)

// Server is the proxy HTTP server.
type Server struct {
	Config Config
	Store  *store.Store
	Logger *log.Logger
	Client *http.Client
}

// Config wraps proxy configuration relevant to the HTTP server.
type Config = config.Config

// New constructs a Server with sensible HTTP defaults.
func New(cfg Config, st *store.Store, logger *log.Logger) *Server {
	if logger == nil {
		logger = log.Default()
	}
	timeout := time.Duration(cfg.RequestTimeoutSeconds * float64(time.Second))
	if timeout <= 0 {
		timeout = 300 * time.Second
	}
	return &Server{
		Config: cfg,
		Store:  st,
		Logger: logger,
		Client: &http.Client{
			Timeout: timeout,
			Transport: &http.Transport{
				DialContext: (&net.Dialer{
					Timeout:   30 * time.Second,
					KeepAlive: 30 * time.Second,
				}).DialContext,
				ResponseHeaderTimeout: timeout,
				IdleConnTimeout:       90 * time.Second,
				DisableCompression:    true,
				MaxIdleConns:          100,
			},
		},
	}
}

// ServeHTTP routes incoming requests.
func (s *Server) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	switch r.Method {
	case http.MethodOptions:
		s.handleOptions(w, r)
	case http.MethodGet:
		s.handleGet(w, r)
	case http.MethodPost:
		s.handlePost(w, r)
	default:
		s.writeJSON(w, http.StatusMethodNotAllowed, map[string]any{
			"error": map[string]any{"message": "Method not allowed"},
		})
	}
}

func (s *Server) handleOptions(w http.ResponseWriter, r *http.Request) {
	if s.Config.Verbose {
		s.Logger.Printf("incoming OPTIONS %s from %s", r.URL.Path, clientIP(r))
	}
	s.writeCORSHeaders(w)
	w.WriteHeader(http.StatusNoContent)
}

func (s *Server) handleGet(w http.ResponseWriter, r *http.Request) {
	if s.Config.Verbose {
		s.Logger.Printf("incoming GET %s from %s", r.URL.Path, clientIP(r))
	}
	switch r.URL.Path {
	case "/healthz", "/v1/healthz":
		s.writeJSON(w, http.StatusOK, map[string]any{"ok": true})
	case "/models", "/v1/models":
		s.writeModels(w)
	default:
		s.writeJSON(w, http.StatusNotFound, map[string]any{
			"error": map[string]any{"message": "Not found"},
		})
	}
}

func (s *Server) handlePost(w http.ResponseWriter, r *http.Request) {
	started := time.Now()
	if s.Config.Verbose {
		s.Logger.Printf(
			"incoming POST %s from %s content_length=%s user_agent=%s",
			r.URL.Path, clientIP(r),
			r.Header.Get("Content-Length"),
			r.Header.Get("User-Agent"),
		)
	}
	if r.URL.Path != "/chat/completions" && r.URL.Path != "/v1/chat/completions" {
		s.Logger.Printf("rejected unsupported POST path=%s status=404", r.URL.Path)
		s.writeJSON(w, http.StatusNotFound, map[string]any{
			"error": map[string]any{"message": "Only /v1/chat/completions is supported"},
		})
		return
	}
	authorization := cursorAuthorization(r)
	if authorization == "" {
		s.Logger.Printf("rejected request path=%s status=401 reason=missing_bearer_token", r.URL.Path)
		s.writeJSON(w, http.StatusUnauthorized, map[string]any{
			"error": map[string]any{"message": "Missing Authorization bearer token"},
		})
		return
	}

	payload, err := s.readJSONBody(r)
	if err != nil {
		var tooLarge requestBodyTooLargeError
		if errors.As(err, &tooLarge) {
			s.Logger.Printf("rejected request path=%s status=413 reason=%s", r.URL.Path, err)
			s.writeJSON(w, http.StatusRequestEntityTooLarge, map[string]any{
				"error": map[string]any{"message": err.Error()},
			})
			return
		}
		s.Logger.Printf("rejected request path=%s status=400 reason=%s", r.URL.Path, err)
		s.writeJSON(w, http.StatusBadRequest, map[string]any{
			"error": map[string]any{"message": err.Error()},
		})
		return
	}

	if s.Config.Verbose {
		s.logJSON("cursor request body", payload)
	}
	s.Logger.Printf("cursor request: %s", summarizeChatPayload(payload))

	prepared := transform.PrepareUpstreamRequest(payload, s.Config, s.Store, authorization)
	if prepared.PatchedReasoningMessages > 0 {
		s.Logger.Printf("restored reasoning_content on %d assistant message(s)", prepared.PatchedReasoningMessages)
	}
	if prepared.RecoveredReasoningMessages > 0 {
		if prepared.RecoveryNotice != "" {
			s.Logger.Printf(
				"recovered request because cached reasoning_content was unavailable for %d assistant message(s); omitted %d older message(s) from forwarded history and will show a Cursor notice",
				prepared.RecoveredReasoningMessages, prepared.RecoveryDroppedMessages,
			)
		} else {
			s.Logger.Printf("continued recovered request; omitted %d old message(s) before the prior recovery boundary", prepared.RecoveryDroppedMessages)
		}
	}
	if prepared.MissingReasoningMessages > 0 {
		s.Logger.Printf(
			"strict missing-reasoning mode rejected request path=%s status=409 reason=missing_reasoning_content count=%d",
			r.URL.Path, prepared.MissingReasoningMessages,
		)
		s.writeJSON(w, http.StatusConflict, map[string]any{
			"error": map[string]any{
				"message": fmt.Sprintf(
					"deepseek-cursor-proxy is running in strict missing-reasoning mode and cannot automatically recover this thinking-mode tool-call history because cached DeepSeek reasoning_content is missing for %d assistant message(s). Restart without `--missing-reasoning-strategy reject`, or pass `--missing-reasoning-strategy recover`, so the proxy can recover from partial chat history automatically.",
					prepared.MissingReasoningMessages,
				),
				"type":                       "missing_reasoning_content",
				"code":                       "missing_reasoning_content",
				"missing_reasoning_messages": prepared.MissingReasoningMessages,
			},
		})
		return
	}
	streamRequested, _ := prepared.Payload["stream"].(bool)
	s.Logger.Printf(
		"deepseek send: %s patched=%d recovered=%d",
		compactRequestStats(prepared.Payload),
		prepared.PatchedReasoningMessages,
		prepared.RecoveredReasoningMessages,
	)
	if s.Config.Verbose {
		s.logJSON("upstream request body", prepared.Payload)
	}

	upstreamBody, err := json.Marshal(prepared.Payload)
	if err != nil {
		s.writeJSON(w, http.StatusInternalServerError, map[string]any{
			"error": map[string]any{"message": "marshal upstream payload: " + err.Error()},
		})
		return
	}
	upstreamURL := s.Config.UpstreamBaseURL + "/chat/completions"
	upstreamReq, err := http.NewRequestWithContext(r.Context(), http.MethodPost, upstreamURL, bytes.NewReader(upstreamBody))
	if err != nil {
		s.writeJSON(w, http.StatusInternalServerError, map[string]any{
			"error": map[string]any{"message": err.Error()},
		})
		return
	}
	upstreamReq.Header.Set("Authorization", authorization)
	upstreamReq.Header.Set("Content-Type", "application/json")
	if streamRequested {
		upstreamReq.Header.Set("Accept", "text/event-stream")
	} else {
		upstreamReq.Header.Set("Accept", "application/json")
	}
	upstreamReq.Header.Set("Accept-Encoding", "identity")
	upstreamReq.Header.Set("User-Agent", "DeepSeekGoProxy/0.1")
	if v := r.Header.Get("Accept-Language"); v != "" {
		upstreamReq.Header.Set("Accept-Language", v)
	}

	resp, err := s.Client.Do(upstreamReq)
	if err != nil {
		s.Logger.Printf("upstream request failed elapsed_ms=%d reason=%s", elapsedMs(started), err)
		s.writeJSON(w, http.StatusBadGateway, map[string]any{
			"error": map[string]any{"message": "Upstream request failed: " + err.Error()},
		})
		return
	}
	defer resp.Body.Close()
	upstreamStatus := resp.StatusCode
	if s.Config.Verbose {
		s.Logger.Printf("upstream response status=%d stream=%v elapsed_ms=%d", upstreamStatus, streamRequested, elapsedMs(started))
	}

	if upstreamStatus >= 400 {
		s.proxyUpstreamError(w, resp)
		return
	}

	requestMessages := messagesFromAny(prepared.Payload["messages"])
	var sent bool
	if streamRequested {
		sent = s.proxyStreamingResponse(w, resp, prepared.OriginalModel, requestMessages, prepared.CacheNamespace, prepared.RecoveryNotice)
	} else {
		sent = s.proxyRegularResponse(w, resp, prepared.OriginalModel, requestMessages, prepared.CacheNamespace, prepared.RecoveryNotice)
	}
	if !sent {
		return
	}
	s.Logger.Printf(
		"request complete status=%d stream=%v elapsed_ms=%d patched_reasoning=%d missing_reasoning=%d recovered_reasoning=%d",
		upstreamStatus, streamRequested, elapsedMs(started),
		prepared.PatchedReasoningMessages, prepared.MissingReasoningMessages, prepared.RecoveredReasoningMessages,
	)
}

func (s *Server) proxyRegularResponse(
	w http.ResponseWriter,
	resp *http.Response,
	originalModel string,
	requestMessages []map[string]any,
	cacheNamespace string,
	recoveryNotice string,
) bool {
	body, err := readResponseBody(resp)
	if err != nil {
		s.Logger.Printf("failed to read upstream response: %s", err)
		s.writeJSON(w, http.StatusBadGateway, map[string]any{
			"error": map[string]any{"message": "Upstream read failed: " + err.Error()},
		})
		return false
	}
	body = transform.RewriteResponseBody(body, originalModel, s.Store, requestMessages, cacheNamespace, recoveryNotice)
	if s.Config.Verbose {
		s.logBytes("cursor response body", body)
	}
	contentType := resp.Header.Get("Content-Type")
	if contentType == "" {
		contentType = "application/json"
	}
	s.writeCORSHeaders(w)
	w.Header().Set("Content-Type", contentType)
	w.Header().Set("Content-Length", fmt.Sprintf("%d", len(body)))
	w.WriteHeader(resp.StatusCode)
	_, err = w.Write(body)
	return err == nil
}

func (s *Server) proxyStreamingResponse(
	w http.ResponseWriter,
	resp *http.Response,
	originalModel string,
	requestMessages []map[string]any,
	cacheNamespace string,
	recoveryNotice string,
) bool {
	flusher, ok := w.(http.Flusher)
	if !ok {
		s.Logger.Printf("response writer does not support streaming flush")
		s.writeJSON(w, http.StatusInternalServerError, map[string]any{
			"error": map[string]any{"message": "streaming not supported"},
		})
		return false
	}
	s.writeCORSHeaders(w)
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "close")
	w.WriteHeader(resp.StatusCode)
	flusher.Flush()

	accumulator := streaming.NewStreamAccumulator()
	var displayAdapter *streaming.CursorReasoningDisplayAdapter
	if s.Config.CursorDisplayReasoning {
		displayAdapter = streaming.NewCursorReasoningDisplayAdapter()
	}
	scope := store.ConversationScope(requestMessages, cacheNamespace)
	pendingNotice := recoveryNotice
	finalized := false
	reader := bufio.NewReaderSize(resp.Body, 32*1024)

	for {
		line, err := reader.ReadBytes('\n')
		if len(line) > 0 {
			rewritten, doneFlag, newNotice := s.rewriteSSELine(line, originalModel, accumulator, scope, displayAdapter, pendingNotice)
			pendingNotice = newNotice
			if _, werr := w.Write(rewritten); werr != nil {
				s.Logger.Printf("client disconnected while writing stream: %s", werr)
				return false
			}
			flusher.Flush()
			if doneFlag {
				finalized = true
				break
			}
		}
		if err != nil {
			if errors.Is(err, io.EOF) {
				break
			}
			s.Logger.Printf("upstream streaming read failed: %s", err)
			return false
		}
	}
	if !finalized {
		stored := accumulator.StoreReasoning(s.Store, scope)
		if stored > 0 {
			s.Logger.Printf("stored %d streaming reasoning cache key(s)", stored)
		}
	}
	return true
}

func (s *Server) rewriteSSELine(
	line []byte,
	originalModel string,
	accumulator *streaming.StreamAccumulator,
	scope string,
	displayAdapter *streaming.CursorReasoningDisplayAdapter,
	recoveryNotice string,
) (output []byte, finalized bool, newRecoveryNotice string) {
	stripped := bytes.TrimSpace(line)
	if !bytes.HasPrefix(stripped, []byte("data:")) {
		return line, false, recoveryNotice
	}
	data := bytes.TrimSpace(stripped[len("data:"):])
	if bytes.Equal(data, []byte("[DONE]")) {
		stored := accumulator.StoreReasoning(s.Store, scope)
		if stored > 0 {
			s.Logger.Printf("stored %d streaming reasoning cache key(s)", stored)
		}
		var prefix []byte
		if displayAdapter != nil {
			closing := displayAdapter.FlushChunk(originalModel)
			if closing != nil {
				prefix = append(prefix, sseData(closing)...)
			}
		}
		if recoveryNotice != "" {
			prefix = append(prefix, sseData(recoveryNoticeChunk(originalModel, recoveryNotice))...)
		}
		return append(prefix, []byte("data: [DONE]\n\n")...), true, ""
	}
	var chunk map[string]any
	if err := json.Unmarshal(data, &chunk); err != nil {
		return line, false, recoveryNotice
	}
	if chunk != nil {
		if recoveryNotice != "" && injectRecoveryNotice(chunk, recoveryNotice) {
			recoveryNotice = ""
		}
		accumulator.IngestChunk(chunk)
		stored := accumulator.StoreReadyReasoning(s.Store, scope)
		if stored > 0 {
			s.Logger.Printf("stored %d streaming reasoning cache key(s)", stored)
		}
		if displayAdapter != nil {
			displayAdapter.RewriteChunk(chunk)
		}
		if _, ok := chunk["model"]; ok {
			chunk["model"] = originalModel
		}
		ending := []byte("\n")
		if bytes.HasSuffix(line, []byte("\r\n")) {
			ending = []byte("\r\n")
		}
		out := encodeJSONNoEscape(chunk)
		buf := append([]byte("data: "), out...)
		buf = append(buf, ending...)
		return buf, false, recoveryNotice
	}
	return line, false, recoveryNotice
}

func (s *Server) proxyUpstreamError(w http.ResponseWriter, resp *http.Response) {
	body, err := readResponseBody(resp)
	if err != nil {
		s.writeJSON(w, http.StatusBadGateway, map[string]any{
			"error": map[string]any{"message": "Upstream read failed: " + err.Error()},
		})
		return
	}
	if s.Config.Verbose {
		s.logBytes("upstream error body", body)
	}
	contentType := resp.Header.Get("Content-Type")
	if contentType == "" {
		contentType = "application/json"
	}
	s.writeCORSHeaders(w)
	w.Header().Set("Content-Type", contentType)
	w.Header().Set("Content-Length", fmt.Sprintf("%d", len(body)))
	w.WriteHeader(resp.StatusCode)
	_, _ = w.Write(body)
}

func (s *Server) writeModels(w http.ResponseWriter) {
	created := time.Now().Unix()
	seen := map[string]struct{}{}
	var ids []string
	for _, id := range []string{s.Config.UpstreamModel, "deepseek-v4-pro", "deepseek-v4-flash"} {
		if _, ok := seen[id]; ok {
			continue
		}
		seen[id] = struct{}{}
		ids = append(ids, id)
	}
	models := make([]map[string]any, 0, len(ids))
	for _, id := range ids {
		models = append(models, map[string]any{
			"id":       id,
			"object":   "model",
			"created":  created,
			"owned_by": "deepseek",
		})
	}
	s.writeJSON(w, http.StatusOK, map[string]any{"object": "list", "data": models})
}

func (s *Server) writeJSON(w http.ResponseWriter, status int, payload map[string]any) {
	body, err := json.Marshal(payload)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	s.writeCORSHeaders(w)
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Content-Length", fmt.Sprintf("%d", len(body)))
	w.WriteHeader(status)
	_, _ = w.Write(body)
}

func (s *Server) writeCORSHeaders(w http.ResponseWriter) {
	if !s.Config.CORS {
		return
	}
	h := w.Header()
	h.Set("Access-Control-Allow-Origin", "*")
	h.Set("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
	h.Set("Access-Control-Allow-Headers", "Origin, Content-Type, Accept, Authorization")
	h.Set("Access-Control-Expose-Headers", "Content-Length")
	h.Set("Access-Control-Allow-Credentials", "true")
}

type requestBodyTooLargeError struct{ msg string }

func (e requestBodyTooLargeError) Error() string { return e.msg }

func (s *Server) readJSONBody(r *http.Request) (map[string]any, error) {
	limit := s.Config.MaxRequestBodyBytes
	if limit <= 0 {
		limit = config.DefaultMaxRequestBody
	}
	limitedReader := http.MaxBytesReader(nil, r.Body, limit+1)
	defer limitedReader.Close()
	data, err := io.ReadAll(limitedReader)
	if err != nil {
		var maxErr *http.MaxBytesError
		if errors.As(err, &maxErr) {
			return nil, requestBodyTooLargeError{
				msg: fmt.Sprintf("Request body is too large; limit is %d bytes", limit),
			}
		}
		return nil, fmt.Errorf("read body: %w", err)
	}
	if int64(len(data)) > limit {
		return nil, requestBodyTooLargeError{
			msg: fmt.Sprintf("Request body is too large; limit is %d bytes", limit),
		}
	}
	if len(data) == 0 {
		return nil, errors.New("Request body is empty")
	}
	var payload map[string]any
	if err := json.Unmarshal(data, &payload); err != nil {
		return nil, fmt.Errorf("Invalid JSON: %w", err)
	}
	if payload == nil {
		return nil, errors.New("Request body must be a JSON object")
	}
	return payload, nil
}

func cursorAuthorization(r *http.Request) string {
	header := strings.TrimSpace(r.Header.Get("Authorization"))
	if header == "" {
		return ""
	}
	scheme, token, found := strings.Cut(header, " ")
	if !found {
		return ""
	}
	if !strings.EqualFold(scheme, "bearer") {
		return ""
	}
	token = strings.TrimSpace(token)
	if token == "" {
		return ""
	}
	return "Bearer " + token
}

func clientIP(r *http.Request) string {
	if host, _, err := net.SplitHostPort(r.RemoteAddr); err == nil {
		return host
	}
	return r.RemoteAddr
}

func elapsedMs(t time.Time) int64 {
	return time.Since(t).Milliseconds()
}

func readResponseBody(resp *http.Response) ([]byte, error) {
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}
	encoding := strings.ToLower(resp.Header.Get("Content-Encoding"))
	switch encoding {
	case "gzip":
		gz, err := gzip.NewReader(bytes.NewReader(body))
		if err != nil {
			return body, nil
		}
		defer gz.Close()
		return io.ReadAll(gz)
	case "deflate":
		zr, err := zlib.NewReader(bytes.NewReader(body))
		if err != nil {
			return body, nil
		}
		defer zr.Close()
		return io.ReadAll(zr)
	}
	return body, nil
}

func messagesFromAny(value any) []map[string]any {
	if list, ok := value.([]any); ok {
		out := make([]map[string]any, 0, len(list))
		for _, item := range list {
			if m, ok := item.(map[string]any); ok {
				out = append(out, m)
			}
		}
		return out
	}
	if list, ok := value.([]map[string]any); ok {
		return list
	}
	return nil
}

func (s *Server) logJSON(label string, payload any) {
	out, err := json.MarshalIndent(payload, "", "  ")
	if err != nil {
		s.Logger.Printf("%s: %v", label, payload)
		return
	}
	s.Logger.Printf("%s:\n%s", label, out)
}

func (s *Server) logBytes(label string, body []byte) {
	var payload any
	if err := json.Unmarshal(body, &payload); err == nil {
		s.logJSON(label, payload)
		return
	}
	s.Logger.Printf("%s:\n%s", label, body)
}

func summarizeChatPayload(payload map[string]any) string {
	messages, _ := payload["messages"].([]any)
	tools, _ := payload["tools"].([]any)
	functions, _ := payload["functions"].([]any)
	stream, _ := payload["stream"].(bool)
	model, _ := payload["model"].(string)
	return fmt.Sprintf(
		"model=%q stream=%v messages=%d tools=%d functions=%d tool_choice=%v",
		model, stream, len(messages), len(tools), len(functions), payload["tool_choice"],
	)
}

func compactRequestStats(payload map[string]any) string {
	messages, _ := payload["messages"].([]any)
	tools, _ := payload["tools"].([]any)
	reasoningCount := 0
	reasoningChars := 0
	rounds := 0
	for _, raw := range messages {
		message, ok := raw.(map[string]any)
		if !ok {
			continue
		}
		role, _ := message["role"].(string)
		if role == "user" {
			rounds++
		}
		if role != "assistant" {
			continue
		}
		if reasoning, ok := message["reasoning_content"].(string); ok {
			reasoningCount++
			reasoningChars += len(reasoning)
		}
	}
	streamFlag := 0
	if v, _ := payload["stream"].(bool); v {
		streamFlag = 1
	}
	return fmt.Sprintf(
		"model=%v stream=%d rounds=%d msgs=%d tools=%d reasoning=%d/%dch",
		payload["model"], streamFlag, rounds, len(messages), len(tools), reasoningCount, reasoningChars,
	)
}

func sseData(payload map[string]any) []byte {
	out := encodeJSONNoEscape(payload)
	buf := append([]byte("data: "), out...)
	return append(buf, '\n', '\n')
}

// encodeJSONNoEscape marshals JSON without HTML escaping, matching Python's
// `json.dumps(..., ensure_ascii=False)` output for our streaming use case.
func encodeJSONNoEscape(value any) []byte {
	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	enc.SetEscapeHTML(false)
	if err := enc.Encode(value); err != nil {
		fallback, _ := json.Marshal(value)
		return fallback
	}
	out := buf.Bytes()
	// json.Encoder.Encode appends a newline; trim it.
	if len(out) > 0 && out[len(out)-1] == '\n' {
		out = out[:len(out)-1]
	}
	return out
}

func injectRecoveryNotice(chunk map[string]any, notice string) bool {
	choices, ok := chunk["choices"].([]any)
	if !ok {
		return false
	}
	for _, raw := range choices {
		choice, ok := raw.(map[string]any)
		if !ok {
			continue
		}
		delta, ok := choice["delta"].(map[string]any)
		if !ok {
			continue
		}
		_, hasContent := delta["content"]
		toolCalls, _ := delta["tool_calls"].([]any)
		if !hasContent && len(toolCalls) == 0 {
			continue
		}
		existing, _ := delta["content"].(string)
		delta["content"] = notice + existing
		return true
	}
	return false
}

func recoveryNoticeChunk(model, notice string) map[string]any {
	if notice == "" {
		notice = transform.RecoveryNoticeContent
	}
	return map[string]any{
		"id":      "chatcmpl-deepseek-cursor-proxy-recovery",
		"object":  "chat.completion.chunk",
		"created": time.Now().Unix(),
		"model":   model,
		"choices": []any{
			map[string]any{
				"index":         0,
				"delta":         map[string]any{"content": notice},
				"finish_reason": nil,
			},
		},
	}
}

// WarnIfInsecureUpstream logs a warning if the upstream URL is plaintext HTTP
// to a non-loopback host.
func (s *Server) WarnIfInsecureUpstream() {
	parsed, err := url.Parse(s.Config.UpstreamBaseURL)
	if err != nil || parsed.Scheme != "http" {
		return
	}
	host := parsed.Hostname()
	if host == "127.0.0.1" || host == "localhost" || host == "::1" {
		return
	}
	s.Logger.Printf("upstream base_url uses plain HTTP; bearer tokens may be exposed")
}

// Run starts the HTTP server until the context is cancelled.
func (s *Server) Run(ctx context.Context) error {
	address := fmt.Sprintf("%s:%d", s.Config.Host, s.Config.Port)
	httpServer := &http.Server{
		Addr:        address,
		Handler:     s,
		ReadTimeout: 0,
	}
	listener, err := net.Listen("tcp", address)
	if err != nil {
		return fmt.Errorf("listen: %w", err)
	}
	s.Logger.Printf("listening on http://%s/v1", listener.Addr().String())
	s.Logger.Printf("forwarding to %s/chat/completions default_model=%s", s.Config.UpstreamBaseURL, s.Config.UpstreamModel)
	s.Logger.Printf(
		"thinking=%s reasoning_effort=%s cursor_display_reasoning=%v missing_reasoning_strategy=%s reasoning_content_path=%s",
		s.Config.Thinking, s.Config.ReasoningEffort, s.Config.CursorDisplayReasoning, s.Config.MissingReasoningStrategy, s.Config.ReasoningContentPath,
	)
	if s.Config.Verbose {
		s.Logger.Print("logging mode=verbose metadata=detailed bodies=true")
		s.Logger.Print("verbose logging enabled; prompts and code may be written to stdout")
	} else {
		s.Logger.Print("logging mode=normal metadata=safe_summaries bodies=false")
	}
	s.WarnIfInsecureUpstream()

	errCh := make(chan error, 1)
	var once sync.Once
	go func() {
		err := httpServer.Serve(listener)
		if err != nil && !errors.Is(err, http.ErrServerClosed) {
			once.Do(func() { errCh <- err })
		} else {
			once.Do(func() { errCh <- nil })
		}
	}()
	select {
	case <-ctx.Done():
		shutdownCtx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		_ = httpServer.Shutdown(shutdownCtx)
		<-errCh
		return nil
	case err := <-errCh:
		return err
	}
}
