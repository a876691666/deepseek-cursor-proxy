// Package server implements the OpenAI-compatible HTTP proxy integrated with
// PocketBase for API key management and token usage recording.
package server

import (
	"bufio"
	"bytes"
	"compress/gzip"
	"compress/zlib"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net"
	"net/http"
	"net/url"
	"strings"
	"time"

	"github.com/pocketbase/pocketbase"

	"github.com/pocketbase/pocketbase/core"
	"github.com/pocketbase/pocketbase/tools/router"

	"github.com/a876691666/deepseek-cursor-proxy/internal/config"
	pbPkg "github.com/a876691666/deepseek-cursor-proxy/internal/pocketbase"
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
	PB     *pocketbase.PocketBase
	apiKey string // cached env API key
}

// Config wraps proxy configuration relevant to the HTTP server.
type Config = config.Config

// New constructs a Server with sensible HTTP defaults.
func New(cfg Config, st *store.Store, logger *log.Logger, pb *pocketbase.PocketBase) *Server {
	if logger == nil {
		logger = log.Default()
	}
	timeout := time.Duration(cfg.RequestTimeoutSeconds * float64(time.Second))
	if timeout <= 0 {
		timeout = 300 * time.Second
	}
	apiKey := cfg.DeepSeekAPIKey
	return &Server{
		Config: cfg,
		Store:  st,
		Logger: logger,
		PB:     pb,
		apiKey: apiKey,
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

// RegisterRoutes registers all proxy routes on the PocketBase router.
func (s *Server) RegisterRoutes(r *router.Router[*core.RequestEvent]) {
	// OpenAI-format routes.
	r.POST("/v1/chat/completions", s.HandleChatCompletions)
	r.POST("/chat/completions", s.HandleChatCompletions)
	r.GET("/v1/models", s.HandleModels)
	r.GET("/models", s.HandleModels)
	r.GET("/healthz", s.HandleHealth)
	r.GET("/v1/healthz", s.HandleHealth)
	r.POST("/v1/api_keys", s.HandleCreateAPIKey)
	r.GET("/v1/api_keys", s.HandleListAPIKeys)
	r.OPTIONS("/v1/chat/completions", s.HandleOptions)
	r.OPTIONS("/chat/completions", s.HandleOptions)
	r.OPTIONS("/v1/api_keys", s.HandleOptions)

	// Anthropic-format routes (only when upstream is configured).
	if s.Config.AnthropicBaseURL != "" {
		r.POST("/v1/messages", s.HandleMessages)
		r.POST("/messages", s.HandleMessages)
		r.OPTIONS("/v1/messages", s.HandleOptions)
		r.OPTIONS("/messages", s.HandleOptions)
	}
}

// ServeHTTP routes incoming requests (kept for direct http.Handler compatibility).
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

// ---- PocketBase route handlers ----

func (s *Server) HandleChatCompletions(e *core.RequestEvent) error {
	r := e.Request
	w := e.Response

	if s.Config.CORS {
		s.writeCORSHeaders(w)
	}

	return s.handleChatCompletions(w, r)
}

func (s *Server) HandleModels(e *core.RequestEvent) error {
	s.writeModels(e.Response)
	return nil
}

func (s *Server) HandleHealth(e *core.RequestEvent) error {
	e.Response.Header().Set("Content-Type", "application/json")
	e.Response.WriteHeader(http.StatusOK)
	e.Response.Write([]byte(`{"ok":true}`))
	return nil
}

func (s *Server) HandleOptions(e *core.RequestEvent) error {
	if s.Config.CORS {
		s.writeCORSHeaders(e.Response)
	}
	e.Response.WriteHeader(http.StatusNoContent)
	return nil
}

func (s *Server) HandleMessages(e *core.RequestEvent) error {
	r := e.Request
	w := e.Response

	if s.Config.CORS {
		s.writeCORSHeaders(w)
	}

	return s.handleMessages(w, r)
}

func (s *Server) HandleCreateAPIKey(e *core.RequestEvent) error {
	w := e.Response
	r := e.Request

	if s.PB == nil {
		s.writeJSON(w, http.StatusInternalServerError, map[string]any{
			"error": map[string]any{"message": "PocketBase not available"},
		})
		return nil
	}

	// Require superuser auth for creating API keys
	info, err := e.RequestInfo()
	if err != nil || info.Auth == nil || !info.Auth.IsSuperuser() {
		s.writeJSON(w, http.StatusForbidden, map[string]any{
			"error": map[string]any{"message": "Superuser authentication required"},
		})
		return nil
	}

	var body struct {
		Name string `json:"name"`
	}
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil || body.Name == "" {
		s.writeJSON(w, http.StatusBadRequest, map[string]any{
			"error": map[string]any{"message": "A 'name' field is required in the request body"},
		})
		return nil
	}

	record, err := pbPkg.CreateAPIKey(s.PB, body.Name)
	if err != nil {
		s.writeJSON(w, http.StatusInternalServerError, map[string]any{
			"error": map[string]any{"message": "Failed to create API key: " + err.Error()},
		})
		return nil
	}

	s.writeJSON(w, http.StatusCreated, map[string]any{
		"id":      record.Id,
		"key":     record.GetString("key"),
		"name":    record.GetString("name"),
		"active":  record.GetBool("active"),
		"created": record.GetString("created"),
	})
	return nil
}

func (s *Server) HandleListAPIKeys(e *core.RequestEvent) error {
	w := e.Response

	if s.PB == nil {
		s.writeJSON(w, http.StatusInternalServerError, map[string]any{
			"error": map[string]any{"message": "PocketBase not available"},
		})
		return nil
	}

	info, err := e.RequestInfo()
	if err != nil || info.Auth == nil || !info.Auth.IsSuperuser() {
		s.writeJSON(w, http.StatusForbidden, map[string]any{
			"error": map[string]any{"message": "Superuser authentication required"},
		})
		return nil
	}

	records, err := s.PB.FindAllRecords(pbPkg.CollectionAPIKeys)
	if err != nil {
		s.writeJSON(w, http.StatusInternalServerError, map[string]any{
			"error": map[string]any{"message": err.Error()},
		})
		return nil
	}

	items := make([]map[string]any, 0, len(records))
	for _, r := range records {
		items = append(items, map[string]any{
			"id":      r.Id,
			"key":     r.GetString("key"),
			"name":    r.GetString("name"),
			"active":  r.GetBool("active"),
			"created": r.GetString("created"),
		})
	}

	s.writeJSON(w, http.StatusOK, map[string]any{
		"data":  items,
		"total": len(items),
	})
	return nil
}

// ---- Core proxy logic ----

func (s *Server) handleChatCompletions(w http.ResponseWriter, r *http.Request) error {
	started := time.Now()
	if s.Config.Verbose {
		s.Logger.Printf(
			"incoming POST %s from %s content_length=%s user_agent=%s",
			r.URL.Path, clientIP(r),
			r.Header.Get("Content-Length"),
			r.Header.Get("User-Agent"),
		)
	}

	// Authorization check first (before reading body).
	upstreamAuth := s.resolveUpstreamAuth(r)
	if upstreamAuth == "" {
		s.Logger.Printf("rejected request path=%s status=401 reason=missing_bearer_token", r.URL.Path)
		s.writeJSON(w, http.StatusUnauthorized, map[string]any{
			"error": map[string]any{"message": "Missing Authorization bearer token (set DEEPSEEK_API_KEY env var or provide Bearer token)"},
		})
		return nil
	}

	// Extract distributed API key from header or query parameter.
	queryKey := s.extractDistributedKey(r)
	var apiKeyRecordID string
	if queryKey != "" && s.PB != nil {
		record, err := pbPkg.LookupAPIKey(s.PB, queryKey)
		if err != nil {
			s.Logger.Printf("api key lookup error: %v", err)
		}
		if record != nil {
			apiKeyRecordID = record.Id
		}
	}

	payload, err := s.readJSONBody(r)
	if err != nil {
		var tooLarge requestBodyTooLargeError
		if errors.As(err, &tooLarge) {
			s.Logger.Printf("rejected request path=%s status=413 reason=%s", r.URL.Path, err)
			s.writeJSON(w, http.StatusRequestEntityTooLarge, map[string]any{
				"error": map[string]any{"message": err.Error()},
			})
			return nil
		}
		s.Logger.Printf("rejected request path=%s status=400 reason=%s", r.URL.Path, err)
		s.writeJSON(w, http.StatusBadRequest, map[string]any{
			"error": map[string]any{"message": err.Error()},
		})
		return nil
	}

	modelName := s.modelName(payload)

	if s.Config.Verbose {
		s.logJSON("cursor request body", payload)
	}
	s.Logger.Printf("cursor request: %s", summarizeChatPayload(payload))

	prepared := transform.PrepareUpstreamRequest(payload, s.Config, s.Store, upstreamAuth)
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
		return nil
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
		return nil
	}
	upstreamURL := s.Config.UpstreamBaseURL + "/chat/completions"
	upstreamReq, err := http.NewRequestWithContext(r.Context(), http.MethodPost, upstreamURL, bytes.NewReader(upstreamBody))
	if err != nil {
		s.writeJSON(w, http.StatusInternalServerError, map[string]any{
			"error": map[string]any{"message": err.Error()},
		})
		return nil
	}
	upstreamReq.Header.Set("Authorization", upstreamAuth)
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
		return nil
	}
	defer resp.Body.Close()
	upstreamStatus := resp.StatusCode
	if s.Config.Verbose {
		s.Logger.Printf("upstream response status=%d stream=%v elapsed_ms=%d", upstreamStatus, streamRequested, elapsedMs(started))
	}

	if upstreamStatus >= 400 {
		s.proxyUpstreamError(w, resp)
		return nil
	}

	requestMessages := messagesFromAny(prepared.Payload["messages"])
	var sent bool
	if streamRequested {
		sent = s.proxyStreamingResponse(w, resp, prepared.OriginalModel, requestMessages, prepared.CacheNamespace, prepared.RecoveryNotice, queryKey, modelName)
	} else {
		sent = s.proxyRegularResponse(w, resp, prepared.OriginalModel, requestMessages, prepared.CacheNamespace, prepared.RecoveryNotice, queryKey, modelName)
	}
	if !sent {
		return nil
	}
	s.Logger.Printf(
		"request complete status=%d stream=%v elapsed_ms=%d patched_reasoning=%d missing_reasoning=%d recovered_reasoning=%d api_key=%s",
		upstreamStatus, streamRequested, elapsedMs(started),
		prepared.PatchedReasoningMessages, prepared.MissingReasoningMessages, prepared.RecoveredReasoningMessages,
		apiKeyRecordID,
	)
	return nil
}

func (s *Server) resolveUpstreamAuth(r *http.Request) string {
	// Check if the Authorization header carries a distributed API key (sk-dcp- prefix).
	// Those keys are not real DeepSeek API keys �?use the configured DeepSeek key instead.
	if key := distributedKeyFromAuth(r); key != "" {
		if s.apiKey != "" {
			return "Bearer " + s.apiKey
		}
		return key // No upstream API key configured; upstream auth fails.
	}
	// If DEEPSEEK_API_KEY env var is set, use it for all upstream requests.
	if s.apiKey != "" {
		return "Bearer " + s.apiKey
	}
	return cursorAuthorization(r)
}

// distributedKeyFromAuth extracts a distributed API key from the Authorization
// header when the Bearer token starts with the sk-dcp- prefix.
func distributedKeyFromAuth(r *http.Request) string {
	header := strings.TrimSpace(r.Header.Get("Authorization"))
	if header == "" {
		return ""
	}
	scheme, token, found := strings.Cut(header, " ")
	if !found || !strings.EqualFold(scheme, "bearer") {
		return ""
	}
	token = strings.TrimSpace(token)
	if strings.HasPrefix(token, "sk-dcp-") {
		return token
	}
	return ""
}

// extractDistributedKey returns the distributed API key from either the
// ?api_key= query parameter or the Authorization header (sk-dcp- prefix).
func (s *Server) extractDistributedKey(r *http.Request) string {
	if qk := r.URL.Query().Get("api_key"); qk != "" {
		return qk
	}
	return distributedKeyFromAuth(r)
}

func (s *Server) modelName(payload map[string]any) string {
	if m, _ := payload["model"].(string); m != "" {
		return m
	}
	return s.Config.UpstreamModel
}

func (s *Server) proxyRegularResponse(
	w http.ResponseWriter,
	resp *http.Response,
	originalModel string,
	requestMessages []map[string]any,
	cacheNamespace string,
	recoveryNotice string,
	queryKey string,
	modelName string,
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

	// Extract usage and record it.
	if queryKey != "" {
		usage := extractUsage(body)
		s.recordUsage(queryKey, modelName, usage)
	}

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
	queryKey string,
	modelName string,
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
	var trackUsage usageInfo

	for {
		line, err := reader.ReadBytes('\n')
		if len(line) > 0 {
			rewritten, doneFlag, newNotice := s.rewriteSSELine(line, originalModel, accumulator, scope, displayAdapter, pendingNotice, &trackUsage)
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
	if (trackUsage.PromptTokens > 0 || trackUsage.CompletionTokens > 0) && queryKey != "" {
		s.recordUsage(queryKey, modelName, &trackUsage)
	}
	return true
}

type usageInfo struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

func extractUsage(body []byte) *usageInfo {
	var payload map[string]any
	if err := json.Unmarshal(body, &payload); err != nil {
		return nil
	}
	rawUsage, ok := payload["usage"]
	if !ok {
		return nil
	}
	u, ok := rawUsage.(map[string]any)
	if !ok {
		return nil
	}
	info := &usageInfo{}
	if v, ok := u["prompt_tokens"].(float64); ok {
		info.PromptTokens = int(v)
	}
	if v, ok := u["completion_tokens"].(float64); ok {
		info.CompletionTokens = int(v)
	}
	if v, ok := u["total_tokens"].(float64); ok {
		info.TotalTokens = int(v)
	}
	return info
}

func (s *Server) recordUsage(queryKey, model string, usage *usageInfo) {
	if usage == nil || s.PB == nil {
		return
	}
	record, err := pbPkg.LookupAPIKey(s.PB, queryKey)
	if err != nil || record == nil {
		return
	}
	now := time.Now()
	if err := pbPkg.RecordTokenUsage(s.PB, queryKey, model, usage.PromptTokens, usage.CompletionTokens, usage.TotalTokens, now); err != nil {
		s.Logger.Printf("failed to record token usage: %s", err)
		return
	}
	s.Logger.Printf("token_usage time=%s key=%s model=%s prompt=%d completion=%d total=%d",
		now.UTC().Format(time.RFC3339), record.GetString("name"), model,
		usage.PromptTokens, usage.CompletionTokens, usage.TotalTokens)
}

func (s *Server) rewriteSSELine(
	line []byte,
	originalModel string,
	accumulator *streaming.StreamAccumulator,
	scope string,
	displayAdapter *streaming.CursorReasoningDisplayAdapter,
	recoveryNotice string,
	acc *usageInfo,
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
		// Extract usage from the final streaming chunk.
		if rawUsage, ok := chunk["usage"]; ok {
			if u, ok := rawUsage.(map[string]any); ok {
				if v, ok := u["prompt_tokens"].(float64); ok {
					acc.PromptTokens = int(v)
				}
				if v, ok := u["completion_tokens"].(float64); ok {
					acc.CompletionTokens = int(v)
				}
				if v, ok := u["total_tokens"].(float64); ok {
					acc.TotalTokens = int(v)
				}
			}
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

// ---- Anthropic Messages API handlers ----

func (s *Server) handleMessages(w http.ResponseWriter, r *http.Request) error {
	started := time.Now()
	if s.Config.Verbose {
		s.Logger.Printf(
			"incoming POST %s (anthropic) from %s content_length=%s user_agent=%s",
			r.URL.Path, clientIP(r),
			r.Header.Get("Content-Length"),
			r.Header.Get("User-Agent"),
		)
	}

	upstreamAuth := s.resolveUpstreamAuth(r)
	if upstreamAuth == "" {
		s.Logger.Printf("rejected anthropic request path=%s status=401 reason=missing_bearer_token", r.URL.Path)
		s.writeJSON(w, http.StatusUnauthorized, map[string]any{
			"error": map[string]any{"message": "Missing Authorization bearer token (set DEEPSEEK_API_KEY env var or provide Bearer token)"},
		})
		return nil
	}

	queryKey := s.extractDistributedKey(r)
	var apiKeyRecordID string
	if queryKey != "" && s.PB != nil {
		record, err := pbPkg.LookupAPIKey(s.PB, queryKey)
		if err != nil {
			s.Logger.Printf("api key lookup error: %v", err)
		}
		if record != nil {
			apiKeyRecordID = record.Id
		}
	}

	payload, err := s.readJSONBody(r)
	if err != nil {
		var tooLarge requestBodyTooLargeError
		if errors.As(err, &tooLarge) {
			s.Logger.Printf("rejected anthropic request path=%s status=413 reason=%s", r.URL.Path, err)
			s.writeJSON(w, http.StatusRequestEntityTooLarge, map[string]any{
				"error": map[string]any{"message": err.Error()},
			})
			return nil
		}
		s.Logger.Printf("rejected anthropic request path=%s status=400 reason=%s", r.URL.Path, err)
		s.writeJSON(w, http.StatusBadRequest, map[string]any{
			"error": map[string]any{"message": err.Error()},
		})
		return nil
	}

	modelName := s.Config.UpstreamModel
	originalModel := modelName
	payload["model"] = modelName

	streamRequested, _ := payload["stream"].(bool)
	s.Logger.Printf("anthropic request: model=%q stream=%v messages=%d api_key=%s",
		modelName, streamRequested, anthropicMessageCount(payload), apiKeyRecordID)

	if s.Config.Verbose {
		s.logJSON("anthropic request body", payload)
	}

	upstreamBody, err := json.Marshal(payload)
	if err != nil {
		s.writeJSON(w, http.StatusInternalServerError, map[string]any{
			"error": map[string]any{"message": "marshal upstream payload: " + err.Error()},
		})
		return nil
	}

	upstreamURL := s.Config.AnthropicBaseURL + s.Config.AnthropicAPIPath
	upstreamReq, err := http.NewRequestWithContext(r.Context(), http.MethodPost, upstreamURL, bytes.NewReader(upstreamBody))
	if err != nil {
		s.writeJSON(w, http.StatusInternalServerError, map[string]any{
			"error": map[string]any{"message": err.Error()},
		})
		return nil
	}
	upstreamReq.Header.Set("Authorization", upstreamAuth)
	upstreamReq.Header.Set("Content-Type", "application/json")
	if streamRequested {
		upstreamReq.Header.Set("Accept", "text/event-stream")
	} else {
		upstreamReq.Header.Set("Accept", "application/json")
	}
	upstreamReq.Header.Set("Accept-Encoding", "identity")
	upstreamReq.Header.Set("User-Agent", "DeepSeekGoProxy/0.1")
	if s.apiKey != "" {
		upstreamReq.Header.Set("x-api-key", s.apiKey)
	}
	if v := r.Header.Get("anthropic-version"); v != "" {
		upstreamReq.Header.Set("anthropic-version", v)
	}

	resp, err := s.Client.Do(upstreamReq)
	if err != nil {
		s.Logger.Printf("anthropic upstream request failed elapsed_ms=%d reason=%s", elapsedMs(started), err)
		s.writeJSON(w, http.StatusBadGateway, map[string]any{
			"error": map[string]any{"message": "Upstream request failed: " + err.Error()},
		})
		return nil
	}
	defer resp.Body.Close()
	upstreamStatus := resp.StatusCode
	if s.Config.Verbose {
		s.Logger.Printf("anthropic upstream response status=%d stream=%v elapsed_ms=%d", upstreamStatus, streamRequested, elapsedMs(started))
	}

	if upstreamStatus >= 400 {
		s.proxyUpstreamError(w, resp)
		return nil
	}

	var sent bool
	if streamRequested {
		sent = s.proxyAnthropicStreaming(w, resp, originalModel, queryKey, modelName)
	} else {
		sent = s.proxyAnthropicRegular(w, resp, originalModel, queryKey, modelName)
	}
	if !sent {
		return nil
	}
	s.Logger.Printf(
		"anthropic request complete status=%d stream=%v elapsed_ms=%d api_key=%s",
		upstreamStatus, streamRequested, elapsedMs(started), apiKeyRecordID,
	)
	return nil
}

func (s *Server) proxyAnthropicRegular(w http.ResponseWriter, resp *http.Response, originalModel, queryKey, modelName string) bool {
	body, err := readResponseBody(resp)
	if err != nil {
		s.Logger.Printf("failed to read anthropic upstream response: %s", err)
		s.writeJSON(w, http.StatusBadGateway, map[string]any{
			"error": map[string]any{"message": "Upstream read failed: " + err.Error()},
		})
		return false
	}

	// Rewrite model name in response and extract usage.
	body = rewriteAnthropicResponseModel(body, originalModel)
	if queryKey != "" {
		usage := extractAnthropicUsage(body)
		s.recordUsage(queryKey, modelName, usage)
	}

	if s.Config.Verbose {
		s.logBytes("anthropic response body", body)
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

func (s *Server) proxyAnthropicStreaming(w http.ResponseWriter, resp *http.Response, originalModel, queryKey, modelName string) bool {
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

	reader := bufio.NewReaderSize(resp.Body, 32*1024)
	var trackUsage usageInfo

	for {
		line, err := reader.ReadBytes('\n')
		if len(line) > 0 {
			rewritten := s.rewriteAnthropicSSELine(line, originalModel, &trackUsage)
			if _, werr := w.Write(rewritten); werr != nil {
				s.Logger.Printf("client disconnected while writing anthropic stream: %s", werr)
				return false
			}
			flusher.Flush()
		}
		if err != nil {
			if errors.Is(err, io.EOF) {
				break
			}
			s.Logger.Printf("anthropic upstream streaming read failed: %s", err)
			return false
		}
	}
	if (trackUsage.PromptTokens > 0 || trackUsage.CompletionTokens > 0) && queryKey != "" {
		s.recordUsage(queryKey, modelName, &trackUsage)
	}
	return true
}

func (s *Server) rewriteAnthropicSSELine(line []byte, originalModel string, acc *usageInfo) []byte {
	stripped := bytes.TrimSpace(line)
	if len(stripped) == 0 {
		return line
	}

	// Pass through event: lines as-is.
	if bytes.HasPrefix(stripped, []byte("event:")) {
		return line
	}

	if !bytes.HasPrefix(stripped, []byte("data:")) {
		return line
	}

	data := bytes.TrimSpace(stripped[len("data:"):])
	var chunk map[string]any
	if err := json.Unmarshal(data, &chunk); err != nil {
		return line
	}

	// Rewrite model name in message_start events.
	if typ, _ := chunk["type"].(string); typ == "message_start" {
		if msg, ok := chunk["message"].(map[string]any); ok {
			msg["model"] = originalModel
		}
		// Extract input_tokens and cache tokens from message_start.
		if msg, ok := chunk["message"].(map[string]any); ok {
			if rawUsage, ok := msg["usage"].(map[string]any); ok {
				if v, ok := rawUsage["input_tokens"].(float64); ok {
					acc.PromptTokens = int(v)
				}
			}
		}
		// Also rewrite model at top level of the event.
		if _, ok := chunk["model"]; ok {
			chunk["model"] = originalModel
		}
	} else if typ, _ := chunk["type"].(string); typ == "message_delta" {
		if rawUsage, ok := chunk["usage"].(map[string]any); ok {
			if v, ok := rawUsage["output_tokens"].(float64); ok {
				acc.CompletionTokens = int(v)
			}
		}
	}

	acc.TotalTokens = acc.PromptTokens + acc.CompletionTokens

	// Re-serialize the modified data.
	ending := []byte("\n")
	if bytes.HasSuffix(line, []byte("\r\n")) {
		ending = []byte("\r\n")
	}
	out := encodeJSONNoEscape(chunk)
	buf := append([]byte("data: "), out...)
	buf = append(buf, ending...)
	return buf
}

func extractAnthropicUsage(body []byte) *usageInfo {
	var payload map[string]any
	if err := json.Unmarshal(body, &payload); err != nil {
		return nil
	}
	rawUsage, ok := payload["usage"].(map[string]any)
	if !ok {
		return nil
	}
	info := &usageInfo{}
	if v, ok := rawUsage["input_tokens"].(float64); ok {
		info.PromptTokens = int(v)
	}
	if v, ok := rawUsage["output_tokens"].(float64); ok {
		info.CompletionTokens = int(v)
	}
	info.TotalTokens = info.PromptTokens + info.CompletionTokens
	return info
}

func rewriteAnthropicResponseModel(body []byte, originalModel string) []byte {
	var payload map[string]any
	if err := json.Unmarshal(body, &payload); err != nil {
		return body
	}
	if _, ok := payload["model"]; ok {
		payload["model"] = originalModel
	}
	out, err := json.Marshal(payload)
	if err != nil {
		return body
	}
	return out
}

func anthropicMessageCount(payload map[string]any) int {
	if messages, ok := payload["messages"].([]any); ok {
		return len(messages)
	}
	return 0
}

// ---- Legacy http.Handler methods (keep for compatibility) ----

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
	if r.URL.Path == "/chat/completions" || r.URL.Path == "/v1/chat/completions" {
		s.handleChatCompletions(w, r)
		return
	}
	if s.Config.AnthropicBaseURL != "" &&
		(r.URL.Path == "/v1/messages" || r.URL.Path == "/messages") {
		s.handleMessages(w, r)
		return
	}
	s.Logger.Printf("rejected unsupported POST path=%s status=404", r.URL.Path)
	s.writeJSON(w, http.StatusNotFound, map[string]any{
		"error": map[string]any{"message": "Only /v1/chat/completions and /v1/messages are supported"},
	})
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

func encodeJSONNoEscape(value any) []byte {
	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	enc.SetEscapeHTML(false)
	if err := enc.Encode(value); err != nil {
		fallback, _ := json.Marshal(value)
		return fallback
	}
	out := buf.Bytes()
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

// Run starts the PocketBase HTTP server and blocks until it exits.
// Returns an error if PocketBase is not available.
func (s *Server) Run() error {
	if s.PB == nil {
		return errors.New("PocketBase not initialized; Run requires a PocketBase instance")
	}

	s.Logger.Printf("listening on http://%s:%d/v1", s.Config.Host, s.Config.Port)
	s.Logger.Printf("forwarding (openai) to %s/chat/completions default_model=%s", s.Config.UpstreamBaseURL, s.Config.UpstreamModel)
	if s.Config.AnthropicBaseURL != "" {
		s.Logger.Printf("forwarding (anthropic) to %s%s", s.Config.AnthropicBaseURL, s.Config.AnthropicAPIPath)
	}
	s.Logger.Printf(
		"thinking=%s reasoning_effort=%s cursor_display_reasoning=%v missing_reasoning_strategy=%s",
		s.Config.Thinking, s.Config.ReasoningEffort, s.Config.CursorDisplayReasoning, s.Config.MissingReasoningStrategy,
	)
	if s.apiKey != "" {
		s.Logger.Printf("deepseek API key from DEEPSEEK_API_KEY env (length=%d)", len(s.apiKey))
	}
	s.Logger.Printf("pocketbase admin: %s", s.Config.PBAdminEmail)
	s.Logger.Printf("pocketbase data dir: %s", s.Config.PBDataDir)
	if s.Config.Verbose {
		s.Logger.Print("logging mode=verbose metadata=detailed bodies=true")
	} else {
		s.Logger.Print("logging mode=normal metadata=safe_summaries bodies=false")
	}
	s.WarnIfInsecureUpstream()

	return s.PB.Start()
}
