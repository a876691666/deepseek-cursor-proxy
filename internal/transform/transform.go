// Package transform handles request normalization (Cursor → DeepSeek), recovery
// of missing reasoning_content, and response post-processing.
package transform

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"regexp"
	"strings"

	"github.com/a876691666/deepseek-cursor-proxy/internal/config"
	"github.com/a876691666/deepseek-cursor-proxy/internal/store"
)

// Recovery notice constants surfaced to upstream and clients.
const (
	RecoveryNoticeText = "[deepseek-cursor-proxy] Recovered this DeepSeek chat because older " +
		"tool-call reasoning was unavailable; continuing with recent context only."
	LegacyRecoveryNoticeText = "Note: recovered this DeepSeek chat because older tool-call reasoning " +
		"was unavailable; continuing with recent context only."
	RecoveryNoticeContent = RecoveryNoticeText + "\n\n"
	RecoverySystemContent = "deepseek-cursor-proxy recovered this request because older DeepSeek " +
		"thinking-mode tool-call reasoning_content was unavailable. Older " +
		"unrecoverable tool-call history was omitted; continue using only the " +
		"remaining recovered context."
)

var supportedRequestFields = map[string]struct{}{
	"model":             {},
	"messages":          {},
	"stream":            {},
	"stream_options":    {},
	"max_tokens":        {},
	"response_format":   {},
	"stop":              {},
	"tools":             {},
	"tool_choice":       {},
	"thinking":          {},
	"reasoning_effort":  {},
	"temperature":       {},
	"top_p":             {},
	"presence_penalty":  {},
	"frequency_penalty": {},
	"logprobs":          {},
	"top_logprobs":      {},
}

var roleMessageFields = map[string]map[string]struct{}{
	"system":    {"role": {}, "content": {}, "name": {}},
	"user":      {"role": {}, "content": {}, "name": {}},
	"assistant": {"role": {}, "content": {}, "name": {}, "tool_calls": {}, "reasoning_content": {}, "prefix": {}},
	"tool":      {"role": {}, "content": {}, "tool_call_id": {}},
}

var allMessageFields = map[string]struct{}{
	"role":              {},
	"content":           {},
	"name":              {},
	"tool_call_id":      {},
	"tool_calls":        {},
	"reasoning_content": {},
	"prefix":            {},
}

var effortAliases = map[string]string{
	"low":    "high",
	"medium": "high",
	"high":   "high",
	"max":    "max",
	"xhigh":  "max",
}

var cursorThinkingBlockRE = regexp.MustCompile(`(?i)<(?:think|thinking)>[\s\S]*?(?:</(?:think|thinking)>|$)\s*`)

// PreparedRequest is the result of normalizing a Cursor request for upstream forwarding.
type PreparedRequest struct {
	Payload                      map[string]any
	OriginalModel                string
	UpstreamModel                string
	CacheNamespace               string
	PatchedReasoningMessages     int
	MissingReasoningMessages     int
	RecoveredReasoningMessages   int
	RecoveryDroppedMessages      int
	RecoveryNotice               string // empty when no client-visible notice required
}

// NormalizeReasoningEffort maps user-supplied effort to canonical alias.
func NormalizeReasoningEffort(value any) string {
	if s, ok := value.(string); ok {
		if v, ok := effortAliases[strings.ToLower(strings.TrimSpace(s))]; ok {
			return v
		}
	}
	return "high"
}

// ExtractTextContent flattens content arrays/objects to a single string.
func ExtractTextContent(content any) string {
	switch v := content.(type) {
	case nil:
		return ""
	case string:
		return v
	case []any:
		parts := make([]string, 0, len(v))
		for _, item := range v {
			switch piece := item.(type) {
			case string:
				if piece != "" {
					parts = append(parts, piece)
				}
			case map[string]any:
				itemType, _ := piece["type"].(string)
				text, _ := piece["text"].(string)
				if text == "" {
					text, _ = piece["content"].(string)
				}
				if (itemType == "text" || itemType == "input_text") && text != "" {
					parts = append(parts, text)
				} else if text != "" {
					parts = append(parts, text)
				} else if itemType != "" {
					parts = append(parts, fmt.Sprintf("[%s omitted by DeepSeek text proxy]", itemType))
				}
			default:
				parts = append(parts, fmt.Sprintf("%v", piece))
			}
		}
		return strings.Join(parts, "\n")
	case map[string]any:
		out, _ := json.Marshal(v)
		return string(out)
	default:
		return fmt.Sprintf("%v", v)
	}
}

// StripCursorThinkingBlocks removes <think>/<thinking> blocks from assistant content.
func StripCursorThinkingBlocks(content string) string {
	stripped := cursorThinkingBlockRE.ReplaceAllString(content, "")
	return strings.TrimLeft(stripped, "\r\n")
}

func normalizeToolCall(tc any) map[string]any {
	call, _ := tc.(map[string]any)
	if call == nil {
		call = map[string]any{}
	}
	function, _ := call["function"].(map[string]any)
	if function == nil {
		function = map[string]any{}
	}
	var arguments string
	switch v := function["arguments"].(type) {
	case string:
		arguments = v
	case nil:
		arguments = ""
	default:
		raw, _ := json.Marshal(v)
		arguments = string(raw)
	}
	id, _ := call["id"].(string)
	typ, _ := call["type"].(string)
	if typ == "" {
		typ = "function"
	}
	name, _ := function["name"].(string)
	out := map[string]any{
		"type": typ,
		"function": map[string]any{
			"name":      name,
			"arguments": arguments,
		},
	}
	if id != "" {
		out["id"] = id
	}
	return out
}

func normalizeTool(tool any) map[string]any {
	call, _ := tool.(map[string]any)
	if call == nil {
		return map[string]any{
			"type":     "function",
			"function": map[string]any{"name": "", "description": "", "parameters": map[string]any{}},
		}
	}
	out := make(map[string]any, len(call))
	for k, v := range call {
		out[k] = v
	}
	if t, _ := out["type"].(string); t == "" {
		out["type"] = "function"
	}
	return out
}

func legacyFunctionToTool(fn any) map[string]any {
	function, ok := fn.(map[string]any)
	if !ok {
		function = map[string]any{}
	}
	return map[string]any{"type": "function", "function": function}
}

func convertFunctionCall(fc any) any {
	switch v := fc.(type) {
	case string:
		if v == "auto" || v == "none" || v == "required" {
			return v
		}
		return nil
	case map[string]any:
		if name, ok := v["name"].(string); ok && name != "" {
			return map[string]any{
				"type":     "function",
				"function": map[string]any{"name": name},
			}
		}
	}
	return nil
}

func normalizeToolChoice(tc any) any {
	switch v := tc.(type) {
	case string:
		if v == "auto" || v == "none" || v == "required" {
			return v
		}
		return nil
	case map[string]any:
		if t, _ := v["type"].(string); t == "function" {
			if function, ok := v["function"].(map[string]any); ok {
				if name, ok := function["name"].(string); ok && name != "" {
					return map[string]any{
						"type":     "function",
						"function": map[string]any{"name": name},
					}
				}
			}
		}
		return v
	}
	return tc
}

func normalizeMessage(
	rawMessage any,
	st *store.Store,
	priorMessages []map[string]any,
	cacheNamespace string,
	repairReasoning, keepReasoning bool,
) (normalized map[string]any, patched, missing bool) {
	message, _ := rawMessage.(map[string]any)
	if message == nil {
		message = map[string]any{"role": "user", "content": fmt.Sprintf("%v", rawMessage)}
	}
	out := map[string]any{}
	for k, v := range message {
		if _, ok := allMessageFields[k]; ok {
			out[k] = v
		}
	}
	role, _ := out["role"].(string)
	if role == "" {
		role = "user"
	}
	if role == "function" {
		role = "tool"
	}
	out["role"] = role

	if _, hasContent := out["content"]; hasContent {
		out["content"] = ExtractTextContent(out["content"])
	} else if role == "assistant" || role == "tool" || role == "system" || role == "user" {
		out["content"] = ""
	}
	if role == "assistant" {
		if s, ok := out["content"].(string); ok {
			out["content"] = StripCursorThinkingBlocks(s)
		}
	}

	if rawCalls, ok := out["tool_calls"].([]any); ok && len(rawCalls) > 0 {
		converted := make([]any, 0, len(rawCalls))
		for _, tc := range rawCalls {
			converted = append(converted, normalizeToolCall(tc))
		}
		out["tool_calls"] = converted
	}

	if role == "assistant" {
		if !keepReasoning {
			delete(out, "reasoning_content")
		} else if repairReasoning {
			reasoning, ok := out["reasoning_content"].(string)
			if !ok || reasoning == "" && !ok {
				delete(out, "reasoning_content")
			}
			if !ok {
				needs := assistantNeedsReasoningForToolContext(out, priorMessages)
				if needs && st != nil {
					scope := store.ConversationScope(priorMessages, cacheNamespace)
					if restored, ok := st.LookupForMessage(out, scope); ok {
						out["reasoning_content"] = restored
						patched = true
					}
				}
				if needs && !patched {
					missing = true
				}
			}
		}
	}

	allowed, ok := roleMessageFields[role]
	if !ok {
		allowed = allMessageFields
	}
	filtered := map[string]any{}
	for k, v := range out {
		if _, ok := allowed[k]; ok {
			filtered[k] = v
		}
	}
	return filtered, patched, missing
}

func assistantNeedsReasoningForToolContext(message map[string]any, priorMessages []map[string]any) bool {
	if calls, ok := message["tool_calls"].([]any); ok && len(calls) > 0 {
		return true
	}
	for i := len(priorMessages) - 1; i >= 0; i-- {
		role, _ := priorMessages[i]["role"].(string)
		if role == "tool" {
			return true
		}
		if role == "user" || role == "system" {
			return false
		}
	}
	return false
}

// NormalizeMessages normalizes a list of messages and returns the normalized
// slice plus the count of patched messages and the indexes of missing-reasoning
// messages that need recovery.
func NormalizeMessages(
	rawMessages any,
	st *store.Store,
	cacheNamespace string,
	repairReasoning, keepReasoning bool,
) (normalized []map[string]any, patched int, missingIdx []int) {
	list, ok := rawMessages.([]any)
	if !ok {
		return nil, 0, nil
	}
	for _, raw := range list {
		msg, p, m := normalizeMessage(raw, st, normalized, cacheNamespace, repairReasoning, keepReasoning)
		normalized = append(normalized, msg)
		if p {
			patched++
		}
		if m {
			missingIdx = append(missingIdx, len(normalized)-1)
		}
	}
	return normalized, patched, missingIdx
}

func hasRecoveryNotice(message map[string]any) bool {
	if role, _ := message["role"].(string); role != "assistant" {
		return false
	}
	content, _ := message["content"].(string)
	return strings.HasPrefix(content, RecoveryNoticeText) || strings.HasPrefix(content, LegacyRecoveryNoticeText)
}

func leadingSystemMessages(messages []map[string]any) []map[string]any {
	var out []map[string]any
	for _, m := range messages {
		if role, _ := m["role"].(string); role == "system" {
			out = append(out, m)
			continue
		}
		break
	}
	return out
}

// recoverMessagesFromMissingReasoning produces a recovered message list,
// returning the number of dropped messages and an optional client-visible notice.
func recoverMessagesFromMissingReasoning(
	messages []map[string]any,
	missingIndexes []int,
) (recovered []map[string]any, dropped int, notice string) {
	recoveryBoundaryIndex := -1
	for index := len(messages) - 1; index >= 0; index-- {
		if !hasRecoveryNotice(messages[index]) {
			continue
		}
		hasMissingBefore := false
		for _, mi := range missingIndexes {
			if mi < index {
				hasMissingBefore = true
				break
			}
		}
		if hasMissingBefore {
			recoveryBoundaryIndex = index
			break
		}
	}
	if recoveryBoundaryIndex != -1 {
		contextUserIndex := -1
		for i := recoveryBoundaryIndex - 1; i >= 0; i-- {
			if role, _ := messages[i]["role"].(string); role == "user" {
				contextUserIndex = i
				break
			}
		}
		leading := leadingSystemMessages(messages)
		recovered = append(recovered, leading...)
		recovered = append(recovered, map[string]any{"role": "system", "content": RecoverySystemContent})
		if contextUserIndex != -1 {
			recovered = append(recovered, messages[contextUserIndex])
		}
		recovered = append(recovered, messages[recoveryBoundaryIndex:]...)
		kept := 0
		if contextUserIndex != -1 {
			kept = 1
		}
		dropped = recoveryBoundaryIndex - len(leading) - kept
		return recovered, dropped, ""
	}
	lastUserIndex := -1
	for i := len(messages) - 1; i >= 0; i-- {
		if role, _ := messages[i]["role"].(string); role == "user" {
			lastUserIndex = i
			break
		}
	}
	if lastUserIndex == -1 {
		return messages, 0, ""
	}
	recovered = leadingSystemMessages(messages)
	dropped = len(messages) - len(recovered) - 1
	recovered = append(recovered, map[string]any{"role": "system", "content": RecoverySystemContent})
	recovered = append(recovered, messages[lastUserIndex])
	return recovered, dropped, RecoveryNoticeContent
}

func upstreamModelFor(originalModel string, cfg config.Config) string {
	if strings.HasPrefix(originalModel, "deepseek-") {
		return originalModel
	}
	return cfg.UpstreamModel
}

// ReasoningCacheNamespace returns a stable namespace for the (config, model,
// thinking, reasoning_effort, authorization) tuple. The authorization header
// is hashed so secrets are never written to disk.
func ReasoningCacheNamespace(cfg config.Config, upstreamModel string, thinking, reasoningEffort any, authorization string) string {
	authHash := ""
	if authorization != "" {
		sum := sha256.Sum256([]byte(authorization))
		authHash = hex.EncodeToString(sum[:])
	}
	payload := map[string]any{
		"base_url":           cfg.UpstreamBaseURL,
		"model":              upstreamModel,
		"thinking":           thinking,
		"reasoning_effort":   reasoningEffort,
		"authorization_hash": authHash,
	}
	canonical, _ := json.Marshal(canonicalize(payload))
	sum := sha256.Sum256(canonical)
	return hex.EncodeToString(sum[:])
}

// PrepareUpstreamRequest is the main entry point for request transformation.
func PrepareUpstreamRequest(
	payload map[string]any,
	cfg config.Config,
	st *store.Store,
	authorization string,
) PreparedRequest {
	originalModelStr, _ := payload["model"].(string)
	if originalModelStr == "" {
		originalModelStr = cfg.UpstreamModel
	}
	upstreamModel := upstreamModelFor(originalModelStr, cfg)

	prepared := map[string]any{}
	for k, v := range payload {
		if _, ok := supportedRequestFields[k]; ok {
			prepared[k] = v
		}
	}
	if _, ok := prepared["max_tokens"]; !ok {
		if v, ok := payload["max_completion_tokens"]; ok {
			prepared["max_tokens"] = v
		}
	}
	prepared["model"] = upstreamModel
	if streamVal, _ := prepared["stream"].(bool); streamVal {
		options, _ := prepared["stream_options"].(map[string]any)
		newOptions := map[string]any{}
		for k, v := range options {
			newOptions[k] = v
		}
		newOptions["include_usage"] = true
		prepared["stream_options"] = newOptions
	}

	if tools, ok := prepared["tools"].([]any); ok {
		converted := make([]any, len(tools))
		for i, tool := range tools {
			converted[i] = normalizeTool(tool)
		}
		prepared["tools"] = converted
	} else if functions, ok := payload["functions"].([]any); ok {
		converted := make([]any, len(functions))
		for i, fn := range functions {
			converted[i] = legacyFunctionToTool(fn)
		}
		prepared["tools"] = converted
	}

	if _, ok := prepared["tool_choice"]; ok {
		choice := normalizeToolChoice(prepared["tool_choice"])
		if choice == nil {
			delete(prepared, "tool_choice")
		} else {
			prepared["tool_choice"] = choice
		}
	} else if v, ok := payload["function_call"]; ok {
		if choice := convertFunctionCall(v); choice != nil {
			prepared["tool_choice"] = choice
		}
	}

	if cfg.Thinking != "pass-through" {
		prepared["thinking"] = map[string]any{"type": cfg.Thinking}
	}
	thinking, _ := prepared["thinking"].(map[string]any)
	thinkingType, _ := thinking["type"].(string)
	thinkingEnabled := thinkingType == "enabled"
	thinkingDisabled := thinkingType == "disabled"

	if thinkingEnabled {
		var rawEffort any
		if v, ok := prepared["reasoning_effort"]; ok && v != nil {
			rawEffort = v
		} else {
			rawEffort = cfg.ReasoningEffort
		}
		prepared["reasoning_effort"] = NormalizeReasoningEffort(rawEffort)
	}

	cacheNamespace := ReasoningCacheNamespace(cfg, upstreamModel, prepared["thinking"], prepared["reasoning_effort"], authorization)

	messages, patchedCount, missingIndexes := NormalizeMessages(
		payload["messages"], st, cacheNamespace,
		thinkingEnabled, !thinkingDisabled,
	)

	recoveredCount := 0
	recoveryDropped := 0
	var recoveryNotice string
	for len(missingIndexes) > 0 && cfg.MissingReasoningStrategy == "recover" {
		recoveredMessages, dropped, notice := recoverMessagesFromMissingReasoning(messages, missingIndexes)
		if dropped == 0 {
			break
		}
		recoveredCount += len(missingIndexes)
		recoveryDropped += dropped
		if notice != "" {
			recoveryNotice = notice
		}
		// Re-run normalization on the recovered message slice (converted to []any).
		converted := make([]any, len(recoveredMessages))
		for i, m := range recoveredMessages {
			converted[i] = m
		}
		messages, patchedCount, missingIndexes = NormalizeMessages(
			converted, st, cacheNamespace,
			thinkingEnabled, !thinkingDisabled,
		)
	}

	preparedMessages := make([]any, len(messages))
	for i, m := range messages {
		preparedMessages[i] = m
	}
	prepared["messages"] = preparedMessages

	return PreparedRequest{
		Payload:                    prepared,
		OriginalModel:              originalModelStr,
		UpstreamModel:              upstreamModel,
		CacheNamespace:             cacheNamespace,
		PatchedReasoningMessages:   patchedCount,
		MissingReasoningMessages:   len(missingIndexes),
		RecoveredReasoningMessages: recoveredCount,
		RecoveryDroppedMessages:    recoveryDropped,
		RecoveryNotice:             recoveryNotice,
	}
}

// RecordResponseReasoning persists assistant `reasoning_content` from a non-streaming response.
func RecordResponseReasoning(
	responsePayload map[string]any,
	st *store.Store,
	requestMessages []map[string]any,
	cacheNamespace string,
) int {
	if st == nil {
		return 0
	}
	choices, ok := responsePayload["choices"].([]any)
	if !ok {
		return 0
	}
	scope := store.ConversationScope(requestMessages, cacheNamespace)
	stored := 0
	for _, choice := range choices {
		c, ok := choice.(map[string]any)
		if !ok {
			continue
		}
		message, ok := c["message"].(map[string]any)
		if !ok {
			continue
		}
		n, _ := st.StoreAssistantMessage(message, scope)
		stored += n
	}
	return stored
}

// RewriteResponseBody parses, optionally rewrites the response body, and returns the new body bytes.
func RewriteResponseBody(
	body []byte,
	originalModel string,
	st *store.Store,
	requestMessages []map[string]any,
	cacheNamespace string,
	contentPrefix string,
) []byte {
	var responsePayload map[string]any
	dec := json.NewDecoder(strings.NewReader(string(body)))
	dec.UseNumber()
	if err := dec.Decode(&responsePayload); err != nil || responsePayload == nil {
		return body
	}
	if contentPrefix != "" {
		PrefixResponseContent(responsePayload, contentPrefix)
	}
	RecordResponseReasoning(responsePayload, st, requestMessages, cacheNamespace)
	if _, ok := responsePayload["model"]; ok {
		responsePayload["model"] = originalModel
	}
	out, err := json.Marshal(responsePayload)
	if err != nil {
		return body
	}
	return out
}

// PrefixResponseContent inserts a leading prefix into the first choice message.
func PrefixResponseContent(responsePayload map[string]any, prefix string) bool {
	choices, ok := responsePayload["choices"].([]any)
	if !ok {
		return false
	}
	for _, choice := range choices {
		c, ok := choice.(map[string]any)
		if !ok {
			continue
		}
		message, ok := c["message"].(map[string]any)
		if !ok {
			continue
		}
		content, _ := message["content"].(string)
		message["content"] = prefix + content
		return true
	}
	return false
}

// canonicalize normalizes a value for canonical JSON encoding (recursively).
// We delegate to encoding/json after sort-stabilising maps via the store package.
func canonicalize(v any) any {
	switch val := v.(type) {
	case map[string]any:
		out := map[string]any{}
		for k, vv := range val {
			out[k] = canonicalize(vv)
		}
		return out
	case []any:
		out := make([]any, len(val))
		for i, vv := range val {
			out[i] = canonicalize(vv)
		}
		return out
	default:
		return v
	}
}
