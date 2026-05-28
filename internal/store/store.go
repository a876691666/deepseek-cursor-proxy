// Package store implements a PocketBase-backed cache of DeepSeek
// reasoning_content keyed by canonical conversation/tool-call signatures.
package store

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"sort"
	"time"

	"github.com/pocketbase/pocketbase/core"
)

const tableName = "reasoning_cache"

// Store wraps PocketBase for reasoning cache operations.
type Store struct {
	app           core.App
	maxAgeSeconds int64
	maxRows       int64
}

// New creates a Store backed by the PocketBase reasoning_cache collection.
func New(app core.App, maxAgeSeconds, maxRows int64) *Store {
	return &Store{app: app, maxAgeSeconds: maxAgeSeconds, maxRows: maxRows}
}

// Close is a no-op (PocketBase manages the DB lifecycle).
func (s *Store) Close() error { return nil }

// Put inserts or replaces a cache entry.
func (s *Store) Put(key, reasoning string, message map[string]any) error {
	msgJSON, err := canonicalJSON(message)
	if err != nil {
		return err
	}
	rec, findErr := s.app.FindFirstRecordByData(tableName, "key", key)
	if findErr != nil {
		c, err := s.app.FindCollectionByNameOrId(tableName)
		if err != nil {
			return err
		}
		rec = core.NewRecord(c)
		rec.Set("key", key)
	}
	rec.Set("reasoning", reasoning)
	rec.Set("message_json", msgJSON)
	if err := s.app.Save(rec); err != nil {
		return err
	}
	s.prune()
	return nil
}

// Get returns the reasoning blob for a key.
func (s *Store) Get(key string) (string, bool) {
	rec, err := s.app.FindFirstRecordByData(tableName, "key", key)
	if err != nil || rec == nil {
		return "", false
	}
	return rec.GetString("reasoning"), true
}

// Clear deletes all rows.
func (s *Store) Clear() (int64, error) {
	records, err := s.app.FindAllRecords(tableName)
	if err != nil {
		return 0, err
	}
	count := int64(len(records))
	for _, r := range records {
		if err := s.app.Delete(r); err != nil {
			return count, err
		}
	}
	return count, nil
}

func (s *Store) prune() {
	if s.maxRows <= 0 && s.maxAgeSeconds <= 0 {
		return
	}

	allRecords, err := s.app.FindAllRecords(tableName)
	if err != nil {
		return
	}
	if len(allRecords) == 0 {
		return
	}

	// Sort by created descending (newest first).
	sort.Slice(allRecords, func(i, j int) bool {
		return allRecords[i].GetString("created") > allRecords[j].GetString("created")
	})

	// Prune by row count.
	if s.maxRows > 0 && len(allRecords) > int(s.maxRows) {
		for _, r := range allRecords[s.maxRows:] {
			s.app.Delete(r)
		}
		allRecords = allRecords[:int(s.maxRows)]
	}

	// Prune by age.
	if s.maxAgeSeconds > 0 {
		cutoff := time.Now().UTC().Add(-time.Duration(s.maxAgeSeconds) * time.Second).Format("2006-01-02 15:04:05.000Z")
		var remaining []*core.Record
		for _, r := range allRecords {
			if r.GetString("created") < cutoff {
				s.app.Delete(r)
			} else {
				remaining = append(remaining, r)
			}
		}
		allRecords = remaining
	}
}

// StoreAssistantMessage persists reasoning for an assistant message.
func (s *Store) StoreAssistantMessage(message map[string]any, scope string) (int, error) {
	if role, _ := message["role"].(string); role != "assistant" {
		return 0, nil
	}
	reasoning, ok := message["reasoning_content"].(string)
	if !ok {
		return 0, nil
	}
	keys := []string{fmt.Sprintf("scope:%s:signature:%s", scope, MessageSignature(message))}
	for _, id := range ToolCallIDs(message) {
		keys = append(keys, fmt.Sprintf("scope:%s:tool_call:%s", scope, id))
	}
	if rawCalls, ok := message["tool_calls"].([]any); ok {
		for _, tc := range rawCalls {
			if call, ok := tc.(map[string]any); ok {
				keys = append(keys, fmt.Sprintf("scope:%s:tool_call_signature:%s", scope, ToolCallSignature(call)))
			}
		}
	}
	for _, key := range keys {
		if err := s.Put(key, reasoning, message); err != nil {
			return 0, err
		}
	}
	return len(keys), nil
}

// LookupForMessage returns cached reasoning for a message.
func (s *Store) LookupForMessage(message map[string]any, scope string) (string, bool) {
	if reasoning, ok := s.Get(fmt.Sprintf("scope:%s:signature:%s", scope, MessageSignature(message))); ok {
		return reasoning, true
	}
	for _, id := range ToolCallIDs(message) {
		if reasoning, ok := s.Get(fmt.Sprintf("scope:%s:tool_call:%s", scope, id)); ok {
			return reasoning, true
		}
	}
	if rawCalls, ok := message["tool_calls"].([]any); ok {
		for _, tc := range rawCalls {
			if call, ok := tc.(map[string]any); ok {
				if reasoning, ok := s.Get(fmt.Sprintf("scope:%s:tool_call_signature:%s", scope, ToolCallSignature(call))); ok {
					return reasoning, true
				}
			}
		}
	}
	return "", false
}

// --- helpers ---

func NormalizeToolCall(call map[string]any) map[string]any {
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
		raw, _ := canonicalJSON(v)
		arguments = raw
	}
	name, _ := function["name"].(string)
	id, _ := call["id"].(string)
	typ, _ := call["type"].(string)
	if typ == "" {
		typ = "function"
	}
	return map[string]any{
		"id":   id,
		"type": typ,
		"function": map[string]any{
			"name":      name,
			"arguments": arguments,
		},
	}
}

func ToolCallSignature(call map[string]any) string {
	normalized := NormalizeToolCall(call)
	delete(normalized, "id")
	canonical, _ := canonicalJSON(normalized)
	return sha256Hex([]byte(canonical))
}

func MessageSignature(message map[string]any) string {
	content, _ := message["content"].(string)
	var calls []map[string]any
	if rawCalls, ok := message["tool_calls"].([]any); ok {
		for _, tc := range rawCalls {
			if call, ok := tc.(map[string]any); ok {
				calls = append(calls, NormalizeToolCall(call))
			}
		}
	}
	if calls == nil {
		calls = []map[string]any{}
	}
	canonical, _ := canonicalJSON(map[string]any{
		"content":    content,
		"tool_calls": calls,
	})
	return sha256Hex([]byte(canonical))
}

func ToolCallIDs(message map[string]any) []string {
	var ids []string
	if rawCalls, ok := message["tool_calls"].([]any); ok {
		for _, tc := range rawCalls {
			if call, ok := tc.(map[string]any); ok {
				if id, ok := call["id"].(string); ok && id != "" {
					ids = append(ids, id)
				}
			}
		}
	}
	return ids
}

func CanonicalScopeMessage(message map[string]any) map[string]any {
	canonical := map[string]any{}
	if role, ok := message["role"]; ok {
		canonical["role"] = role
	} else {
		canonical["role"] = nil
	}
	for _, key := range []string{"content", "name", "tool_call_id", "prefix"} {
		if v, ok := message[key]; ok {
			canonical[key] = v
		}
	}
	if rawCalls, ok := message["tool_calls"].([]any); ok && len(rawCalls) > 0 {
		var calls []map[string]any
		for _, tc := range rawCalls {
			if call, ok := tc.(map[string]any); ok {
				calls = append(calls, NormalizeToolCall(call))
			}
		}
		canonical["tool_calls"] = calls
	}
	return canonical
}

func ConversationScope(messages []map[string]any, namespace string) string {
	scoped := make([]map[string]any, 0, len(messages))
	for _, m := range messages {
		scoped = append(scoped, CanonicalScopeMessage(m))
	}
	var payload any = scoped
	if namespace != "" {
		payload = map[string]any{
			"namespace": namespace,
			"messages":  scoped,
		}
	}
	canonical, _ := canonicalJSON(payload)
	return sha256Hex([]byte(canonical))
}

// --- canonical JSON helpers ---

func canonicalJSON(value any) (string, error) {
	out, err := encodeCanonical(value)
	if err != nil {
		return "", err
	}
	return string(out), nil
}

func encodeCanonical(value any) ([]byte, error) {
	switch v := value.(type) {
	case nil:
		return []byte("null"), nil
	case bool:
		if v {
			return []byte("true"), nil
		}
		return []byte("false"), nil
	case string:
		return jsonEncodeString(v), nil
	case json.Number:
		return []byte(v.String()), nil
	case int:
		return []byte(fmt.Sprintf("%d", v)), nil
	case int64:
		return []byte(fmt.Sprintf("%d", v)), nil
	case float64:
		return json.Marshal(v)
	case []any:
		buf := []byte{'['}
		for i, item := range v {
			if i > 0 {
				buf = append(buf, ',')
			}
			enc, err := encodeCanonical(item)
			if err != nil {
				return nil, err
			}
			buf = append(buf, enc...)
		}
		buf = append(buf, ']')
		return buf, nil
	case []map[string]any:
		converted := make([]any, len(v))
		for i, m := range v {
			converted[i] = m
		}
		return encodeCanonical(converted)
	case map[string]any:
		keys := make([]string, 0, len(v))
		for k := range v {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		buf := []byte{'{'}
		for i, k := range keys {
			if i > 0 {
				buf = append(buf, ',')
			}
			buf = append(buf, jsonEncodeString(k)...)
			buf = append(buf, ':')
			enc, err := encodeCanonical(v[k])
			if err != nil {
				return nil, err
			}
			buf = append(buf, enc...)
		}
		buf = append(buf, '}')
		return buf, nil
	default:
		return json.Marshal(v)
	}
}

func jsonEncodeString(s string) []byte {
	buf := make([]byte, 0, len(s))
	buf = append(buf, '"')
	for _, r := range s {
		switch r {
		case '\\':
			buf = append(buf, '\\', '\\')
		case '"':
			buf = append(buf, '\\', '"')
		case '\n':
			buf = append(buf, '\\', 'n')
		case '\r':
			buf = append(buf, '\\', 'r')
		case '\t':
			buf = append(buf, '\\', 't')
		case '\b':
			buf = append(buf, '\\', 'b')
		case '\f':
			buf = append(buf, '\\', 'f')
		default:
			if r < 0x20 {
				buf = append(buf, []byte(fmt.Sprintf("\\u%04x", r))...)
			} else {
				buf = append(buf, []byte(string(r))...)
			}
		}
	}
	buf = append(buf, '"')
	return buf
}

func sha256Hex(data []byte) string {
	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:])
}
