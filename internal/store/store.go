// Package store implements the SQLite-backed cache of DeepSeek reasoning_content
// keyed by canonical conversation/tool-call signatures.
package store

import (
	"crypto/sha256"
	"database/sql"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"sync"
	"time"

	_ "modernc.org/sqlite"
)

// Store wraps a SQLite database holding reasoning_content blobs.
type Store struct {
	mu            sync.Mutex
	db            *sql.DB
	maxAgeSeconds int64
	maxRows       int64
	path          string
}

// New opens (creating directories if necessary) the SQLite cache. Pass ":memory:"
// for an in-memory DB used in tests.
func New(path string, maxAgeSeconds, maxRows int64) (*Store, error) {
	dsn := path
	if path != ":memory:" {
		dir := filepath.Dir(path)
		if err := os.MkdirAll(dir, 0o700); err != nil {
			return nil, fmt.Errorf("create cache dir: %w", err)
		}
		_ = os.Chmod(dir, 0o700)
		dsn = "file:" + path + "?_pragma=journal_mode(WAL)&_pragma=busy_timeout(5000)"
	} else {
		dsn = "file::memory:?cache=shared"
	}
	db, err := sql.Open("sqlite", dsn)
	if err != nil {
		return nil, fmt.Errorf("open sqlite: %w", err)
	}
	db.SetMaxOpenConns(1)
	if path != ":memory:" {
		_ = os.Chmod(path, 0o600)
	}
	if _, err := db.Exec(`
        CREATE TABLE IF NOT EXISTS reasoning_cache (
            key TEXT PRIMARY KEY,
            reasoning TEXT NOT NULL,
            message_json TEXT NOT NULL,
            created_at REAL NOT NULL
        )
    `); err != nil {
		_ = db.Close()
		return nil, fmt.Errorf("init schema: %w", err)
	}
	s := &Store{db: db, maxAgeSeconds: maxAgeSeconds, maxRows: maxRows, path: path}
	if _, err := s.Prune(); err != nil {
		_ = db.Close()
		return nil, fmt.Errorf("prune: %w", err)
	}
	return s, nil
}

// Close releases the underlying database handle.
func (s *Store) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.db == nil {
		return nil
	}
	err := s.db.Close()
	s.db = nil
	return err
}

// Put inserts or replaces a single cache entry.
func (s *Store) Put(key, reasoning string, message map[string]any) error {
	messageJSON, err := canonicalJSON(message)
	if err != nil {
		return err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, err := s.db.Exec(`
        INSERT INTO reasoning_cache(key, reasoning, message_json, created_at)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(key) DO UPDATE SET
            reasoning = excluded.reasoning,
            message_json = excluded.message_json,
            created_at = excluded.created_at
    `, key, reasoning, messageJSON, float64(time.Now().UnixNano())/1e9); err != nil {
		return err
	}
	if _, err := s.pruneLocked(); err != nil {
		return err
	}
	return nil
}

// Get returns the reasoning blob for a key, or "" with ok=false if missing.
func (s *Store) Get(key string) (string, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	row := s.db.QueryRow("SELECT reasoning FROM reasoning_cache WHERE key = ?", key)
	var reasoning string
	if err := row.Scan(&reasoning); err != nil {
		return "", false
	}
	return reasoning, true
}

// Clear deletes all rows and returns the number of rows that were present.
func (s *Store) Clear() (int64, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	var count int64
	_ = s.db.QueryRow("SELECT COUNT(*) FROM reasoning_cache").Scan(&count)
	if _, err := s.db.Exec("DELETE FROM reasoning_cache"); err != nil {
		return 0, err
	}
	return count, nil
}

// Prune trims expired rows and enforces the row cap.
func (s *Store) Prune() (int64, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.pruneLocked()
}

func (s *Store) pruneLocked() (int64, error) {
	var deleted int64
	if s.maxAgeSeconds > 0 {
		cutoff := float64(time.Now().UnixNano())/1e9 - float64(s.maxAgeSeconds)
		res, err := s.db.Exec("DELETE FROM reasoning_cache WHERE created_at < ?", cutoff)
		if err != nil {
			return deleted, err
		}
		if rows, err := res.RowsAffected(); err == nil {
			deleted += rows
		}
	}
	if s.maxRows > 0 {
		res, err := s.db.Exec(`
            DELETE FROM reasoning_cache
            WHERE key NOT IN (
                SELECT key FROM reasoning_cache
                ORDER BY created_at DESC LIMIT ?
            )
        `, s.maxRows)
		if err != nil {
			return deleted, err
		}
		if rows, err := res.RowsAffected(); err == nil {
			deleted += rows
		}
	}
	return deleted, nil
}

// StoreAssistantMessage persists reasoning for an assistant message under all
// derived cache keys (signature, tool_call_id, tool_call_signature). Returns
// the number of keys written.
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

// LookupForMessage returns cached reasoning for a message, checking signature,
// then per-tool-call IDs, then per-tool-call signatures.
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

// NormalizeToolCall returns a canonical form of a tool call entry. The
// `function.arguments` field is serialised to a sorted-key JSON string when
// supplied as anything other than a string.
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
	out := map[string]any{
		"id":   id,
		"type": typ,
		"function": map[string]any{
			"name":      name,
			"arguments": arguments,
		},
	}
	return out
}

// ToolCallSignature hashes a tool call (excluding its id).
func ToolCallSignature(call map[string]any) string {
	normalized := NormalizeToolCall(call)
	delete(normalized, "id")
	canonical, _ := canonicalJSON(normalized)
	return sha256Hex([]byte(canonical))
}

// MessageSignature hashes the (content, tool_calls) pair of a message.
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
	payload := map[string]any{
		"content":    content,
		"tool_calls": calls,
	}
	canonical, _ := canonicalJSON(payload)
	return sha256Hex([]byte(canonical))
}

// ToolCallIDs returns the list of non-empty tool_call ids in a message.
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

// CanonicalScopeMessage builds the canonical projection used for conversation
// scope hashing — preserves role/content/tool_calls metadata only.
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

// ConversationScope returns a stable hex digest scoping all messages in the
// supplied namespace.
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

// canonicalJSON encodes data in a Python-`json.dumps(..., sort_keys=True,
// separators=(",", ":"), ensure_ascii=False)` compatible way: sorted keys, no
// spaces, UTF-8 passthrough (non-ASCII not escaped).
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
		// Match Python's repr() behaviour for whole numbers ("1.0" vs "1") only
		// matters for tests; encoding/json handles general case well enough.
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
	// Use encoding/json for a safe escape of control characters and quotes,
	// but with HTMLEscape disabled would still escape <, >, &. Python's
	// json.dumps with ensure_ascii=False does not escape <, >, &. Compose
	// manually to match Python output.
	buf := make([]byte, 0, len(s)+2)
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
