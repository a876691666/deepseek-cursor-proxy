package store

import (
	"testing"
	"time"

	"github.com/pocketbase/pocketbase"
	"github.com/pocketbase/pocketbase/core"
)

func newStore(t *testing.T, maxRows int64) *Store {
	t.Helper()
	dir := t.TempDir()
	pb := pocketbase.NewWithConfig(pocketbase.Config{
		DefaultDataDir:  dir,
		HideStartBanner: true,
	})
	if err := pb.Bootstrap(); err != nil {
		t.Fatalf("bootstrap: %v", err)
	}
	t.Cleanup(func() { pb.ResetBootstrapState() })
	if _, err := pb.FindCollectionByNameOrId(tableName); err != nil {
		c := core.NewBaseCollection(tableName)
		c.Fields = core.FieldsList{
			&core.TextField{Name: "key", Required: true},
			&core.TextField{Name: "reasoning"},
			&core.TextField{Name: "message_json"},
		}
		if err := pb.Save(c); err != nil {
			t.Fatalf("create collection: %v", err)
		}
	}
	t.Cleanup(func() { pb.ResetBootstrapState() })
	return New(pb, 0, maxRows)
}

func TestPutGetRoundtrip(t *testing.T) {
	s := newStore(t, 0)
	if err := s.Put("k1", "thinking...", map[string]any{"role": "assistant"}); err != nil {
		t.Fatal(err)
	}
	v, ok := s.Get("k1")
	if !ok || v != "thinking..." {
		t.Errorf("got (%q,%v) want (thinking...,true)", v, ok)
	}
	if _, ok := s.Get("missing"); ok {
		t.Errorf("expected missing key")
	}
}

func TestStoreAssistantMessageKeys(t *testing.T) {
	s := newStore(t, 0)
	message := map[string]any{
		"role":              "assistant",
		"content":           "answer",
		"reasoning_content": "thinking",
		"tool_calls": []any{
			map[string]any{
				"id":   "call_1",
				"type": "function",
				"function": map[string]any{
					"name":      "do_thing",
					"arguments": `{"x":1}`,
				},
			},
		},
	}
	count, err := s.StoreAssistantMessage(message, "scope-A")
	if err != nil {
		t.Fatal(err)
	}
	if count != 3 {
		t.Errorf("expected 3 keys (signature + tool_call id + tool_call signature), got %d", count)
	}
	got, ok := s.LookupForMessage(message, "scope-A")
	if !ok || got != "thinking" {
		t.Errorf("LookupForMessage: %q ok=%v", got, ok)
	}
	if _, ok := s.LookupForMessage(message, "other-scope"); ok {
		t.Errorf("scope must isolate keys")
	}
}

func TestLookupByToolCallID(t *testing.T) {
	s := newStore(t, 0)
	original := map[string]any{
		"role":              "assistant",
		"content":           "ans",
		"reasoning_content": "REASON",
		"tool_calls": []any{
			map[string]any{
				"id":   "abc",
				"type": "function",
				"function": map[string]any{
					"name":      "f",
					"arguments": "{}",
				},
			},
		},
	}
	if _, err := s.StoreAssistantMessage(original, "S"); err != nil {
		t.Fatal(err)
	}
	probe := map[string]any{
		"role":    "assistant",
		"content": "different content",
		"tool_calls": []any{
			map[string]any{
				"id":   "abc",
				"type": "function",
				"function": map[string]any{
					"name":      "f",
					"arguments": "{}",
				},
			},
		},
	}
	got, ok := s.LookupForMessage(probe, "S")
	if !ok || got != "REASON" {
		t.Errorf("expected lookup-by-id hit; got=%q ok=%v", got, ok)
	}
}

func TestNonAssistantStoreNoOp(t *testing.T) {
	s := newStore(t, 0)
	count, err := s.StoreAssistantMessage(map[string]any{"role": "user", "reasoning_content": "x"}, "S")
	if err != nil {
		t.Fatal(err)
	}
	if count != 0 {
		t.Errorf("expected non-assistant to be skipped")
	}
}

func TestPruneByRows(t *testing.T) {
	s := newStore(t, 2)
	for i := 0; i < 5; i++ {
		if err := s.Put("k"+string(rune('a'+i)), "x", map[string]any{}); err != nil {
			t.Fatal(err)
		}
		time.Sleep(time.Millisecond * 2)
	}
	// After inserting 5 records with maxRows=2, at most 2 should remain.
	records, err := s.app.FindAllRecords(tableName)
	if err != nil {
		t.Fatal(err)
	}
	if len(records) > 2 {
		t.Errorf("expected at most 2 records after prune, got %d", len(records))
	}
}

func TestClear(t *testing.T) {
	s := newStore(t, 0)
	_ = s.Put("k", "v", map[string]any{})
	count, err := s.Clear()
	if err != nil {
		t.Fatal(err)
	}
	if count != 1 {
		t.Errorf("expected 1 cleared row, got %d", count)
	}
	if _, ok := s.Get("k"); ok {
		t.Errorf("key should be cleared")
	}
}

func TestConversationScopeStable(t *testing.T) {
	msgs := []map[string]any{
		{"role": "user", "content": "hi"},
		{"role": "assistant", "content": "hello"},
	}
	a := ConversationScope(msgs, "ns-1")
	b := ConversationScope(msgs, "ns-1")
	if a != b {
		t.Errorf("scope should be stable: %s vs %s", a, b)
	}
	c := ConversationScope(msgs, "ns-2")
	if a == c {
		t.Errorf("different namespaces should produce different scopes")
	}
}

func TestMessageSignatureIgnoresNonContentFields(t *testing.T) {
	a := MessageSignature(map[string]any{
		"role":    "assistant",
		"content": "x",
	})
	b := MessageSignature(map[string]any{
		"role":    "assistant",
		"content": "x",
		"name":    "ignored",
	})
	if a != b {
		t.Errorf("signature should ignore role/name fields")
	}
}

func TestToolCallSignatureIgnoresID(t *testing.T) {
	tc1 := map[string]any{"id": "id-1", "type": "function", "function": map[string]any{"name": "n", "arguments": "{}"}}
	tc2 := map[string]any{"id": "id-2", "type": "function", "function": map[string]any{"name": "n", "arguments": "{}"}}
	if ToolCallSignature(tc1) != ToolCallSignature(tc2) {
		t.Errorf("signature should ignore tool_call id")
	}
}
