package streaming

import (
	"path/filepath"
	"strings"
	"testing"

	"github.com/a876691666/deepseek-cursor-proxy/internal/store"
)

func newStore(t *testing.T) *store.Store {
	t.Helper()
	dir := t.TempDir()
	s, err := store.New(filepath.Join(dir, "cache.sqlite3"), 0, 0)
	if err != nil {
		t.Fatalf("store: %v", err)
	}
	t.Cleanup(func() { _ = s.Close() })
	return s
}

func TestStreamAccumulatorMergesContent(t *testing.T) {
	a := NewStreamAccumulator()
	a.IngestChunk(map[string]any{
		"choices": []any{
			map[string]any{
				"index": float64(0),
				"delta": map[string]any{"role": "assistant", "content": "hel"},
			},
		},
	})
	a.IngestChunk(map[string]any{
		"choices": []any{
			map[string]any{
				"index": float64(0),
				"delta": map[string]any{"content": "lo"},
			},
		},
	})
	msgs := a.Messages()
	if len(msgs) != 1 {
		t.Fatalf("expected 1 message, got %d", len(msgs))
	}
	if msgs[0]["content"] != "hello" {
		t.Errorf("content: %v", msgs[0]["content"])
	}
}

func TestStreamAccumulatorReasoningAndToolCalls(t *testing.T) {
	a := NewStreamAccumulator()
	a.IngestChunk(map[string]any{
		"choices": []any{
			map[string]any{
				"index": float64(0),
				"delta": map[string]any{"reasoning_content": "thinking..."},
			},
		},
	})
	a.IngestChunk(map[string]any{
		"choices": []any{
			map[string]any{
				"index": float64(0),
				"delta": map[string]any{
					"tool_calls": []any{
						map[string]any{
							"index":    float64(0),
							"id":       "tc-1",
							"type":     "function",
							"function": map[string]any{"name": "do", "arguments": `{"x":`},
						},
					},
				},
			},
		},
	})
	a.IngestChunk(map[string]any{
		"choices": []any{
			map[string]any{
				"index": float64(0),
				"delta": map[string]any{
					"tool_calls": []any{
						map[string]any{
							"index":    float64(0),
							"function": map[string]any{"arguments": `1}`},
						},
					},
				},
			},
		},
	})
	msgs := a.Messages()
	msg := msgs[0]
	if msg["reasoning_content"] != "thinking..." {
		t.Errorf("reasoning: %v", msg["reasoning_content"])
	}
	calls, _ := msg["tool_calls"].([]any)
	if len(calls) != 1 {
		t.Fatalf("tool_calls: %v", calls)
	}
	call, _ := calls[0].(map[string]any)
	function, _ := call["function"].(map[string]any)
	if function["name"] != "do" {
		t.Errorf("name: %v", function["name"])
	}
	if function["arguments"] != `{"x":1}` {
		t.Errorf("arguments concat: %v", function["arguments"])
	}
}

func TestStreamAccumulatorStoresOnFinish(t *testing.T) {
	st := newStore(t)
	a := NewStreamAccumulator()
	a.IngestChunk(map[string]any{
		"choices": []any{
			map[string]any{
				"index": float64(0),
				"delta": map[string]any{"role": "assistant", "reasoning_content": "rc", "content": "answer"},
			},
		},
	})
	a.IngestChunk(map[string]any{
		"choices": []any{
			map[string]any{
				"index":         float64(0),
				"delta":         map[string]any{},
				"finish_reason": "stop",
			},
		},
	})
	stored := a.StoreReadyReasoning(st, "scope")
	if stored == 0 {
		t.Errorf("expected store on finish")
	}
	// Storing again should be a no-op.
	if again := a.StoreReadyReasoning(st, "scope"); again != 0 {
		t.Errorf("expected dedup, got %d", again)
	}
}

func TestStreamAccumulatorStoresOnIdentifiedToolCalls(t *testing.T) {
	st := newStore(t)
	a := NewStreamAccumulator()
	a.IngestChunk(map[string]any{
		"choices": []any{
			map[string]any{
				"index": float64(0),
				"delta": map[string]any{
					"reasoning_content": "rc",
					"tool_calls": []any{
						map[string]any{"index": float64(0), "id": "ID-1", "function": map[string]any{"name": "f"}},
					},
				},
			},
		},
	})
	stored := a.StoreReadyReasoning(st, "scope")
	if stored == 0 {
		t.Errorf("expected to store once tool_call ids known")
	}
}

func TestCursorReasoningDisplayAdapterOpensAndClosesBlock(t *testing.T) {
	a := NewCursorReasoningDisplayAdapter()
	chunk := map[string]any{
		"id": "x",
		"choices": []any{
			map[string]any{
				"index": float64(0),
				"delta": map[string]any{"reasoning_content": "thinking..."},
			},
		},
	}
	a.RewriteChunk(chunk)
	choices, _ := chunk["choices"].([]any)
	first, _ := choices[0].(map[string]any)
	delta, _ := first["delta"].(map[string]any)
	content, _ := delta["content"].(string)
	if !strings.HasPrefix(content, "<think>") || !strings.Contains(content, "thinking...") {
		t.Errorf("expected open think block in content, got %q", content)
	}

	chunk2 := map[string]any{
		"choices": []any{
			map[string]any{
				"index": float64(0),
				"delta": map[string]any{"content": "real answer"},
			},
		},
	}
	a.RewriteChunk(chunk2)
	c2, _ := chunk2["choices"].([]any)
	d2, _ := c2[0].(map[string]any)["delta"].(map[string]any)
	cText := d2["content"].(string)
	if !strings.Contains(cText, "</think>") {
		t.Errorf("expected closing block, got %q", cText)
	}
	if !strings.Contains(cText, "real answer") {
		t.Errorf("expected answer preserved, got %q", cText)
	}
}

func TestCursorReasoningDisplayAdapterFlushChunk(t *testing.T) {
	a := NewCursorReasoningDisplayAdapter()
	a.RewriteChunk(map[string]any{
		"id":      "X",
		"created": 100,
		"choices": []any{
			map[string]any{
				"index": float64(0),
				"delta": map[string]any{"reasoning_content": "stuff"},
			},
		},
	})
	closing := a.FlushChunk("rebrand")
	if closing == nil {
		t.Fatalf("expected flush chunk for open block")
	}
	if closing["model"] != "rebrand" {
		t.Errorf("model: %v", closing["model"])
	}
	choices, _ := closing["choices"].([]any)
	if len(choices) != 1 {
		t.Errorf("choices: %v", choices)
	}
	if a.FlushChunk("rebrand") != nil {
		t.Errorf("expected nil after flush")
	}
}
