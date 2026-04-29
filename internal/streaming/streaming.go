// Package streaming implements the SSE accumulator for DeepSeek streaming
// responses and the Cursor "thinking-block" display adapter.
package streaming

import (
	"sort"
	"time"

	"github.com/a876691666/deepseek-cursor-proxy/internal/store"
)

const (
	thinkingBlockStart = "<think>\n"
	thinkingBlockEnd   = "\n</think>\n\n"
)

// StreamingChoice is the per-index assistant accumulator for streamed deltas.
type StreamingChoice struct {
	Role                string
	Content             string
	ReasoningContent    string
	HasReasoningContent bool
	ToolCalls           []map[string]any
	FinishReason        string
	HasFinishReason     bool
}

// ToMessage produces a message dict suitable for the reasoning store.
func (c *StreamingChoice) ToMessage() map[string]any {
	role := c.Role
	if role == "" {
		role = "assistant"
	}
	msg := map[string]any{
		"role":    role,
		"content": c.Content,
	}
	if c.HasReasoningContent {
		msg["reasoning_content"] = c.ReasoningContent
	}
	if len(c.ToolCalls) > 0 {
		toolCalls := make([]any, len(c.ToolCalls))
		for i, tc := range c.ToolCalls {
			toolCalls[i] = tc
		}
		msg["tool_calls"] = toolCalls
	}
	return msg
}

// StreamAccumulator merges streaming chunks into per-choice state and tracks
// which choices have already had their reasoning persisted.
type StreamAccumulator struct {
	Choices        map[int]*StreamingChoice
	storedChoices  map[int]string // index -> stored stage ("tool_call" | "final")
}

// NewStreamAccumulator returns a fresh accumulator.
func NewStreamAccumulator() *StreamAccumulator {
	return &StreamAccumulator{
		Choices:       map[int]*StreamingChoice{},
		storedChoices: map[int]string{},
	}
}

// IngestChunk merges a single SSE chunk into the accumulator.
func (a *StreamAccumulator) IngestChunk(chunk map[string]any) {
	rawChoices, ok := chunk["choices"].([]any)
	if !ok {
		return
	}
	for _, raw := range rawChoices {
		rawChoice, ok := raw.(map[string]any)
		if !ok {
			continue
		}
		index := chunkIndex(rawChoice)
		choice := a.Choices[index]
		if choice == nil {
			choice = &StreamingChoice{}
			a.Choices[index] = choice
		}
		if reason, ok := rawChoice["finish_reason"].(string); ok && reason != "" {
			choice.FinishReason = reason
			choice.HasFinishReason = true
		}
		delta, ok := rawChoice["delta"].(map[string]any)
		if !ok {
			continue
		}
		if role, ok := delta["role"].(string); ok && role != "" {
			choice.Role = role
		}
		if content, ok := delta["content"].(string); ok {
			choice.Content += content
		}
		if reasoning, ok := delta["reasoning_content"].(string); ok {
			choice.HasReasoningContent = true
			choice.ReasoningContent += reasoning
		}
		mergeToolCallDeltas(choice, delta["tool_calls"])
	}
}

// StoreReasoning persists reasoning for every accumulated choice.
func (a *StreamAccumulator) StoreReasoning(st *store.Store, scope string) int {
	stored := 0
	for index, choice := range a.Choices {
		stored += a.storeChoice(index, choice, st, scope, "final")
	}
	return stored
}

// StoreReadyReasoning persists reasoning for choices that are either finished
// or have fully-identified tool calls.
func (a *StreamAccumulator) StoreReadyReasoning(st *store.Store, scope string) int {
	stored := 0
	for index, choice := range a.Choices {
		if choice.HasFinishReason {
			stored += a.storeChoice(index, choice, st, scope, "final")
		} else if hasIdentifiedToolCalls(choice) {
			stored += a.storeChoice(index, choice, st, scope, "tool_call")
		}
	}
	return stored
}

// Messages returns the accumulated assistant messages sorted by choice index.
func (a *StreamAccumulator) Messages() []map[string]any {
	indexes := make([]int, 0, len(a.Choices))
	for i := range a.Choices {
		indexes = append(indexes, i)
	}
	sort.Ints(indexes)
	out := make([]map[string]any, 0, len(indexes))
	for _, i := range indexes {
		out = append(out, a.Choices[i].ToMessage())
	}
	return out
}

func (a *StreamAccumulator) storeChoice(index int, choice *StreamingChoice, st *store.Store, scope, stage string) int {
	rank := map[string]int{"tool_call": 1, "final": 2}
	if r := rank[a.storedChoices[index]]; r >= rank[stage] {
		return 0
	}
	stored, _ := st.StoreAssistantMessage(choice.ToMessage(), scope)
	if stored > 0 {
		a.storedChoices[index] = stage
	}
	return stored
}

func hasIdentifiedToolCalls(choice *StreamingChoice) bool {
	if !choice.HasReasoningContent || len(choice.ToolCalls) == 0 {
		return false
	}
	for _, tc := range choice.ToolCalls {
		id, _ := tc["id"].(string)
		if id == "" {
			return false
		}
	}
	return true
}

func chunkIndex(rawChoice map[string]any) int {
	switch v := rawChoice["index"].(type) {
	case float64:
		return int(v)
	case int:
		return v
	case int64:
		return int(v)
	}
	return 0
}

func mergeToolCallDeltas(choice *StreamingChoice, deltas any) {
	list, ok := deltas.([]any)
	if !ok {
		return
	}
	for _, raw := range list {
		rawDelta, ok := raw.(map[string]any)
		if !ok {
			continue
		}
		var index int
		if iv, ok := rawDelta["index"].(float64); ok {
			index = int(iv)
		} else if iv, ok := rawDelta["index"].(int); ok {
			index = iv
		} else {
			index = len(choice.ToolCalls)
		}
		for len(choice.ToolCalls) <= index {
			choice.ToolCalls = append(choice.ToolCalls, map[string]any{
				"type":     "function",
				"function": map[string]any{"name": "", "arguments": ""},
			})
		}
		toolCall := choice.ToolCalls[index]
		if id, ok := rawDelta["id"].(string); ok && id != "" {
			toolCall["id"] = id
		}
		if typ, ok := rawDelta["type"].(string); ok && typ != "" {
			toolCall["type"] = typ
		}
		functionDelta, ok := rawDelta["function"].(map[string]any)
		if !ok {
			continue
		}
		function, ok := toolCall["function"].(map[string]any)
		if !ok {
			function = map[string]any{"name": "", "arguments": ""}
			toolCall["function"] = function
		}
		if name, ok := functionDelta["name"].(string); ok && name != "" {
			existing, _ := function["name"].(string)
			if existing == "" {
				function["name"] = name
			} else {
				function["name"] = existing + name
			}
		}
		if v, ok := functionDelta["arguments"]; ok && v != nil {
			existing, _ := function["arguments"].(string)
			switch a := v.(type) {
			case string:
				function["arguments"] = existing + a
			default:
				// Preserve via fmt.Sprintf to remain compatible with non-string deltas.
				function["arguments"] = existing + jsonString(a)
			}
		}
	}
}

func jsonString(v any) string {
	switch s := v.(type) {
	case string:
		return s
	default:
		return ""
	}
}

// CursorReasoningDisplayAdapter mirrors `reasoning_content` chunks into the
// `content` channel using <think>…</think> blocks so Cursor can render them.
type CursorReasoningDisplayAdapter struct {
	openChoices       map[int]struct{}
	lastChunkMetadata map[string]any
}

// NewCursorReasoningDisplayAdapter returns a new adapter ready for use.
func NewCursorReasoningDisplayAdapter() *CursorReasoningDisplayAdapter {
	return &CursorReasoningDisplayAdapter{
		openChoices:       map[int]struct{}{},
		lastChunkMetadata: map[string]any{},
	}
}

// RewriteChunk mutates the chunk in place to mirror reasoning content into the
// content delta stream.
func (a *CursorReasoningDisplayAdapter) RewriteChunk(chunk map[string]any) {
	a.rememberChunkMetadata(chunk)
	rawChoices, ok := chunk["choices"].([]any)
	if !ok {
		return
	}
	for _, raw := range rawChoices {
		rawChoice, ok := raw.(map[string]any)
		if !ok {
			continue
		}
		index := chunkIndex(rawChoice)
		delta, ok := rawChoice["delta"].(map[string]any)
		if !ok {
			delta = map[string]any{}
			rawChoice["delta"] = delta
		}
		var mirroredParts []string
		reasoning, _ := delta["reasoning_content"].(string)
		if reasoning != "" {
			if _, open := a.openChoices[index]; !open {
				mirroredParts = append(mirroredParts, thinkingBlockStart)
				a.openChoices[index] = struct{}{}
			}
			mirroredParts = append(mirroredParts, reasoning)
		}
		existingContent, hasContent := delta["content"].(string)
		_, openNow := a.openChoices[index]
		shouldClose := openNow && (hasContent && existingContent != "" || hasNonEmptyToolCalls(delta) || hasNonEmptyFinishReason(rawChoice))
		if shouldClose {
			mirroredParts = append(mirroredParts, thinkingBlockEnd)
			delete(a.openChoices, index)
		}
		if len(mirroredParts) == 0 {
			continue
		}
		if hasContent {
			mirroredParts = append(mirroredParts, existingContent)
		}
		delta["content"] = joinParts(mirroredParts)
	}
}

// FlushChunk produces a closing chunk for any open thinking blocks. Returns
// nil if there is nothing to flush.
func (a *CursorReasoningDisplayAdapter) FlushChunk(model string) map[string]any {
	if len(a.openChoices) == 0 {
		return nil
	}
	indexes := make([]int, 0, len(a.openChoices))
	for i := range a.openChoices {
		indexes = append(indexes, i)
	}
	sort.Ints(indexes)
	choices := make([]any, 0, len(indexes))
	for _, i := range indexes {
		choices = append(choices, map[string]any{
			"index":         i,
			"delta":         map[string]any{"content": thinkingBlockEnd},
			"finish_reason": nil,
		})
	}
	a.openChoices = map[int]struct{}{}
	id, _ := a.lastChunkMetadata["id"].(string)
	if id == "" {
		id = "chatcmpl-reasoning-close"
	}
	object, _ := a.lastChunkMetadata["object"].(string)
	if object == "" {
		object = "chat.completion.chunk"
	}
	created, ok := a.lastChunkMetadata["created"]
	if !ok {
		created = time.Now().Unix()
	}
	return map[string]any{
		"id":      id,
		"object":  object,
		"created": created,
		"model":   model,
		"choices": choices,
	}
}

func (a *CursorReasoningDisplayAdapter) rememberChunkMetadata(chunk map[string]any) {
	for _, key := range []string{"id", "object", "created"} {
		if v, ok := chunk[key]; ok {
			a.lastChunkMetadata[key] = v
		}
	}
}

func hasNonEmptyToolCalls(delta map[string]any) bool {
	if calls, ok := delta["tool_calls"].([]any); ok {
		return len(calls) > 0
	}
	return false
}

func hasNonEmptyFinishReason(rawChoice map[string]any) bool {
	if v, ok := rawChoice["finish_reason"]; ok && v != nil {
		if s, ok := v.(string); ok {
			return s != ""
		}
		return true
	}
	return false
}

func joinParts(parts []string) string {
	total := 0
	for _, p := range parts {
		total += len(p)
	}
	out := make([]byte, 0, total)
	for _, p := range parts {
		out = append(out, p...)
	}
	return string(out)
}
