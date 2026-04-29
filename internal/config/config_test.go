package config

import (
	"os"
	"path/filepath"
	"testing"
)

func TestLoadFileMissingReturnsDefaults(t *testing.T) {
	dir := t.TempDir()
	cfg, _, err := LoadFile(filepath.Join(dir, "missing.yaml"))
	if err != nil {
		t.Fatalf("LoadFile returned error for missing path: %v", err)
	}
	if cfg.Host != DefaultHost {
		t.Errorf("expected default host, got %q", cfg.Host)
	}
	if cfg.Port != DefaultPort {
		t.Errorf("expected default port, got %d", cfg.Port)
	}
	if cfg.Thinking != DefaultThinking {
		t.Errorf("expected default thinking, got %q", cfg.Thinking)
	}
}

func TestLoadFileFromYAML(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "config.yaml")
	body := []byte(`
host: 127.0.0.1
port: 1234
base_url: https://example.com/
model: deepseek-foo
thinking: pass-through
display_reasoning: false
verbose: true
cors: true
reasoning_content_path: cache.sqlite3
missing_reasoning_strategy: reject
reasoning_cache_max_age_seconds: 60
reasoning_cache_max_rows: 5
request_timeout: 12.5
max_request_body_bytes: 9999
`)
	if err := os.WriteFile(path, body, 0o600); err != nil {
		t.Fatalf("write config: %v", err)
	}
	cfg, _, err := LoadFile(path)
	if err != nil {
		t.Fatalf("LoadFile: %v", err)
	}
	if cfg.Host != "127.0.0.1" {
		t.Errorf("host: %q", cfg.Host)
	}
	if cfg.Port != 1234 {
		t.Errorf("port: %d", cfg.Port)
	}
	if cfg.UpstreamBaseURL != "https://example.com" {
		t.Errorf("base_url should drop trailing slash, got %q", cfg.UpstreamBaseURL)
	}
	if cfg.UpstreamModel != "deepseek-foo" {
		t.Errorf("model: %q", cfg.UpstreamModel)
	}
	if cfg.Thinking != "pass-through" {
		t.Errorf("thinking: %q", cfg.Thinking)
	}
	if cfg.CursorDisplayReasoning {
		t.Errorf("display_reasoning should be false")
	}
	if !cfg.Verbose {
		t.Errorf("verbose should be true")
	}
	if !cfg.CORS {
		t.Errorf("cors should be true")
	}
	if cfg.MissingReasoningStrategy != "reject" {
		t.Errorf("missing_reasoning_strategy: %q", cfg.MissingReasoningStrategy)
	}
	if cfg.ReasoningContentPath != filepath.Join(dir, "cache.sqlite3") {
		t.Errorf("reasoning_content_path should be relative to config dir, got %q", cfg.ReasoningContentPath)
	}
	if cfg.ReasoningCacheMaxAgeSeconds != 60 {
		t.Errorf("max_age_seconds: %d", cfg.ReasoningCacheMaxAgeSeconds)
	}
	if cfg.ReasoningCacheMaxRows != 5 {
		t.Errorf("max_rows: %d", cfg.ReasoningCacheMaxRows)
	}
	if cfg.RequestTimeoutSeconds != 12.5 {
		t.Errorf("request_timeout: %v", cfg.RequestTimeoutSeconds)
	}
	if cfg.MaxRequestBodyBytes != 9999 {
		t.Errorf("max body bytes: %d", cfg.MaxRequestBodyBytes)
	}
}

func TestLoadFileInvalidYAML(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "bad.yaml")
	if err := os.WriteFile(path, []byte("not: valid: yaml: ::"), 0o600); err != nil {
		t.Fatalf("write: %v", err)
	}
	if _, _, err := LoadFile(path); err == nil {
		t.Fatalf("expected error for invalid YAML")
	}
}

func TestNormalizeThinking(t *testing.T) {
	cases := map[string]string{
		"enabled":      "enabled",
		"disabled":     "disabled",
		"PASS-THROUGH": "pass-through",
		"passthrough":  "pass-through",
		"pass_through": "pass-through",
		"unknown":      DefaultThinking,
	}
	for input, want := range cases {
		if got := NormalizeThinking(input); got != want {
			t.Errorf("NormalizeThinking(%q)=%q want %q", input, got, want)
		}
	}
}

func TestNormalizeMissingReasoningStrategy(t *testing.T) {
	if NormalizeMissingReasoningStrategy("recover") != "recover" {
		t.Fail()
	}
	if NormalizeMissingReasoningStrategy("REJECT") != "reject" {
		t.Fail()
	}
	if NormalizeMissingReasoningStrategy("nonsense") != DefaultMissingStrategy {
		t.Fail()
	}
}

func TestParseBool(t *testing.T) {
	cases := []struct {
		in       string
		fallback bool
		want     bool
	}{
		{"true", false, true},
		{"FALSE", true, false},
		{"yes", false, true},
		{"no", true, false},
		{"on", false, true},
		{"OFF", true, false},
		{"1", false, true},
		{"0", true, false},
		{"garbage", true, true},
		{"garbage", false, false},
	}
	for _, c := range cases {
		if got := ParseBool(c.in, c.fallback); got != c.want {
			t.Errorf("ParseBool(%q,%v)=%v want %v", c.in, c.fallback, got, c.want)
		}
	}
}

func TestPopulateDefaultConfigFile(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "sub", "cfg.yaml")
	if err := PopulateDefaultConfigFile(path); err != nil {
		t.Fatalf("populate: %v", err)
	}
	stat, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	if stat.Mode().Perm() != 0o600 {
		t.Errorf("expected 0600 perm, got %v", stat.Mode().Perm())
	}
}
