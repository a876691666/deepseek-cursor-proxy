// Package config loads YAML configuration with sensible defaults and supports
// command-line overrides for the DeepSeek Cursor proxy.
package config

import (
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"gopkg.in/yaml.v3"
)

const (
	AppDirName                = ".deepseek-cursor-proxy"
	ConfigFileName            = "config.yaml"
	ReasoningContentFileName  = "reasoning_content.sqlite3"
	DefaultHost               = "0.0.0.0"
	DefaultPort               = 9000
	DefaultUpstreamBaseURL    = "https://api.deepseek.com"
	DefaultUpstreamModel      = "deepseek-v4-pro"
	DefaultThinking           = "enabled"
	DefaultReasoningEffort    = "high"
	DefaultDisplayReasoning   = true
	DefaultVerbose            = false
	DefaultRequestTimeout     = 300.0
	DefaultMaxRequestBody     = 20 * 1024 * 1024
	DefaultCORS               = false
	DefaultMissingStrategy    = "recover"
	DefaultCacheMaxAgeSeconds = 30 * 24 * 60 * 60
	DefaultCacheMaxRows       = 100_000
)

// DefaultConfigText is written to disk on first run.
const DefaultConfigText = `# This file was created automatically at ~/.deepseek-cursor-proxy/config.yaml.
# API keys are read from Cursor's Authorization header and forwarded upstream.

# ` + "`model`" + ` is the fallback when a request has no model; Cursor's requested
# DeepSeek model name is otherwise respected.
base_url: https://api.deepseek.com
model: deepseek-v4-pro
thinking: enabled
reasoning_effort: high
display_reasoning: true

host: 0.0.0.0
port: 9000
verbose: false
request_timeout: 300
max_request_body_bytes: 20971520
cors: false

reasoning_content_path: reasoning_content.sqlite3
missing_reasoning_strategy: recover
reasoning_cache_max_age_seconds: 2592000
reasoning_cache_max_rows: 100000
`

// Config holds resolved proxy settings.
type Config struct {
	Host                        string
	Port                        int
	UpstreamBaseURL             string
	UpstreamModel               string
	Thinking                    string
	ReasoningEffort             string
	RequestTimeoutSeconds       float64
	MaxRequestBodyBytes         int64
	ReasoningContentPath        string
	MissingReasoningStrategy    string
	ReasoningCacheMaxAgeSeconds int64
	ReasoningCacheMaxRows       int64
	CursorDisplayReasoning      bool
	CORS                        bool
	Verbose                     bool
}

// rawConfig matches the YAML structure on disk.
type rawConfig struct {
	Host                        *string  `yaml:"host"`
	Port                        *int     `yaml:"port"`
	BaseURL                     *string  `yaml:"base_url"`
	Model                       *string  `yaml:"model"`
	Thinking                    *string  `yaml:"thinking"`
	ReasoningEffort             *string  `yaml:"reasoning_effort"`
	DisplayReasoning            *bool    `yaml:"display_reasoning"`
	Verbose                     *bool    `yaml:"verbose"`
	RequestTimeout              *float64 `yaml:"request_timeout"`
	MaxRequestBodyBytes         *int64   `yaml:"max_request_body_bytes"`
	CORS                        *bool    `yaml:"cors"`
	ReasoningContentPath        *string  `yaml:"reasoning_content_path"`
	MissingReasoningStrategy    *string  `yaml:"missing_reasoning_strategy"`
	ReasoningCacheMaxAgeSeconds *int64   `yaml:"reasoning_cache_max_age_seconds"`
	ReasoningCacheMaxRows       *int64   `yaml:"reasoning_cache_max_rows"`
}

// DefaultAppDir returns ~/.deepseek-cursor-proxy.
func DefaultAppDir() string {
	home, err := os.UserHomeDir()
	if err != nil {
		return AppDirName
	}
	return filepath.Join(home, AppDirName)
}

// DefaultConfigPath returns the default config file location.
func DefaultConfigPath() string {
	return filepath.Join(DefaultAppDir(), ConfigFileName)
}

// DefaultReasoningContentPath returns the default SQLite cache location.
func DefaultReasoningContentPath() string {
	return filepath.Join(DefaultAppDir(), ReasoningContentFileName)
}

// PopulateDefaultConfigFile writes the default YAML config to disk and applies
// restrictive permissions.
func PopulateDefaultConfigFile(path string) error {
	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return err
	}
	_ = os.Chmod(dir, 0o700)
	if err := os.WriteFile(path, []byte(DefaultConfigText), 0o600); err != nil {
		return err
	}
	return nil
}

// ResolveConfigPath expands ~ in user-supplied paths.
func ResolveConfigPath(path string) string {
	if path == "" {
		return DefaultConfigPath()
	}
	return expandUser(path)
}

func expandUser(path string) string {
	if !strings.HasPrefix(path, "~") {
		return path
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return path
	}
	if path == "~" {
		return home
	}
	if strings.HasPrefix(path, "~/") {
		return filepath.Join(home, path[2:])
	}
	return path
}

// LoadFile loads (and creates if missing) a config file then returns the
// resolved Config along with the path that was read.
func LoadFile(path string) (Config, string, error) {
	resolved := ResolveConfigPath(path)
	autoCreate := path == ""
	if autoCreate {
		if _, err := os.Stat(resolved); os.IsNotExist(err) {
			if err := PopulateDefaultConfigFile(resolved); err != nil {
				return Config{}, resolved, fmt.Errorf("create default config: %w", err)
			}
		}
	}
	cfg := Defaults()
	data, err := os.ReadFile(resolved)
	if err != nil {
		if os.IsNotExist(err) {
			return cfg, resolved, nil
		}
		return cfg, resolved, fmt.Errorf("read config %s: %w", resolved, err)
	}
	var raw rawConfig
	if len(strings.TrimSpace(string(data))) > 0 {
		if err := yaml.Unmarshal(data, &raw); err != nil {
			return cfg, resolved, fmt.Errorf("invalid YAML config at %s: %w", resolved, err)
		}
	}
	applyRaw(&cfg, raw, filepath.Dir(resolved))
	return cfg, resolved, nil
}

// Defaults returns a Config populated with all default values.
func Defaults() Config {
	return Config{
		Host:                        DefaultHost,
		Port:                        DefaultPort,
		UpstreamBaseURL:             DefaultUpstreamBaseURL,
		UpstreamModel:               DefaultUpstreamModel,
		Thinking:                    DefaultThinking,
		ReasoningEffort:             DefaultReasoningEffort,
		RequestTimeoutSeconds:       DefaultRequestTimeout,
		MaxRequestBodyBytes:         DefaultMaxRequestBody,
		ReasoningContentPath:        DefaultReasoningContentPath(),
		MissingReasoningStrategy:    DefaultMissingStrategy,
		ReasoningCacheMaxAgeSeconds: DefaultCacheMaxAgeSeconds,
		ReasoningCacheMaxRows:       DefaultCacheMaxRows,
		CursorDisplayReasoning:      DefaultDisplayReasoning,
		CORS:                        DefaultCORS,
		Verbose:                     DefaultVerbose,
	}
}

func applyRaw(cfg *Config, raw rawConfig, configDir string) {
	if raw.Host != nil {
		cfg.Host = *raw.Host
	}
	if raw.Port != nil {
		cfg.Port = *raw.Port
	}
	if raw.BaseURL != nil {
		cfg.UpstreamBaseURL = strings.TrimRight(*raw.BaseURL, "/")
	}
	if raw.Model != nil {
		cfg.UpstreamModel = *raw.Model
	}
	if raw.Thinking != nil {
		cfg.Thinking = NormalizeThinking(*raw.Thinking)
	}
	if raw.ReasoningEffort != nil {
		cfg.ReasoningEffort = *raw.ReasoningEffort
	}
	if raw.DisplayReasoning != nil {
		cfg.CursorDisplayReasoning = *raw.DisplayReasoning
	}
	if raw.Verbose != nil {
		cfg.Verbose = *raw.Verbose
	}
	if raw.RequestTimeout != nil {
		cfg.RequestTimeoutSeconds = *raw.RequestTimeout
	}
	if raw.MaxRequestBodyBytes != nil {
		cfg.MaxRequestBodyBytes = *raw.MaxRequestBodyBytes
	}
	if raw.CORS != nil {
		cfg.CORS = *raw.CORS
	}
	if raw.ReasoningContentPath != nil && *raw.ReasoningContentPath != "" {
		cfg.ReasoningContentPath = resolvePath(*raw.ReasoningContentPath, configDir)
	}
	if raw.MissingReasoningStrategy != nil {
		cfg.MissingReasoningStrategy = NormalizeMissingReasoningStrategy(*raw.MissingReasoningStrategy)
	}
	if raw.ReasoningCacheMaxAgeSeconds != nil {
		cfg.ReasoningCacheMaxAgeSeconds = *raw.ReasoningCacheMaxAgeSeconds
	}
	if raw.ReasoningCacheMaxRows != nil {
		cfg.ReasoningCacheMaxRows = *raw.ReasoningCacheMaxRows
	}
}

func resolvePath(value, base string) string {
	expanded := expandUser(value)
	if filepath.IsAbs(expanded) {
		return expanded
	}
	return filepath.Join(base, expanded)
}

// NormalizeThinking maps user-supplied thinking values to the canonical form.
func NormalizeThinking(value string) string {
	switch strings.ToLower(strings.TrimSpace(value)) {
	case "passthrough", "pass-through", "pass_through":
		return "pass-through"
	case "enabled":
		return "enabled"
	case "disabled":
		return "disabled"
	}
	return DefaultThinking
}

// NormalizeMissingReasoningStrategy maps user-supplied strategies to canonical form.
func NormalizeMissingReasoningStrategy(value string) string {
	v := strings.ToLower(strings.TrimSpace(value))
	if v == "recover" || v == "reject" {
		return v
	}
	return DefaultMissingStrategy
}

// ParseBool accepts a wider range of truthy/falsey strings than strconv.
func ParseBool(value string, fallback bool) bool {
	switch strings.ToLower(strings.TrimSpace(value)) {
	case "1", "true", "yes", "on":
		return true
	case "0", "false", "no", "off":
		return false
	}
	if b, err := strconv.ParseBool(value); err == nil {
		return b
	}
	return fallback
}
