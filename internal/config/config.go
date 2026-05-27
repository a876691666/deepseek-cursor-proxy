// Package config loads YAML configuration with sensible defaults and supports
// command-line overrides for the DeepSeek Cursor proxy.
//
// Configuration priority (highest to lowest):
//  1. CLI flags
//  2. config.yaml
//  3. Environment variables (DEEPSEEK_API_KEY, PB_ADMIN_EMAIL, PB_ADMIN_PASSWORD, PB_DATA_DIR)
//  4. Hard-coded defaults
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
	PocketBaseDataDirName     = "pb_data"
	DefaultHost               = "0.0.0.0"
	DefaultPort               = 9000
	DefaultUpstreamBaseURL      = "https://api.deepseek.com"
	DefaultUpstreamModel        = "deepseek-v4-pro"
	DefaultAnthropicBaseURL     = ""
	DefaultAnthropicAPIPath     = "/v1/messages"
	DefaultThinking             = "enabled"
	DefaultReasoningEffort    = "max"
	DefaultDisplayReasoning   = true
	DefaultVerbose            = false
	DefaultRequestTimeout     = 300.0
	DefaultMaxRequestBody     = 20 * 1024 * 1024
	DefaultCORS               = false
	DefaultMissingStrategy    = "recover"
	DefaultCacheMaxAgeSeconds = 30 * 24 * 60 * 60
	DefaultCacheMaxRows       = 100_000
	DefaultPBAdminEmail       = "admin@admin.com"
	DefaultPBAdminPassword    = "admin123"
)

// DefaultConfigText is written to disk on first run.
const DefaultConfigText = `# deepseek-cursor-proxy configuration
# Values here override environment variables. Use CLI flags for final overrides.

# ---- Upstream DeepSeek API (OpenAI format) ----
base_url: https://api.deepseek.com
model: deepseek-v4-pro
thinking: enabled
reasoning_effort: max
display_reasoning: true

# ---- Upstream DeepSeek API (Anthropic format) ----
# When set, the proxy exposes a /v1/messages endpoint that forwards Anthropic-format
# requests to this upstream. Leave empty to disable the Anthropic endpoint.
# anthropic_base_url: https://api.deepseek.com/anthropic
# anthropic_api_path: /v1/messages

# API key for upstream DeepSeek requests (overrides DEEPSEEK_API_KEY env var).
# deepseek_api_key: sk-xxx

# ---- Proxy server ----
host: 0.0.0.0
port: 9000
verbose: false
request_timeout: 300
max_request_body_bytes: 20971520
cors: false

# ---- Reasoning cache ----
missing_reasoning_strategy: recover
reasoning_cache_max_age_seconds: 2592000
reasoning_cache_max_rows: 100000

# ---- PocketBase ----
# pb_data_dir: ~/.deepseek-cursor-proxy/pb_data
# pb_admin_email: admin@admin.com
# pb_admin_password: admin123
`

// Config holds resolved proxy settings.
type Config struct {
	Host                        string
	Port                        int
	UpstreamBaseURL             string // OpenAI-format upstream
	UpstreamModel               string
	AnthropicBaseURL            string // Anthropic-format upstream (empty = Anthropic endpoint disabled)
	AnthropicAPIPath            string // path appended to AnthropicBaseURL (default /v1/messages)
	Thinking                    string
	ReasoningEffort             string
	RequestTimeoutSeconds       float64
	MaxRequestBodyBytes         int64
	MissingReasoningStrategy    string
	ReasoningCacheMaxAgeSeconds int64
	ReasoningCacheMaxRows       int64
	CursorDisplayReasoning      bool
	CORS                        bool
	Verbose                     bool
	DeepSeekAPIKey              string
	PBAdminEmail                string
	PBAdminPassword             string
	PBDataDir                   string
}

// rawConfig matches the YAML structure on disk.
type rawConfig struct {
	Host                        *string  `yaml:"host"`
	Port                        *int     `yaml:"port"`
	BaseURL                     *string  `yaml:"base_url"`
	AnthropicBaseURL            *string  `yaml:"anthropic_base_url"`
	AnthropicAPIPath            *string  `yaml:"anthropic_api_path"`
	Model                       *string  `yaml:"model"`
	Thinking                    *string  `yaml:"thinking"`
	ReasoningEffort             *string  `yaml:"reasoning_effort"`
	DisplayReasoning            *bool    `yaml:"display_reasoning"`
	Verbose                     *bool    `yaml:"verbose"`
	RequestTimeout              *float64 `yaml:"request_timeout"`
	MaxRequestBodyBytes         *int64   `yaml:"max_request_body_bytes"`
	CORS                        *bool    `yaml:"cors"`
	MissingReasoningStrategy    *string  `yaml:"missing_reasoning_strategy"`
	ReasoningCacheMaxAgeSeconds *int64   `yaml:"reasoning_cache_max_age_seconds"`
	ReasoningCacheMaxRows       *int64   `yaml:"reasoning_cache_max_rows"`
	DeepSeekAPIKey              *string  `yaml:"deepseek_api_key"`
	PBAdminEmail                *string  `yaml:"pb_admin_email"`
	PBAdminPassword             *string  `yaml:"pb_admin_password"`
	PBDataDir                   *string  `yaml:"pb_data_dir"`
}

func DefaultAppDir() string {
	home, err := os.UserHomeDir()
	if err != nil {
		return AppDirName
	}
	return filepath.Join(home, AppDirName)
}

func DefaultConfigPath() string {
	return filepath.Join(DefaultAppDir(), ConfigFileName)
}

func DefaultPBDataDir() string {
	return filepath.Join(DefaultAppDir(), PocketBaseDataDirName)
}

func PopulateDefaultConfigFile(path string) error {
	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return err
	}
	_ = os.Chmod(dir, 0o700)
	return os.WriteFile(path, []byte(DefaultConfigText), 0o600)
}

func ResolveConfigPath(path string) string {
	if path != "" {
		return expandUser(path)
	}
	// When no explicit path, check current working directory first,
	// then fall back to ~/.deepseek-cursor-proxy/config.yaml.
	if cwd, err := os.Getwd(); err == nil {
		local := filepath.Join(cwd, ConfigFileName)
		if _, err := os.Stat(local); err == nil {
			return local
		}
	}
	return DefaultConfigPath()
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

// LoadFile loads (and creates if missing) a config file, returning the resolved Config.
// Env vars set default values; YAML overrides them; CLI flags override both.
func LoadFile(path string) (Config, string, error) {
	resolved := ResolveConfigPath(path)
	if path == "" {
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

// Defaults populates Config from constants and environment variables.
func Defaults() Config {
	return Config{
		Host:                        DefaultHost,
		Port:                        DefaultPort,
		UpstreamBaseURL:             DefaultUpstreamBaseURL,
		UpstreamModel:               DefaultUpstreamModel,
		AnthropicBaseURL:            DefaultAnthropicBaseURL,
		AnthropicAPIPath:            DefaultAnthropicAPIPath,
		Thinking:                    DefaultThinking,
		ReasoningEffort:             DefaultReasoningEffort,
		RequestTimeoutSeconds:       DefaultRequestTimeout,
		MaxRequestBodyBytes:         DefaultMaxRequestBody,
		MissingReasoningStrategy:    DefaultMissingStrategy,
		ReasoningCacheMaxAgeSeconds: DefaultCacheMaxAgeSeconds,
		ReasoningCacheMaxRows:       DefaultCacheMaxRows,
		CursorDisplayReasoning:      DefaultDisplayReasoning,
		CORS:                        DefaultCORS,
		Verbose:                     DefaultVerbose,
		DeepSeekAPIKey:              envOr("DEEPSEEK_API_KEY", ""),
		PBAdminEmail:                envOr("PB_ADMIN_EMAIL", DefaultPBAdminEmail),
		PBAdminPassword:             envOr("PB_ADMIN_PASSWORD", DefaultPBAdminPassword),
		PBDataDir:                   envOr("PB_DATA_DIR", DefaultPBDataDir()),
	}
}

func envOr(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}

func applyRaw(cfg *Config, raw rawConfig, configDir string) {
	setIf(&cfg.Host, raw.Host)
	setIf(&cfg.Port, raw.Port)
	if raw.BaseURL != nil {
		cfg.UpstreamBaseURL = strings.TrimRight(*raw.BaseURL, "/")
	}
	if raw.AnthropicBaseURL != nil {
		cfg.AnthropicBaseURL = strings.TrimRight(*raw.AnthropicBaseURL, "/")
	}
	setIf(&cfg.AnthropicAPIPath, raw.AnthropicAPIPath)
	setIf(&cfg.UpstreamModel, raw.Model)
	if raw.Thinking != nil {
		cfg.Thinking = NormalizeThinking(*raw.Thinking)
	}
	setIf(&cfg.ReasoningEffort, raw.ReasoningEffort)
	setIf(&cfg.CursorDisplayReasoning, raw.DisplayReasoning)
	setIf(&cfg.Verbose, raw.Verbose)
	setIf(&cfg.RequestTimeoutSeconds, raw.RequestTimeout)
	setIf(&cfg.MaxRequestBodyBytes, raw.MaxRequestBodyBytes)
	setIf(&cfg.CORS, raw.CORS)
	if raw.MissingReasoningStrategy != nil {
		cfg.MissingReasoningStrategy = NormalizeMissingReasoningStrategy(*raw.MissingReasoningStrategy)
	}
	setIf(&cfg.ReasoningCacheMaxAgeSeconds, raw.ReasoningCacheMaxAgeSeconds)
	setIf(&cfg.ReasoningCacheMaxRows, raw.ReasoningCacheMaxRows)
	// YAML values take priority over env vars.
	setIf(&cfg.DeepSeekAPIKey, raw.DeepSeekAPIKey)
	setIf(&cfg.PBAdminEmail, raw.PBAdminEmail)
	setIf(&cfg.PBAdminPassword, raw.PBAdminPassword)
	if raw.PBDataDir != nil && *raw.PBDataDir != "" {
		cfg.PBDataDir = resolvePath(*raw.PBDataDir, configDir)
	}
}

func setIf[T any](dst *T, src *T) {
	if src != nil {
		*dst = *src
	}
}

func resolvePath(value, base string) string {
	expanded := expandUser(value)
	if filepath.IsAbs(expanded) {
		return expanded
	}
	return filepath.Join(base, expanded)
}

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

func NormalizeMissingReasoningStrategy(value string) string {
	v := strings.ToLower(strings.TrimSpace(value))
	if v == "recover" || v == "reject" {
		return v
	}
	return DefaultMissingStrategy
}

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
