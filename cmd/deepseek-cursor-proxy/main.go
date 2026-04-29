// Command deepseek-cursor-proxy runs the DeepSeek Cursor proxy HTTP server.
package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"os/signal"
	"syscall"

	"github.com/a876691666/deepseek-cursor-proxy/internal/config"
	"github.com/a876691666/deepseek-cursor-proxy/internal/server"
	"github.com/a876691666/deepseek-cursor-proxy/internal/store"
)

// stringFlag is a tri-state flag that records whether it was set by the user.
type stringFlag struct {
	value string
	set   bool
}

func (s *stringFlag) String() string { return s.value }
func (s *stringFlag) Set(v string) error {
	s.value = v
	s.set = true
	return nil
}

// boolFlag is a tri-state bool flag.
type boolFlag struct {
	value bool
	set   bool
}

func (b *boolFlag) String() string {
	if b == nil {
		return ""
	}
	return fmt.Sprintf("%t", b.value)
}
func (b *boolFlag) Set(v string) error {
	b.value = config.ParseBool(v, false)
	b.set = true
	return nil
}
func (b *boolFlag) IsBoolFlag() bool { return true }

// intFlag is a tri-state int flag.
type intFlag struct {
	value int
	set   bool
}

func (i *intFlag) String() string {
	if i == nil {
		return ""
	}
	return fmt.Sprintf("%d", i.value)
}
func (i *intFlag) Set(v string) error {
	var x int
	if _, err := fmt.Sscanf(v, "%d", &x); err != nil {
		return err
	}
	i.value = x
	i.set = true
	return nil
}

// int64Flag is a tri-state int64 flag.
type int64Flag struct {
	value int64
	set   bool
}

func (i *int64Flag) String() string {
	if i == nil {
		return ""
	}
	return fmt.Sprintf("%d", i.value)
}
func (i *int64Flag) Set(v string) error {
	var x int64
	if _, err := fmt.Sscanf(v, "%d", &x); err != nil {
		return err
	}
	i.value = x
	i.set = true
	return nil
}

// floatFlag is a tri-state float flag.
type floatFlag struct {
	value float64
	set   bool
}

func (f *floatFlag) String() string {
	if f == nil {
		return ""
	}
	return fmt.Sprintf("%g", f.value)
}
func (f *floatFlag) Set(v string) error {
	var x float64
	if _, err := fmt.Sscanf(v, "%f", &x); err != nil {
		return err
	}
	f.value = x
	f.set = true
	return nil
}

func main() {
	if err := run(os.Args[1:]); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func run(args []string) error {
	logger := log.New(os.Stdout, "", log.LstdFlags)

	fs := flag.NewFlagSet("deepseek-cursor-proxy", flag.ContinueOnError)
	configPath := fs.String("config", "", fmt.Sprintf("YAML config file, default %s", config.DefaultConfigPath()))
	host := stringFlag{}
	fs.Var(&host, "host", "Bind host, default from config or 0.0.0.0")
	port := intFlag{}
	fs.Var(&port, "port", "Bind port, default from config or 9000")
	model := stringFlag{}
	fs.Var(&model, "model", "Fallback DeepSeek model when the request has no model")
	baseURL := stringFlag{}
	fs.Var(&baseURL, "base-url", "DeepSeek base URL, default from config or https://api.deepseek.com")
	thinking := stringFlag{}
	fs.Var(&thinking, "thinking", "DeepSeek thinking mode (enabled|disabled|pass-through)")
	reasoningEffort := stringFlag{}
	fs.Var(&reasoningEffort, "reasoning-effort", "DeepSeek reasoning effort (low|medium|high|max|xhigh)")
	reasoningContentPath := stringFlag{}
	fs.Var(&reasoningContentPath, "reasoning-content-path", fmt.Sprintf("SQLite reasoning_content cache path, default %s", config.DefaultReasoningContentPath()))
	verbose := boolFlag{}
	fs.Var(&verbose, "verbose", "Log detailed request lifecycle metadata and full payloads")
	displayReasoning := boolFlag{}
	fs.Var(&displayReasoning, "display-reasoning", "Mirror reasoning_content into Cursor-visible <think> content")
	cors := boolFlag{}
	fs.Var(&cors, "cors", "Send permissive CORS headers")
	requestTimeout := floatFlag{}
	fs.Var(&requestTimeout, "request-timeout", "Upstream request timeout in seconds, default from config or 300")
	maxBodyBytes := int64Flag{}
	fs.Var(&maxBodyBytes, "max-request-body-bytes", "Maximum accepted request body size, default from config")
	cacheMaxAgeSeconds := int64Flag{}
	fs.Var(&cacheMaxAgeSeconds, "reasoning-cache-max-age-seconds", "Maximum reasoning cache row age in seconds")
	cacheMaxRows := int64Flag{}
	fs.Var(&cacheMaxRows, "reasoning-cache-max-rows", "Maximum reasoning cache rows")
	missingStrategy := stringFlag{}
	fs.Var(&missingStrategy, "missing-reasoning-strategy", "What to do when reasoning_content is missing (recover|reject)")
	clearCache := fs.Bool("clear-reasoning-cache", false, "Clear the local reasoning_content SQLite cache and exit")

	if err := fs.Parse(args); err != nil {
		return err
	}

	cfg, _, err := config.LoadFile(*configPath)
	if err != nil {
		return err
	}

	if host.set {
		cfg.Host = host.value
	}
	if port.set {
		cfg.Port = port.value
	}
	if model.set {
		cfg.UpstreamModel = model.value
	}
	if baseURL.set {
		cfg.UpstreamBaseURL = trimTrailingSlash(baseURL.value)
	}
	if thinking.set {
		cfg.Thinking = config.NormalizeThinking(thinking.value)
	}
	if reasoningEffort.set {
		cfg.ReasoningEffort = reasoningEffort.value
	}
	if reasoningContentPath.set {
		cfg.ReasoningContentPath = reasoningContentPath.value
	}
	if verbose.set {
		cfg.Verbose = verbose.value
	}
	if displayReasoning.set {
		cfg.CursorDisplayReasoning = displayReasoning.value
	}
	if cors.set {
		cfg.CORS = cors.value
	}
	if requestTimeout.set {
		cfg.RequestTimeoutSeconds = requestTimeout.value
	}
	if maxBodyBytes.set {
		cfg.MaxRequestBodyBytes = maxBodyBytes.value
	}
	if cacheMaxAgeSeconds.set {
		cfg.ReasoningCacheMaxAgeSeconds = cacheMaxAgeSeconds.value
	}
	if cacheMaxRows.set {
		cfg.ReasoningCacheMaxRows = cacheMaxRows.value
	}
	if missingStrategy.set {
		cfg.MissingReasoningStrategy = config.NormalizeMissingReasoningStrategy(missingStrategy.value)
	}

	st, err := store.New(cfg.ReasoningContentPath, cfg.ReasoningCacheMaxAgeSeconds, cfg.ReasoningCacheMaxRows)
	if err != nil {
		return err
	}
	defer st.Close()

	if *clearCache {
		deleted, err := st.Clear()
		if err != nil {
			return err
		}
		logger.Printf("cleared %d reasoning cache row(s)", deleted)
		return nil
	}

	srv := server.New(cfg, st, logger)
	ctx, cancel := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer cancel()
	return srv.Run(ctx)
}

func trimTrailingSlash(s string) string {
	for len(s) > 0 && s[len(s)-1] == '/' {
		s = s[:len(s)-1]
	}
	return s
}
