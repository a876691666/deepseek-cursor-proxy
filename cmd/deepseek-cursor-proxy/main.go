// Command deepseek-cursor-proxy runs the DeepSeek Cursor proxy HTTP server
// backed by PocketBase for API key management, token usage, and reasoning cache.
package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"os/signal"
	"syscall"

	"github.com/pocketbase/pocketbase/core"

	"github.com/a876691666/deepseek-cursor-proxy/internal/config"
	pbPkg "github.com/a876691666/deepseek-cursor-proxy/internal/pocketbase"
	"github.com/a876691666/deepseek-cursor-proxy/internal/server"
	"github.com/a876691666/deepseek-cursor-proxy/internal/store"
)

// ---- tri-state flag types ----

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

type boolFlag struct {
	value bool
	set   bool
}

func (b *boolFlag) String() string { return fmt.Sprintf("%t", b.value) }
func (b *boolFlag) Set(v string) error {
	b.value = config.ParseBool(v, false)
	b.set = true
	return nil
}
func (b *boolFlag) IsBoolFlag() bool { return true }

type intFlag struct {
	value int
	set   bool
}

func (i *intFlag) String() string { return fmt.Sprintf("%d", i.value) }
func (i *intFlag) Set(v string) error {
	var x int
	if _, err := fmt.Sscanf(v, "%d", &x); err != nil {
		return err
	}
	i.value = x
	i.set = true
	return nil
}

type int64Flag struct {
	value int64
	set   bool
}

func (i *int64Flag) String() string { return fmt.Sprintf("%d", i.value) }
func (i *int64Flag) Set(v string) error {
	var x int64
	if _, err := fmt.Sscanf(v, "%d", &x); err != nil {
		return err
	}
	i.value = x
	i.set = true
	return nil
}

type floatFlag struct {
	value float64
	set   bool
}

func (f *floatFlag) String() string { return fmt.Sprintf("%g", f.value) }
func (f *floatFlag) Set(v string) error {
	var x float64
	if _, err := fmt.Sscanf(v, "%f", &x); err != nil {
		return err
	}
	f.value = x
	f.set = true
	return nil
}

// ---- main ----

func main() {
	if err := run(os.Args[1:]); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func run(args []string) error {
	logger := log.New(os.Stdout, "", log.LstdFlags)

	fs := flag.NewFlagSet("deepseek-cursor-proxy", flag.ContinueOnError)
	configPath := fs.String("config", "", "YAML config file")
	host := stringFlag{}
	fs.Var(&host, "host", "Bind host")
	port := intFlag{}
	fs.Var(&port, "port", "Bind port")
	model := stringFlag{}
	fs.Var(&model, "model", "Fallback DeepSeek model")
	baseURL := stringFlag{}
	fs.Var(&baseURL, "base-url", "DeepSeek base URL (OpenAI format)")
	anthropicBaseURL := stringFlag{}
	fs.Var(&anthropicBaseURL, "anthropic-base-url", "DeepSeek Anthropic base URL")
	anthropicAPIPath := stringFlag{}
	fs.Var(&anthropicAPIPath, "anthropic-api-path", "Anthropic API path (default /v1/messages)")
	thinking := stringFlag{}
	fs.Var(&thinking, "thinking", "Thinking mode: enabled|disabled|pass-through")
	reasoningEffort := stringFlag{}
	fs.Var(&reasoningEffort, "reasoning-effort", "Reasoning effort: low|medium|high|max|xhigh")
	verbose := boolFlag{}
	fs.Var(&verbose, "verbose", "Verbose logging")
	displayReasoning := boolFlag{}
	fs.Var(&displayReasoning, "display-reasoning", "Show reasoning in <think> blocks")
	cors := boolFlag{}
	fs.Var(&cors, "cors", "Enable CORS headers")
	requestTimeout := floatFlag{}
	fs.Var(&requestTimeout, "request-timeout", "Upstream request timeout (seconds)")
	maxBodyBytes := int64Flag{}
	fs.Var(&maxBodyBytes, "max-request-body-bytes", "Max request body size")
	cacheMaxAgeSeconds := int64Flag{}
	fs.Var(&cacheMaxAgeSeconds, "reasoning-cache-max-age-seconds", "Max reasoning cache age")
	cacheMaxRows := int64Flag{}
	fs.Var(&cacheMaxRows, "reasoning-cache-max-rows", "Max reasoning cache rows")
	missingStrategy := stringFlag{}
	fs.Var(&missingStrategy, "missing-reasoning-strategy", "Missing reasoning strategy: recover|reject")
	clearCache := fs.Bool("clear-reasoning-cache", false, "Clear reasoning cache and exit")
	pbDataDir := stringFlag{}
	fs.Var(&pbDataDir, "pb-data-dir", "PocketBase data directory")

	if err := fs.Parse(args); err != nil {
		return err
	}

	cfg, _, err := config.LoadFile(*configPath)
	if err != nil {
		return err
	}

	// Apply CLI overrides.
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
	if anthropicBaseURL.set {
		cfg.AnthropicBaseURL = trimTrailingSlash(anthropicBaseURL.value)
	}
	if anthropicAPIPath.set {
		cfg.AnthropicAPIPath = anthropicAPIPath.value
	}
	if thinking.set {
		cfg.Thinking = config.NormalizeThinking(thinking.value)
	}
	if reasoningEffort.set {
		cfg.ReasoningEffort = reasoningEffort.value
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
	if pbDataDir.set {
		cfg.PBDataDir = pbDataDir.value
	}

	// Bootstrap PocketBase (API keys, token usage, reasoning cache).
	pb, err := pbPkg.Setup(cfg)
	if err != nil {
		return err
	}

	// Reasoning cache backed by PocketBase.
	st := store.New(pb, cfg.ReasoningCacheMaxAgeSeconds, cfg.ReasoningCacheMaxRows)

	if *clearCache {
		deleted, err := st.Clear()
		if err != nil {
			return err
		}
		logger.Printf("cleared %d reasoning cache row(s)", deleted)
		return nil
	}

	srv := server.New(cfg, st, logger, pb)

	// Register proxy routes and set listen address.
	address := fmt.Sprintf("%s:%d", cfg.Host, cfg.Port)
	pb.OnServe().BindFunc(func(se *core.ServeEvent) error {
		se.Server.Addr = address
		srv.RegisterRoutes(se.Router)
		return se.Next()
	})

	// Graceful shutdown on SIGINT/SIGTERM.
	ctx, cancel := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer cancel()
	go func() {
		<-ctx.Done()
		logger.Printf("shutting down...")
		pb.ResetBootstrapState()
		os.Exit(0)
	}()

	return srv.Run()
}

func trimTrailingSlash(s string) string {
	for len(s) > 0 && s[len(s)-1] == '/' {
		s = s[:len(s)-1]
	}
	return s
}
