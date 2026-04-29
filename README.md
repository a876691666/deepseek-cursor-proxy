<h1 align="center"><img src="assets/logo.png" width="150" alt="deepseek-cursor-proxy logo"><br>DeepSeek Cursor Proxy</h1>

A compatibility proxy (rewritten in **Go**) that connects Cursor to DeepSeek
thinking models (`deepseek-v4-pro` and `deepseek-v4-flash`) by properly
handling the `reasoning_content` field for DeepSeek tool-call reasoning
requests.

The proxy listens on `:9000` by default and exposes an OpenAI-compatible
endpoint at `http://<host>:9000/v1`. Cursor's bearer token is forwarded to
DeepSeek upstream as-is — the proxy never stores keys.

## Features

- OpenAI-compatible `/v1/chat/completions`, `/v1/models`, `/healthz` endpoints
- Streaming (SSE) and non-streaming responses
- Persistent SQLite-backed cache of DeepSeek `reasoning_content` so multi-turn
  tool-call conversations work even when Cursor strips it from history
- Automatic recovery when reasoning is missing (or strict `reject` mode for
  debugging)
- Optional Cursor `<think>` mirror so the editor can render reasoning
- CORS, request size limit, request timeout, configurable cache retention

## Quick start

### Run with Docker Compose (recommended)

```bash
docker compose up -d
```

Cursor → `Settings → Models → Add custom model` → Base URL:
`http://<docker-host-ip>:9000/v1` (use any DeepSeek API key in the API Key field).

### Run with Docker

```bash
docker build -t deepseek-cursor-proxy .
docker run --rm -p 9000:9000 \
  -v deepseek-cursor-proxy-data:/home/app/.deepseek-cursor-proxy \
  deepseek-cursor-proxy
```

### Run from source

Requires Go 1.25+.

```bash
go build -o deepseek-cursor-proxy ./cmd/deepseek-cursor-proxy
./deepseek-cursor-proxy --host 0.0.0.0 --port 9000
```

On the first run, a default config is written to
`~/.deepseek-cursor-proxy/config.yaml` and the SQLite cache is created at
`~/.deepseek-cursor-proxy/reasoning_content.sqlite3`.

## CLI flags

| Flag | Description |
| --- | --- |
| `--config <path>` | YAML config file (defaults to `~/.deepseek-cursor-proxy/config.yaml`) |
| `--host <addr>` | Bind host (default `0.0.0.0`) |
| `--port <n>` | Bind port (default `9000`) |
| `--base-url <url>` | DeepSeek upstream base URL |
| `--model <name>` | Fallback DeepSeek model when the request omits one |
| `--thinking <enabled\|disabled\|pass-through>` | DeepSeek thinking mode |
| `--reasoning-effort <low\|medium\|high\|max\|xhigh>` | DeepSeek reasoning effort |
| `--reasoning-content-path <path>` | SQLite reasoning cache path |
| `--missing-reasoning-strategy <recover\|reject>` | Behavior when cached reasoning is missing |
| `--reasoning-cache-max-age-seconds <n>` | Cache row age cap |
| `--reasoning-cache-max-rows <n>` | Cache row count cap |
| `--request-timeout <seconds>` | Upstream request timeout |
| `--max-request-body-bytes <n>` | Maximum accepted request body size |
| `--display-reasoning` / `--display-reasoning=false` | Mirror reasoning into Cursor `<think>` blocks |
| `--cors` / `--cors=false` | Send permissive CORS headers |
| `--verbose` / `--verbose=false` | Log full request/response payloads |
| `--clear-reasoning-cache` | Wipe the SQLite cache and exit |

All flags can also be set in the YAML config file. CLI flags override the YAML.

## Configuration file

The default `config.yaml` (created on first run):

```yaml
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
```

## Endpoints

- `GET /healthz`, `GET /v1/healthz` → `{"ok":true}`
- `GET /models`, `GET /v1/models` → list of advertised model IDs
- `POST /v1/chat/completions` → main proxy endpoint (OpenAI-compatible)

## Project layout

```
cmd/deepseek-cursor-proxy/    main entry (CLI + signal handling)
internal/config/              YAML config + defaults
internal/store/               SQLite reasoning_content cache
internal/transform/           request normalization + recovery
internal/streaming/           SSE accumulator + Cursor display adapter
internal/server/              HTTP handlers, streaming proxy, CORS
Dockerfile                    multi-stage build (alpine runtime, pure Go)
docker-compose.yml            persistent-volume service definition
```

## Development

```bash
go vet ./...
go test -race ./...
```

## License

MIT — see `LICENSE`.
