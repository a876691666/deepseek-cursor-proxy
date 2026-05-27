# DeepSeek Cursor Proxy

一个用 Go 编写的兼容性代理，将 Cursor 连接到 DeepSeek 思考模型，内置
**PocketBase** 后端，支持 API 密钥管理、Token 用量追踪和推理缓存持久化。

## 功能特性

- **OpenAI 兼容** — `/v1/chat/completions`、`/v1/models`、`/healthz` 接口
- **Anthropic 兼容** — `/v1/messages` 接口，可转发到 DeepSeek Anthropic API<br>
  _配置 `anthropic_base_url` 即可启用，与 OpenAI 接口同时运行_
- 支持流式 (SSE) 和非流式响应（两种格式均支持）
- **推理缓存** — 基于 PocketBase 的 DeepSeek `reasoning_content` 缓存，
  确保多轮工具调用对话在 Cursor 剔除推理内容后仍能正常工作
- **API 密钥分发** — 向下游用户分发代理密钥，按密钥统计 Token 用量
- **API 密钥覆盖** — 设置 `DEEPSEEK_API_KEY` 环境变量，统一使用一个上游密钥
- **配置优先级**: CLI 参数 > config.yaml > 环境变量 > 默认值
- PocketBase 管理后台：`http://host:port/_/`
- 支持 CORS、请求大小限制、超时设置、缓存保留策略

## 快速开始

### 环境变量

| 变量 | 默认值 | 说明 |
|---|---|---|
| `DEEPSEEK_API_KEY` | _(空)_ | 上游 DeepSeek API 密钥（覆盖客户端提供的认证） |
| `PB_ADMIN_EMAIL` | `admin` | PocketBase 超级管理员邮箱 |
| `PB_ADMIN_PASSWORD` | `admin123` | PocketBase 超级管理员密码 |
| `PB_DATA_DIR` | `~/.deepseek-cursor-proxy/pb_data` | PocketBase 数据目录 |

### Docker 运行

```bash
docker compose up -d
```

### 从源码运行

需要 Go 1.25+。

```bash
go build -o deepseek-cursor-proxy ./cmd/deepseek-cursor-proxy
./deepseek-cursor-proxy --host 0.0.0.0 --port 9000
```

首次运行时，会自动创建 `~/.deepseek-cursor-proxy/config.yaml` 配置文件，
并在 `~/.deepseek-cursor-proxy/pb_data/` 初始化 PocketBase 数据库。

## 配置说明

### config.yaml（优先级高于环境变量）

```yaml
# 上游 DeepSeek API (OpenAI 格式)
base_url: https://api.deepseek.com
model: deepseek-v4-pro
thinking: enabled
reasoning_effort: max
display_reasoning: true
deepseek_api_key: sk-xxx          # 覆盖 DEEPSEEK_API_KEY 环境变量

# 上游 DeepSeek API (Anthropic 格式，可选)
# 设置后代理同时暴露 /v1/messages 端点
# anthropic_base_url: https://api.deepseek.com/anthropic
# anthropic_api_path: /v1/messages

# 代理服务
host: 0.0.0.0
port: 9000
verbose: false
request_timeout: 300
max_request_body_bytes: 20971520
cors: false

# 推理缓存
missing_reasoning_strategy: recover   # recover | reject
reasoning_cache_max_age_seconds: 2592000
reasoning_cache_max_rows: 100000

# PocketBase
pb_data_dir: ~/.deepseek-cursor-proxy/pb_data
pb_admin_email: admin
pb_admin_password: admin123
```

### CLI 参数

所有配置项都可通过命令行参数覆盖，运行 `--help` 查看完整列表。

| 参数 | 说明 |
|---|---|
| `--config <path>` | YAML 配置文件路径 |
| `--host <addr>` | 绑定地址（默认 `0.0.0.0`） |
| `--port <n>` | 绑定端口（默认 `9000`） |
| `--base-url <url>` | DeepSeek 上游地址（OpenAI 格式） |
| `--anthropic-base-url <url>` | DeepSeek Anthropic 上游地址 |
| `--anthropic-api-path <path>` | Anthropic API 路径（默认 `/v1/messages`） |
| `--model <name>` | 默认模型名称 |
| `--thinking <enabled\|disabled\|pass-through>` | 思考模式 |
| `--reasoning-effort <max\|high\|medium\|low>` | 推理强度 |
| `--display-reasoning` | 将推理内容镜像到 `<think>` 标签 |
| `--cors` | 启用 CORS 头 |
| `--verbose` | 详细日志（打印完整请求/响应） |
| `--missing-reasoning-strategy <recover\|reject>` | 缓存缺失时的处理策略 |
| `--reasoning-cache-max-age-seconds <n>` | 缓存过期时间（秒） |
| `--reasoning-cache-max-rows <n>` | 最大缓存条数 |
| `--clear-reasoning-cache` | 清空缓存并退出 |
| `--pb-data-dir <path>` | PocketBase 数据目录 |

## API 接口

### 代理接口（OpenAI 兼容）

### 代理接口（OpenAI 兼容）

| 方法 | 路径 | 说明 |
|---|---|---|
| `POST` | `/v1/chat/completions` | 聊天代理（支持 `?api_key=` 查询参数） |
| `GET` | `/v1/models` | 列出可用模型 |
| `GET` | `/healthz` | 健康检查 |

### 代理接口（Anthropic 兼容）

> 需要配置 `anthropic_base_url`，例如 `https://api.deepseek.com/anthropic`。

| 方法 | 路径 | 说明 |
|---|---|---|
| `POST` | `/v1/messages` | Anthropic Messages API（支持 `?api_key=` 查询参数） |

### API 密钥管理（需要超级管理员认证）

| 方法 | 路径 | 说明 |
|---|---|---|
| `POST` | `/v1/api_keys` | 创建分发密钥。请求体: `{"name": "密钥名称"}` |
| `GET` | `/v1/api_keys` | 列出所有 API 密钥 |

### PocketBase 管理

| 路径 | 说明 |
|---|---|
| `/_/` | 管理后台（使用 PB_ADMIN_EMAIL / PB_ADMIN_PASSWORD 登录） |
| `/api/` | REST API（集合增删改查） |

## 使用场景

### 场景一：API 密钥覆盖（统一上游密钥）

设置环境变量 `DEEPSEEK_API_KEY=sk-你的真实密钥`，代理将所有请求转发到上游时
统一使用此密钥，忽略客户端传入的 Authorization 头。

```bash
curl http://localhost:9000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer 任意值" \
  -d '{"model":"deepseek-v4-pro","messages":[{"role":"user","content":"你好"}]}'
```

### 场景二：API 密钥分发（按用户追踪用量）

1. 通过管理后台或 API 创建分发密钥：
```bash
# 先从管理后台获取超级管理员 Token，然后：
curl -X POST http://localhost:9000/v1/api_keys \
  -H "Authorization: Bearer <管理员Token>" \
  -H "Content-Type: application/json" \
  -d '{"name":"alice的密钥"}'
# 响应: {"id":"...","key":"sk-dcp-<hex>","name":"alice的密钥","active":true}
```

2. 用户通过查询参数使用分发密钥：
```bash
curl "http://localhost:9000/v1/chat/completions?api_key=sk-dcp-<hex>" \
  -H "Content-Type: application/json" \
  -d '{"model":"deepseek-v4-pro","messages":[{"role":"user","content":"你好"}]}'
```

3. Token 用量自动记录到 `token_usage` 集合，可在管理后台查看。

### 场景三：Anthropic 格式代理

配置 `anthropic_base_url` 后，代理同时支持 Anthropic Messages API 格式:

```bash
# 在 Cursor 中配置 Anthropic 提供商，地址指向代理
curl http://localhost:9000/v1/messages \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-xxx" \
  -H "x-api-key: sk-xxx" \
  -d '{
    "model": "claude-sonnet-4-6",
    "max_tokens": 1024,
    "messages": [{"role": "user", "content": "你好"}]
  }'
```

### 场景四：直接透传

不设置 `DEEPSEEK_API_KEY` 时，代理将客户端的 Authorization 头原样转发给
DeepSeek。此模式下不记录按密钥统计的用量。

## 项目架构

```
cmd/deepseek-cursor-proxy/    命令行入口，参数解析，信号处理
internal/config/               YAML + 环境变量配置
internal/pocketbase/           PocketBase 初始化，集合定义，API 密钥管理
internal/server/               HTTP 代理处理器，路由注册
internal/store/                基于 PocketBase 的推理缓存
internal/transform/            请求/响应转换，推理恢复
internal/streaming/            SSE 累加器，Cursor 思考标签适配器
```

### 模块说明

**config** — 加载 `config.yaml`（首次运行自动创建）。合并环境变量和 CLI 参数。
优先级：CLI > YAML > 环境变量 > 默认值。

**pocketbase** — 初始化嵌入式 PocketBase 应用。创建集合（`api_keys`、
`token_usage`、`reasoning_cache`）。设置超级管理员账户。对外提供：
`CreateAPIKey`、`LookupAPIKey`、`RecordTokenUsage`、`GenerateAPIKey`。

**server** — 注册代理路由到 PocketBase 路由器。处理认证覆盖、API 密钥验证、
Token 用量记录、SSE 流式传输。

**store** — 基于 PocketBase Record API 的推理缓存。键格式为
`scope:signature:...`、`scope:tool_call:...`、`scope:tool_call_signature:...`。
支持 Get、Put、Clear 以及自动 TTL/条数裁剪。

**transform** — 将 Cursor 聊天请求规范化为 DeepSeek 格式。处理从缓存恢复
`reasoning_content`、截断对话恢复、思考模式配置、工具/函数格式转换。

**streaming** — SSE 数据块累加器，将增量 delta 合并为完整消息以供推理存储。
`CursorReasoningDisplayAdapter` 将 `reasoning_content` 包装在 `<think>` 标签中，
供 Cursor 编辑器渲染。

## 开发

```bash
go vet ./...
go test -race ./...
go build -o deepseek-cursor-proxy ./cmd/deepseek-cursor-proxy
```

## 许可

MIT — 详见 `LICENSE`。
