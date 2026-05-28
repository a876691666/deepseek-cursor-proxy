# DeepSeek Cursor Proxy —— 下载安装 + 配置指南

> 用大白话写的，照着做就行，不废话。
>
> **先说清楚：** Claude Code（Anthropic 家的）和 Codex（OpenAI 家的）是两码事，
> 下面分别教你俩怎么接这个代理。

---

## 一、这东西是干嘛的？

简单说：**让你在本机跑一个小代理，把 Claude Code / Codex 的请求转发给 DeepSeek**。
代理会做格式转换、推理缓存、API Key 管理这些事情。

原理图：

```
Claude Code  ──(Anthropic格式)──┐
                                ├──► 代理 (localhost:9000) ──► DeepSeek API
Codex CLI    ──(OpenAI格式)  ──┘
```

---

## 二、下载

去 GitHub 的 [Releases 页面](https://github.com/a876691666/deepseek-cursor-proxy/releases) 下载你系统对应的包。

| 你的系统 | 下载哪个 |
|----------|---------|
| Windows 64位（常见） | `deepseek-cursor-proxy-windows-amd64.exe` |
| Windows ARM（Surface等） | `deepseek-cursor-proxy-windows-arm64.exe` |
| Mac Intel 芯片 | `deepseek-cursor-proxy-darwin-amd64` |
| Mac Apple M 芯片 | `deepseek-cursor-proxy-darwin-arm64` |
| Linux x86_64 | `deepseek-cursor-proxy-linux-amd64` |
| Linux ARM（树莓派等） | `deepseek-cursor-proxy-linux-arm64` |

> 不确定就用 **windows-amd64** / **darwin-arm64**（M 芯片 Mac）/ **linux-amd64**，99% 的人都是这些。

---

## 三、安装 & 第一次启动

### Windows

1. 把下载的 `.exe` 放到一个文件夹，比如 `C:\dcp\`
2. 双击运行，或者打开终端：

```powershell
cd C:\dcp
.\deepseek-cursor-proxy-windows-amd64.exe
```

### Mac / Linux

1. 先把文件改成可执行：

```bash
chmod +x deepseek-cursor-proxy-darwin-arm64   # M芯片Mac
# 或
chmod +x deepseek-cursor-proxy-linux-amd64    # Linux
```

2. 运行：

```bash
./deepseek-cursor-proxy-darwin-arm64
```

首次运行会在 `~/.deepseek-cursor-proxy/`（Mac/Linux）或 `C:\Users\你的用户名\.deepseek-cursor-proxy\`（Windows）下
自动生成 `config.yaml`。

---

## 四、配置 config.yaml（最重要的一步！）

打开自动生成的 `config.yaml`，把下面几个关键项填好：

```yaml
# ---- 上游 DeepSeek API（OpenAI 格式）----
base_url: https://api.deepseek.com
model: deepseek-v4-pro

# ---- 上游 DeepSeek API（Anthropic 格式）----
# Claude Code 走这条线路
anthropic_base_url: https://api.deepseek.com/anthropic
anthropic_api_path: /v1/messages
thinking: enabled
reasoning_effort: max
display_reasoning: true

# API Key！！必须换成你自己的！！
deepseek_api_key: sk-你的DeepSeek密钥

# ---- 代理服务器 ----
host: 0.0.0.0
port: 9000
cors: true

# ---- PocketBase 管理后台 ----
pb_data_dir: ./pb_data
pb_admin_email: admin@dcp.com
pb_admin_password: admin123
```

### 必改的有三项：

1. **`deepseek_api_key`**：去 [DeepSeek 开放平台](https://platform.deepseek.com) 注册拿一个 key 填上。
2. **`model`**：想用哪个模型就填哪个，推荐 `deepseek-v4-pro`。
3. **`pb_admin_email` / `pb_admin_password`**：管理后台的登录账号，建议改掉默认密码。

其他保持默认就行。

---

## 五、启动代理

配好 `config.yaml` 后，运行程序：

```bash
# Windows
.\deepseek-cursor-proxy-windows-amd64.exe

# Mac/Linux
./deepseek-cursor-proxy-darwin-arm64
```

看到类似这样的输出就说明成功了：

```
listening on http://0.0.0.0:9000/v1
forwarding (openai) to https://api.deepseek.com/chat/completions default_model=deepseek-v4-pro
forwarding (anthropic) to https://api.deepseek.com/anthropic/v1/messages
```

---

## 六、创建 API Key（必做！）

Claude Code 和 Codex 都需要一个 API Key 才能跟代理通信。
虽然代理底层会用你在 `config.yaml` 里配的 DeepSeek 密钥去请求 DeepSeek，
但客户端这边还是得传一个 key 过来——代理会拿它做身份识别和用量统计。

所以我们需要在代理里**手动创建一个 Key**，然后让 Claude Code / Codex 用这个 Key。

### 6.1 打开管理后台

代理启动后，浏览器打开：

```
http://localhost:9000/_/
```

### 6.2 登录

用 `config.yaml` 里配的账号密码登录：

- 邮箱：`admin@dcp.com`（或你改过的）
- 密码：`admin123`（或你改过的）

### 6.3 创建 Key

1. 登录后会看到左侧菜单，点击 **`api_keys`** 这个集合
2. 点击右上角的 **`+` 新建记录** 按钮
3. 在弹出的表单里填写：

| 字段 | 填什么 |
|------|--------|
| `key` | `sk-dcp-123`（随便起，但一定要以 `sk-dcp-` 开头，方便识别） |
| `name` | 随便，比如 `我的ClaudeCode` |
| `active` | ✅ 勾上 |

4. 点 **Save / 保存**

> 你也可以创建多个 Key，比如给不同电脑用的 `sk-dcp-macbook`、`sk-dcp-server`。
> 每个 Key 的 Token 用量会分开记录在 `token_usage` 表里，方便你查看谁用得多。

### 6.4 这个 Key 的作用

- 如果你在 `config.yaml` 里填了 `deepseek_api_key`，代理会一律用那个密钥访问 DeepSeek，客户端的 Key 只是个**身份标识**，不验证。
- 如果你没填 `deepseek_api_key`，代理会要求客户端的 Key 必须是 `api_keys` 表里存在且 `active=true` 的，否则拒绝请求。

**推荐**：填上 `deepseek_api_key`，这样省事。客户端随便传个 Key 都行，但有 Key 就能统计用量。

---

## 七、把 Claude Code 接上代理

Claude Code 是 Anthropic 家的命令行 AI 工具。它用 **Anthropic Messages 格式** 跟大模型通信，
代理的 `/v1/messages` 端点就是给它准备的。

### 7.1 找到配置文件

Claude Code 的配置放在用户目录下：

- **Windows**：`C:\Users\你的用户名\.claude\`
- **Mac/Linux**：`~/.claude/`

新建一个 `settings.local.json`（优先级高于 `settings.json`，不会被 Claude Code 自己覆盖）。

> 如果你只想在某个项目里用代理，就在项目根目录建 `.claude/settings.local.json`。

### 7.2 写入配置

```json
{
  "env": {
    "ANTHROPIC_AUTH_TOKEN": "sk-dcp-123",
    "ANTHROPIC_BASE_URL": "http://localhost:9000",
    "ANTHROPIC_DEFAULT_OPUS_MODEL": "deepseek-v4-pro",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL": "deepseek-v4-flash",
    "ANTHROPIC_DEFAULT_SONNET_MODEL": "deepseek-v4-pro",
    "CLAUDE_CODE_SUBAGENT_MODEL": "deepseek-v4-flash",
    "CLAUDE_CODE_EFFORT_LEVEL": "max",
    "DISABLE_AUTOUPDATER": "1"
  }
}
```

### 7.3 每个配置是啥意思

| 配置项 | 干啥的 |
|--------|-------|
| `ANTHROPIC_BASE_URL` | 让 Claude Code 把请求发到代理 `http://localhost:9000` |
| `ANTHROPIC_AUTH_TOKEN` | 填你在第 6 步创建的 Key，比如 `sk-dcp-123` |
| `ANTHROPIC_DEFAULT_OPUS_MODEL` | Claude Code 里 Opus 角色用的模型名，填 DeepSeek 的模型 |
| `ANTHROPIC_DEFAULT_SONNET_MODEL` | Sonnet 角色用的模型名 |
| `ANTHROPIC_DEFAULT_HAIKU_MODEL` | Haiku 角色（轻量任务）用的模型名，建议用便宜的 flash |
| `CLAUDE_CODE_SUBAGENT_MODEL` | 子 agent 用的模型，建议用便宜的 flash |
| `CLAUDE_CODE_EFFORT_LEVEL` | 推理强度，`max` 效果最好 |
| `DISABLE_AUTOUPDATER` | 关掉自动更新，避免覆盖配置 |

### 7.4 验证

终端里跑：

```bash
claude
```

进去后随便问一句话，正常回复就说明链路通了。代理的终端窗口里应该能看到请求日志。

---

## 八、把 Codex 接上代理

Codex 是 OpenAI 家的命令行 AI 工具（你现在用的这个就是）。它用 **OpenAI Responses 格式**，
代理的 `/v1/responses` 端点就是给它准备的。

### 8.1 配置 auth.json（API Key）

- **Windows**：`C:\Users\你的用户名\.codex\`
- **Mac/Linux**：`~/.codex/`

在 `.codex/` 下新建或编辑 `auth.json`：

```json
{
  "OPENAI_API_KEY": "sk-dcp-123"
}
```

> 填你在第 6 步创建的 Key，用的是同一个 `sk-dcp-123`。

### 8.2 配置 config.toml（模型和地址）

在 `.codex/` 下新建或编辑 `config.toml`：

```toml
base_url = "http://localhost:9000/v1"

model = "deepseek-v4-pro"
model_reasoning_effort = "xhigh"

model_provider = "openai-deepseek"

[model_providers.openai-deepseek]
name = "DeepSeek"
base_url = "http://localhost:9000/v1"
wire_api = "responses"
```

### 8.3 每个配置是啥意思

| 配置项 | 干啥的 |
|--------|-------|
| `base_url` | Codex 发请求的地址，填代理 `http://localhost:9000/v1` |
| `model` | 告诉代理你想用 DeepSeek 的哪个模型 |
| `model_reasoning_effort` | 推理力度，`xhigh` 最强 |
| `model_provider` | 自定义一个提供商叫 `openai-deepseek` |
| `[model_providers.openai-deepseek]` | 这个提供商的详细配置 |
| `wire_api` | `"responses"` 表示用 OpenAI Responses API（Codex 默认协议） |

### 8.4 验证

终端里跑：

```bash
codex
```

进去后问一句话，正常回复就通了。

---

## 九、Claude Code vs Codex 配置速查

| | Claude Code | Codex |
|---|---|---|
| 厂商 | Anthropic | OpenAI |
| 配置文件 | `~/.claude/settings.local.json` | `~/.codex/config.toml` + `~/.codex/auth.json` |
| 代理端点 | `/v1/messages`（Anthropic 格式） | `/v1/responses`（OpenAI Responses 格式） |
| 地址写法 | `http://localhost:9000` | `http://localhost:9000/v1` |
| API Key 写法 | `"ANTHROPIC_AUTH_TOKEN": "sk-dcp-123"` | `"OPENAI_API_KEY": "sk-dcp-123"` |
| 配置格式 | JSON | TOML + JSON |

---

## 十、常见问题

### Q: 启动代理时报错 "bind: address already in use"

9000 端口被占了。改 `config.yaml` 里 `port` 为别的（比如 `9001`），
然后把 Claude Code / Codex 配置里的地址对应改成 `http://localhost:9001`。

### Q: 怎么停止代理？

终端里按 `Ctrl + C`。

### Q: 想开机自启怎么办？

- **Windows**：把 `.exe` 快捷方式放到 `shell:startup` 文件夹
- **Mac**：用 launchd 注册服务
- **Linux**：写 systemd service

### Q: 代理能跑在另一台机器上吗？

可以。`config.yaml` 里 `host: 0.0.0.0` 就是允许外部访问。
其他机器把 Claude Code / Codex 配置里的 `localhost` 换成代理机器的 IP 就行。

### Q: 管理后台除了创建 Key 还能干嘛？

- **`token_usage` 表**：看每个 Key 用了多少 Token，谁在用、用了多少一目了然。
- **`reasoning_cache` 表**：推理缓存，代理自动管理，一般不用管。
- **右上角齿轮**：可以修改管理员密码、添加其他管理员。

### Q: 想换模型怎么办？

改 `config.yaml` 里的 `model`，重启代理就行。Claude Code / Codex 的配置不用动。

### Q: 我能创建多个 Key 给不同人用吗？

完全可以。在管理后台 `api_keys` 表里多建几条记录就行，每条 Key 的 Token 用量会分开统计。
然后告诉不同的人用不同的 Key，谁用多少一查就知道。

---

搞定！有问题提 Issue。
