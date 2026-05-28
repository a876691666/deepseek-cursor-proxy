## auth.json

```json
{
  "OPENAI_API_KEY": "sk-dcp-123"
}
```

## config.toml

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