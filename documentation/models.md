# Model Providers

The LLM Behaviour Lab supports both **local** and **cloud** model providers, allowing you to seamlessly switch between self-hosted models and cloud APIs.

## Provider Overview

| Provider | Platform | Models | RAPL Support | Use Case |
|----------|----------|--------|--------------|----------|
| **Local** | Ollama | All locally pulled models | ✅ Yes | Development, privacy, power measurement |
| **Cloud** | Groq | Llama, Mixtral, Gemma, etc. | ❌ No (estimates only) | Fast inference, larger models |

---

## Local Provider (Ollama)

### Setup

1. **Install Ollama**: Download from [ollama.ai](https://ollama.ai)
2. **Start the server**:
   ```bash
   ollama serve
   ```
3. **Pull models**:
   ```bash
   ollama pull llama3.1:8b
   ollama pull qwen3:0.6b
   ```

### Configuration

Local provider settings in `.env`:

```bash
# Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434      # Ollama API endpoint
OLLAMA_DEFAULT_MODEL=llama3.1:8b            # Default instruct model
OLLAMA_DEFAULT_BASE_MODEL=llama3.1:8b-base  # Default base model

# Performance tuning
MAX_INPUT_LENGTH=16000
MAX_CONTEXT_TOKENS=16384
MAX_OUTPUT_TOKENS=8192
REQUEST_TIMEOUT=300.0
```

### Features

- **RAPL Power Monitoring**: Direct measurement of energy consumption using Intel RAPL
- **Custom Context Windows**: Control over context length per request
- **Full Model Control**: Parameter tuning, seed control, thinking mode
- **Offline Operation**: No internet required after model download

### Supported Models

Any model available in Ollama's library can be used. Popular options:
- `llama3.1:8b`, `llama3.1:70b`
- `qwen3:0.6b`, `qwen3:8b`
- `mistral:7b`, `mixtral:8x7b`
- `gemma2:9b`, `gemma2:27b`

---

## Cloud Provider (Groq)

### Setup

1. **Get API Key**: Sign up at [console.groq.com](https://console.groq.com)
2. **Configure credentials**: Add to your `.env` file:
   ```bash
   GROQ_API_KEY=gsk_your_api_key_here
   ```

### Configuration

Cloud provider settings in `.env`:

```bash
# Groq Cloud API Configuration
GROQ_API_KEY=gsk_your_api_key_here
GROQ_API_BASE_URL=https://api.groq.com/openai/v1  # OpenAI-compatible endpoint
GROQ_DEFAULT_MODEL=llama-3.1-70b-versatile        # Default model
GROQ_REQUEST_TIMEOUT=180.0                         # Request timeout in seconds
```

### Features

- **Ultra-Fast Inference**: Groq's LPU architecture provides exceptional speed
- **Large Models**: Access to 70B+ parameter models without local hardware requirements
- **OpenAI-Compatible API**: Standard chat completion format with streaming
- **Token Counting**: Accurate token usage from API responses

### Limitations

- **No RAPL Support**: Cloud inference cannot be measured with local power monitoring
- **Energy Estimates Only**: Uses benchmark-based energy estimation instead of direct measurement
- **Network Required**: Constant internet connection needed
- **API Costs**: Usage may incur charges based on token consumption

### Supported Models

Models available through Groq API:
- `llama-3.1-70b-versatile`, `llama-3.1-8b-instant`
- `llama3-70b-8192`, `llama3-8b-8192`
- `mixtral-8x7b-32768`
- `gemma2-9b-it`

---

## Using Providers in the UI

### Switching Providers

1. Navigate to the **Prompt Model** section
2. Use the **Provider** dropdown to select:
   - 🖥️ **Local (Ollama)** - for self-hosted models
   - ☁️ **Cloud (Groq)** - for cloud inference

3. The **Model Selection** dropdown will automatically refresh with available models

### RAPL Behavior

When using **Cloud** provider:
- RAPL checkbox is automatically disabled
- A warning note appears: "RAPL power monitoring disabled for cloud models"
- Energy tracking falls back to benchmark-based estimates
- RAPL Calibration is blocked with an alert

When using **Local** provider:
- RAPL checkbox is available (if RAPL is supported on your system)
- Direct power measurement is possible
- RAPL Calibration can be run to create custom benchmarks

---

## API Reference

### `/api/models`

Lists available models based on provider selection.

**Query Parameters:**
- `provider` (optional): `"local"` | `"cloud"` (default: `"local"`)

**Response:**
```json
{
  "models": ["model-name-1", "model-name-2"],
  "current": {
    "base": "default-base-model",
    "instruct": "default-instruct-model"
  },
  "provider": "local"
}
```

**Error Response (cloud not configured):**
```json
{
  "models": [],
  "current": {"base": "", "instruct": ""},
  "provider": "cloud",
  "error": "GROQ_API_KEY not configured"
}
```

---

## Adding New Providers

The architecture is designed to be extensible. To add a new cloud provider:

1. **Create a client module** (e.g., `openai_client.py`):
   - Implement `check_connection()`, `list_models()`, `stream_chat()`
   - Follow the async generator pattern for streaming

2. **Update backend apps**:
   - Import the new client in `app_energy.py`, `app_model_comparison.py`
   - Add provider branching in `/api/models` and inference functions

3. **Update frontend**:
   - Add new option to provider dropdown
   - Handle any provider-specific UI behavior

---

## Best Practices

### For Development
- Use **Local (Ollama)** for:
  - Initial testing and debugging
  - Power consumption analysis
  - Offline development
  - Cost control

### For Production/Benchmarking
- Use **Cloud (Groq)** for:
  - High-throughput testing
  - Accessing larger models
  - Speed benchmarks
  - Consistent baseline comparisons

### Energy Tracking
- **Local**: Enable RAPL for accurate power measurement
- **Cloud**: Select an appropriate energy benchmark for estimation
  - Use `conservative_estimate` for worst-case scenarios
  - Use `inference_optimized` for typical cloud efficiency
