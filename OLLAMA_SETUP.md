# Ollama Setup Guide

## Why Ollama?

- ✅ **No manual GGUF downloads** - automatic model management
- ✅ **No authentication issues** - everything local
- ✅ **Faster setup** - just `ollama pull <model>`
- ✅ **Better performance** - optimized for your hardware
- ✅ **Easy switching** - change models with one command

## Installation

### Linux
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

### macOS
```bash
brew install ollama
```

### Windows
Download from: https://ollama.ai/download

## Setup Steps

### 1. Start Ollama Server
```bash
ollama serve
```

### 2. Pull Models

**Qwen 2.5 7B (Recommended):**
```bash
ollama pull qwen2.5:7b           # Instruct (RLHF)
ollama pull qwen2.5:7b-base      # Base (if available)
```

**Mistral 7B:**
```bash
ollama pull mistral:7b           # Instruct
ollama pull mistral:7b-base      # Base (if available)
```

**Llama 3 8B:**
```bash
ollama pull llama3:8b            # Instruct
ollama pull llama3:8b-base       # Base (if available)
```

**Note**: Not all models have `-base` variants. Check available models:
```bash
ollama list
```

### 3. Install Python Dependencies
```bash
pip install httpx
```

### 4. Run the App
```bash
python app_ollama.py
```

### 5. Open UI
Navigate to: http://localhost:8000

## Configuration

Edit `app_ollama.py` to change models:

```python
DEFAULT_MODEL = "qwen2.5:7b"          # Instruct model
DEFAULT_BASE_MODEL = "qwen2.5:7b-base"  # Base model
```

Available options:
- `qwen2.5:7b` / `qwen2.5:14b` / `qwen2.5:32b`
- `mistral:7b` / `mistral:latest`
- `llama3:8b` / `llama3:70b`
- `gemma2:9b` / `gemma2:27b`

## Checking Available Models

```bash
# List pulled models
ollama list

# Search for models
ollama search qwen

# Get model info
ollama show qwen2.5:7b
```

## Troubleshooting

### "Cannot connect to Ollama"
```bash
# Make sure Ollama is running
ollama serve

# Check if it's running
curl http://localhost:11434/api/tags
```

### "Model not found"
```bash
# Pull the model first
ollama pull qwen2.5:7b
```

### Change Ollama Port
```bash
# Set environment variable
export OLLAMA_HOST=0.0.0.0:11435

# Then in app_ollama.py
OLLAMA_BASE_URL = "http://localhost:11435"
```

## Performance Tips

1. **GPU Acceleration**: Ollama automatically uses GPU if available
2. **Memory**: Models stay in memory after first use (faster subsequent requests)
3. **Concurrent Requests**: Ollama handles multiple requests efficiently

## Comparison: Ollama vs Manual GGUF

| Feature | Ollama | Manual GGUF |
|---------|--------|-------------|
| **Setup** | `ollama pull` | Download + manage files |
| **Size** | Automatic quantization | Manual selection |
| **Updates** | `ollama pull` again | Re-download |
| **GPU** | Automatic | Manual config |
| **Memory** | Optimized | Manual tuning |
| **Speed** | Fast | Fast |

## Next Steps

1. Pull your preferred models
2. Start `ollama serve`
3. Run `python app_ollama.py`
4. Compare base vs instruct models in the UI!
