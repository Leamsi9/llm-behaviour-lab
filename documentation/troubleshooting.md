# Troubleshooting

This document collects common issues and resolutions when running the lab.

## "Cannot connect to Ollama"

```bash
# Ensure Ollama is running
ollama serve

# Check connection
curl http://localhost:11434/api/tags

# Change port if needed
export OLLAMA_HOST=0.0.0.0:11435
```

## "No models found"

```bash
# Pull models
ollama pull qwen2.5:7b
ollama pull llama3.2:3b

# List available
ollama list
```

## pip install blocked (externally-managed environment)

Create and use a virtual environment (recommended):

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

If you must install system-wide, you can use `--break-system-packages` at your own risk. Prefer a venv to avoid conflicts.

## System Freezes

1. **Check model context**:

   ```bash
   # Verify model is responding
   curl http://localhost:11434/api/tags
   
   # Check specific model info
   curl http://localhost:8002/api/model-info/qwen3:0.6b
   ```

2. **Use smaller models**:

   ```bash
   ollama pull llama3.2:1b
   ```

3. **Monitor resources**:

   ```bash
   htop                    # CPU/RAM
   watch -n 1 nvidia-smi   # GPU (if available)
   ```

4. **Ollama Constraints**

To modify constraints directly in Ollama for better stability, set these environment variables before running `ollama serve`:

```bash
export OLLAMA_NUM_THREADS=4       # Limit CPU threads to 4
export OLLAMA_GPU_LAYERS=35       # Limit GPU layers (0 disables GPU)
export OLLAMA_MAX_LOADED_MODELS=3 # Limit concurrent loaded models
```

These environment variables allow fine-tuning Ollama's resource consumption to match your system's capabilities, preventing freezes and ensuring stable operation.

## WebSocket Errors

- Check browser console for connection issues.
- Ensure no firewall blocks WebSocket connections.
- Try a different browser (Chrome recommended).
