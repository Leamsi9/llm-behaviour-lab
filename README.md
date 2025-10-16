# LLM Behaviour Lab

A FastAPI-based web application for comparing multiple language models side by side using Ollama. Features dynamic model selection, per-pane controls, and stability limits to prevent system freezes.

## Table of Contents

- [Basic Workflow](#basic-workflow)
- [Quick Start](#quick-start)
- [Features](#features)
- [Usage](#usage)
- [Stability Features](#stability-features)
- [API Endpoints](#api-endpoints)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Project Structure](#project-structure)
- [Dependencies](#dependencies)
- [Performance Tips](#performance-tips)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)


At its core, the LLM Behaviour Lab enables systematic exploration of how **deterministic, interpretable and corrigible human-defined parameters extrinsic to the model** interact with the **intrinsic, probabilistic model outputs**. These deterministic parameters include both the direct inference time configuration and code scaffolds (e.g. system/user prompts, temperature, token limits), and the post training inputs (e.g. Q&A, instructions, preferences, reinforcements).


## Basic Workflow
1. **Select models** from the multi-select dropdown (hold Ctrl/Cmd for multiple)
2. **Click "Add Selected"** to create comparison panes
3. **Craft prompts** in the system/user input fields - this is where you control the deterministic variables
4. **Adjust parameters** like temperature (0.0-2.0) and max tokens to see their effects
5. **Click "Generate"** on individual panes or **"Generate All"** for batch
6. **Use "Stop"** buttons to cancel generation
7. **Add aliases** to distinguish between similar models

## Quick Start

### 1. Install Ollama
```bash
# Linux
curl -fsSL https://ollama.ai/install.sh | sh

# macOS
brew install ollama

# Windows: Download from https://ollama.ai/download
```

### 2. Pull Models
```bash
ollama serve  # Start Ollama in another terminal

# Pull some models to compare
ollama pull qwen2.5:7b       # Instruct model
ollama pull llama3.2:3b      # Smaller model
ollama pull gemma2:9b        # Different architecture
```

**Browse all available models**: Visit [https://ollama.com/search](https://ollama.com/search) to explore the full catalog of models available for comparison.

### 3. Setup Python Environment
```bash
# Clone and setup
git clone https://github.com/Leamsi9/llm-behaviour-lab.git
cd llm-behaviour-lab

# Use the setup script
./setup.sh

# Or manually:
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 4. Configure Stability Limits (Optional)
Edit `.env` file with your system specs:
```bash
# For 32GB RAM systems (default)
MAX_INPUT_LENGTH=12000
MAX_CONTEXT_TOKENS=8192
MAX_OUTPUT_TOKENS=4096

# For 16GB RAM systems
MAX_INPUT_LENGTH=8000
MAX_CONTEXT_TOKENS=4096
MAX_OUTPUT_TOKENS=2048
```

### 5. Run the Application
```bash
source venv/bin/activate
uvicorn app_ollama:app --host 0.0.0.0 --port 8000 --reload
```

### 6. Open UI
Navigate to: `http://localhost:8000`

## Features

- ✅ **Multi-model comparison**: Compare any number of Ollama models simultaneously
- ✅ **Dynamic model loading**: Automatically detects and lists all pulled Ollama models
- ✅ **Per-pane controls**: Individual Generate/Stop/Clear/Remove buttons for each model
- ✅ **Global controls**: Generate All and Stop All buttons for batch operations
- ✅ **Real-time streaming**: Token-by-token generation with visual indicators
- ✅ **Stability limits**: Configurable limits to prevent system freezes (.env file)
- ✅ **Cancellation support**: Properly interrupts generation without leaving orphaned processes
- ✅ **Token counting**: Detailed metrics (prompt tokens, completion tokens, latency, TPS)
- ✅ **Model aliases**: Tag each model pane with custom labels
- ✅ **Responsive UI**: Works on desktop and mobile devices

## Usage

### Comparison Strategies
Each model comparison reveals insights about:

#### The Deterministic Elements (Human-Controlled)
- **System Prompt**: Defines the AI's role, personality, and behavioral constraints. To compare behaviour under the system prompts of major LLMs, see https://github.com/elder-plinius/CL4R1T4S for a collection of system prompts for major LLMs and tools which you can use.
- **User Prompt**: The specific task or question being asked
- **Temperature**: Controls randomness (0.0 = deterministic, 1.0 = creative, 2.0 = chaotic)
- **Token limits**: Limits output length and computational cost
- **Post training inputs**: Fine-tuning ([instruction, preference and reinforcement](https://www.interconnects.ai/p/the-state-of-post-training-2025)) comparable in this app by selecting base and fine tuned models (e.g. `qwen2.5:7b` vs `qwen2.5:7b-instruct`) or different fine-tuning approaches to the same base model (e.g. tool-using vs instruction-tuned vs abliterated (uncensored) vs niche-tuned). The post-training becomes intrinsic to the model, but the process of post-training relies on deterministic human artefacts extrinsic to the base models, in the form of explicit instructions, preferences and reinforcements that are fully interpretable and corrigible.

#### The Probabilistic Elements (Model-Dependent)
- **Architecture differences**: Transformer variants, attention mechanisms, parameter counts
- **Training data**: What knowledge and patterns each model has learned
- **Fine-tuning approach**: Base models vs instruction-tuned vs tool-using variants
- **Token generation**: How each model chooses the next word given identical inputs

#### Temperature Testing
Compare the same model at different temperatures:
- `llama3.2:3b [Temp 0.1]` - Precise, factual responses
- `llama3.2:3b [Temp 0.7]` - Balanced creativity and coherence  
- `llama3.2:3b [Temp 1.5]` - Highly creative, more unpredictable

#### Architecture Comparison
Compare different model families:
- `qwen2.5:7b` vs `llama3.2:3b` vs `gemma2:9b` - Same prompt, different architectures
- Base vs instruction-tuned variants of the same model
- Small vs large parameter counts within the same family

#### Fine-tuning Analysis
Compare different training approaches:
- Base models (raw, pre-training only)
- Instruction-tuned (RLHF, aligned for helpfulness)
- Tool-using variants (function calling, API integration)
- Domain-specific fine-tunes (coding, medical, legal)

### Advanced Features

#### Model Aliases
Each pane can have a custom alias (displayed in brackets):
- `qwen2.5:7b [Base]` - for base model comparisons
- `qwen2.5:7b [Creative]` - for creative writing tests
- `llama3.2:3b [Fast]` - for quick iterations
- `mistral:7b [Temp 0.1]` - for precise, factual responses

#### Global Controls
- **Generate All**: Start generation on all panes simultaneously
- **Stop All**: Cancel all active generations
- **Model Status**: Shows number of active WebSocket connections

#### Per-Pane Controls
- **Generate**: Start generation for this model
- **Stop**: Cancel generation (with "Stopping..." feedback)
- **Clear**: Reset output and metrics
- **Remove**: Delete this pane and close its WebSocket

## Stability Features

### Input Validation
- **Character limits**: `MAX_INPUT_LENGTH` prevents memory exhaustion
- **Token capping**: `MAX_OUTPUT_TOKENS` limits generation length
- **Context windows**: `MAX_CONTEXT_TOKENS` prevents overflow

### System Protection
- **Thread limiting**: Caps CPU usage to 4 threads
- **Request timeouts**: `REQUEST_TIMEOUT` prevents infinite hangs
- **HTTP cleanup**: Properly closes connections on cancellation

### Emergency Recovery
If you experience freezes:
```bash
# Kill processes
pkill -9 ollama
pkill -9 python

# Reduce limits in .env
MAX_INPUT_LENGTH=4000
MAX_CONTEXT_TOKENS=2048

# Restart
ollama serve
uvicorn app_ollama:app --reload
```

## API Endpoints

### WebSocket `/ws`
Streaming inference endpoint with cancellation support.

**Request payload:**
```json
{
  "model_name": "qwen2.5:7b",
  "system": "You are a helpful assistant.",
  "user": "Explain quantum computing.",
  "temp": 0.7,
  "max_tokens": 1024,
  "stop": ["USER:", "ASSISTANT:", "</s>"]
}
```

**Response stream:**
```json
{"token": "Quantum"}
{"token": " computing"}
{"token": " is"}
{"token": "..."}
{"token": "[DONE]", "done": true, "metrics": {...}}
```

### GET `/api/models`
Returns available Ollama models.

**Response:**
```json
{
  "models": ["qwen2.5:7b", "llama3.2:3b", "gemma2:9b"],
  "current": {
    "base": "qwen2.5:7b-base",
    "instruct": "qwen2.5:7b"
  }
}
```

### GET `/api/health`
Health check endpoint.

**Response:**
```json
{
  "status": "ok",
  "ollama": true,
  "websocket": true,
  "models": {
    "base": "qwen2.5:7b-base",
    "instruct": "qwen2.5:7b"
  }
}
```

## Configuration

### Environment Variables
Create a `.env` file in the project root from the `.env-example` file:

```bash
# Stability limits
MAX_INPUT_LENGTH=8000          # Character limit for prompts
MAX_CONTEXT_TOKENS=4096        # Ollama context window
MAX_OUTPUT_TOKENS=2048         # Maximum generation length
REQUEST_TIMEOUT=180.0          # Seconds before timeout
```

### System Recommendations

| RAM | Input Length | Context Tokens | Output Tokens | Example Models |
|-----|-------------|----------------|----------------|-------------------|
| 8GB | 4,000 | 2,048 | 1,024 | `llama3.2:1b`, `phi3:mini` |
| 16GB | 8,000 | 4,096 | 2,048 | `llama3.2:3b`, `mistral:7b` |
| 32GB | 16,000 | 16,384 | 8,192 | `llama3:8b`, `mixtral:8x7b` |
| 64GB | 32,000 | 32,768 | 16,384 | `llama3:70b`, `qwen2.5:72b` |

## Troubleshooting

### "Cannot connect to Ollama"
```bash
# Ensure Ollama is running
ollama serve

# Check connection
curl http://localhost:11434/api/tags

# Change port if needed
export OLLAMA_HOST=0.0.0.0:11435
```

### "No models found"
```bash
# Pull models
ollama pull qwen2.5:7b
ollama pull llama3.2:3b

# List available
ollama list
```

### System Freezes
1. **Reduce limits** in `.env`:
   ```bash
   MAX_INPUT_LENGTH=4000
   MAX_CONTEXT_TOKENS=2048
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

### WebSocket Errors
- Check browser console for connection issues
- Ensure no firewall blocks WebSocket connections
- Try different browser (Chrome recommended)

## Project Structure

```
llm-behaviour-lab/
├── app_ollama.py          # FastAPI application with Ollama integration
├── static/
│   └── ui_multi.html      # Multi-model comparison UI
├── .env-example          # Environment configuration template
├── .gitignore            # Git ignore rules
├── requirements.txt      # Python dependencies
├── setup.sh             # Automated setup script
├── README.md            # This file
└── Stability.md         # Detailed stability configuration
```

## Dependencies

- **FastAPI**: Web framework
- **Uvicorn**: ASGI server
- **httpx**: HTTP client for Ollama API
- **python-dotenv**: Environment configuration
- **Ollama**: Local LLM inference server

## Performance Tips

1. **Model caching**: Pull frequently used models for faster startup
2. **Concurrent limits**: Don't run too many large models simultaneously
3. **GPU acceleration**: Ollama automatically uses GPU if available
4. **Memory management**: Clear unused models with `ollama stop <model>`

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test with different model configurations
5. Submit a pull request

## License

This project is licensed under the **MIT License** 

**This software is fully Free and Open Source.** You are free to:
- ✅ Use it for any purpose (personal, commercial, educational)
- ✅ Modify and distribute your changes
- ✅ Include it in other projects
- ✅ Use it in production environments

### Author
**Ismael Velasco** - Original developer and maintainer

## Acknowledgments

- [Ollama](https://ollama.ai/) for efficient local LLM inference
- [FastAPI](https://fastapi.tiangolo.com/) for the web framework
- [Meta Llama](https://ai.meta.com/llama/) and other model providers
