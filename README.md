# LLM Behaviour Lab

A FastAPI-based lab for analyzing LLM behavior with two primary experiences:

- **Energy Testing Lab (primary)**: Measure energy consumption per 1000 output tokens ("Wh/1000 e-tokens"), run live hardware tests via RAPL, and benchmark/compare energy profiles across models and prompts.
- **Model Comparison (secondary)**: Side-by-side model comparison UI.

Alignment Testing Lab is available in the integrated app (see /alignment).





## Table of Contents

- [Methodology & Metrics](#methodology--metrics)
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

## Methodology & Metrics

### Energy Measurement
- **Live (RAPL)**: Direct hardware measurement using Intel/AMD RAPL interface. Measures CPU/DRAM energy during inference. Accurate but requires local execution on Linux with RAPL available.
- **Estimated**: Uses benchmark coefficients (Wh/1000 tokens) derived from hardware specs or calibration. Good for approximation when live monitoring isn't available.

### Metrics Explained
- **Wh/1000 Energy-Weighted Output Tokens (E-Tokens**: Energy intensity metric. Amortizes total energy (input + output) over generated output tokens. Useful for comparing the “cost of production.”
- **Input/Output Split**: Separate energy costs for processing the prompt (prefill) and generating tokens (decode). Used in estimated mode for more accurate attribution.
- **Injection Overhead**: Extra tokens added by system prompts, tools, or conversation context that the user doesn’t see but still pay for in energy.

See the “RAPL Workflow” section for step-by-step live measurement and batch procedures.

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

### 5. Run the Application (Integrated Lab)
```bash
source venv/bin/activate
uvicorn app_llm_behaviour_lab:app --host 0.0.0.0 --port 8001 --reload
```

### 6. Open UI
- Energy Testing Lab: `http://localhost:8001/energy`
- Model Comparison: `http://localhost:8001/comparison`
- Alignment Testing Lab: `http://localhost:8001/alignment`

Optional standalone Energy app (same UI/endpoints on a separate port):

```bash
uvicorn app_energy:app --host 0.0.0.0 --port 8002 --reload
open http://localhost:8002/energy
```

## Features

### Energy Testing Lab (primary)
- ✅ **Per-output-token energy metrics**: All "Wh/1000" metrics are per 1000 output tokens ("Wh/1000 e-tokens").
- ✅ **Live Hardware Tests (RAPL)**: Integrated continuous sampling with baseline and active power, and measured Wh/1K.
- ✅ **Estimated Hardware Tests**: Switch and create benchmarks; recalculate session with new benchmarks.
- ✅ **Middleware Injections**: Fixed System Prompt (presets from `./system_prompts`), Conversation Context (with "Inject conversation"), and free-form injections.
- ✅ **RAPL Batch Runner**: Multiple live runs using current UI query + injections with CLI logs and stop support; results table appears in Test Results.
- ✅ **Session Summary**: Cumulative energy/carbon across session, plus Energy Weight (Wh/1000 e-tokens).
- ✅ **Real-time streaming**: Token streaming over WebSocket (supports reasoning "thinking" streams when present).

### Model Comparison (secondary)
- ✅ Multi-model comparison panes with per-pane controls.
- ✅ Streaming outputs, token counts (prompt/completion), and TPS.

## Usage

### Energy Testing Lab Overview

Layout & ordering (top to bottom):

- **Header**
- **Test Configuration**
  - Model & Energy Benchmark selection
  - User Query
  - Temperature / Max Tokens
  - Middleware Injections:
    - System Prompt dropdown + textarea + Clear button
    - Conversation Context + Inject Conversation button
    - Free‑form injection list
- **Controls**
  - Enable Live Power Monitoring (RAPL) toggle + helper text
  - Run Energy Test / Stop Test / Clear Results / Export Data / View Live Logs
- **Live Hardware Tests**
  - RAPL Batch Runs (left)
  - RAPL Calibration (right)
- **Estimated Hardware Tests**
  - Current Benchmark / Switch Benchmark / Add Custom Benchmark
- **Test Results**
  - RAPL Batch Results (UI Prompt + Injections) table
  - Energy Consumption (incl. Wh/1000 e-tokens, Carbon, RAPL Measured Wh/1K)
  - Token Analysis (Input, Output, Injection, Tool Overhead)
  - Strategy / tokens / latency / tokens/sec
- **Response Output**
- **Session Summary** (Total Energy, Carbon Footprint, Energy Weight – Wh/1000 e‑tokens, Total Tokens)

Notes:

- RAPL toggle controls live power monitoring for single runs. When enabled, logs modal opens automatically.
- Energy Weight equals average Wh/1000 e‑tokens (per output tokens only).
- System prompt presets are served by `/api/system-prompts`; conversation context injection is wrapped in `<conversation_context>` tags.

### RAPL Workflow

#### Prerequisites
- Linux host with RAPL available at `/sys/class/powercap/*-rapl*`.
- Sufficient permissions to read RAPL energy counters (root/sudo may be required on some systems).
- If RAPL is unavailable, the app gracefully degrades to estimated benchmarks (no measured values).

#### Single Run (live measurement)
1. Enable "Enable Live Power (RAPL)" in the Energy UI.
2. Enter your query, configure injections/context as desired.
3. Click "Run Energy Test".
4. The Live Logs modal will open and stream:
   - Baseline/active watts snapshots
   - Integrated RAPL energy (Wh total)
   - Measured Wh/1000 e‑tokens (per 1000 output tokens)
5. On completion:
   - A single-row summary is added to "RAPL Batch Results (UI Prompt + Injections)".
   - A dynamic benchmark named `rapl_live_dynamic` is created/updated and used for the energy metrics of that run.

#### Batch Mode (N runs)
1. Set "Number of Runs" > 1 in Batch Testing.
2. Keep "Enable Live Power (RAPL)" enabled.
3. Click "Run Energy Test".
4. The UI performs N back-to-back live runs with current query + injections.
5. CLI logs show per-run tokens, latency, energy (Wh), and measured Wh/1K.
6. At the end, a summary row is added to "RAPL Batch Results (UI Prompt + Injections)" containing:
   - Mean/Median/Std and 5–95% Wh/1000 e‑tokens
   - Coefficient of Variation (CV)
   - Input/Output token statistics (μ/median)
7. Use the red Remove action to delete rows. Results are not persisted as benchmarks.

Tip: Use the first-run duration to estimate time remaining; the CLI log prints an ETA.

#### Calibration (Benchmark creation)
- Use `POST /api/rapl-calibrate` (or the Calibration UI button when available) to run many integrated measurements and create/update a named benchmark (e.g., `rapl_calibrated_<model>`). See API Endpoints for request/response.

#### Output Interpretation
- Measured Wh/1000 e‑tokens are normalized per 1000 output tokens only (completion tokens).
- The Energy chart shows:
  - Total Energy (blue)
  - Intensity (Wh/1000 e‑tokens) (purple)
- Session Summary aggregates energy/carbon across runs, independent of RAPL availability.

#### Troubleshooting RAPL
- No rows appear in the RAPL table:
  - Ensure the RAPL toggle is enabled for the run(s).
  - Check Live Logs for "Integrated RAPL" and "Measured" lines; if absent, RAPL may be unavailable.
  - If logs show "insufficient tokens", increase output tokens or prompt complexity.
- Permission denied reading `energy_uj`:
  - Try running with elevated permissions or adjust udev permissions for powercap.

### Model Comparison Strategies
Each model comparison reveals insights about:

#### The Deterministic Elements (Human-Controlled)
- **System Prompt**: Defines the AI's role, personality, and behavioral constraints. Presets sourced from https://github.com/elder-plinius/CL4R1T4S are available in the Energy UI.
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
uvicorn app_llm_behaviour_lab:app --host 0.0.0.0 --port 8001 --reload
```

## API Endpoints

### WebSocket `/ws`
Streaming inference endpoint with cancellation support. Used by Energy UI.

**Request payload (Energy):**
```json
{
  "model_name": "qwen3:0.6b",
  "system_prompt": "You are a helpful assistant.",
  "user_prompt": "Explain quantum computing.",
  "conversation_context": "<prior_msgs>...</prior_msgs>",
  "injections": [
    { "description": "Safety guardrails", "content": "..." }
  ],
  "temp": 0.7,
  "max_tokens": 512,
  "energy_benchmark": "conservative_estimate",
  "enable_live_power_monitoring": true
}
```

Legacy fields `system` and `user` are still accepted if `system_prompt`/`user_prompt` are omitted. The server composes the final system prompt as: Base system + `<conversation_context>` wrapper + free‑form injections.

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

### POST `/api/rapl-calibrate`
Run N integrated RAPL measurements and create/update a calibrated benchmark.

Request:
```json
{ "runs": 30, "model_name": "qwen3:0.6b", "prompt": "Explain transformers in 3 sentences." }
```

Response (abridged):
```json
{
  "metric": "wh_per_1000_tokens",
  "successful_runs": 28,
  "stats": { "mean": 0.115, "median": 0.112, "std": 0.010, "cv": 0.087 },
  "benchmark": { "name": "rapl_calibrated_qwen3_0.6b", "watt_hours_per_1000_tokens": 0.112 }
}
```

Additional endpoints used by the Energy UI:
- `GET /api/energy-benchmarks` – list benchmarks
- `GET /api/system-prompts` – list preset system prompts
- `GET /api/benchmark-info` – metadata about benchmarks and CO2
- `POST /api/switch-benchmark` – switch current benchmark
- `POST /api/add-custom-benchmark` – add a custom benchmark
- `POST /api/export-session` – export session readings

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

### pip install blocked (externally-managed environment)
Create and use a virtual environment (recommended):
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```
If you must install system-wide, you can use `--break-system-packages` at your own risk. Prefer a venv to avoid conflicts.

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
├── app_llm_behaviour_lab.py   # Integrated FastAPI app (Energy primary, Comparison secondary)
├── app_energy.py              # Standalone Energy app (optional)
├── app_model_comparison.py    # Model comparison app
├── static/
│   ├── ui_energy.html         # Energy Testing UI
│   └── ui_multi.html          # Model comparison UI
├── system_prompts/    # Preset system prompts (txt/md)
├── tests/                     # Test suite
├── .env-example               # Environment configuration template
├── requirements.txt           # Python dependencies
├── setup.sh                   # Automated setup script
└── README.md                  # This file
```

Alignment app: Placeholder/WIP (not yet implemented).

## Dependencies

- **FastAPI**: Web framework
- **Uvicorn**: ASGI server
- **httpx**: HTTP client for Ollama API
- **python-dotenv**: Environment configuration
- **Ollama**: Local LLM inference server

Optional for live power monitoring (Linux/Intel RAPL): kernel RAPL support and read permissions. The app will gracefully degrade if RAPL is unavailable.

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
