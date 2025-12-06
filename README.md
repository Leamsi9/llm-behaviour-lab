# LLM Energy Lab

_Version 0.1.0_

The **LLM Energy Lab (primary)**: The Energy Lab is designed to surface the difference (computational non-equivalence) between the energy impacts of the model in isolation, and the energy impacts of the model as a conversational system powered by (generally invisible), and fully deterministic middleware.  It measures energy consumption per 1000 energy-weighted tokens (e-tokens), and can run live hardware tests via RAPL (on Linux), and rigorous benchmark estimates to compare energy profiles across models, prompts, and prompt-injection strategies.

- **LLM Behaviour Lab (WIP)**: Complementary side-by-side model comparison UI for studying output quality, variability, and behaviour across models and temperatures. Helpful to understand deterministic factors that affect model behaviour, and assess fit for purpose and fine tuning potential of environmentally optimal small models.

## At a Glance

- **How to run**
  - Docker (Linux, reproducible): see [Quick Start](#quick-start) for `docker build` and `docker run` examples.
  - Local (Linux/macOS): run `./install.sh` and then start `uvicorn app_llm_behaviour_lab:app` as shown in [Quick Start](#quick-start).
- **Main UIs**
  - Integrated Lab: `http://localhost:8001/`
  - LLM Energy Lab: `http://localhost:8001/energy`
  - LLM Behaviour Lab: `http://localhost:8001/comparison`
- **Model Providers**
  - **Local**: Ollama instance at `http://localhost:11434` (default, with RAPL power monitoring)
  - **Cloud**: Groq API (optional, requires `GROQ_API_KEY` - see [Model Providers](./documentation/models.md))

## Table of Contents

- [Features](#features)
- [Conceptual Overview](#conceptual-overview)
- [System Recommendations](#system-recommendations)
- [Quick Start](#quick-start)
- [Basic Workflow](#basic-workflow)
- [Methodology & Metrics](#methodology--metrics)
- [Installation Options](#installation-options)
- [Stability Features](#stability-features)
- [API Endpoints](#api-endpoints)
- [Performance Tips](#performance-tips)
- [Troubleshooting](#troubleshooting)
- [Dependencies](#dependencies)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)


 ## Features

### Energy Testing Lab (primary)
 
 ✅ **Per-input/output-token energy metrics**: All "Wh/1000" metrics are per 1000 energy weighted tokens (e-tokens) normalised by input or by output ("Wh/1000 e-tokens").
 
 ✅ **Live Hardware Tests (RAPL)**: Snapshot-based cumulative energy counters with start/split/end snapshots to separate Prefill (input) vs Decode (output) energy, plus measured Wh/1K.
 
 ✅ **Estimated Benchmark Tests**: Switch and create benchmarks; recalculate session with new benchmarks.
 
 ✅ **Middleware Injections and Configuration**: Custom and Frontier System Prompts, Conversation Context (with "Inject conversation"), free-form injections, thinking and temperature toggles.
 
 ✅ **RAPL Batch Runner**: Multiple live runs using current UI query + injections with CLI logs and stop support; results table appears in Test Results.
 
 ✅ **Real-time streaming**: Token streaming over WebSocket (supports reasoning "thinking" streams when present).

### LLM Behaviour Lab (WIP)
 
 ✅ Multi-model comparison panes with per-pane controls.
 
 ✅ Streaming outputs, token counts (prompt/completion), and TPS.
 
 ## Conceptual Overview
 
 At its core, the LLM Behaviour Lab enables systematic exploration of how **deterministic, interpretable and corrigible human-defined parameters extrinsic to the model** interact with the **intrinsic, probabilistic model outputs**. These deterministic parameters include both the direct inference time configuration and code scaffolds (e.g. system/user prompts, temperature, token limits), and the post training inputs (e.g. Q&A, instructions, preferences, reinforcements).
 
 High-level data flow:
 
 - **Browser UIs** (`/energy`, `/comparison`, plus the main lab selector) collect prompts, temperatures, injection strategies, tool settings, and energy benchmark choices.
 - **FastAPI apps** (`app_llm_behaviour_lab.py` and the standalone apps) compose final system/user prompts, apply prompt injections and tool integrations, and stream tokens over WebSockets.
 - **Ollama** performs inference for the selected model(s) and reports prompt/completion token counts and timing.
 - **Energy and behaviour analysis modules** wrap inference to:
  - measure or estimate Wh/1000 weighted output tokens and derived CO2 (RAPL or benchmarks), and
  - analyse token breakdown (original vs injected vs tool-related vs thinking tokens).
 - **Results** are aggregated into session metrics (energy, carbon, tokens, variability) and surfaced back to the UIs and export endpoints.

## System Recommendations

The app runs on any OS but **live power monitoring requires Linux/Intel RAPL** with kernel RAPL support and read permissions. The app will gracefully degrade to benchmark-only estimates if RAPL is unavailable.

The app is designed to work with local models and automatically adapts to each model's capabilities. These are general guidelines for smooth operation:

| RAM | Recommended Models | Notes |
|-----|-------------------|-------|
| 8GB | `llama3.2:1b`, `phi3:mini`, `smollm:360m` | Small models for basic testing |
| 16GB | `llama3.2:3b`, `mistral:7b`, `qwen3:0.6b` | Good balance of speed and capability |
| 32GB | `llama3:8b`, `mixtral:8x7b`, `qwen2.5:7b` | Larger models for serious work |
| 64GB | `llama3:70b`, `qwen2.5:72b` | Full-featured models |

**Context Lengths by Model** (automatically detected):
- `qwen3:0.6b`: 40,960 tokens
- `llama3.2:3b`: 128,000 tokens
- `mistral:7b`: 32,768 tokens
- `smollm:360m`: 2048 tokens
- (Check `/api/model-info/{model}` for exact values)



## Quick Start

For a minimal set of commands, see this section. For a more detailed matrix of options, trade‑offs, and OS‑specific notes, read [Installation Options](#installation-options).

### Docker (Linux, recommended for reproducible runs)

```bash
git clone https://github.com/Leamsi9/llm-behaviour-lab.git
cd llm-behaviour-lab

docker build -t llm-behaviour-lab .
docker run --rm \
  --network=host \
  --privileged \
  -v /sys/class/powercap:/sys/class/powercap:ro \
  llm-behaviour-lab
```

Then open `http://localhost:8001/energy` in your browser.

### Local install (Linux/macOS)

```bash
# 1. Clone the repo
git clone https://github.com/Leamsi9/llm-behaviour-lab.git
cd llm-behaviour-lab

# 2, Run the install script
./install.sh

# 3. Run the integrated lab
uvicorn app_llm_behaviour_lab:app --host 0.0.0.0 --port 8001 --reload
```

Then open:

- Integrated Lab: `http://localhost:8001/`
- LLM Energy Lab: `http://localhost:8001/energy`
- LLM Behaviour Lab: `http://localhost:8001/comparison`

For Windows, alternative Docker networking setups, and the full `install.sh` helper script behaviour (including Scaphandre integration), see [Installation Options](#installation-options).

## Basic Workflow

### LLM Energy Lab

At a high level, a single Energy test walks through the same structure as the Energy UI:

- **Test Configuration**
  - Select **Model** and **Energy Benchmark**.
  - Set **User Query**.
  - Configure **Temperature / Max Tokens**.
  - Add **Middleware Injections** if desired:
    - System Prompt dropdown + textarea + Clear button.
    - Conversation Context + Inject Conversation button.
    - Free‑form injection list.
- **Controls**
  - Toggle **Enable Live Power Monitoring (RAPL)** when you want measured energy instead of pure benchmark estimates.
  - Use **Run Energy Test / Stop Test / Clear Results / Export Data / View Live Logs**.
- **Live Hardware Tests (optional)**
  - RAPL Batch Runs and Calibration tools for repeated or calibration-style testing.
- **Estimated Hardware Tests**
  - Current Benchmark / Switch Benchmark / Add Custom Benchmark.
- **Test Results**
  - RAPL Batch Results (UI Prompt + Injections) table.
  - Energy Consumption (Wh/1000 e-tokens, Carbon, RAPL Measured Wh/1K).
  - Token Analysis (Input, Output, Injection, Tool Overhead, Thinking) and RAPL energy metrics (Prefill / Decode / Energy‑weighted Wh/1K).
  - Strategy / tokens / latency / tokens/sec.
- **Response Output** and **Session Summary** (Total Energy, Carbon Footprint, Energy Weight – Wh/1000 e‑tokens, Total Tokens).

Notes:

- RAPL toggle controls live power monitoring for single runs. When enabled, logs modal opens automatically.
- Energy Weight equals average Wh/1000 e-tokens (per output tokens only).
- System prompt presets are served by `/api/system-prompts`; conversation context injection is wrapped in `<conversation_context>` tags.
- Model context limits are fetched dynamically from Ollama's `/api/show` endpoint.

### LLM Behaviour Lab

For the behavioural UI, a basic workflow is:

1. **Select models** from the multi-select dropdown (hold Ctrl/Cmd for multiple) and click **Add Selected** to create panes.
2. **Craft prompts** (system and user) in each pane – this is where you control deterministic variables like style, safety, or reasoning instructions.
3. **Adjust parameters** such as temperature and max tokens per pane.
4. Use **Generate** on individual panes or **Generate All** for batch comparisons.
5. Inspect **token counts, tokens/sec, and streamed outputs** across models and temperatures.
6. Use **aliases** (e.g., `[Base]`, `[Creative]`, `[Temp 0.1]`) to keep experiments readable.

The "Model Comparison Strategies" and "Advanced Features" sections below outline how to interpret these comparisons in terms of deterministic (human-controlled) vs probabilistic (model-intrinsic) elements.

## Methodology (_Energy Lab_)


### Inference Energy Distribution: Prefill, Decode, and Crossover

During a single LLM inference you can roughly separate work into two phases:

- **Prefill (input tokens)**: The model ingests the full prompt/context and builds up internal key/value caches. Each new input token attends over all previous tokens, so the compute cost grows **roughly quadratically with prompt length**.
- **Decode (output tokens)**: The model autoregressively generates new tokens. Each output token is expensive (it attends over the entire existing context), but the number of decode steps is usually bounded by your `max_tokens` or stopping criteria.

At short prompt lengths, **decode is typically more energy-intensive per token** than prefill: you pay a high cost for each output token you ask for. As prompts become very long, the quadratic attention cost in the prefill phase grows so much that there is a **crossover point** where:

> Additional input tokens (longer prompts, heavy middleware injections) dominate the total energy, and prefill becomes more expensive than decode.

The Energy Testing Lab is designed to help you see and reason about this crossover in practice:

- It surfaces **separate energy metrics** for prefilling the prompt vs decoding the answer (when live measurements are enabled).
- It always reports **Wh/1000 weighted output tokens ("e‑tokens")**, so you can compare runs with different prompt and output lengths on a common scale.
- By systematically increasing prompt size (e.g., via conversation context or prompt injections) while holding the requested output behavior approximately fixed, you can observe:
  - Prefill Wh and **Prefill Wh/1000 input tokens** rising with longer prompts.
  - Decode Wh and **Decode Wh/1000 output tokens** staying relatively stable for similar-length answers.
  - The point where **prefill Wh overtakes decode Wh** for your model and hardware — your empirical crossover region.

### Energy-Weighted Tokens (e-Tokens)

_For full rationale and maths, see [Energy-Weighted Tokens documentation](./documentation/energy_weighted_tokens.md)._ 

> **E‑tokens express the total end to end energy of an inference run per 1,000 tokens**.
> 
> **E-tokens ammortize the full energy cost of inference over either input or output tokens.**
>
> **E-tokens turn raw Wh measurements into human-interpretable, model-comparable quantities**.

---

Let

$$
E = \text{total inference energy (Wh)}
$$

$$
T_{\text{input}}, T_{\text{output}} = \text{number of input/output tokens}
$$

Then:

- **e-token-i (input)**
  $$
  \text{e-token-i} = \frac{E}{T_{\text{input}}} \times 1000
  $$
- **e-token-o (output)**
  $$
  \text{e-token-o} = \frac{E}{T_{\text{output}}} \times 1000
  $$

---

#### **E-tokens provide:**

 ✅ A universal scale for comparing models

 ✅ A universal scale for comparing tasks

 ✅ A universal scale for comparing hardware

 ✅ A way to express task intensity ("how heavy is this task?")

 ✅ A predictive metric for relative cost at scale

**_Examples:_**

- Two models perform the same task on the same hardware, with the same input and output tokens. The model with the lower Wh/1000 e-tokens value will be more energy-efficient.
- One model performs two tasks on the same hardware, each task yielding a different Wh/1000 e-tokens value. The task with the lower Wh/1000 e-tokens value will be more energy-efficient.
- The same model runs the same task with the same input and output tokens on different hardware. The hardware yielding the lower Wh/1000 e-tokens value will be more energy-efficient.

In each of these examples the e-token value is a **direct metric of the total energy** consumed by the query and its response.

#### **How to use e-tokens in the Energy Lab**

- Use **Wh/1000 e‑tokens** in the UI as your primary energy efficiency metric.
- Prefer **input‑normalised** values when you have more visibility into the input tokens, or wish to emphasize *query length*.
- Prefer **output‑normalised** values when you have more visibility into output tokens, or wish to emphasize *final inference output*.

### Benchmark e-Token Estimates

_This section summarises how external benchmarks are turned into e‑token metrics. For full methodology, including Hugging Face AI Energy Score and Jegham et al., see [Benchmark e‑Token Estimates documentation](./documentation/benchmark_etoken_estimates.md)._ 

> **Benchmarks at a glance**  
> - **Hugging Face AI Energy Score (HF)** – ground‑truth GPU energy measurements for open models on H100 hardware.  
> - **Jegham et al. (2025)** – modelled energy estimates for many closed and open models under a standardised 1k/1k token workload.  
> - Together they provide rigorous and replicable examples of appying e-tokens to both measured and inferred energy use.
> See [Benchmark e‑Token Estimates](./documentation/benchmark_etoken_estimates.md) for details.


The lab currently uses two benchmark families:

- **Hugging Face AI Energy Score (HF)** – GPU energy **measured** on an H100, reported as Wh/1,000 queries and converted here to **Wh/1,000 input tokens (e‑token‑i)**.
- **Jegham et al. (2025)** – API benchmarks for many open and closed models; energy is **modelled** from assumed infrastructure and a standardised **1,000 input / 1,000 output** workload, giving e‑token‑style values for both input and output.

In practice:

- HF anchors **ground‑truth GPU measurements** for open models.
- Jegham provides **plausible proxies** and coverage for closed models and additional I/O regimes.
- The UI exposes these as selectable **energy benchmarks** that drive estimated Wh/1000 e‑token values when live power is disabled.

### Live Energy Measurements

_This section explains live power monitoring at a glance; see [Live Energy Measurements documentation](./documentation/live_energy_measurements.md) for full RAPL/Scaphandre workflows and calibration._ 

The lab supports three energy modes:

- **Live (RAPL)** – reads CPU/DRAM RAPL counters on Linux to measure **actual Wh** per run.
- **Live (RAPL + Scaphandre, optional)** – attributes power to the Ollama process tree to estimate **LLM‑only Wh/1000 tokens (`E_llm`, `M_eff`)**.
- **Estimated** – uses benchmark coefficients (HF / Jegham / calibrations) when live monitoring is unavailable.

To use live energy:

1. Run on **Linux** with RAPL available under `/sys/class/powercap/*-rapl*`.
2. Start the lab and enable **“Enable Live Power (RAPL)”** in the Energy UI.
3. Optionally run a Scaphandre Prometheus exporter so `E_llm` and `M_eff` are available.

When live power is off, the lab falls back to **benchmark-based estimates only**.

### API Endpoints

_This section maps out the main backend surfaces. For complete request/response schemas and more endpoints, see the [API Endpoints documentation](./documentation/api_endpoints.md)._ 

Core endpoints:

- **WebSocket `/ws`** – streaming inference used by the Energy UI.
  - Send model, prompts, injections, energy benchmark, and flags like `enable_live_power_monitoring`.
  - Receive streamed tokens plus a final message containing basic metrics and (optionally) live power metrics.

- **GET `/api/models`** – list available Ollama models and current defaults.
- **GET `/api/model-info/{model_name}`** – model metadata, including context length.
- **GET `/api/health`** – simple health check (backend + Ollama connectivity).
- **POST `/api/rapl-calibrate`** – run N live RAPL measurements and produce a calibrated benchmark.

The Energy UI also calls helper endpoints such as `/api/energy-benchmarks`, `/api/system-prompts`, and `/api/export-session`; these are described in the API documentation.

## Emissions & Carbon Footprint

The lab converts energy consumption (Wh) into carbon footprint estimates (gCO2) using **electricity grid emission factors**. By default, it uses the IEA global average of **445 gCO2/kWh**, but users can select country-specific factors for more accurate local estimates.

### Features

- **Global Average**: Default emission factor of 0.445 kgCO2e/kWh (IEA Electricity 2025)
- **Country-Specific**: 209 countries with emission factors based on their electricity generation fuel mix
- **UI Selection**: "Electricity Grid" dropdown in the Results Overview to select your region
- **Real-time Updates**: Carbon footprint recalculates instantly when you change the grid selection

### Data Sources

| Source | Coverage | Description |
|--------|----------|-------------|
| [IEA Electricity 2025](https://www.iea.org/reports/electricity-2025/emissions) | Global | Worldwide average carbon intensity |
| [CaDI](https://www.carbondi.com) | 209 Countries | Country-specific factors from fuel mix data |

For detailed documentation on the methodology, data sources, and emission factor calculations, see the [Emissions Documentation](./documentation/emissions.md).

## Model Comparison Lab (WIP)

 `localhost:8001/comparison`

The LLM Comparison Lab complements the Energy Lab by helping assess fit for purpose for small language models. The Energy Lab can help surface energy efficient alternatives to frontier models, but it may be unclear how suitable they would be in either general or specialist applications. The Comparison Lab enables side by side comparison and intuitions about fine tuning potential. It is also (and primarily) designed to reveal insights about:

#### The Deterministic Elements of LLM behavioural patterns (Human-Controlled)
- **System Prompt**: Defines the AI's role, personality, and behavioral constraints. Presets sourced from https://github.com/elder-plinius/CL4R1T4S are available in the Energy UI.
- **User Prompt**: The specific task or question being asked
- **Temperature**: Controls randomness (0.0 = deterministic, 1.0 = creative, 2.0 = chaotic)
- **Token limits**: Limits output length and computational cost
- **Post training inputs**: Fine-tuning ([instruction, preference and reinforcement](https://www.interconnects.ai/p/the-state-of-post-training-2025)) comparable in this app by selecting base and fine tuned models (e.g. `qwen2.5:7b` vs `qwen2.5:7b-instruct`) or different fine-tuning approaches to the same base model (e.g. tool-using vs instruction-tuned vs abliterated (uncensored) vs niche-tuned). The post-training becomes intrinsic to the model, but the process of post-training relies on deterministic human artefacts extrinsic to the base models, in the form of explicit instructions, preferences and reinforcements that are fully interpretable and corrigible.

#### The Probabilistic Elements of LLM behavioural patterns(Model-Dependent)
- **Architecture differences**: Transformer variants, attention mechanisms, parameter counts
- **Training data**: What knowledge and patterns each model has learned
- **Fine-tuning approach**: Base models vs instruction-tuned vs abliterated (uncensored) vs niche-tuned vs tool-using variants
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

### Dynamic Model Limits
- **Automatic detection**: Context limits are fetched from Ollama's `/api/show` endpoint for each model
- **Model-aware limits**: Context limits are derived from model metadata instead of arbitrary constants
- **User visibility**: UI displays actual model context length (e.g., "40,960" for qwen3:0.6b)
- **Flexible defaults**: Max tokens defaults to 1000 but can be overridden by user

### System Protection
- **Thread limiting**: Caps CPU usage to 4 threads (configurable via OLLAMA_NUM_THREADS)
- **Request timeouts**: `REQUEST_TIMEOUT` prevents infinite hangs (default: 180s)
- **HTTP cleanup**: Properly closes connections on cancellation
- **Error handling**: Errors properly close WebSockets and exit loading states

### Emergency Recovery
If you experience freezes:
```bash
# Kill processes
pkill -9 ollama
pkill -9 python

# Restart with sensible defaults
ollama serve
uvicorn app_llm_behaviour_lab:app --host 0.0.0.0 --port 8001 --reload
```

Note: The app uses model-specific limits from Ollama, so manual limit adjustments are rarely needed.


## Troubleshooting

_Refer to this section for quick fixes. For a fuller catalogue of issues and remedies, see the [Troubleshooting documentation](./documentation/troubleshooting.md)._ 

**Connection to Ollama fails**

- Make sure `ollama serve` is running.
- Verify `curl http://localhost:11434/api/tags` works from the same machine.
- If using a non‑default host/port, set `OLLAMA_HOST` accordingly.

**No models found**

- Pull at least one model via `ollama pull ...` (e.g. `qwen2.5:7b`, `llama3.2:3b`).
- Check `ollama list` to confirm they are available.

**Install or environment issues**

- Prefer a **virtual environment** when installing Python deps:
  - `python3 -m venv .venv && source .venv/bin/activate`.
- Install requirements with `pip install -r requirements.txt` (and `requirements-dev.txt` if needed).

**System freezes or resource pressure**

- Try smaller models (e.g. `llama3.2:1b`).
- Monitor CPU/GPU usage (`htop`, `watch -n 1 nvidia-smi`).
- Tune Ollama limits via env vars like `OLLAMA_NUM_THREADS`, `OLLAMA_GPU_LAYERS`, `OLLAMA_MAX_LOADED_MODELS`.

**WebSocket/UI problems**

- Check the browser console for errors.
- Ensure no local firewall or reverse proxy is blocking WebSocket connections.
- Try a different browser if problems persist.

## Dependencies

- **FastAPI**: Web framework.
- **Uvicorn**: ASGI server.
- **httpx**: HTTP client for Ollama API.
- **Ollama**: Local LLM inference server (separate install; see Quick Start).
- **Scaphandre** (optional): for per‑process LLM energy (`E_llm`) when you want to distinguish LLM energy from other processes on the same host.

## Contributing
This is a **work in progress** project and contributions are welcome. Please follow these steps:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test with different model configurations
5. Submit a pull request

## License

This project is licensed under the **MIT License** 

**This software is fully Free and Open Source.** You are free to:
 
 ✅ Use it for any purpose (personal, commercial, educational)
 
 ✅ Modify and distribute your changes
 
 ✅ Include it in other projects
 
 ✅ Use it in production environments

### Author
**Ismael Velasco** - Original developer and maintainer

## Acknowledgments

- [Ollama](https://ollama.ai/) for efficient local LLM inference
- [FastAPI](https://fastapi.tiangolo.com/) for the web framework
- [Meta Llama](https://ai.meta.com/llama/) and other model providers
