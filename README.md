# LLM Behaviour Lab

_Version 0.1.0_

LLM Behaviour Lab is an experimental environment for systematically probing how **human-controlled, deterministic parameters** interact with the **intrinsic, probabilistic behaviour** of Large Language Models via middleware.
 
It focuses on two related labs, both exposed through a FastAPI backend:

- **Energy Testing Lab (primary)**: Measure energy consumption per 1000 energy weighted output tokens ("Wh/1000 e-tokens"), run live hardware tests via RAPL, and benchmark or compare energy profiles across models, prompts, and prompt-injection strategies.

- **Model Comparison Lab (secondary)**: Side-by-side model comparison UI for studying output quality, variability, and behaviour across models and temperatures.


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
- ✅ **Per-output-token energy metrics**: All "Wh/1000" metrics are per 1000 output tokens ("Wh/1000 e-tokens").
- ✅ **Live Hardware Tests (RAPL)**: Snapshot-based cumulative energy counters with start/split/end snapshots to separate Prefill (input) vs Decode (output) energy, plus measured Wh/1K.
- ✅ **Estimated Hardware Tests**: Switch and create benchmarks; recalculate session with new benchmarks.
- ✅ **Middleware Injections**: Fixed System Prompt (presets from `./system_prompts`), Conversation Context (with "Inject conversation"), and free-form injections.
- ✅ **RAPL Batch Runner**: Multiple live runs using current UI query + injections with CLI logs and stop support; results table appears in Test Results.
- ✅ **Real-time streaming**: Token streaming over WebSocket (supports reasoning "thinking" streams when present).

### Model Comparison (secondary)
- ✅ Multi-model comparison panes with per-pane controls.
- ✅ Streaming outputs, token counts (prompt/completion), and TPS.
 
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
- Energy Testing Lab: `http://localhost:8001/energy`
- Model Comparison: `http://localhost:8001/comparison`

For Windows, alternative Docker networking setups, and the full `install.sh` helper script behaviour (including Scaphandre integration), see [Installation Options](#installation-options).

## Basic Workflow

### Energy Testing Lab

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

### Model Comparison Lab

For the comparison UI, a basic workflow is:

1. **Select models** from the multi-select dropdown (hold Ctrl/Cmd for multiple) and click **Add Selected** to create panes.
2. **Craft prompts** (system and user) in each pane – this is where you control deterministic variables like style, safety, or reasoning instructions.
3. **Adjust parameters** such as temperature and max tokens per pane.
4. Use **Generate** on individual panes or **Generate All** for batch comparisons.
5. Inspect **token counts, tokens/sec, and streamed outputs** across models and temperatures.
6. Use **aliases** (e.g., `[Base]`, `[Creative]`, `[Temp 0.1]`) to keep experiments readable.

The "Model Comparison Strategies" and "Advanced Features" sections below outline how to interpret these comparisons in terms of deterministic (human-controlled) vs probabilistic (model-intrinsic) elements.

## Methodology & Metrics

### Inference Energy Workflow: Prefill, Decode, and Crossover

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

### Energy Measurement
- **Live Power Measurements (RAPL)**: Uses Intel/AMD **Running Average Power Limit (RAPL)** counters to directly measure CPU/DRAM energy during inference. RAPL is a hardware feature that exposes cumulative energy readings (in microjoules) per power domain (package, cores, DRAM). The Linux kernel surfaces these counters under `/sys/class/powercap/*-rapl*`. The lab reads these counters before, during, and after each run to compute actual watt‑hours consumed on your machine. Accurate but requires local execution on Linux with RAPL available.
- **Live (RAPL + Scaphandre, optional)**: [Scaphandre](https://github.com/hubblo-org/scaphandre) is an open‑source power/energy agent that also reads RAPL counters but attributes power **per process** and exports metrics in Prometheus format. When its Prometheus exporter is running, the lab samples `scaph_process_power_consumption_microwatts` and filters for the Ollama process tree to approximate **per‑process energy for the LLM service itself (`E_llm`)**. RAPL remains the ground‑truth energy source; Scaphandre adds a process‑level attribution layer on top of it so you can separate LLM energy from other workloads on the same host.
- **Estimated**: Uses benchmark coefficients (Wh/1000 tokens) derived from hardware specs or calibration. Good for approximation when live monitoring isn't available, or for comparing runs against a consistent "virtual hardware" profile (e.g., a specific GPU or CPU baseline).

In all live modes, RAPL is read as **cumulative energy counters** (in microjoules) from the CPU package/DRAM. The app never guesses instantaneous power: instead it takes three snapshots per run and uses differences to derive Wh:

- `start` – just before sending the Ollama request.
- `split` – when the **first token** (thinking or content) is observed.
- `end` – after the last token is streamed.

From these three cumulative readings we compute:

- **Prefill energy**: energy between `start` → `split` (includes full TTFT: model prefill plus any queueing or stalls before the first token).
- **Decode energy**: energy between `split` → `end` (token generation phase).
- **Total energy**: energy between `start` → `end` (entire run window).

We also measure a short **idle baseline power** before each run and use it to derive optional *baseline-subtracted* "active" energies (prefill/decode/total minus idle × time). This helps separate model work from host background load on noisy machines, while still reporting the full package-level energy that the user actually pays.

### Metrics Explained
- **Wh/1000 Energy-Weighted Output Tokens (E-Tokens)**: Energy intensity metric. Amortizes total energy (input + output) over generated output tokens. Useful for comparing the “cost of production.”
- **Input/Output Split**: Separate energy costs for processing the prompt (prefill) and generating tokens (decode). Used in estimated mode for more accurate attribution.
- **Injection Overhead**: Extra tokens added by system prompts, tools, or conversation context that the user doesn’t see but still pay for in energy.
- **E_llm (LLM-only energy)**: When Scaphandre is available, the app integrates per-process power for the Ollama process tree over the whole WebSocket run and computes **Wh/1000 output tokens for the LLM service itself**. If Scaphandre is unavailable, this falls back to package-level RAPL energy. *Intuitively: “what did the Ollama server actually burn in Wh to serve this request, including stalls and waiting, per 1000 output tokens?”*
- **M_eff (model compute efficiency)**: An *active-time–scaled* variant of `E_llm`. It uses Ollama’s `prompt_eval_duration` + `eval_duration` (nanoseconds) as an approximation of **active model compute time** within the wall-clock run, and scales `E_llm` by the ratio `(active_time / wall_time)`. Intuitively: “if only the time the model was actively computing tokens counted, what would Wh/1000 tokens look like?” This is the primary live **efficiency** metric shown in the Results Overview card.

The UI reports both metrics when live monitoring is enabled, but they answer slightly different questions:

- `E_llm` captures **what the LLM service actually cost on this hardware, under real-world contention**. On a noisy CPU host with other processes, or when the OS intermittently suspends the Ollama process, `E_llm` will rise because tokens took longer to produce at roughly the same power draw. Use `E_llm` when you care about "what did this request actually cost me in practice?".
- `M_eff` instead asks **"how intense is compute during the time Ollama reports as actively evaluating?"** by scaling `E_llm` by `(Ollama active time / wall-clock latency)`. This is closer to an "intrinsic" model efficiency and is useful when wall-clock is badly polluted by stalls, queueing, or I/O. The Results Overview card surfaces `M_eff` as the default live efficiency score.
- The **deviation between `M_eff` and `E_llm`** is displayed as ` M_eff vs E_llm` and highlighted in red when large (e.g. > ~20%). A large deviation typically means that a significant portion of the wall-clock time was not reported by Ollama as prompt/decoding (queueing, stalls, I/O waits, or instrumentation gaps). On a stable, lightly loaded machine these two values should be close; large gaps are a signal that system-level factors, not pure model compute, are dominating your energy bill.
- The **deviation between `M_eff` and `E_llm`** is displayed as ` M_eff vs E_llm` and highlighted in red when large (e.g. > ~20%). A large deviation typically means that a significant portion of the wall-clock time was not reported by Ollama as prompt/decoding (queueing, stalls, I/O waits, or instrumentation gaps). On a stable, lightly loaded machine these two values should be close; large gaps are a signal that system-level factors, not pure model compute, are dominating your energy bill.

See the “RAPL Workflow” section for step-by-step live measurement and batch procedures, including how RAPL snapshots and Scaphandre sampling are combined.

#### Prerequisites
- Linux host with RAPL available at `/sys/class/powercap/*-rapl*`.
- Sufficient permissions to read RAPL energy counters (root/sudo may be required on some systems).
- If RAPL is unavailable, the app gracefully degrades to estimated benchmarks (no measured values).

Optional, for per-process LLM energy (`E_llm`):

- A running Scaphandre Prometheus exporter (for example: `scaphandre prometheus --bind :8080`).
- Environment variables configured so the Energy app can query the exporter:
  - `SCAPHANDRE_URL` – full URL of the `/metrics` endpoint (e.g. `http://localhost:8080/metrics`).
  - `SCAPHANDRE_OLLAMA_MATCH` – substring to match in the `scaph_process_power_consumption_microwatts` metric lines to select the Ollama process (default: `"ollama"`, but you can narrow this to a specific command line).

#### Single Run (live measurement)
1. Enable "Enable Live Power (RAPL)" in the Energy UI.
2. Enter your query, configure injections/context as desired.
3. Click "Run Energy Test".
4. The Live Logs modal will open and stream:
   - RAPL Snapshots: start → split (first token) → end
   - Prefill energy (Wh), Decode energy (Wh), and Total energy (Wh)
   - `E_llm` (per 1000 output tokens): LLM-service energy from Scaphandre when available, else package-level RAPL energy
   - `M_eff` (per 1000 output tokens): active-time–scaled efficiency using Ollama’s `prompt_eval_duration` + `eval_duration`
   - Measured Wh/1000 e‑tokens (legacy RAPL-per-1k metric, still logged for comparison)
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

## Installation Options

### Option A: Docker (portable, good for CI and pinned environments)

If you prefer to run the lab inside a container, the repository includes a `Dockerfile` that:

- Builds the integrated lab (FastAPI + Uvicorn) into an image.
- Compiles and **bakes in a `scaphandre` binary** via a Rust build stage.
- Configures the app so that, when live power monitoring is enabled, it will try to start a Scaphandre Prometheus exporter inside the container and use it for per-process `E_llm` measurements (falling back to package-level RAPL if this is not possible).

#### Building the image

From the project root:

```bash
docker build -t llm-behaviour-lab .
```

This produces an image with:

- Python runtime and all dependencies from `requirements.txt` / `requirements-dev.txt`.
- The integrated lab entrypoint (`app_llm_behaviour_lab:app`).
- A `scaphandre` binary on `PATH`, configured by default as:
  - `SCAPHANDRE_CMD="scaphandre prometheus --bind :8080"`
  - `SCAPHANDRE_DEFAULT_URL="http://127.0.0.1:8080/metrics"`
  - `SCAPHANDRE_AUTOSTART=1` (the app will try to start the exporter automatically).

#### Recommended: Linux with `--network=host`

On a Linux host where **Ollama is running directly on the host** and **RAPL is available**, the simplest and most accurate setup is to share the host network and expose RAPL counters into the container:

```bash
docker run --rm \
  --network=host \
  --privileged \
  -v /sys/class/powercap:/sys/class/powercap:ro \
  llm-behaviour-lab
```

This configuration means:

- **Networking**
  - The container shares the host network namespace, so the integrated lab is reachable at `http://localhost:8001/` on the host without extra `-p` flags.
  - Ollama is reachable from the container at the same address it uses on the host (e.g. `http://localhost:11434`), so you usually do *not* need to set `OLLAMA_HOST` explicitly.
- **Energy / RAPL access**
  - `/sys/class/powercap` is mounted read-only from the host, allowing Scaphandre and the app to read RAPL counters.
  - `--privileged` is the simplest way to grant the necessary capabilities for RAPL/Scaphandre in local experiments; for stricter environments you can replace this with a more minimal set of capabilities or device mounts as appropriate.

With this setup, when you enable **Enable Live Power (RAPL)** in the Energy UI:

- The app reads RAPL counters directly from the host to compute total energy (`total_wh`, `prefill_wh`, `decode_wh`).
- Scaphandre inside the container attributes power per process and exposes it on `SCAPHANDRE_DEFAULT_URL`; the app uses this to compute `E_llm` and `M_eff` per 1000 output tokens.

#### Alternative networking setups

If you cannot or do not want to use `--network=host`, you can instead publish only the lab’s port and point the container at a reachable Ollama endpoint explicitly. For example:

```bash
docker run --rm \
  -p 8001:8001 \
  -e OLLAMA_HOST=http://host.docker.internal:11434 \
  llm-behaviour-lab
```

In this mode:

- The integrated lab is available at `http://localhost:8001/` on the host.
- `OLLAMA_HOST` tells the container how to reach the host Ollama instance (the `host.docker.internal` name works on Docker for macOS/Windows and many recent Linux setups; otherwise you can inject the host IP instead).
- You can still mount `/sys/class/powercap` and grant additional capabilities if you want live RAPL measurements inside the container:

```bash
docker run --rm \
  -p 8001:8001 \
  -e OLLAMA_HOST=http://host.docker.internal:11434 \
  --privileged \
  -v /sys/class/powercap:/sys/class/powercap:ro \
  llm-behaviour-lab
```

On macOS and Windows hosts, RAPL is typically not available to the container at all, so live hardware energy metrics will be limited. The lab will still function, but energy values will come from **benchmarks and calibration profiles** rather than direct RAPL reads; Scaphandre may start, but without real RAPL counters it cannot provide meaningful LLM-only energy.

#### Tuning Scaphandre behaviour via environment variables

The container (and bare-metal app) honour the following environment variables:

- `SCAPHANDRE_AUTOSTART` (default `1`): set to `0`, `false`, or `no` to disable automatic Scaphandre startup and rely only on package-level RAPL unless you start an exporter yourself.
- `SCAPHANDRE_CMD`: the exact command the app will run to start the exporter (defaults to `scaphandre prometheus --bind :8080`). You can change this to use a different port, flags, or wrapper script.
- `SCAPHANDRE_DEFAULT_URL`: the URL the app probes for the exporter metrics endpoint (defaults to `http://127.0.0.1:8080/metrics`).

If Scaphandre cannot be started or reached, the app logs a warning and gracefully falls back to RAPL-only measurements; if RAPL is also unavailable, it falls back again to benchmark-based estimates.


### Option B: Local install (Linux/macOS)

Local install is the most straightforward path when:

- You are running on a **single Linux box with RAPL** and want the app to manage Scaphandre for you.
- You want Scaphandre to see **host‑level tools** (browser, code executor, RAG stack) as well as Ollama.

#### 1. Clone the repository

```bash
git clone https://github.com/Leamsi9/llm-behaviour-lab.git
cd llm-behaviour-lab
```

#### 2. Use the `install.sh` helper (recommended on Linux/macOS)

Run:

```bash
./install.sh
```

The `./install.sh` helper script is designed to set up an **end‑to‑end local environment**:

- Creates a local `.venv` virtual environment if it does not exist.
- Upgrades `pip` and installs both `requirements.txt` and `requirements-dev.txt` into that environment.
- Checks for **Ollama** on Linux/macOS and can help install it if missing.
- Offers to pull a default test model (`qwen3:0.6b`) via Ollama.
- On Linux, optionally downloads a local **Scaphandre** binary so per‑process live power works out of the box.
- Prints next steps for starting the integrated lab.

If you prefer not to use the script, you can follow the equivalent manual steps:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
```

#### 3. Run the integrated lab locally

After either `./install.sh` or the manual steps:

```bash
source .venv/bin/activate
uvicorn app_llm_behaviour_lab:app --host 0.0.0.0 --port 8001 --reload
```

Then open:

- Integrated Lab: `http://localhost:8001/`
- Energy Testing Lab: `http://localhost:8001/energy`
- Model Comparison: `http://localhost:8001/comparison`

Optional standalone Energy app (same UI/endpoints on a separate port):

```bash
uvicorn app_energy:app --host 0.0.0.0 --port 8002 --reload
```

#### 4. Manual Ollama install (all OS)

If you want to install Ollama yourself (or are on a non‑Linux OS where `install.sh` does not manage it), use the official instructions:

```bash
# Linux
curl -fsSL https://ollama.ai/install.sh | sh

# macOS
brew install ollama

# Windows
# Download and install from the official website:
# https://ollama.ai/download
```

Once Ollama is installed and serving (`ollama serve`), the lab will discover it at `http://localhost:11434` by default, or use the `OLLAMA_HOST` environment variable if you need a custom endpoint.

#### When to prefer local `install.sh` over Docker

Running the lab via Docker is convenient for **reproducible environments**, **CI**, and **pinned environments**, but a direct local install via `./install.sh` is often preferable when you care about **end‑to‑end energy for tool‑using agents**, not just the LLM server:

- **Whole‑workflow energy vs. LLM‑only energy**
  - In many agentic setups, the LLM response triggers **non‑inference compute**: browser automation, code execution, RAG queries, database calls, etc.
  - A host‑level Scaphandre install (with a local lab) can see **all of these processes** and lets you attribute energy to both the LLM (Ollama) and the **tools it calls**.
  - A containerised Scaphandre instance mostly sees processes **inside the container**; host‑side browsers, DBs, and tools remain invisible unless you add more advanced `--pid=host` style configurations.
- **Measuring “cost of a tool‑augmented answer”**
  - If you want to ask “how much extra energy did this agentic answer cost once it started browsing, running code, or hitting RAG?”, running the lab and Scaphandre directly on the host is the cleanest path.
  - You can then use Scaphandre’s Prometheus metrics to disaggregate **LLM vs non‑LLM** energy for the same request window (e.g. Ollama vs browser vs Python workers).
- **Version 0.1.0 limitation: tool metrics not yet surfaced in the UI**
  - As of **version 0.1.0**, the lab does **not yet expose dedicated “tool energy” or “agentic overhead” metrics in the UI**.
  - You can still collect those numbers from your Scaphandre/Prometheus stack by looking at other processes during the run window, but the app currently focuses on LLM‑centric metrics (`E_llm`, `M_eff`, RAPL prefill/decode).

In short: use `./install.sh` when you want the **clearest view of end‑to‑end energy for tool‑using workflows on a single host**; reach for Docker when you primarily need a **portable, reproducible environment** and are happy to focus on LLM‑centric measurements.

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
  "enable_live_power_monitoring": true,
  "include_thinking": false
}
```

Legacy fields `system` and `user` are still accepted if `system_prompt`/`user_prompt` are omitted. The server composes the final system prompt as: Base system + `<conversation_context>` wrapper + free‑form injections.

**Response stream:**
```json
{"token": "Okay, the user is…", "thinking": true}
{"token": "I'm an AI assistant…", "thinking": false}
{"token": "..."}
{"token": "[DONE]", "done": true, "basic_metrics": {...}, "live_power_metrics": {...}}
```

`live_power_metrics` (when RAPL is enabled) includes:
- `prefill_wh`, `decode_wh`, `total_wh`
- `prefill_wh_per_1000_input_tokens`
- `decode_wh_per_1000_output_tokens`
- `energy_weighted_output_wh_per_1000_tokens`

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

### GET `/api/model-info/{model_name}`
Returns model-specific information including context length.

**Response:**
```json
{
  "model_name": "qwen3:0.6b",
  "context_length": 40960,
  "modelfile_info": {
    "num_ctx": 40960
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

## Performance Tips

1. **Model caching**: Pull frequently used models for faster startup
2. **Concurrent limits**: Don't run too many large models simultaneously
3. **GPU acceleration**: Ollama automatically uses GPU if available
4. **Memory management**: Clear unused models with `ollama stop <model>`

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

### WebSocket Errors
- Check browser console for connection issues
- Ensure no firewall blocks WebSocket connections
- Try different browser (Chrome recommended)

## Dependencies

- **FastAPI**: Web framework.
- **Uvicorn**: ASGI server.
- **httpx**: HTTP client for Ollama API.
- **Ollama**: Local LLM inference server (separate install; see Quick Start).
- **Scaphandre** (optional): for per‑process LLM energy (`E_llm`) when you want to distinguish LLM energy from other processes on the same host.

## Performance Tips

1. **Model caching**: Pull frequently used models for faster startup
2. **Concurrent limits**: Don't run too many large models simultaneously
3. **GPU acceleration**: Ollama automatically uses GPU if available
4. **Memory management**: Clear unused models with `ollama stop <model>`

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
