# Live Energy Measurements

## Overview

The Energy Lab supports several modes for estimating and measuring inference energy:

- **Live Power Measurements (RAPL)**: Uses Intel/AMD **Running Average Power Limit (RAPL)** counters to directly measure CPU/DRAM energy during inference.
- **Live (RAPL + Scaphandre, optional)**: Uses [Scaphandre](https://github.com/hubblo-org/scaphandre) to attribute power **per process** and approximate **LLM-only energy (`E_llm`)**.
- **Estimated**: Uses benchmark coefficients (Wh/1000 tokens) derived from hardware specs or calibration.

In all live modes, RAPL is read as **cumulative energy counters** (in microjoules) from the CPU package/DRAM. The app never guesses instantaneous power: instead it takes three snapshots per run and uses differences to derive Wh.

## RAPL Workflow

RAPL counters are read at three points:

- `start` – just before sending the Ollama request.
- `split` – when the **first token** (thinking or content) is observed.
- `end` – after the last token is streamed.

From these three cumulative readings we compute:

- **Prefill energy**: energy between `start` → `split` (includes full TTFT: model prefill plus any queueing or stalls before the first token).
- **Decode energy**: energy between `split` → `end` (token generation phase).
- **Total energy**: energy between `start` → `end` (entire run window).

We also measure a short **idle baseline power** before each run and use it to derive optional *baseline-subtracted* "active" energies (prefill/decode/total minus idle × time).

## Metrics Explained

- **Wh/1000 Energy-Weighted Output Tokens (E-Tokens)**: Amortizes total energy (input + output) over generated output tokens.
- **Input/Output Split**: Separate energy costs for processing the prompt (prefill) and generating tokens (decode).
- **Injection Overhead**: Extra tokens added by system prompts, tools, or conversation context that the user doesn’t see but still pay for in energy.
- **`E_llm` (LLM-only energy)**: Per-process energy for the Ollama process tree (via Scaphandre), or package-level RAPL fallback.
- **`M_eff` (model compute efficiency)**: Active-time–scaled variant of `E_llm` based on Ollama’s `prompt_eval_duration` + `eval_duration`.

The UI reports both `E_llm` and `M_eff` when live monitoring is enabled, but they answer slightly different questions:

- `E_llm` captures **what the LLM service actually cost on this hardware, under real-world contention**.
- `M_eff` asks **"how intense is compute during the time Ollama reports as actively evaluating?"**

The deviation between `M_eff` and `E_llm` is displayed as `M_eff vs E_llm` and highlighted in red when large (e.g. > ~20%).

## Prerequisites

- Linux host with RAPL available at `/sys/class/powercap/*-rapl*`.
- Sufficient permissions to read RAPL energy counters (root/sudo may be required on some systems).
- If RAPL is unavailable, the app gracefully degrades to estimated benchmarks (no measured values).

Optional, for per-process LLM energy (`E_llm`):

- A running Scaphandre Prometheus exporter (for example: `scaphandre prometheus --bind :8080`).
- Environment variables configured so the Energy app can query the exporter:
  - `SCAPHANDRE_URL` – full URL of the `/metrics` endpoint (e.g. `http://localhost:8080/metrics`).
  - `SCAPHANDRE_OLLAMA_MATCH` – substring to match in the `scaph_process_power_consumption_microwatts` metric lines to select the Ollama process.

## Single Run (Live Measurement)

1. Enable "Enable Live Power (RAPL)" in the Energy UI.
2. Enter your query, configure injections/context as desired.
3. Click "Run Energy Test".
4. The Live Logs modal will open and stream:
   - RAPL snapshots: start → split (first token) → end.
   - Prefill energy (Wh), decode energy (Wh), and total energy (Wh).
   - `E_llm` (per 1000 output tokens) and `M_eff` (per 1000 output tokens).
   - Measured Wh/1000 e‑tokens (legacy RAPL-per-1k metric, still logged for comparison).
5. On completion:
   - A single-row summary is added to "RAPL Batch Results (UI Prompt + Injections)".
   - A dynamic benchmark named `rapl_live_dynamic` is created/updated and used for the energy metrics of that run.

## Batch Mode (N Runs)

1. Set "Number of Runs" > 1 in Batch Testing.
2. Keep "Enable Live Power (RAPL)" enabled.
3. Click "Run Energy Test".
4. The UI performs N back-to-back live runs with current query + injections.
5. CLI logs show per-run tokens, latency, energy (Wh), and measured Wh/1K.
6. At the end, a summary row is added to "RAPL Batch Results (UI Prompt + Injections)" containing:
   - Mean/Median/Std and 5–95% Wh/1000 e‑tokens.
   - Coefficient of Variation (CV).
   - Input/Output token statistics (μ/median).
7. Use the red Remove action to delete rows. Results are not persisted as benchmarks.

Tip: Use the first-run duration to estimate time remaining; the CLI log prints an ETA.

## Calibration (Benchmark Creation)

- Use `POST /api/rapl-calibrate` (or the Calibration UI button when available) to run many integrated measurements and create/update a named benchmark (e.g., `rapl_calibrated_<model>`).

## Output Interpretation

- Measured Wh/1000 e‑tokens are normalized per 1000 output tokens only (completion tokens).
- The Energy chart shows:
  - Total Energy (blue).
  - Intensity (Wh/1000 e‑tokens) (purple).
- Session Summary aggregates energy/carbon across runs, independent of RAPL availability.

## Troubleshooting RAPL

- **No rows appear in the RAPL table**:
  - Ensure the RAPL toggle is enabled for the run(s).
  - Check Live Logs for "Integrated RAPL" and "Measured" lines; if absent, RAPL may be unavailable.
  - If logs show "insufficient tokens", increase output tokens or prompt complexity.
- **Permission denied reading `energy_uj`**:
  - Try running with elevated permissions or adjust udev permissions for powercap.
