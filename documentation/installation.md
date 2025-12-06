# Installation

This document covers detailed installation options beyond the quick start.

## Summary

- **Option A: Docker** – portable, good for CI and pinned environments.
- **Option B: Local install (Linux/macOS)** – best for host-level energy tracking and integration with other tools.

## Option A: Docker (Portable, Good for CI and Pinned Environments)

If you prefer to run the lab inside a container, the repository includes a `Dockerfile` that:

- Builds the integrated lab (FastAPI + Uvicorn) into an image.
- Compiles and **bakes in a `scaphandre` binary** via a Rust build stage.
- Configures the app so that, when live power monitoring is enabled, it will try to start a Scaphandre Prometheus exporter inside the container and use it for per-process `E_llm` measurements (falling back to package-level RAPL if this is not possible).

### Building the Image

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

### Recommended: Linux with `--network=host`

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
  - `--privileged` is the simplest way to grant the necessary capabilities for RAPL/Scaphandre in local experiments.

With this setup, when you enable **Enable Live Power (RAPL)** in the Energy UI:

- The app reads RAPL counters directly from the host to compute total energy (`total_wh`, `prefill_wh`, `decode_wh`).
- Scaphandre inside the container attributes power per process and exposes it on `SCAPHANDRE_DEFAULT_URL`; the app uses this to compute `E_llm` and `M_eff` per 1000 output tokens.

### Alternative Networking Setups

If you cannot or do not want to use `--network=host`, you can instead publish only the lab’s port and point the container at a reachable Ollama endpoint explicitly. For example:

```bash
docker run --rm \
  -p 8001:8001 \
  -e OLLAMA_HOST=http://host.docker.internal:11434 \
  llm-behaviour-lab
```

In this mode:

- The integrated lab is available at `http://localhost:8001/` on the host.
- `OLLAMA_HOST` tells the container how to reach the host Ollama instance.

You can still mount `/sys/class/powercap` and grant additional capabilities if you want live RAPL measurements inside the container:

```bash
docker run --rm \
  -p 8001:8001 \
  -e OLLAMA_HOST=http://host.docker.internal:11434 \
  --privileged \
  -v /sys/class/powercap:/sys/class/powercap:ro \
  llm-behaviour-lab
```

On macOS and Windows hosts, RAPL is typically not available to the container at all, so live hardware energy metrics will be limited. The lab will still function, but energy values will come from **benchmarks and calibration profiles** rather than direct RAPL reads.

### Tuning Scaphandre Behaviour via Environment Variables

The container (and bare-metal app) honour the following environment variables:

- `SCAPHANDRE_AUTOSTART` (default `1`): set to `0`, `false`, or `no` to disable automatic Scaphandre startup and rely only on package-level RAPL unless you start an exporter yourself.
- `SCAPHANDRE_CMD`: the exact command the app will run to start the exporter (defaults to `scaphandre prometheus --bind :8080`).
- `SCAPHANDRE_DEFAULT_URL`: the URL the app probes for the exporter metrics endpoint (defaults to `http://127.0.0.1:8080/metrics`).

If Scaphandre cannot be started or reached, the app logs a warning and gracefully falls back to RAPL-only measurements; if RAPL is also unavailable, it falls back again to benchmark-based estimates.

## Option B: Local Install (Linux/macOS)

Local install is the most straightforward path when:

- You are running on a **single Linux box with RAPL** and want the app to manage Scaphandre for you.
- You want Scaphandre to see **host‑level tools** (browser, code executor, RAG stack) as well as Ollama.

### Clone the Repository

```bash
git clone https://github.com/Leamsi9/llm-behaviour-lab.git
cd llm-behaviour-lab
```

### Use the `install.sh` Helper (Recommended on Linux/macOS)

Run:

```bash
./install.sh
```

The `./install.sh` helper script is designed to set up an **end‑to‑end local environment**:

- Creates a local `.venv` virtual environment if it does not exist.
- Upgrades `pip` and installs both `requirements.txt` and `requirements-dev.txt` into that environment.
- Checks for **Ollama** on Linux/macOS and can help install it if missing.
- Offers to pull a default test model (`qwen3:0.6b`) via Ollama.
- On Linux, optionally downloads a local **Scaphandre** binary.
- Prints next steps for starting the integrated lab.

If you prefer not to use the script, you can follow the equivalent manual steps:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
```

### Run the Integrated Lab Locally

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

### Manual Ollama Install (All OS)

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

### When to Prefer Local `install.sh` over Docker

Running the lab via Docker is convenient for **reproducible environments**, **CI**, and **pinned environments**, but a direct local install via `./install.sh` is often preferable when you care about **end‑to‑end energy for tool‑using agents**, not just the LLM server:

- **Whole‑workflow energy vs. LLM‑only energy**
  - In many agentic setups, the LLM response triggers **non‑inference compute**: browser automation, code execution, RAG queries, database calls, etc.
  - A host‑level Scaphandre install (with a local lab) can see **all of these processes** and lets you attribute energy to both the LLM (Ollama) and the **tools it calls**.
- **Measuring “cost of a tool‑augmented answer”**
  - If you want to ask “how much extra energy did this agentic answer cost once it started browsing, running code, or hitting RAG?”, running the lab and Scaphandre directly on the host is the cleanest path.
- **Version 0.1.0 limitation: tool metrics not yet surfaced in the UI**
  - As of **version 0.1.0**, the lab does **not yet expose dedicated “tool energy” or “agentic overhead” metrics in the UI**.

In short: use `./install.sh` when you want the **clearest view of end‑to‑end energy for tool‑using workflows on a single host**; reach for Docker when you primarily need a **portable, reproducible environment** and are happy to focus on LLM‑centric measurements.
