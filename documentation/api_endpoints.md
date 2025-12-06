# API Endpoints

This document describes the main HTTP and WebSocket endpoints exposed by the lab.

## Summary

- WebSocket `/ws` – streaming inference used by the Energy UI.
- GET `/api/models` – list available models.
- GET `/api/model-info/{model_name}` – model metadata (e.g., context length).
- GET `/api/health` – health check.
- POST `/api/rapl-calibrate` – run RAPL calibration and create benchmarks.
- Additional Energy UI endpoints for benchmarks, system prompts, and exports.

## WebSocket `/ws`

Streaming inference endpoint with cancellation support. Used by Energy UI.

**Request payload (Energy example):**

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

Legacy fields `system` and `user` are still accepted if `system_prompt`/`user_prompt` are omitted. The server composes the final system prompt as: base system + `<conversation_context>` wrapper + free‑form injections.

**Response stream (abridged):**

```json
{"token": "Okay, the user is…", "thinking": true}
{"token": "I'm an AI assistant…", "thinking": false}
{"token": "..."}
{"token": "[DONE]", "done": true, "basic_metrics": {"...": "..."}, "live_power_metrics": {"...": "..."}}
```

`live_power_metrics` (when RAPL is enabled) includes fields such as:

- `prefill_wh`, `decode_wh`, `total_wh`.
- `prefill_wh_per_1000_input_tokens`.
- `decode_wh_per_1000_output_tokens`.
- `energy_weighted_output_wh_per_1000_tokens`.

## GET `/api/models`

Returns available Ollama models.

**Response example:**

```json
{
  "models": ["qwen2.5:7b", "llama3.2:3b", "gemma2:9b"],
  "current": {
    "base": "qwen2.5:7b-base",
    "instruct": "qwen2.5:7b"
  }
}
```

## GET `/api/model-info/{model_name}`

Returns model-specific information including context length and parsed `Modelfile` metadata.

**Response example:**

```json
{
  "model_name": "qwen3:0.6b",
  "context_length": 40960,
  "modelfile_info": {
    "num_ctx": 40960
  }
}
```

## GET `/api/health`

Simple health check endpoint.

**Response example:**

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

## POST `/api/rapl-calibrate`

Run N integrated RAPL measurements and create/update a calibrated benchmark.

**Request example:**

```json
{ "runs": 30, "model_name": "qwen3:0.6b", "prompt": "Explain transformers in 3 sentences." }
```

**Response example (abridged):**

```json
{
  "metric": "wh_per_1000_tokens",
  "successful_runs": 28,
  "stats": { "mean": 0.115, "median": 0.112, "std": 0.010, "cv": 0.087 },
  "benchmark": { "name": "rapl_calibrated_qwen3_0.6b", "watt_hours_per_1000_tokens": 0.112 }
}
```

## Additional Endpoints Used by the Energy UI

- `GET /api/energy-benchmarks` – list benchmarks.
- `GET /api/system-prompts` – list preset system prompts.
- `GET /api/benchmark-info` – metadata about benchmarks and CO2.
- `POST /api/switch-benchmark` – switch current benchmark.
- `POST /api/add-custom-benchmark` – add a custom benchmark.
- `POST /api/export-session` – export session readings.
