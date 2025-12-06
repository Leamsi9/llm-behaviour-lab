#!/usr/bin/env python3
"""
Integrated Energy and Behaviour Testing Lab for LLM analysis.

A focused application for testing energy consumption and behavioural effects
across different prompt injection strategies and tool integrations.
Supports 1-2 models simultaneously for detailed comparative analysis.
"""

import os
import json
import asyncio
import httpx
import contextlib
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import uvicorn

# Import comparison functionality at module level
from app_model_comparison import Payload as ComparisonPayload, run_generation as run_comparison_generation

# Import our specialized modules
from energy_tracker import (
    energy_tracker,
    estimate_energy_impact,
    get_available_benchmarks,
    ENERGY_BENCHMARKS,
    get_hf_benchmarks,
    get_benchmark_sources,
)
from rapl_benchmark import measure_wh_per_1000_tokens
from prompt_injection import injection_manager, InjectionConfig, InjectionType
from tool_integration import tool_manager, ToolIntegrationConfig, ToolIntegrationMethod, ToolCall
from ollama_client import (
    check_ollama_connection,
    list_ollama_models,
    get_models_with_defaults,
    get_model_info,
    print_startup_info,
    OLLAMA_BASE_URL,
    DEFAULT_MODEL,
    DEFAULT_BASE_MODEL,
    DEFAULT_MAX_OUTPUT_TOKENS,
    REQUEST_TIMEOUT
)

# Import Groq cloud client for cloud model support
from groq_client import (
    check_groq_connection,
    get_groq_models_with_defaults,
    is_groq_configured,
)

# Import payload classes and test functions from standalone apps
from app_energy import EnergyPayload, run_energy_test, ensure_scaphandre_ready

# ---------- Core Testing Logic ----------
# Test functions are now imported from standalone apps (app_energy.py, app_alignment.py)
# This ensures consistency - any changes to standalone apps automatically propagate here

# ---------- FastAPI App ----------

@asynccontextmanager
async def lifespan(_: FastAPI):
    """Check Ollama connection and warm up Scaphandre on startup."""
    await print_startup_info("LLM Behavior Lab (Integrated)")

    # Best-effort Scaphandre warm-up so the first live power run already has
    # an exporter available for E_llm sampling. This uses a lightweight faux
    # WebSocket that prints log messages to stdout instead of sending them
    # over a real client connection.
    class _StartupWebSocket:
        async def send_json(self, msg: dict) -> None:
            try:
                print(json.dumps({"scaphandre_startup": msg}, ensure_ascii=False))
            except Exception:
                print(f"scaphandre_startup: {msg}")

    try:
        await ensure_scaphandre_ready(_StartupWebSocket())
    except Exception:
        # Any failure here just means we won't have per-process E_llm until
        # Scaphandre is manually started or becomes available later.
        pass

    yield
    print("Shutting down LLM Behavior Lab...")

app = FastAPI(
    title="LLM Behavior Lab (Integrated)",
    description="Comprehensive testing environment for LLM behavior analysis - Energy and Comparison testing",
    version="1.0.0",
    lifespan=lifespan
)
app.mount("/static", StaticFiles(directory="static"), name="static")

SYSTEM_PROMPTS_DIR = Path("./data/system_prompts")
DOCS_DIR = Path("./documentation")
README_PATH = Path("./README.md")

# ---------- Routes ----------

@app.get("/")
async def index():
    """Serve the main lab selection UI"""
    try:
        with open("static/ui_main.html", "r") as f:
            content = f.read()
        return HTMLResponse(content)
    except FileNotFoundError:
        # Fallback to energy UI if main UI doesn't exist
        try:
            with open("static/ui_energy.html", "r") as f:
                content = f.read()
            return HTMLResponse(content)
        except FileNotFoundError:
            return HTMLResponse("""
            <h1>LLM Behavior Lab</h1>
            <p>Welcome to the comprehensive LLM testing environment!</p>
            <p>Please create static/ui_main.html for the main interface.</p>
            """)

@app.get("/energy")
async def energy_ui():
    """Serve the energy testing UI"""
    try:
        with open("static/ui_energy.html", "r") as f:
            content = f.read()
        return HTMLResponse(content)
    except FileNotFoundError:
        return HTMLResponse("<h1>Energy UI not found. Please check static/ui_energy.html</h1>")

@app.get("/behaviour")
async def behaviour_ui():
    """Serve the model comparison UI (basic model testing)"""
    try:
        with open("static/ui_multi.html", "r") as f:
            content = f.read()
        return HTMLResponse(content)
    except FileNotFoundError:
        return HTMLResponse("<h1>Model Behaviour UI not found. Please check static/ui_multi.html</h1>")


@app.get("/documentation")
async def documentation_ui():
    """Serve the integrated documentation page for the lab."""
    try:
        with open("static/ui_docs.html", "r") as f:
            content = f.read()
        return HTMLResponse(content)
    except FileNotFoundError:
        return HTMLResponse("<h1>Documentation UI not found. Please check static/ui_docs.html</h1>")

@app.get("/api/models")
async def get_available_models(provider: str = "local"):
    """List available models based on provider (local=Ollama, cloud=Groq)"""
    if provider == "cloud":
        if not is_groq_configured():
            return {
                "models": [],
                "current": {"base": "", "instruct": ""},
                "provider": "cloud",
                "error": "GROQ_API_KEY not configured"
            }
        return await get_groq_models_with_defaults()
    else:
        # Default to local (Ollama)
        result = await get_models_with_defaults()
        result["provider"] = "local"
        return result


@app.get("/api/model-info/{model_name}")
async def get_model_info_route(model_name: str):
    """Get model information including context length (via shared ollama_client.get_model_info).

    This mirrors the endpoint in app_energy so the integrated lab on port 8001
    exposes the same API surface as the standalone energy lab on port 8002.
    """
    try:
        info = await get_model_info(model_name)
        if info and isinstance(info, dict):
            ctx = info.get("context_length") or 40960
            details = info.get("details", {}) or {}
            return {
                "model_name": info.get("name", model_name),
                "context_length": ctx,
                "architecture": details.get("architecture", "unknown"),
                "source": "ollama_client.get_model_info",
            }
        # Fallback if info is None or malformed
        return {
            "model_name": model_name,
            "context_length": 40960,
            "architecture": "unknown",
            "source": "fallback_estimate",
        }
    except Exception as e:
        # Return fallback on any error
        return {
            "model_name": model_name,
            "context_length": 40960,
            "architecture": "unknown",
            "source": "error_fallback",
            "error": str(e),
        }

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    ollama_ok = await check_ollama_connection()
    return {
        "status": "ok" if ollama_ok else "degraded",
        "ollama": ollama_ok,
        "websocket": True,
        "models": {
            "base": DEFAULT_BASE_MODEL,
            "instruct": DEFAULT_MODEL
        }
    }

@app.get("/api/energy-benchmarks")
async def get_energy_benchmarks():
    """Get available energy benchmarks"""
    return {"benchmarks": get_available_benchmarks()}

@app.get("/api/emissions")
async def get_emissions():
    """Get electricity grid emission factors for all countries."""
    try:
        emissions_path = Path("data/emissions.json")
        if emissions_path.exists():
            with emissions_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            return {"emissions": data}
        else:
            return {"emissions": [], "error": "emissions.json not found"}
    except Exception as e:
        return {"emissions": [], "error": str(e)}


@app.get("/api/system-prompts")
async def get_system_prompts():
    """List available reference system prompts from ./system_prompts.

    Returns a list of objects with:
      - id: filename without extension
      - name: same as id (for display)
      - content: full file contents
    """

    prompts: List[Dict[str, Any]] = []

    if SYSTEM_PROMPTS_DIR.is_dir():
        for path in sorted(SYSTEM_PROMPTS_DIR.iterdir()):
            if not path.is_file():
                continue
            # Accept common text/markdown extensions but don't expose them in the name
            if path.suffix.lower() not in {".txt", ".md", ".mkd", ".markdown"}:
                continue
            try:
                content = path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                content = ""

            prompts.append({
                "id": path.stem,
                "name": path.stem,
                "content": content,
            })

    return {"prompts": prompts}

@app.get("/api/docs")
async def get_docs():
    """Return lab documentation (README + markdown files in ./documentation) as raw Markdown.

    Docs are ordered to match the order in which they are first referenced in README.md,
    with the Overview (README) first and any unreferenced docs appended alphabetically.
    """
    docs: List[Dict[str, Any]] = []

    # Cache README contents so we can both expose it and use it for ordering.
    readme_text: str = ""
    if README_PATH.exists():
        try:
            readme_text = README_PATH.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            readme_text = ""

        docs.append({
            "id": "overview",
            "label": "Overview",
            "file": "README.md",
            "content": readme_text,
        })

    # Additional markdown documents under ./documentation
    if DOCS_DIR.is_dir():
        for path in sorted(DOCS_DIR.iterdir()):
            if not path.is_file():
                continue
            if path.suffix.lower() not in {".md"}:
                continue
            try:
                content = path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                content = ""

            # Derive a human-friendly label from the filename if possible
            stem = path.stem.replace("_", " ")
            label = stem[:1].upper() + stem[1:]

            docs.append({
                "id": path.stem,
                "label": label,
                "file": f"documentation/{path.name}",
                "content": content,
            })

    # Sort docs to match a preferred logical order for key energy docs, then
    # README order of appearance, then fall back to label.
    def doc_sort_key(doc: Dict[str, Any]) -> tuple:
        # Overview (README) always first
        doc_id = doc.get("id")
        if doc_id == "overview":
            return (0, "")

        # Explicit preferred order for core energy documentation sections
        preferred_order = {
            "energy_weighted_tokens": 1_000,
            "benchmark_etoken_estimates": 2_000,
            "live_energy_measurements": 3_000,
            "emissions": 4_000,
            "models": 5_000,
            "api_endpoints": 6_000,
            "installation": 7_000,
            "troubleshooting": 8_000,
        }

        if doc_id in preferred_order:
            base_pos = preferred_order[doc_id]
            label_key = (doc.get("label") or doc_id or "").lower()
            return (base_pos, label_key)

        # If we don't have README text, group all non-overview docs after it alphabetically.
        if not readme_text:
            return (10_000_000, (doc.get("label") or doc.get("id") or "").lower())

        # Find first mention of this doc's file or filename in README (case-insensitive).
        file_ref = (doc.get("file") or "")
        candidates: List[str] = []
        if file_ref:
            candidates.append(file_ref)
            candidates.append(file_ref.lstrip("./"))
            # Bare filename, e.g. models.md
            candidates.append(file_ref.split("/")[-1])

        best_index: Optional[int] = None
        readme_lower = readme_text.lower()
        for candidate in candidates:
            cand = candidate.lower()
            if not cand:
                continue
            idx = readme_lower.find(cand)
            if idx != -1 and (best_index is None or idx < best_index):
                best_index = idx

        if best_index is None:
            # Not explicitly referenced in README: group after referenced docs.
            position = 20_000_000
        else:
            # Shift by 1 to leave room for overview at 0, and add offset to come after preferred docs.
            position = best_index + 1 + 10_000

        label_key = (doc.get("label") or doc.get("id") or "").lower()
        return (position, label_key)

    docs.sort(key=doc_sort_key)

    return {"docs": docs}

@app.get("/api/injection-methods")
async def get_injection_methods():
    """Get available prompt injection methods"""
    return {"methods": injection_manager.get_available_injections()}

@app.get("/api/tool-methods")
async def get_tool_methods():
    """Get available tool integration methods"""
    return {"methods": tool_manager.get_available_methods()}

@app.get("/api/session-summary")
async def get_session_summary():
    """Get current session energy summary"""
    return energy_tracker.get_session_summary()

@app.post("/api/export-session")
async def export_session(request: dict):
    """Export session data to file"""
    try:
        filepath = request.get("filepath", "energy_session.json")
        energy_tracker.export_readings(filepath)
        return {"success": True, "filepath": filepath}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")

@app.post("/api/switch-benchmark")
async def switch_benchmark(request: dict):
    """Switch to a different energy benchmark and recalculate all readings."""
    try:
        benchmark_name = request.get("benchmark_name") if isinstance(request, dict) else request
        # Ensure external benchmarks are registered before switching
        from energy_tracker import ensure_external_energy_benchmarks_registered
        ensure_external_energy_benchmarks_registered()
        result = energy_tracker.recalculate_with_benchmark(benchmark_name)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/add-custom-benchmark")
async def add_custom_benchmark(request: dict):
    """Add a custom energy benchmark."""
    try:
        name = request.get("name")
        description = request.get("description")
        watt_hours = request.get("watt_hours_per_1000_tokens")
        source = request.get("source", "Custom")
        hardware_specs = request.get("hardware_specs", "User defined")

        if not name or not description or watt_hours is None:
            raise HTTPException(status_code=400, detail="Missing required fields")

        success = energy_tracker.add_custom_benchmark(
            name=name,
            description=description,
            watt_hours_per_1000_tokens=float(watt_hours),
            source=source,
            hardware_specs=hardware_specs
        )

        if success:
            return {"success": True, "message": f"Custom benchmark '{name}' added"}
        else:
            raise HTTPException(status_code=400, detail=f"Benchmark '{name}' already exists")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to add benchmark: {str(e)}")

@app.get("/api/benchmark-info")
async def get_benchmark_info():
    """Get information about benchmarks and CO2 conversion."""
    return {
        "benchmarks": get_available_benchmarks(),
        "hf_benchmarks": get_hf_benchmarks(),
        "co2_info": {
            "global_average_gco2_per_kwh": 400,
            "description": "Global average carbon intensity for electricity generation",
            "source": "IEA 2023 Global Energy Review",
            "units": "grams of CO2 equivalent per kilowatt-hour",
            "calculation": "Energy consumption (Wh) × Carbon intensity (gCO2/kWh) ÷ 1000",
        },
        # Expose all external benchmark JSON sources (e.g. Hugging Face, Jegham et al.)
        "benchmark_sources": get_benchmark_sources(),
    }


@app.post("/api/rapl-calibrate")
async def rapl_calibrate(request: dict):
    """Run N integrated-RAPL measurements and create/update a calibrated benchmark.

    Request JSON:
      {
        "runs": 30,
        "model_name": "qwen3:0.6b",
        "prompt": "..."  # optional, defaults to same as create_custom_benchmark
      }
    """

    runs = max(int(request.get("runs", 10)), 1)
    model_name = request.get("model_name") or DEFAULT_MODEL
    prompt = request.get("prompt") or "Explain how transformers work in machine learning in 3 sentences."
    include_thinking = bool(request.get("include_thinking", False))
    target_words = int(request.get("target_words", 120))

    values = []

    for _ in range(runs):
        try:
            result = await measure_wh_per_1000_tokens(
                prompt=prompt,
                model=model_name,
                max_tokens=512,
                include_thinking=include_thinking,
                target_words=target_words,
            )
        except Exception as exc:
            # Best-effort: skip failed runs, continue
            print(f"RAPL calibration run failed: {exc}")
            continue

        if result and result.wh_per_1000_tokens is not None:
            values.append(float(result.wh_per_1000_tokens))

    if not values:
        raise HTTPException(status_code=500, detail="No successful RAPL measurements. Check RAPL and Ollama.")

    # Basic statistics
    values_sorted = sorted(values)
    n = len(values_sorted)
    mean = sum(values_sorted) / n
    median = values_sorted[n // 2] if n % 2 == 1 else 0.5 * (values_sorted[n // 2 - 1] + values_sorted[n // 2])
    # Population std dev
    var = sum((x - mean) ** 2 for x in values_sorted) / n
    std = var ** 0.5
    p5 = values_sorted[max(int(n * 0.05) - 1, 0)]
    p95 = values_sorted[min(int(n * 0.95), n - 1)]
    cv = std / mean if mean else None

    # Create/update a calibrated benchmark using the median
    base_key = model_name.replace(":", "_").replace(" ", "_")
    benchmark_name = f"rapl_calibrated_{base_key}{'_thinking' if include_thinking else ''}"
    energy_tracker.add_custom_benchmark(
        name=benchmark_name,
        description=(
            f"RAPL-calibrated benchmark for {model_name} (median of {n} runs: {median:.4f} Wh/1K)"
        ),
        watt_hours_per_1000_tokens=median,
        source="Integrated RAPL calibration",
        hardware_specs="Current system (RAPL measured)",
        # Use force_update if available; safe to ignore if not
    )

    return {
        "metric": "wh_per_1000_tokens",
        "model": model_name,
        "prompt": prompt,
        "requested_runs": runs,
        "successful_runs": n,
        "values": values_sorted,
        "stats": {
            "mean": mean,
            "median": median,
            "std": std,
            "p5": p5,
            "p95": p95,
            "cv": cv,
        },
        "benchmark": {
            "name": benchmark_name,
            "watt_hours_per_1000_tokens": median,
            "thinking": include_thinking,
            "target_words": target_words,
        },
    }

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for energy testing"""
    await websocket.accept()
    print("🔌 WebSocket /ws connected - waiting for messages...")

    current_task: Optional[asyncio.Task] = None
    cancel_event: Optional[asyncio.Event] = None

    def reset_task(_: asyncio.Task):
        nonlocal current_task, cancel_event
        cancel_event = None
        current_task = None

    try:
        while True:
            try:
                raw = await websocket.receive_json()
                print(f"📨 Received message on /ws: {raw.get('test_type', 'unknown')}")
            except json.JSONDecodeError:
                await websocket.send_json({"error": "Invalid JSON"})
                continue
            except Exception as exc:
                await websocket.send_json({"error": f"Invalid payload: {str(exc)}"})
                continue

            # Handle control commands (e.g., cancel)
            if isinstance(raw, dict) and raw.get("command") == "cancel":
                if cancel_event:
                    cancel_event.set()
                if current_task and not current_task.done():
                    current_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError, Exception):
                        await current_task
                continue

            # Convert payload - accept any dict for energy testing
            try:
                # For flexibility, accept any payload with required fields
                required_fields = ['system', 'user', 'model_name']
                if not all(field in raw for field in required_fields):
                    await websocket.send_json({"error": f"Missing required fields: {required_fields}"})
                    continue

            except Exception as exc:
                await websocket.send_json({"error": f"Invalid payload: {str(exc)}"})
                continue

            if not raw.get('model_name'):
                await websocket.send_json({
                    "error": "No model selected. Please choose a model in the UI first.",
                    "done": True,
                })
                continue

            cancel_event = asyncio.Event()
            # For the integrated lab, treat all tests as energy-style runs
            test_type = raw.get('test_type', 'energy')  # Retained for logging only
            print(f"🧪 [INTEGRATED LAB /ws] Handling {test_type.upper()} test - INTEGRATED APP (delegates to standalone)")
            print(f"🧪 [INTEGRATED LAB /ws] Creating task for payload: {raw.get('model_name')}")
            await websocket.send_json({"log": f"🧪 [INTEGRATED LAB /ws] Handling {test_type.upper()} test - INTEGRATED APP (delegates to standalone)"})

            # Create EnergyPayload and delegate to the standalone energy app (include new composition fields)
            payload_obj = EnergyPayload(
                    # New fields
                    system_prompt=raw.get('system_prompt', ''),
                    user_prompt=raw.get('user_prompt', ''),
                    conversation_context=raw.get('conversation_context', ''),
                    injections=raw.get('injections', []),

                    # Legacy fields
                    system=raw.get('system', ''),
                    user=raw.get('user', ''),

                    model_name=raw.get('model_name', ''),
                    strategy_name=raw.get('strategy_name', 'baseline'),
                    energy_benchmark=raw.get('energy_benchmark', 'conservative_estimate'),
                    injection_type=raw.get('injection_type', 'none'),
                    injection_params=raw.get('injection_params', {}),
                    tool_integration_method=raw.get('tool_integration_method', 'none'),
                    tool_config=raw.get('tool_config', {}),
                    temp=raw.get('temp', 0.7),
                    max_tokens=raw.get('max_tokens', 512),
                    enable_live_power_monitoring=raw.get('enable_live_power_monitoring', False),
                    include_thinking=raw.get('include_thinking', False),
                    provider=raw.get('provider', 'local')  # Provider for cloud/local routing
            )
            current_task = asyncio.create_task(run_energy_test(payload_obj, websocket, cancel_event))
            current_task.add_done_callback(reset_task)

    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as exc:
        print(f"WebSocket error: {str(exc)}")
    finally:
        if cancel_event:
            cancel_event.set()
        if current_task and not current_task.done():
            current_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await current_task
        with contextlib.suppress(Exception):
            await websocket.close()

@app.websocket("/ws/behaviour")
async def comparison_websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for basic model comparison"""
    await websocket.accept()

    current_task: Optional[asyncio.Task] = None
    cancel_event: Optional[asyncio.Event] = None
    current_model_key: str = "base"

    async def acknowledge_cancel(model_key: str):
        await websocket.send_json({
            "token": "[CANCELLED]",
            "model": model_key,
            "done": True,
            "cancelled": True,
        })

    def reset_task(_: asyncio.Task):
        nonlocal current_task, cancel_event
        cancel_event = None
        current_task = None

    try:
        while True:
            try:
                raw = await websocket.receive_json()
            except json.JSONDecodeError:
                await websocket.send_json({"error": "Invalid JSON"})
                continue
            except Exception as exc:
                await websocket.send_json({"error": f"Invalid payload: {str(exc)}"})
                continue

            # Handle control commands (e.g., cancel)
            if isinstance(raw, dict) and raw.get("command") == "cancel":
                target_model = raw.get("model_key", current_model_key)

                if cancel_event:
                    cancel_event.set()
                if current_task and not current_task.done():
                    current_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError, Exception):
                        await current_task
                else:
                    await acknowledge_cancel(target_model)
                continue

            # Convert payload - comparison lab expects the standalone Payload shape
            try:
                required_fields = ['system', 'user', 'model_name']
                if not all(field in raw for field in required_fields):
                    await websocket.send_json({"error": f"Missing required fields: {required_fields}"})
                    continue

                # Reuse the standalone comparison Payload dataclass so provider is preserved
                payload_obj = ComparisonPayload(**raw)
            except Exception as exc:
                await websocket.send_json({"error": f"Invalid payload: {str(exc)}"})
                continue

            if not payload_obj.model_name:
                await websocket.send_json({
                    "error": "No model selected. Please choose a model in the UI first.",
                    "done": True,
                })
                continue

            cancel_event = asyncio.Event()
            current_model_key = "base" if payload_obj.use_base_model else "instruct"

            # For the integrated lab, comparison WebSocket delegates to the standalone comparison app
            test_type = raw.get('test_type', 'comparison')  # Logging only
            print(f"🧪 [INTEGRATED LAB /ws/behaviour] Handling {test_type.upper()} test - delegating to comparison app")
            await websocket.send_json({"log": f"🧪 [INTEGRATED LAB /ws/behaviour] Handling {test_type.upper()} test - delegating to comparison app"})

            current_task = asyncio.create_task(run_comparison_generation(payload_obj, websocket, cancel_event))
            current_task.add_done_callback(reset_task)

    except WebSocketDisconnect:
        print("Basic comparison client disconnected")
    except Exception as exc:
        print(f"Basic comparison WebSocket error: {str(exc)}")
    finally:
        if cancel_event:
            cancel_event.set()
        if current_task and not current_task.done():
            current_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await current_task
        with contextlib.suppress(Exception):
            await websocket.close()

# ---------- Token Counting ----------

def count_tokens(text: str, model_name: str = "gpt-3.5-turbo") -> int:
    """Count tokens in text using tiktoken."""
    try:
        # Try to get the appropriate encoding for the model
        if "llama" in model_name.lower() or "mistral" in model_name.lower():
            # Use cl100k_base for most modern models
            encoding = tiktoken.get_encoding("cl100k_base")
        else:
            # Default to GPT encoding
            encoding = tiktoken.encoding_for_model(model_name)
        return len(encoding.encode(text))
    except Exception:
        # Fallback to simple estimation: ~4 characters per token
        return len(text) // 4

def analyze_token_breakdown(system_prompt: str, user_prompt: str, final_system: str, final_user: str,
                           injection_result: Dict, tool_integration_result: Dict,
                           ollama_prompt_tokens: int, ollama_completion_tokens: int,
                           model_name: str) -> Dict[str, Any]:
    """
    Analyze token usage breakdown by component.
    Returns detailed token counts for different phases and components.
    """

    # Count tokens for each component using our tokenizer
    token_counts = {
        "original": {
            "system_prompt_tokens": count_tokens(system_prompt, model_name),
            "user_prompt_tokens": count_tokens(user_prompt, model_name),
            "total_original_tokens": 0  # Will calculate
        },
        "injected": {
            "system_injection_tokens": count_tokens(final_system, model_name) - count_tokens(system_prompt, model_name),
            "user_injection_tokens": count_tokens(final_user, model_name) - count_tokens(user_prompt, model_name),
            "total_injection_tokens": 0  # Will calculate
        },
        "tool_integration": {
            "tool_preparation_tokens": 0,  # Tokens used in tool setup/preparation
            "tool_execution_tokens": 0,    # Tokens used during tool calls
            "tool_integration_overhead": 0  # Additional tokens from tool integration
        },
        "generation": {
            "direct_output_tokens": ollama_completion_tokens,  # Assume all from Ollama are direct for now
            "iterative_tokens": 0,     # Tokens from iterative processes (reasoning, etc.)
            "tool_response_tokens": 0  # Tokens from tool responses
        },
        "ollama_reported": {
            "prompt_eval_count": ollama_prompt_tokens,
            "eval_count": ollama_completion_tokens,
            "total_ollama_tokens": ollama_prompt_tokens + ollama_completion_tokens
        }
    }

    # Calculate totals
    token_counts["original"]["total_original_tokens"] = (
        token_counts["original"]["system_prompt_tokens"] +
        token_counts["original"]["user_prompt_tokens"]
    )

    token_counts["injected"]["total_injection_tokens"] = (
        token_counts["injected"]["system_injection_tokens"] +
        token_counts["injected"]["user_injection_tokens"]
    )

    # Calculate tool integration overhead
    if tool_integration_result and tool_integration_result.get("tools_processed", 0) > 0:
        # Estimate tool integration overhead (this is approximate)
        tool_integration_metadata = tool_integration_result.get("integration_metadata", {})
        # Assume some overhead for tool processing
        token_counts["tool_integration"]["tool_integration_overhead"] = max(0,
            ollama_prompt_tokens - token_counts["original"]["total_original_tokens"] -
            token_counts["injected"]["total_injection_tokens"]
        )

    # Add metadata about the analysis
    token_counts["analysis_notes"] = [
        "Token counts use tiktoken for accurate estimation",
        "Ollama reported counts may differ due to different tokenization",
        f"Model used for tokenization: {model_name}",
        "Tool integration overhead estimated from prompt differences"
    ]

    return token_counts

# Ollama connection functions now imported from ollama_client.py

# ---------- Run the Application ----------

if __name__ == "__main__":
    print("=" * 60)
    print("LLM Behavior Lab (Integrated)")
    print("=" * 60)
    print()
    print("This lab integrates multiple specialized testing environments:")
    print("• Energy Testing Lab: http://localhost:8001/energy")
    print("• LLM Behaviour Lab: http://localhost:8001/behaviour")
    print()
    print("Standalone apps also available:")
    print("• python3 app_energy.py (port 8002)")
    print("• python3 app_model_comparison.py (port 8000)")
    print()
    print("Features:")
    print("• Energy consumption tracking (tokens → Wh → CO2)")
    print("• Model comparison (side-by-side benchmarking)")
    print("• Prompt injection effects testing")
    print("• Tool integration impact assessment")
    print("• Custom benchmark creation")
    print("• Dynamic benchmark switching")
    print()
    print("Make sure Ollama is running:")
    print("  ollama serve")
    print()
    print("Pull test models:")
    print(f"  ollama pull {DEFAULT_MODEL or 'llama3.1:8b'}")
    print()
    print("=" * 60)
    print()

    uvicorn.run(
        "app_llm_behaviour_lab:app",
        host="0.0.0.0",
        port=8001,  # Main integrated lab
        reload=True,
        log_level="info"
    )
