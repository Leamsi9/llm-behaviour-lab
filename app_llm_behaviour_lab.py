#!/usr/bin/env python3
"""
Energy & Alignment Testing Lab for LLM Behavior Analysis

A focused application for testing energy consumption and alignment
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
from energy_tracker import energy_tracker, estimate_energy_impact, get_available_benchmarks, ENERGY_BENCHMARKS
from alignment_analyzer import alignment_analyzer, AlignmentScore
from prompt_injection import injection_manager, InjectionConfig, InjectionType
from tool_integration import tool_manager, ToolIntegrationConfig, ToolIntegrationMethod, ToolCall
from ollama_client import (
    check_ollama_connection,
    list_ollama_models,
    get_models_with_defaults,
    print_startup_info,
    OLLAMA_BASE_URL,
    DEFAULT_MODEL,
    DEFAULT_BASE_MODEL,
    MAX_INPUT_LENGTH,
    MAX_CONTEXT_TOKENS,
    MAX_OUTPUT_TOKENS,
    REQUEST_TIMEOUT
)

# Import payload classes and test functions from standalone apps
from app_energy import EnergyPayload, run_energy_test
from app_alignment import AlignmentPayload, run_alignment_test

# ---------- Core Testing Logic ----------
# Test functions are now imported from standalone apps (app_energy.py, app_alignment.py)
# This ensures consistency - any changes to standalone apps automatically propagate here

# ---------- FastAPI App ----------

@asynccontextmanager
async def lifespan(_: FastAPI):
    """Check Ollama connection on startup"""
    await print_startup_info("LLM Behavior Lab (Integrated)")
    yield
    print("Shutting down LLM Behavior Lab...")

app = FastAPI(
    title="LLM Behavior Lab (Integrated)",
    description="Comprehensive testing environment for LLM behavior analysis - Energy, Alignment, and Comparison testing",
    version="1.0.0",
    lifespan=lifespan
)
app.mount("/static", StaticFiles(directory="static"), name="static")

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

@app.get("/alignment")
async def alignment_ui():
    """Serve the alignment testing UI"""
    try:
        with open("static/ui_alignment.html", "r") as f:
            content = f.read()
        return HTMLResponse(content)
    except FileNotFoundError:
        return HTMLResponse("<h1>Alignment UI not found. Please check static/ui_alignment.html</h1>")

@app.get("/comparison")
async def comparison_ui():
    """Serve the model comparison UI (basic model testing)"""
    try:
        with open("static/ui_multi.html", "r") as f:
            content = f.read()
        return HTMLResponse(content)
    except FileNotFoundError:
        return HTMLResponse("<h1>Model Comparison UI not found. Please check static/ui_multi.html</h1>")

@app.get("/api/models")
async def get_available_models():
    """List available Ollama models with defaults"""
    return await get_models_with_defaults()

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
async def export_session(filepath: str = "energy_session.json"):
    """Export session data to file"""
    try:
        energy_tracker.export_readings(filepath)
        return {"success": True, "filepath": filepath}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")

@app.post("/api/switch-benchmark")
async def switch_benchmark(benchmark_name: str):
    """Switch to a different energy benchmark and recalculate all readings."""
    try:
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
        "co2_info": {
            "global_average_gco2_per_kwh": 400,
            "description": "Global average carbon intensity for electricity generation",
            "source": "IEA 2023 Global Energy Review",
            "units": "grams of CO2 equivalent per kilowatt-hour",
            "calculation": "Energy consumption (Wh) × Carbon intensity (gCO2/kWh) ÷ 1000"
        },
        "benchmark_sources": {
            "conservative_estimate": {
                "description": "Conservative estimate based on typical LLM inference workloads",
                "source": "Estimated from various academic papers and industry reports",
                "assumptions": "Assumes efficient GPU utilization, typical model sizes, and modern hardware"
            },
            "nvidia_rtx_4090": {
                "description": "NVIDIA RTX 4090 GPU benchmark",
                "source": "Measured power consumption during LLM inference",
                "specs": "24GB GDDR6X, Ada Lovelace architecture"
            },
            "nvidia_a100": {
                "description": "NVIDIA A100 GPU benchmark (data center)",
                "source": "Manufacturer specifications and research measurements",
                "specs": "40GB HBM2e, Ampere architecture, 400W TDP"
            },
            "apple_m2": {
                "description": "Apple M2 chip benchmark",
                "source": "Measured power consumption during LLM inference",
                "specs": "8-core CPU, 10-core GPU, 20W TDP"
            }
        }
    }

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for energy/alignment testing"""
    await websocket.accept()

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

            # Convert payload - accept any dict for energy/alignment testing
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
            # Determine test type and call appropriate function from standalone apps
            test_type = raw.get('test_type', 'energy')  # Default to energy for backward compatibility
            if test_type == 'alignment':
                # Create AlignmentPayload and delegate to standalone app
                payload_obj = AlignmentPayload(
                    system=raw.get('system', ''),
                    user=raw.get('user', ''),
                    model_name=raw.get('model_name', ''),
                    strategy_name=raw.get('strategy_name', 'baseline'),
                    injection_type=raw.get('injection_type', 'none'),
                    injection_params=raw.get('injection_params', {}),
                    tool_integration_method=raw.get('tool_integration_method', 'none'),
                    tool_config=raw.get('tool_config', {}),
                    temp=raw.get('temp', 0.7),
                    max_tokens=raw.get('max_tokens', 512)
                )
                current_task = asyncio.create_task(run_alignment_test(payload_obj, websocket, cancel_event))
            else:  # energy or default
                # Create EnergyPayload and delegate to standalone app
                payload_obj = EnergyPayload(
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
                    max_tokens=raw.get('max_tokens', 512)
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

@app.websocket("/ws/comparison")
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

            # Convert payload - accept any dict for energy/alignment testing
            try:
                # For flexibility, accept any payload with required fields
                required_fields = ['system', 'user', 'model_name']
                if not all(field in raw for field in required_fields):
                    await websocket.send_json({"error": f"Missing required fields: {required_fields}"})
                    continue

                # Create a dict payload that can be passed to the test function
                payload = {
                    'system': raw['system'],
                    'user': raw['user'],
                    'model_name': raw['model_name'],
                    'strategy_name': raw.get('strategy_name', 'baseline'),
                    'energy_benchmark': raw.get('energy_benchmark', 'conservative_estimate'),
                    'injection_type': raw.get('injection_type', 'none'),
                    'injection_params': raw.get('injection_params', {}),
                    'tool_integration_method': raw.get('tool_integration_method', 'none'),
                    'tool_config': raw.get('tool_config', {}),
                    'temp': raw.get('temp', 0.7),
                    'max_tokens': raw.get('max_tokens', 512)
                }
            except Exception as exc:
                await websocket.send_json({"error": f"Invalid payload: {str(exc)}"})
                continue

            if not payload['model_name']:
                await websocket.send_json({
                    "error": "No model selected. Please choose a model in the UI first.",
                    "done": True,
                })
                continue

            cancel_event = asyncio.Event()
            # Determine test type and call appropriate function from standalone apps
            test_type = raw.get('test_type', 'energy')  # Default to energy for backward compatibility
            if test_type == 'alignment':
                # Create AlignmentPayload and delegate to standalone app
                payload_obj = AlignmentPayload(
                    system=raw.get('system', ''),
                    user=raw.get('user', ''),
                    model_name=raw.get('model_name', ''),
                    strategy_name=raw.get('strategy_name', 'baseline'),
                    injection_type=raw.get('injection_type', 'none'),
                    injection_params=raw.get('injection_params', {}),
                    tool_integration_method=raw.get('tool_integration_method', 'none'),
                    tool_config=raw.get('tool_config', {}),
                    temp=raw.get('temp', 0.7),
                    max_tokens=raw.get('max_tokens', 512)
                )
                current_task = asyncio.create_task(run_alignment_test(payload_obj, websocket, cancel_event))
            else:  # energy or default
                # Create EnergyPayload and delegate to standalone app
                payload_obj = EnergyPayload(
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
                    max_tokens=raw.get('max_tokens', 512)
                )
                current_task = asyncio.create_task(run_energy_test(payload_obj, websocket, cancel_event))
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
    print("• Alignment Testing Lab: http://localhost:8001/alignment")
    print("• Model Comparison Lab: http://localhost:8001/comparison")
    print()
    print("Standalone apps also available:")
    print("• python3 app_energy.py (port 8002)")
    print("• python3 app_alignment.py (port 8003)")
    print("• python3 app_model_comparison.py (port 8000)")
    print()
    print("Features:")
    print("• Energy consumption tracking (tokens → Wh → CO2)")
    print("• Response alignment analysis (goal adherence, consistency)")
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
