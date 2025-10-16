"""
Standalone Energy Testing Lab
Tests energy consumption and carbon footprint of LLM modifications
"""

import os
import json
import asyncio
import uvicorn
import httpx
import contextlib
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager

# Import our specialized modules
from energy_tracker import energy_tracker, estimate_energy_impact, get_available_benchmarks, ENERGY_BENCHMARKS
from prompt_injection import injection_manager, InjectionConfig, InjectionType
from tool_integration import tool_manager, ToolIntegrationConfig, ToolIntegrationMethod, ToolCall
from ollama_client import (
    check_ollama_connection,
    list_ollama_models,
    get_models_with_defaults,
    print_startup_info,
    OLLAMA_BASE_URL,
    DEFAULT_MODEL,
    REQUEST_TIMEOUT
)

@dataclass
class EnergyPayload:
    """Payload for energy testing"""
    system: str = ""
    user: str = ""
    model_name: str = DEFAULT_MODEL
    strategy_name: str = "baseline"
    energy_benchmark: str = "conservative_estimate"
    injection_type: str = "none"
    injection_params: Dict[str, Any] = field(default_factory=dict)
    tool_integration_method: str = "none"
    tool_config: Dict[str, Any] = field(default_factory=dict)
    temp: float = 0.7
    max_tokens: int = 100

# ---------- Ollama Client ----------
# (Now imported from ollama_client.py)

async def run_energy_test(payload: EnergyPayload, websocket: WebSocket, cancel_event: asyncio.Event):
    """Run energy consumption test"""
    try:
        # Prepare messages
        messages = []
        if payload.system:
            messages.append({"role": "system", "content": payload.system})
        messages.append({"role": "user", "content": payload.user})

        # Apply prompt injection if specified
        injection_result = {"injection_metadata": {}}
        if payload.injection_type != "none":
            try:
                injection_config = InjectionConfig(
                    injection_type=InjectionType(payload.injection_type),
                    **payload.injection_params
                )
                injection_result = injection_manager.apply_injection(
                    messages=messages,
                    config=injection_config
                )
                messages = injection_result["messages"]
            except Exception as e:
                try:
                    await websocket.send_json({"error": f"Injection error: {str(e)}", "done": True})
                except Exception:
                    pass
                return

        # Apply tool integration if specified
        tool_integration_result = {"integration_metadata": {}}
        if payload.tool_integration_method != "none":
            try:
                tool_config = ToolIntegrationConfig(
                    method=ToolIntegrationMethod(payload.tool_integration_method),
                    **payload.tool_config
                )
                tool_integration_result = tool_manager.apply_integration(
                    messages=messages,
                    config=tool_config
                )
                messages = tool_integration_result["messages"]
            except Exception as e:
                try:
                    await websocket.send_json({"error": f"Tool integration error: {str(e)}", "done": True})
                except Exception:
                    pass
                return

        # Record start time
        start_time = asyncio.get_event_loop().time()
        
        full_response = ""
        prompt_tokens = 0
        completion_tokens = 0

        # Make request to Ollama
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
            async with client.stream(
                "POST",
                f"{OLLAMA_BASE_URL}/api/chat",
                json={
                    "model": payload.model_name,
                    "messages": messages,
                    "stream": True,
                    "options": {
                        "temperature": payload.temp,
                        "num_predict": payload.max_tokens
                    }
                }
            ) as response:
                if response.status_code != 200:
                    error_text = await response.aread()
                    await websocket.send_json({
                        "error": f"Ollama error: {error_text.decode()}",
                        "done": True,
                    })
                    return

                async for line in response.aiter_lines():
                    if cancel_event.is_set():
                        break

                    if not line:
                        continue

                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    if "message" in chunk and "content" in chunk["message"]:
                        text = chunk["message"]["content"]
                        full_response += text

                        await websocket.send_json({
                            "token": text,
                            "model": payload.model_name,
                            "strategy": payload.strategy_name,
                            "done": False,
                        })

                    if chunk.get("done", False):
                        prompt_tokens = chunk.get("prompt_eval_count", 0)
                        completion_tokens = chunk.get("eval_count", 0)
                        break
                
                # Always close the response stream properly
                await response.aclose()

        if cancel_event.is_set():
            await websocket.send_json({
                "token": "[CANCELLED]",
                "model": payload.model_name,
                "strategy": payload.strategy_name,
                "done": True,
                "cancelled": True,
            })
            return

        # Calculate timing
        end_time = asyncio.get_event_loop().time()
        latency = end_time - start_time

        # Record energy consumption
        energy_reading = energy_tracker.record_usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_seconds=latency,
            model_name=payload.model_name,
            strategy_name=payload.strategy_name
        )

        # Send comprehensive results
        await websocket.send_json({
            "token": "[DONE]",
            "model": payload.model_name,
            "strategy": payload.strategy_name,
            "done": True,

            # Basic metrics
            "basic_metrics": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
                "latency": latency,
                "tokens_per_second": completion_tokens / latency if latency > 0 else 0,
            },

            # Energy metrics
            "energy_metrics": {
                "benchmark_used": energy_reading.benchmark_used,
                "watt_hours_consumed": energy_reading.watt_hours_consumed,
                "carbon_grams_co2": energy_reading.carbon_grams_co2,
                "energy_efficiency_score": energy_reading.watt_hours_consumed / max(energy_reading.total_tokens / 1000, 0.001),
            },

            # Modification tracking
            "modification_info": {
                "injection_applied": injection_result.get("injection_metadata", {}),
                "tool_integration_applied": tool_integration_result.get("integration_metadata", {}),
                "original_prompt_length": len(payload.system) + len(payload.user),
                "final_prompt_length": len(payload.system) + len(payload.user),  # Simplified for standalone
                "modification_overhead": 0,  # Simplified for standalone
            },

            # Session summary
            "session_summary": energy_tracker.get_session_summary(),
        })

    except httpx.ConnectError:
        await websocket.send_json({
            "error": "Cannot connect to Ollama. Make sure it's running: ollama serve",
            "done": True,
        })
    except Exception as exc:
        if isinstance(exc, asyncio.CancelledError):
            await websocket.send_json({
                "token": "[CANCELLED]",
                "model": payload.model_name,
                "strategy": payload.strategy_name,
                "done": True,
                "cancelled": True,
            })
        else:
            await websocket.send_json({
                "error": f"Test error: {str(exc)}",
                "done": True,
            })

# ---------- FastAPI App ----------

@asynccontextmanager
async def lifespan(_: FastAPI):
    """Check Ollama connection on startup"""
    await print_startup_info("Standalone Energy Testing Lab")
    yield
    print("Shutting down Standalone Energy Testing Lab...")

app = FastAPI(
    title="Standalone Energy Testing Lab",
    description="Test energy consumption of LLM modifications",
    version="0.1.0",
    lifespan=lifespan
)
app.mount("/static", StaticFiles(directory="static"), name="static")

# ---------- Routes ----------

@app.get("/")
async def root():
    """Redirect to energy testing UI"""
    return await energy_ui()

@app.get("/energy")
async def energy_ui():
    """Serve the standalone energy testing UI"""
    try:
        with open("static/ui_energy.html", "r") as f:
            content = f.read()
        return HTMLResponse(content)
    except FileNotFoundError:
        return HTMLResponse("<h1>Energy UI not found. Please check static/ui_energy.html</h1>")

@app.get("/api/models")
async def get_available_models():
    """List available Ollama models with defaults"""
    return await get_models_with_defaults()

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
    """WebSocket endpoint for energy testing"""
    await websocket.accept()

    current_task: Optional[asyncio.Task] = None
    cancel_event: Optional[asyncio.Event] = None

    async def acknowledge_cancel():
        await websocket.send_json({
            "token": "[CANCELLED]",
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

            # Handle control commands
            if isinstance(raw, dict) and raw.get("command") == "cancel":
                if cancel_event:
                    cancel_event.set()
                if current_task and not current_task.done():
                    current_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError, Exception):
                        await current_task
                else:
                    await acknowledge_cancel()
                continue

            # Convert payload and run test
            try:
                payload = EnergyPayload(**raw)
            except Exception as exc:
                await websocket.send_json({"error": f"Invalid payload: {str(exc)}"})
                continue

            if not payload.model_name:
                await websocket.send_json({
                    "error": "No model selected. Please choose a model.",
                    "done": True,
                })
                continue

            # Cancel any in-flight test
            if cancel_event:
                cancel_event.set()
            if current_task and not current_task.done():
                current_task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await current_task

            cancel_event = asyncio.Event()
            current_task = asyncio.create_task(run_energy_test(payload, websocket, cancel_event))
            current_task.add_done_callback(reset_task)

    except WebSocketDisconnect:
        print("Energy client disconnected")
    except Exception as exc:
        print(f"Energy WebSocket error: {str(exc)}")
    finally:
        if cancel_event:
            cancel_event.set()
        if current_task and not current_task.done():
            current_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await current_task
        with contextlib.suppress(Exception):
            await websocket.close()

# ---------- Run the Application ----------

if __name__ == "__main__":
    print("=" * 60)
    print("Standalone Energy Testing Lab")
    print("=" * 60)
    print()
    print("This lab focuses on:")
    print("• Energy consumption tracking (tokens → Wh → CO2)")
    print("• Prompt injection effects on energy costs")
    print("• Tool integration impact assessment")
    print("• Benchmark comparison and custom benchmarks")
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
        "app_energy:app",
        host="0.0.0.0",
        port=8002,  # Different port from main lab
        reload=True,
        log_level="info"
    )
