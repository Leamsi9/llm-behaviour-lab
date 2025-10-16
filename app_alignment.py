"""
Standalone Alignment Testing Lab
Tests response alignment and quality across LLM modifications
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
    REQUEST_TIMEOUT
)

@dataclass
class AlignmentPayload:
    """Payload for alignment testing"""
    system: str = ""
    user: str = ""
    model_name: str = DEFAULT_MODEL
    strategy_name: str = "baseline"
    injection_type: str = "none"
    injection_params: Dict[str, Any] = field(default_factory=dict)
    tool_integration_method: str = "none"
    tool_config: Dict[str, Any] = field(default_factory=dict)
    temp: float = 0.7
    max_tokens: int = 100

# ---------- Ollama Client ----------
# (Now imported from ollama_client.py)

async def run_alignment_test(payload: AlignmentPayload, websocket: WebSocket, cancel_event: asyncio.Event):
    """Run alignment quality test"""
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
                await websocket.send_json({"error": f"Injection error: {str(e)}", "done": True})
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
                await websocket.send_json({"error": f"Tool integration error: {str(e)}", "done": True})
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

        # Analyze alignment
        alignment_score = alignment_analyzer.analyze_response(
            response_text=full_response,
            prompt_text=payload.user,
            model_name=payload.model_name,
            strategy_name=payload.strategy_name,
            context={
                "injection_metadata": injection_result.get("injection_metadata", {}),
                "tool_integration_metadata": tool_integration_result.get("integration_metadata", {}),
            }
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

            # Alignment metrics (convert datetime to string)
            "alignment_metrics": {
                "timestamp": alignment_score.timestamp.isoformat(),
                "goal_adherence": alignment_score.goal_adherence,
                "consistency": alignment_score.consistency,
                "relevance": alignment_score.relevance,
                "factual_accuracy": alignment_score.factual_accuracy,
                "hallucination_score": alignment_score.hallucination_score,
                "coherence_score": alignment_score.coherence_score,
                "completeness_score": alignment_score.completeness_score,
                "injection_bleed": alignment_score.injection_bleed,
                "tool_interference": alignment_score.tool_interference,
                "off_topic_penalty": alignment_score.off_topic_penalty,
                "analysis_notes": alignment_score.analysis_notes,
                "detected_issues": alignment_score.detected_issues,
            },

            # Modification tracking
            "modification_info": {
                "injection_applied": injection_result.get("injection_metadata", {}),
                "tool_integration_applied": tool_integration_result.get("integration_metadata", {}),
                "original_prompt_length": len(payload.system) + len(payload.user),
                "final_prompt_length": len(payload.system) + len(payload.user),  # Simplified for standalone
                "modification_overhead": 0,  # Simplified for standalone
            },
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
    await print_startup_info("Standalone Alignment Testing Lab")
    yield
    print("Shutting down Standalone Alignment Testing Lab...")

app = FastAPI(
    title="Standalone Alignment Testing Lab",
    description="Test response alignment and quality across LLM modifications",
    version="0.1.0",
    lifespan=lifespan
)
app.mount("/static", StaticFiles(directory="static"), name="static")

# ---------- Routes ----------

@app.get("/")
async def root():
    """Redirect to alignment testing UI"""
    return await alignment_ui()

@app.get("/alignment")
async def alignment_ui():
    """Serve the standalone alignment testing UI"""
    try:
        with open("static/ui_alignment.html", "r") as f:
            content = f.read()
        return HTMLResponse(content)
    except FileNotFoundError:
        return HTMLResponse("<h1>Alignment UI not found. Please check static/ui_alignment.html</h1>")

@app.get("/api/models")
async def get_available_models():
    """List available Ollama models with defaults"""
    return await get_models_with_defaults()

@app.get("/api/injection-methods")
async def get_injection_methods():
    """Get available prompt injection methods"""
    return {"methods": injection_manager.get_available_injections()}

@app.get("/api/tool-methods")
async def get_tool_methods():
    """Get available tool integration methods"""
    return {"methods": tool_manager.get_available_methods()}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for alignment testing"""
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
                payload = AlignmentPayload(**raw)
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
            current_task = asyncio.create_task(run_alignment_test(payload, websocket, cancel_event))
            current_task.add_done_callback(reset_task)

    except WebSocketDisconnect:
        print("Alignment client disconnected")
    except Exception as exc:
        print(f"Alignment WebSocket error: {str(exc)}")
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
    print("Standalone Alignment Testing Lab")
    print("=" * 60)
    print()
    print("This lab focuses on:")
    print("• Response alignment analysis (goal adherence, consistency)")
    print("• Quality assessment (factual accuracy, coherence)")
    print("• Risk evaluation (hallucination, injection bleed)")
    print("• Tool integration impact on response quality")
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
        "app_alignment:app",
        host="0.0.0.0",
        port=8003,  # Different port from main lab
        reload=True,
        log_level="info"
    )
