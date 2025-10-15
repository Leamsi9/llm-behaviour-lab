#!/usr/bin/env python3
"""
LLM Harness Lab - Compare pre-RLHF and post-RLHF models using Ollama
"""
import os
import json
import asyncio
import uvicorn
import httpx
import contextlib
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager

# ---------- Configuration ----------
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
DEFAULT_MODEL = os.getenv("OLLAMA_DEFAULT_MODEL", "")
DEFAULT_BASE_MODEL = os.getenv("OLLAMA_DEFAULT_BASE_MODEL", "")

# Alternative models you can use:
# - mistral:7b / mistral:7b-base
# - llama3:8b / llama3:8b-base
# - gemma2:9b / gemma2:9b-base

# ---------- Data Models ----------

@dataclass
class Payload:
    system: str
    user: str
    template: str
    use_base_model: bool = False
    model_name: Optional[str] = None  # Allow custom model selection
    temp: float = 0.7
    top_p: float = 0.9
    repeat_penalty: float = 1.1
    max_tokens: int = 1024
    stop: List[str] = field(default_factory=lambda: [])

# ---------- Generation Helpers ----------

async def run_generation(
    payload: Payload,
    websocket: WebSocket,
    cancel_event: asyncio.Event,
):
    """Stream a response from Ollama, respecting cancellation requests."""

    model_name = payload.model_name or ""
    model_key = "base" if payload.use_base_model else "instruct"

    messages = []
    if payload.system.strip():
        messages.append({"role": "system", "content": payload.system})
    messages.append({"role": "user", "content": payload.user})

    full_response = ""
    prompt_tokens = 0
    completion_tokens = 0
    start_time = asyncio.get_event_loop().time()

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream(
                "POST",
                f"{OLLAMA_BASE_URL}/api/chat",
                json={
                    "model": model_name,
                    "messages": messages,
                    "stream": True,
                    "options": {
                        "temperature": payload.temp,
                        "top_p": payload.top_p,
                        "repeat_penalty": payload.repeat_penalty,
                        "num_predict": payload.max_tokens,
                    },
                },
            ) as response:
                if response.status_code != 200:
                    error_text = await response.aread()
                    await websocket.send_json({
                        "error": f"Ollama error: {error_text.decode()}",
                        "done": True,
                    })
                    return

                cancelled = False

                async for line in response.aiter_lines():
                    if cancel_event.is_set():
                        cancelled = True
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
                            "model": model_key,
                            "done": False,
                        })

                    if chunk.get("done", False):
                        prompt_tokens = chunk.get("prompt_eval_count", 0)
                        completion_tokens = chunk.get("eval_count", 0)
                        break

                if not cancel_event.is_set():
                    await response.aclose()

                if cancel_event.is_set():
                    cancelled = True

        if cancel_event.is_set() or cancelled:
            await websocket.send_json({
                "token": "[CANCELLED]",
                "model": model_key,
                "done": True,
                "cancelled": True,
            })
            return

        end_time = asyncio.get_event_loop().time()
        latency = end_time - start_time

        await websocket.send_json({
            "token": "[DONE]",
            "model": model_key,
            "done": True,
            "metrics": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
                "latency": latency,
                "tokens_per_second": completion_tokens / latency if latency > 0 else 0,
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
                "model": model_key,
                "done": True,
                "cancelled": True,
            })
        else:
            await websocket.send_json({
                "error": f"Inference error: {str(exc)}",
                "done": True,
            })

# ---------- Ollama Client ----------

async def check_ollama_connection():
    """Check if Ollama is running"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5.0)
            return response.status_code == 200
    except Exception as e:
        print(f"Failed to connect to Ollama: {e}")
        return False

async def list_ollama_models():
    """List available Ollama models"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
            if response.status_code == 200:
                data = response.json()
                return [model["name"] for model in data.get("models", [])]
    except Exception as e:
        print(f"Error listing models: {e}")
    return []

@asynccontextmanager
async def lifespan(_: FastAPI):
    """Check Ollama connection on startup"""
    print("Starting application...")
    print(f"Ollama URL: {OLLAMA_BASE_URL}")
    
    if await check_ollama_connection():
        print("✓ Connected to Ollama")
        models = await list_ollama_models()
        if models:
            print(f"✓ Available models: {', '.join(models)}")
        else:
            print("⚠️  No models found. Pull models with: ollama pull <model>")
    else:
        print("✗ Could not connect to Ollama")
        print("  Make sure Ollama is running: ollama serve")
        print("  Or install it from: https://ollama.ai")
    
    yield
    
    print("Shutting down...")

app = FastAPI(
    title="LLM Harness Lab (Ollama)",
    description="Compare Base vs RLHF Models using Ollama",
    version="0.2.0",
    lifespan=lifespan
)
app.mount("/static", StaticFiles(directory="static"), name="static")

# ---------- Routes ----------

@app.get("/")
async def index():
    """Serve the main UI"""
    try:
        return HTMLResponse(open("static/ui.html").read())
    except FileNotFoundError:
        return HTMLResponse("<h1>UI not found. Please build the frontend.</h1>")

@app.get("/api/models")
async def get_available_models():
    """List available Ollama models"""
    models = await list_ollama_models()

    # Choose sensible defaults based on available models
    base_default = DEFAULT_BASE_MODEL or (models[0] if models else "")
    instruct_default = DEFAULT_MODEL or (
        models[1] if len(models) > 1 else models[0] if models else ""
    )

    return {
        "models": models,
        "current": {
            "base": base_default,
            "instruct": instruct_default
        }
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

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for model inference with cancellation support"""
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

            # Convert payload
            try:
                payload = Payload(**raw)
            except Exception as exc:
                await websocket.send_json({"error": f"Invalid payload: {str(exc)}"})
                continue

            if not payload.model_name:
                await websocket.send_json({
                    "error": "No model selected. Please choose a model in the UI first.",
                    "done": True,
                })
                continue

            # Cancel any in-flight generation before starting a new one
            if cancel_event:
                cancel_event.set()
            if current_task and not current_task.done():
                current_task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await current_task

            cancel_event = asyncio.Event()
            current_model_key = "base" if payload.use_base_model else "instruct"
            current_task = asyncio.create_task(run_generation(payload, websocket, cancel_event))
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

# ---------- Run the Application ----------

if __name__ == "__main__":
    print("=" * 50)
    print("LLM Harness Lab - Ollama Edition")
    print("=" * 50)
    print()
    print("Make sure Ollama is running:")
    print("  ollama serve")
    print()
    print("Pull models:")
    print(f"  ollama pull {DEFAULT_MODEL}")
    print(f"  ollama pull {DEFAULT_BASE_MODEL}")
    print()
    print("=" * 50)
    print()
    
    # Start the server
    uvicorn.run(
        "app_ollama:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
