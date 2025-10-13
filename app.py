#!/usr/bin/env python3
"""
LLM Harness Lab - Compare pre-RLHF and post-RLHF models
"""
import os
import json
import asyncio
import uvicorn
import pathlib
import jinja2
import llama_cpp
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager

# ---------- Configuration ----------
MODELS_DIR = pathlib.Path("models")
DEFAULT_MODEL = "Qwen2.5-14B-Instruct-Q4_K_M.gguf"
DEFAULT_BASE_MODEL = "Qwen2.5-14B-Q4_K_M.gguf"
GPU_LAYERS = -1 if os.getenv("CUDA") or os.getenv("METAL") else 0
N_CTX = 4096

# Model type for chat template
MODEL_TYPE = "qwen"  # Options: "llama", "qwen"

# Model instances
models: Dict[str, llama_cpp.Llama] = {}

# ---------- Models Management ----------

def load_model(model_path: str) -> Optional[llama_cpp.Llama]:
    """Load a model into memory"""
    try:
        if not os.path.exists(model_path):
            print(f"Model file not found: {model_path}")
            return None
            
        print(f"Loading model: {os.path.basename(model_path)}")
        model = llama_cpp.Llama(
            model_path=model_path,
            n_ctx=N_CTX,
            n_gpu_layers=GPU_LAYERS,
            verbose=False,  # Disable verbose to avoid hanging on vocab errors
        )
        print(f"✓ Model loaded successfully: {os.path.basename(model_path)}")
        return model
    except Exception as e:
        print(f"✗ Error loading model {model_path}: {str(e)}")
        print(f"  This model may be corrupted. Try re-downloading it.")
        return None

@asynccontextmanager
async def lifespan(_: FastAPI):
    """Load models on startup and clean up on shutdown"""
    global models
    
    print("Starting application...")
    print(f"Looking for models in: {os.path.abspath(MODELS_DIR)}")
    
    # Create models directory if it doesn't exist
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # Load default models
    print("Attempting to load models...")
    
    # Try to load base model first
    print(f"Loading base model: {DEFAULT_BASE_MODEL}")
    base_path = MODELS_DIR / DEFAULT_BASE_MODEL
    base_model = load_model(str(base_path))
    if base_model is not None:
        models["base"] = base_model
        print(f"✓ Successfully loaded base model")
    else:
        print(f"✗ Could not load base model from {base_path}")
    
    # Try to load instruct model
    print(f"Loading instruct model: {DEFAULT_MODEL}")
    instruct_path = MODELS_DIR / DEFAULT_MODEL
    instruct_model = load_model(str(instruct_path))
    if instruct_model is not None:
        models["instruct"] = instruct_model
        print(f"✓ Successfully loaded instruct model")
    else:
        print(f"✗ Could not load instruct model from {instruct_path}")
    
    if not models:
        print("⚠️  Warning: No models were loaded.")
        print("   The application will start but won't be able to process inference requests.")
        print("   You can still access the UI at http://localhost:8000")
    else:
        print(f"✓ Successfully loaded {len(models)} model(s): {list(models.keys())}")
    
    yield  # This is where the application runs
    
    # Cleanup
    print("Cleaning up models...")
    for name, model in models.items():
        print(f"Unloading {name} model...")
        del model
    models.clear()

app = FastAPI(
    title="LLM Harness Lab",
    description="Compare Base vs RLHF Models",
    version="0.1.0",
    lifespan=lifespan
)
app.mount("/static", StaticFiles(directory="static"), name="static")

# ---------- Data Models ----------

@dataclass
class Payload:
    system: str
    user: str
    template: str
    use_base_model: bool = False
    temp: float = 0.7
    top_p: float = 0.9
    repeat_penalty: float = 1.1
    max_tokens: int = 1024
    stop: List[str] = field(default_factory=lambda: ["<|im_end|>"])

# ---------- Routes ----------

@app.get("/")
async def index():
    """Serve the main UI"""
    try:
        return HTMLResponse(open("static/ui.html").read())
    except FileNotFoundError:
        return HTMLResponse("<h1>UI not found. Please build the frontend.</h1>")

@app.get("/api/models")
async def list_models():
    """List available models"""
    available_models = []
    if os.path.exists(MODELS_DIR):
        for f in os.listdir(MODELS_DIR):
            if f.endswith(('.gguf', '.bin', '.safetensors')):
                available_models.append({
                    "name": f,
                    "is_loaded": f in [DEFAULT_MODEL, DEFAULT_BASE_MODEL]
                })
    return {"models": available_models}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for model inference with progress streaming"""
    await websocket.accept()
    
    try:
        while True:
            # Receive and parse the request
            try:
                raw = await websocket.receive_json()
                payload = Payload(**raw)
            except json.JSONDecodeError:
                await websocket.send_json({"error": "Invalid JSON"})
                continue
            except Exception as e:
                await websocket.send_json({"error": f"Invalid payload: {str(e)}"})
                continue
                
            # Select model based on payload
            model_key = "base" if payload.use_base_model else "instruct"
            if model_key not in models:
                await websocket.send_json({
                    "error": f"Model not loaded: {model_key}",
                    "available_models": list(models.keys())
                })
                continue
                
            llm = models[model_key]
            
            # Prepare the prompt using the appropriate chat template
            try:
                messages = []
                if payload.system.strip():
                    messages.append({"role": "system", "content": payload.system})
                messages.append({"role": "user", "content": payload.user})
                
                # Format messages according to model type
                if MODEL_TYPE == "qwen":
                    # Qwen 2.5 chat template
                    prompt = ""
                    if payload.system.strip():
                        prompt += f"<|im_start|>system\n{payload.system}<|im_end|>\n"
                    prompt += f"<|im_start|>user\n{payload.user}<|im_end|>\n"
                    prompt += "<|im_start|>assistant\n"
                else:
                    # Llama 3 chat template
                    prompt = ""
                    for message in messages:
                        if message["role"] == "system":
                            prompt += f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{message['content']}<|eot_id|>"
                        elif message["role"] == "user":
                            prompt += f"<|start_header_id|>user<|end_header_id|>\n\n{message['content']}<|eot_id|>"
                        elif message["role"] == "assistant":
                            prompt += f"<|start_header_id|>assistant<|end_header_id|>\n\n{message['content']}<|eot_id|>"
                    # Add the assistant's turn
                    prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n"
                
            except Exception as e:
                await websocket.send_json({"error": f"Prompt formatting error: {str(e)}"})
                continue
            
            # Stream the response
            try:
                full_response = ""
                start_time = asyncio.get_event_loop().time()
                
                for chunk in llm.create_completion(
                    prompt=prompt,
                    max_tokens=payload.max_tokens,
                    temperature=payload.temp,
                    top_p=payload.top_p,
                    repeat_penalty=payload.repeat_penalty,
                    stop=payload.stop,
                    stream=True,
                    echo=False,
                ):
                    if "choices" in chunk and len(chunk["choices"]) > 0:
                        text = chunk["choices"][0].get("text", "")
                        full_response += text
                        await websocket.send_json({
                            "token": text,
                            "model": model_key,
                            "done": False
                        })
                
                # Calculate metrics
                end_time = asyncio.get_event_loop().time()
                # Use the model's tokenizer properly - pass bytes
                try:
                    prompt_tokens = len(llm.tokenize(prompt.encode('utf-8'), add_bos=False))
                    completion_tokens = len(llm.tokenize(full_response.encode('utf-8'), add_bos=False))
                except:
                    # Fallback to estimation if tokenization fails
                    prompt_tokens = len(prompt.split())
                    completion_tokens = len(full_response.split())
                
                await websocket.send_json({
                    "token": "[DONE]",
                    "model": model_key,
                    "done": True,
                    "metrics": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens,
                        "latency": end_time - start_time,
                        "tokens_per_second": completion_tokens / (end_time - start_time) if (end_time - start_time) > 0 else 0
                    }
                })
                
            except Exception as e:
                await websocket.send_json({
                    "error": f"Inference error: {str(e)}",
                    "done": True
                })
                
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"WebSocket error: {str(e)}")
    finally:
        await websocket.close()
@app.get("/api/health")
async def health_check():
    return {"status": "ok", "websocket": True}

# ---------- Run the Application ----------

if __name__ == "__main__":
    # Create models directory if it doesn't exist
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # Start the server
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )