"""
Shared Ollama client utilities
Provides common functions for connecting to and querying Ollama
"""

import os
import httpx
from typing import List, Dict, Any, Optional

# ---------- Configuration ----------
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "llama3.1:8b")
DEFAULT_BASE_MODEL = os.getenv("OLLAMA_DEFAULT_BASE_MODEL", "")

# Default limits (UI should query model-specific limits and allow user override)
# These are fallback values only - actual limits come from model metadata
DEFAULT_MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "1000"))
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "180.0"))


async def check_ollama_connection() -> bool:
    """Check if Ollama is running and accessible"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5.0)
            return response.status_code == 200
    except Exception:
        return False


async def list_ollama_models() -> List[str]:
    """List all available Ollama models"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
            if response.status_code == 200:
                data = response.json()
                return [model["name"] for model in data.get("models", [])]
    except Exception:
        pass
    return []


async def get_model_info(model_name: str) -> Optional[Dict[str, Any]]:
    """Get detailed information about a specific model including context length"""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                f"{OLLAMA_BASE_URL}/api/show",
                json={"name": model_name}
            )
            if response.status_code == 200:
                data = response.json()
                # Extract context length from modelfile or model info
                context_length = None
                
                # Try to get from model_info first
                if "model_info" in data:
                    for key, value in data["model_info"].items():
                        if "context" in key.lower() or "ctx" in key.lower():
                            try:
                                context_length = int(value)
                                break
                            except (ValueError, TypeError):
                                pass
                
                # Fallback: parse modelfile for num_ctx parameter
                if context_length is None and "modelfile" in data:
                    modelfile = data["modelfile"]
                    for line in modelfile.split("\n"):
                        if "num_ctx" in line.lower():
                            try:
                                # Extract number from line like "PARAMETER num_ctx 40960"
                                parts = line.split()
                                for i, part in enumerate(parts):
                                    if "num_ctx" in part.lower() and i + 1 < len(parts):
                                        context_length = int(parts[i + 1])
                                        break
                            except (ValueError, IndexError):
                                pass
                
                return {
                    "name": model_name,
                    "context_length": context_length or 40960,  # Default fallback
                    "details": data.get("details", {}),
                    "modelfile": data.get("modelfile", "")
                }
    except Exception as e:
        print(f"Error getting model info for {model_name}: {e}")
    return None


async def get_models_with_defaults() -> Dict[str, Any]:
    """Get available models with sensible defaults"""
    models = await list_ollama_models()
    
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


async def print_startup_info(app_name: str):
    """Print standardized startup information"""
    print(f"Starting {app_name}...")
    print(f"Ollama URL: {OLLAMA_BASE_URL}")
    
    if await check_ollama_connection():
        print("✓ Connected to Ollama")
        models = await list_ollama_models()
        if models:
            print(f"✓ Available models: {', '.join(models[:5])}{'...' if len(models) > 5 else ''}")
        else:
            print("⚠️  No models found. Pull models with: ollama pull <model>")
    else:
        print("✗ Could not connect to Ollama")
        print("  Make sure Ollama is running: ollama serve")
        print("  Or install it from: https://ollama.ai")
