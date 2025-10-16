"""
Shared Ollama client utilities
Provides common functions for connecting to and querying Ollama
"""

import os
import httpx
from typing import List, Dict, Any

# ---------- Configuration ----------
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "llama3.1:8b")
DEFAULT_BASE_MODEL = os.getenv("OLLAMA_DEFAULT_BASE_MODEL", "")

# Stability limits
MAX_INPUT_LENGTH = int(os.getenv("MAX_INPUT_LENGTH", "8000"))
MAX_CONTEXT_TOKENS = int(os.getenv("MAX_CONTEXT_TOKENS", "4096"))
MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "2048"))
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
