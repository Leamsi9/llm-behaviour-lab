"""
Groq Cloud API Client
Provides functions for connecting to and querying Groq's cloud LLM API.

This module enables cloud-based model inference as an alternative to local Ollama models.
RAPL power monitoring is not available for cloud models - energy estimates are used instead.
"""

import os
import httpx
from pathlib import Path
from typing import List, Dict, Any, Optional, AsyncGenerator

# ---------- Load .env file ----------
# Must be done before reading any environment variables
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass  # dotenv not installed, rely on system environment variables

# ---------- Configuration ----------
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_API_BASE_URL = os.getenv("GROQ_API_BASE_URL", "https://api.groq.com/openai/v1").strip()
GROQ_DEFAULT_MODEL = os.getenv("GROQ_DEFAULT_MODEL", "llama-3.1-70b-versatile").strip()

# Request timeout for Groq API calls
GROQ_REQUEST_TIMEOUT = float(os.getenv("GROQ_REQUEST_TIMEOUT", "180.0"))

# Debug: Log configuration status at module load
if GROQ_API_KEY:
    key_preview = f"{GROQ_API_KEY[:8]}..." if len(GROQ_API_KEY) > 8 else "(short key)"
    if GROQ_API_KEY.startswith("gsk_"):
        print(f"✓ GROQ_API_KEY loaded from .env: {key_preview}")
    else:
        print(f"⚠️  GROQ_API_KEY loaded but doesn't start with 'gsk_': {key_preview}")
        print("   Groq API keys should look like: gsk_xxxxxxxxxxxx")
else:
    print("ℹ️  GROQ_API_KEY not set. Cloud models will not be available.")


def is_groq_configured() -> bool:
    """Check if Groq API is configured with an API key."""
    return bool(GROQ_API_KEY)


async def check_groq_connection() -> bool:
    """Check if Groq API is accessible and API key is valid.
    
    Returns True if we can successfully authenticate with Groq API.
    Returns False if API key is missing or invalid.
    """
    if not GROQ_API_KEY:
        return False
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                f"{GROQ_API_BASE_URL}/models",
                headers={
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type": "application/json"
                }
            )
            return response.status_code == 200
    except Exception as e:
        print(f"Groq connection check failed: {e}")
        return False


async def list_groq_models() -> List[str]:
    """List available models from Groq API.
    
    Returns a list of model IDs available for inference.
    Returns empty list if API key is missing or request fails.
    """
    if not GROQ_API_KEY:
        return []
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                f"{GROQ_API_BASE_URL}/models",
                headers={
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type": "application/json"
                }
            )
            if response.status_code == 200:
                data = response.json()
                # Groq uses OpenAI-style response: { "data": [{ "id": "model-id", ... }, ...] }
                models = data.get("data", [])
                return [model.get("id", "") for model in models if model.get("id")]
    except Exception as e:
        print(f"Error listing Groq models: {e}")
    
    return []


async def get_groq_models_with_defaults() -> Dict[str, Any]:
    """Get available Groq models with sensible defaults.
    
    Returns a dictionary with:
    - models: list of available model IDs
    - current: dict with base and instruct defaults
    - provider: "cloud"
    """
    models = await list_groq_models()
    
    # Use GROQ_DEFAULT_MODEL or first available model
    instruct_default = GROQ_DEFAULT_MODEL if GROQ_DEFAULT_MODEL in models else (
        models[0] if models else GROQ_DEFAULT_MODEL
    )
    base_default = models[1] if len(models) > 1 else instruct_default
    
    return {
        "models": models,
        "current": {
            "base": base_default,
            "instruct": instruct_default
        },
        "provider": "cloud"
    }


async def stream_groq_chat(
    model: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.7,
    max_tokens: int = 1000,
    **kwargs
) -> AsyncGenerator[Dict[str, Any], None]:
    """Stream a chat completion from Groq API.
    
    Yields chunks with structure:
    - {"token": str, "done": False} for content tokens
    - {"done": True, "usage": {...}} for completion
    - {"error": str} on error
    
    Args:
        model: The model ID to use (e.g., "llama-3.1-70b-versatile")
        messages: List of message dicts with "role" and "content"
        temperature: Sampling temperature (0.0-2.0)
        max_tokens: Maximum tokens to generate
        **kwargs: Additional parameters passed to the API
    """
    if not GROQ_API_KEY:
        yield {"error": "GROQ_API_KEY not configured. Please set it in your .env file."}
        return
    
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    
    # Add optional parameters
    if "seed" in kwargs and kwargs["seed"] is not None:
        payload["seed"] = kwargs["seed"]
    
    try:
        async with httpx.AsyncClient(timeout=GROQ_REQUEST_TIMEOUT) as client:
            async with client.stream(
                "POST",
                f"{GROQ_API_BASE_URL}/chat/completions",
                headers=headers,
                json=payload
            ) as response:
                if response.status_code != 200:
                    error_text = await response.aread()
                    yield {"error": f"Groq API error ({response.status_code}): {error_text.decode()}"}
                    return
                
                total_content = ""
                prompt_tokens = 0
                completion_tokens = 0
                
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    
                    # SSE format: lines start with "data: "
                    if line.startswith("data: "):
                        data_str = line[6:]  # Remove "data: " prefix
                        
                        if data_str.strip() == "[DONE]":
                            break
                        
                        try:
                            import json
                            chunk = json.loads(data_str)
                            
                            # Extract content from OpenAI-style response
                            choices = chunk.get("choices", [])
                            if choices:
                                delta = choices[0].get("delta", {})
                                content = delta.get("content", "")
                                
                                if content:
                                    total_content += content
                                    yield {"token": content, "done": False}
                            
                            # Check for usage info in final chunk
                            usage = chunk.get("usage")
                            if usage:
                                prompt_tokens = usage.get("prompt_tokens", 0)
                                completion_tokens = usage.get("completion_tokens", 0)
                        
                        except json.JSONDecodeError:
                            continue
                
                # Final completion message
                yield {
                    "done": True,
                    "content": total_content,
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens
                    }
                }
    
    except httpx.ConnectError:
        yield {"error": "Cannot connect to Groq API. Check your internet connection."}
    except httpx.TimeoutException:
        yield {"error": "Groq API request timed out."}
    except Exception as e:
        yield {"error": f"Groq API error: {str(e)}"}


async def groq_chat(
    model: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.7,
    max_tokens: int = 1000,
    **kwargs
) -> Dict[str, Any]:
    """Non-streaming chat completion from Groq API.
    
    Returns a dictionary with:
    - content: The generated text
    - usage: Token usage stats
    - error: Error message if failed (instead of content)
    """
    if not GROQ_API_KEY:
        return {"error": "GROQ_API_KEY not configured. Please set it in your .env file."}
    
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    
    if "seed" in kwargs and kwargs["seed"] is not None:
        payload["seed"] = kwargs["seed"]
    
    try:
        async with httpx.AsyncClient(timeout=GROQ_REQUEST_TIMEOUT) as client:
            response = await client.post(
                f"{GROQ_API_BASE_URL}/chat/completions",
                headers=headers,
                json=payload
            )
            
            if response.status_code != 200:
                return {"error": f"Groq API error ({response.status_code}): {response.text}"}
            
            data = response.json()
            choices = data.get("choices", [])
            usage = data.get("usage", {})
            
            if choices:
                content = choices[0].get("message", {}).get("content", "")
                return {
                    "content": content,
                    "usage": {
                        "prompt_tokens": usage.get("prompt_tokens", 0),
                        "completion_tokens": usage.get("completion_tokens", 0),
                        "total_tokens": usage.get("total_tokens", 0)
                    }
                }
            else:
                return {"error": "No response from Groq API"}
    
    except httpx.ConnectError:
        return {"error": "Cannot connect to Groq API. Check your internet connection."}
    except httpx.TimeoutException:
        return {"error": "Groq API request timed out."}
    except Exception as e:
        return {"error": f"Groq API error: {str(e)}"}


async def print_groq_startup_info():
    """Print Groq API connection status at startup."""
    print(f"Groq API Base URL: {GROQ_API_BASE_URL}")
    
    if not GROQ_API_KEY:
        print("⚠️  GROQ_API_KEY not set. Cloud models will not be available.")
        print("   Set GROQ_API_KEY in your .env file to enable Groq cloud models.")
        return
    
    if await check_groq_connection():
        print("✓ Connected to Groq API")
        models = await list_groq_models()
        if models:
            print(f"✓ Available Groq models: {', '.join(models[:5])}{'...' if len(models) > 5 else ''}")
        else:
            print("⚠️  No Groq models found.")
    else:
        print("✗ Could not connect to Groq API")
        print("  Verify your GROQ_API_KEY is valid.")
