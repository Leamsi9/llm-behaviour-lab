#!/usr/bin/env python3
"""
Shared Ollama Streaming Utilities

Extracts common streaming logic from app_energy.py and app_model_comparison.py
to eliminate ~70% code duplication.
"""

import asyncio
import json
import httpx
from typing import Dict, Any, List, Optional
from fastapi import WebSocket

from ollama_client import OLLAMA_BASE_URL, REQUEST_TIMEOUT


async def stream_ollama_generation(
    model_name: str,
    messages: List[Dict[str, str]],
    options: Dict[str, Any],
    websocket: WebSocket,
    cancel_event: asyncio.Event,
    pane_id: str = "default"
) -> Dict[str, Any]:
    """
    Shared streaming logic for Ollama chat completions.
    
    Handles:
    - HTTP streaming to Ollama API
    - Token-by-token WebSocket forwarding to client
    - Cancellation support
    - Error handling
    - Metric collection
    
    Args:
        model_name: Ollama model to use (e.g., "llama3:8b")
        messages: Conversation messages [{"role": "system|user|assistant", "content": "..."}]
        options: Ollama generation options (temperature, num_predict, etc.)
        websocket: WebSocket to stream tokens to
        cancel_event: Event to signal cancellation
        pane_id: Identifier for this generation (for multi-model support)
    
    Returns:
        {
            "full_response": str,
            "prompt_tokens": int,
            "completion_tokens": int,
            "latency": float,
            "cancelled": bool,
            "error": Optional[str],
            "prompt_eval_duration_ns": Optional[int],  # For improved energy calc
            "eval_duration_ns": Optional[int]           # For improved energy calc
        }
    """
    start_time = asyncio.get_event_loop().time()
    full_response = ""
    prompt_tokens = 0
    completion_tokens = 0
    prompt_eval_duration = None
    eval_duration = None
    
    try:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
            async with client.stream(
                "POST",
                f"{OLLAMA_BASE_URL}/api/chat",
                json={
                    "model": model_name,
                    "messages": messages,
                    "stream": True,
                    "options": options
                }
            ) as response:
                # Handle HTTP errors
                if response.status_code != 200:
                    error_text = await response.aread()
                    error_msg = f"Ollama error ({response.status_code}): {error_text.decode()}"
                    
                    await websocket.send_json({
                        "pane_id": pane_id,
                        "error": error_msg,
                        "done": True,
                    })
                    
                    return {
                        "full_response": "",
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "latency": 0,
                        "cancelled": False,
                        "error": error_msg
                    }
                
                # Stream response line by line
                async for line in response.aiter_lines():
                    # Check cancellation
                    if cancel_event.is_set():
                        break
                    
                    if not line:
                        continue
                    
                    # Parse JSON chunk
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    
                    # Stream token to client
                    if "message" in chunk and "content" in chunk["message"]:
                        text = chunk["message"]["content"]
                        full_response += text
                        
                        await websocket.send_json({
                            "pane_id": pane_id,
                            "token": text,
                            "done": False,
                        })
                    
                    # Final chunk with metrics
                    if chunk.get("done", False):
                        prompt_tokens = chunk.get("prompt_eval_count", 0)
                        completion_tokens = chunk.get("eval_count", 0)
                        
                        # These durations enable more accurate energy calculation
                        # (prompt processing vs generation have different energy profiles)
                        prompt_eval_duration = chunk.get("prompt_eval_duration")
                        eval_duration = chunk.get("eval_duration")
                        break
                
                # Always close the response stream properly
                await response.aclose()
        
        # Calculate latency
        latency = asyncio.get_event_loop().time() - start_time
        
        # Handle cancellation
        if cancel_event.is_set():
            await websocket.send_json({
                "pane_id": pane_id,
                "token": "[CANCELLED]",
                "done": True,
                "cancelled": True,
            })
            
            return {
                "full_response": full_response,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "latency": latency,
                "cancelled": True,
                "error": None,
                "prompt_eval_duration_ns": prompt_eval_duration,
                "eval_duration_ns": eval_duration
            }
        
        # Success
        return {
            "full_response": full_response,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "latency": latency,
            "cancelled": False,
            "error": None,
            "prompt_eval_duration_ns": prompt_eval_duration,
            "eval_duration_ns": eval_duration
        }
    
    except httpx.ConnectError:
        error_msg = "Cannot connect to Ollama. Make sure it's running: ollama serve"
        
        await websocket.send_json({
            "pane_id": pane_id,
            "error": error_msg,
            "done": True,
        })
        
        return {
            "full_response": "",
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "latency": 0,
            "cancelled": False,
            "error": error_msg
        }
    
    except Exception as exc:
        # Handle asyncio cancellation
        if isinstance(exc, asyncio.CancelledError):
            await websocket.send_json({
                "pane_id": pane_id,
                "token": "[CANCELLED]",
                "done": True,
                "cancelled": True,
            })
            
            return {
                "full_response": full_response,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "latency": asyncio.get_event_loop().time() - start_time,
                "cancelled": True,
                "error": None
            }
        else:
            # Unexpected error
            error_msg = f"Generation error: {str(exc)}"
            
            await websocket.send_json({
                "pane_id": pane_id,
                "error": error_msg,
                "done": True,
            })
            
            return {
                "full_response": "",
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "latency": 0,
                "cancelled": False,
                "error": error_msg
            }
