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
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager

# Import our specialized modules
from energy_tracker import (
    energy_tracker,
    estimate_energy_impact,
    get_available_benchmarks,
    ENERGY_BENCHMARKS,
    get_hf_benchmarks,
    get_benchmark_sources,
)
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
    REQUEST_TIMEOUT,
    DEFAULT_MAX_OUTPUT_TOKENS
)

# Import token tracking utilities (critical fix)
from token_utils import track_middleware_tokens, create_token_breakdown

# Import real-time power monitoring
try:
    from power_monitor import power_monitor
    POWER_MONITORING_AVAILABLE = power_monitor.available
except ImportError:
    POWER_MONITORING_AVAILABLE = False
    power_monitor = None

# Scaphandre per-process energy (optional)
SCAPHANDRE_URL = os.getenv("SCAPHANDRE_URL", "").strip()
SCAPHANDRE_OLLAMA_MATCH = os.getenv("SCAPHANDRE_OLLAMA_MATCH", "ollama")
# Autostart behavior: by default we try to start/use Scaphandre when live power is enabled.
# Set SCAPHANDRE_AUTOSTART=0/false/no to opt out and rely purely on package-level RAPL.
SCAPHANDRE_AUTOSTART = os.getenv("SCAPHANDRE_AUTOSTART", "1").strip().lower()
SCAPHANDRE_CMD = os.getenv("SCAPHANDRE_CMD", "scaphandre prometheus --bind :8080").strip()
SCAPHANDRE_DEFAULT_URL = os.getenv("SCAPHANDRE_DEFAULT_URL", "http://127.0.0.1:8080/metrics").strip()

# Dynamic model context length cache (populated from ollama_client.get_model_info)
MODEL_CONTEXT_CACHE: Dict[str, int] = {}


async def ensure_scaphandre_ready(websocket: WebSocket) -> bool:
    """Best-effort helper to ensure a Scaphandre Prometheus exporter is running.

    Returns True if we have a usable SCAPHANDRE_URL after this call, False otherwise.
    Respects SCAPHANDRE_AUTOSTART so users can opt out of automatic startup.
    """
    global SCAPHANDRE_URL

    # Respect explicit opt-out
    if SCAPHANDRE_AUTOSTART in ("0", "false", "no"):
        return bool(SCAPHANDRE_URL)

    url = SCAPHANDRE_URL or SCAPHANDRE_DEFAULT_URL

    # Quick probe: is there already an exporter listening?
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            resp = await client.get(url)
            if resp.status_code == 200:
                SCAPHANDRE_URL = url
                return True
    except Exception:
        pass

    # If we don't have a command configured, give up quietly and fall back to RAPL only.
    if not SCAPHANDRE_CMD:
        return False

    try:
        await websocket.send_json({"log": f"🚀 Attempting to start Scaphandre exporter: {SCAPHANDRE_CMD}"})
    except Exception:
        pass

    # Fire-and-forget spawn; errors are logged but do not break the run.
    try:
        # Use a simple shell invocation so SCAPHANDRE_CMD can include flags.
        proc = await asyncio.create_subprocess_shell(
            SCAPHANDRE_CMD,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        # Give the exporter a short time to come up, then re-probe.
        await asyncio.sleep(1.5)
        async with httpx.AsyncClient(timeout=2.0) as client:
            resp = await client.get(url)
            if resp.status_code == 200:
                SCAPHANDRE_URL = url
                try:
                    await websocket.send_json({"log": f"✅ Scaphandre exporter detected at {url}"})
                except Exception:
                    pass
                return True
    except Exception as e:
        try:
            await websocket.send_json({"log": f"⚠️ Could not start Scaphandre automatically ({e}); using package-level RAPL only."})
        except Exception:
            pass

    return False

async def fetch_model_context_limit(model_name: str) -> Optional[int]:
    """Fetch context length for a model via shared ollama_client.get_model_info"""
    try:
        info = await get_model_info(model_name)
        if info and isinstance(info, dict):
            ctx = info.get("context_length")
            if isinstance(ctx, int) and ctx > 0:
                return ctx
    except Exception:
        pass
    return None

def get_model_context_limit(model_name: str) -> Optional[int]:
    """Get context length limit for a model (cached only)"""
    # Check cache first
    if model_name in MODEL_CONTEXT_CACHE:
        return MODEL_CONTEXT_CACHE[model_name]
    # If not cached yet, caller should invoke fetch_model_context_limit
    return None

@dataclass
class EnergyPayload:
    """Payload for energy testing"""
    # Raw components for backend composition
    system_prompt: str = ""  # Base system prompt
    user_prompt: str = ""    # Base user query
    conversation_context: str = "" # Context to inject
    injections: List[Dict[str, Any]] = field(default_factory=list) # List of injection objects
    
    # Legacy/Compatibility fields (optional)
    system: str = "" # Deprecated, used if system_prompt empty
    user: str = ""   # Deprecated, used if user_prompt empty
    
    model_name: str = DEFAULT_MODEL
    strategy_name: str = "baseline"
    energy_benchmark: str = "conservative_estimate"
    test_type: str = "energy"
    
    # Advanced config
    injection_type: str = "none" # For tracking metadata
    injection_params: Dict[str, Any] = field(default_factory=dict)
    tool_integration_method: str = "none"
    tool_config: Dict[str, Any] = field(default_factory=dict)
    
    temp: float = 0.7
    max_tokens: int = 1000
    seed: Optional[int] = None
    enable_live_power_monitoring: bool = False
    include_thinking: bool = False

# ---------- Ollama Client ----------
# (Now imported from ollama_client.py)

async def run_energy_test(payload: EnergyPayload, websocket: WebSocket, cancel_event: asyncio.Event):
    """Run energy consumption test"""
    print("🔋 [ENERGY APP] Running energy test - STANDALONE APP")
    await websocket.send_json({"log": "🔋 [ENERGY APP] Running energy test - STANDALONE APP"})
    try:
        # ===== PHASE 1: PROMPT COMPOSITION (BACKEND) =====
        # Use new fields if available, fallback to legacy
        base_system = payload.system_prompt if payload.system_prompt else payload.system
        base_user = payload.user_prompt if payload.user_prompt else payload.user
        
        # 1. Construct System Prompt (Base + Context + Injections)
        final_system_prompt = base_system
        
        # Add Conversation Context
        if payload.conversation_context:
            if final_system_prompt:
                final_system_prompt += "\n\n"
            
            # Check if tags already exist (simple heuristic)
            if "<conversation_context>" in payload.conversation_context:
                final_system_prompt += payload.conversation_context
            else:
                final_system_prompt += f"<conversation_context>\n{payload.conversation_context}\n</conversation_context>"
        
        # Add Injections
        injection_metadata_list = []
        for inj in payload.injections:
            desc = inj.get("description", "Unknown Injection")
            content = inj.get("content", "")
            
            if final_system_prompt:
                final_system_prompt += "\n\n---\n\n"
            final_system_prompt += f"[{desc}]\n{content}"
            
            injection_metadata_list.append({
                "type": desc,
                "tokens": len(content) // 4 # Approximate
            })

        # Do not inject hidden no-CoT text when thinking is disabled; rely on think=false only.

        # 2. Construct Messages
        original_messages = [] # "Original" means just the user query + base system (optional definition)
        # Ideally "Original" is what the user *intended* to send before hidden overhead.
        # Let's define "Original" as Base System + Base User.
        if base_system:
            original_messages.append({"role": "system", "content": base_system})
        original_messages.append({"role": "user", "content": base_user})
        
        # Messages AFTER Injection
        messages_after_injection = []
        if final_system_prompt:
            messages_after_injection.append({"role": "system", "content": final_system_prompt})
        messages_after_injection.append({"role": "user", "content": base_user}) # User prompt usually unchanged by system injections
        
        messages = messages_after_injection.copy()
        try:
            await websocket.send_json({
                "log": f"🧩 Applied context ({len(payload.conversation_context)} chars) and {len(payload.injections)} injection(s); final system length: {len(final_system_prompt)} chars"
            })
        except Exception:
            pass

        # Update injection result for tracking
        injection_result = {
            "injection_metadata": {
                "injection_type": "backend_composite",
                "injections": injection_metadata_list,
                "metadata": {"count": len(payload.injections)}
            }
        }
        print(f"📊 Backend Composition: Base System={len(base_system)} chars, Final System={len(final_system_prompt)} chars")

        # ===== PHASE 2: APPLY TOOL INTEGRATION (TRACK TOKENS) =====
        tool_integration_result = {"integration_metadata": {}}
        messages_after_tools = messages.copy()
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
                messages_after_tools = messages.copy()
            except Exception as e:
                try:
                    await websocket.send_json({"error": f"Tool integration error: {str(e)}", "done": True})
                except Exception:
                    pass
                return

        # ===== PHASE 3: TRACK MIDDLEWARE TOKEN OVERHEAD =====
        middleware_tokens = track_middleware_tokens(
            original_messages=original_messages,
            messages_after_injection=messages_after_injection,
            messages_after_tools=messages_after_tools
        )
        
        # Log token overhead for visibility
        print(f"📊 Token tracking: Original={middleware_tokens['original_tokens']}, "
              f"Injection+{middleware_tokens['injection_added']}, "
              f"Tools+{middleware_tokens['tools_added']}, "
              f"Total={middleware_tokens['final_total']}")

        # ===== PHASE 4: VALIDATE MAX_TOKENS AND CONTEXT WINDOW =====
        # Note: This is a soft limit - UI should allow override based on model capabilities
        safe_max_tokens = min(payload.max_tokens, DEFAULT_MAX_OUTPUT_TOKENS) if payload.max_tokens > DEFAULT_MAX_OUTPUT_TOKENS else payload.max_tokens
        if safe_max_tokens != payload.max_tokens:
            print(f"⚠️ Capping max_tokens from {payload.max_tokens} to {safe_max_tokens}")
        # Estimate context need to avoid truncation of large injections
        try:
            estimated_prompt_tokens_guess = int(middleware_tokens.get("final_total", 0))
        except Exception:
            estimated_prompt_tokens_guess = 0
        # Heuristic: desired ctx = estimated prompt + planned output + small slack
        desired_ctx = estimated_prompt_tokens_guess + safe_max_tokens + 256
        # Clip to a generous maximum (most modern small models support large ctx; engine will cap safely)
        desired_ctx = max(2048, min(desired_ctx, 40960))


        # ===== PHASE 5: INITIALIZE LIVE POWER / BASELINE (IF ENABLED) =====
        power_baseline = None
        power_before = None
        power_after = None
        # Snapshot-based RAPL energy accounting
        rapl_start_snap = None
        rapl_split_snap = None
        rapl_end_snap = None
        prefill_wh = None
        decode_wh = None
        total_wh = None
        # Scaphandre per-process energy accounting (LLM-only)
        scaph_e_llm_wh = 0.0
        scaph_sampler_task = None
        live_power_enabled = payload.enable_live_power_monitoring and POWER_MONITORING_AVAILABLE
        print(f"🔌 Live Power Monitoring: Requested={payload.enable_live_power_monitoring}, Available={POWER_MONITORING_AVAILABLE}, Enabled={live_power_enabled}")
        if payload.enable_live_power_monitoring and not POWER_MONITORING_AVAILABLE:
            await websocket.send_json({"log": "❌ RAPL not available on this system"})
        
        if live_power_enabled:
            try:
                info = power_monitor.get_info() if power_monitor else None
                await websocket.send_json({"log": "🔍 Checking RAPL availability..."})
                if info and info.get("available"):
                    await websocket.send_json({"log": "✅ RAPL Available"})
                    await websocket.send_json({"log": f"📊 Power zones: {', '.join(info.get('zones', []))}"})
                else:
                    await websocket.send_json({"log": "❌ RAPL not available!"})
                loop = asyncio.get_running_loop()
                await websocket.send_json({"log": "Measuring idle power..."})
                power_baseline = await loop.run_in_executor(None, power_monitor.read_power, 0.5)
                
                if power_baseline:
                    await websocket.send_json({
                        "log": f"📊 Baseline power: {power_baseline.total_watts:.2f} W",
                        "power_baseline": power_baseline.total_watts
                    })
                
                # Measure power just before inference (for display only)
                power_before = await loop.run_in_executor(None, power_monitor.read_power, 0.3)
                if power_before:
                    await websocket.send_json({
                        "log": f"🔋 Power before inference: {power_before.total_watts:.2f} W",
                        "power_before": power_before.total_watts
                    })
            except Exception as e:
                await websocket.send_json({
                    "log": f"⚠️ Power monitoring error: {str(e)}"
                })
                live_power_enabled = False

        # With RAPL confirmed, best-effort Scaphandre warm-up when live power is enabled.
        # This will try to start or detect a Scaphandre exporter but will fall back
        # silently to package-level RAPL if it fails.
        if live_power_enabled:
            try:
                await ensure_scaphandre_ready(websocket)
            except Exception:
                # Any failure here just means we won't have per-process E_llm this run.
                pass

        # ===== PHASE 6: RUN GENERATION =====
        start_time = asyncio.get_event_loop().time()
        
        full_response = ""
        prompt_tokens = 0
        completion_tokens = 0
        prompt_eval_duration_ns: Optional[int] = None
        eval_duration_ns: Optional[int] = None
        thinking_chars = 0
        content_chars = 0
        generation_done = False
        had_error = False


        # Scaphandre per-process energy sampler (LLM-only)
        async def scaphandre_sampler():
            nonlocal scaph_e_llm_wh
            if not SCAPHANDRE_URL:
                return
            last_ts = time.time()
            while not generation_done and not cancel_event.is_set():
                now_ts = time.time()
                dt = max(0.0, now_ts - last_ts)
                last_ts = now_ts
                try:
                    async with httpx.AsyncClient(timeout=5.0) as client_scaph:
                        resp = await client_scaph.get(SCAPHANDRE_URL)
                        if resp.status_code != 200:
                            try:
                                await websocket.send_json({"log": f"⚠️ Scaphandre HTTP {resp.status_code}, disabling for this run"})
                            except Exception:
                                pass
                            break
                        text = resp.text
                except Exception as e_sc:
                    try:
                        await websocket.send_json({"log": f"⚠️ Scaphandre error: {str(e_sc)}; disabling for this run"})
                    except Exception:
                        pass
                    break

                total_power_watts = 0.0
                for line in text.splitlines():
                    if not line.startswith("scaph_process_power_consumption_microwatts"):
                        continue
                    if SCAPHANDRE_OLLAMA_MATCH not in line:
                        continue
                    try:
                        value_str = line.strip().split()[-1]
                        microwatts = float(value_str)
                        total_power_watts += microwatts / 1_000_000.0
                    except Exception:
                        continue

                if total_power_watts > 0.0 and dt > 0.0:
                    scaph_e_llm_wh += (total_power_watts * dt) / 3600.0

                await asyncio.sleep(0.5)

        # Make request to Ollama
        if live_power_enabled:
            if SCAPHANDRE_URL and scaph_sampler_task is None:
                try:
                    scaph_sampler_task = asyncio.create_task(scaphandre_sampler())
                    await websocket.send_json({"log": "🧪 Scaphandre per-process sampling enabled (E_llm)"})
                except Exception:
                    scaph_sampler_task = None
            await websocket.send_json({"log": "Running LLM inference with power monitoring..."})
        
        print(f"📤 Sending to Ollama - Model: {payload.model_name}")
        print(f"📤 Messages array: {json.dumps(messages, indent=2)}")
        print(f"📤 Options: temp={payload.temp}, num_predict={safe_max_tokens}, seed={payload.seed}")
        try:
            await websocket.send_json({
                "log": f"🧠 Thinking setting → include_thinking={payload.include_thinking}; sending think={bool(getattr(payload, 'include_thinking', False))}"
            })
        except Exception:
            pass
        
        # Take RAPL start snapshot BEFORE issuing the request to include server-side prefill
        if live_power_enabled and power_monitor and rapl_start_snap is None:
            try:
                loop = asyncio.get_running_loop()
                rapl_start_snap = await loop.run_in_executor(None, power_monitor.take_energy_snapshot)
                await websocket.send_json({"log": "📍 RAPL snapshot taken: start (pre-request)"})
            except Exception as e:
                await websocket.send_json({"log": f"⚠️ RAPL snapshot start (pre-request) error: {str(e)}"})

        # Log the context window setting (calculation done earlier in Phase 4)
        try:
            await websocket.send_json({"log": f"🧮 Setting num_ctx={desired_ctx} (estimated prompt≈{estimated_prompt_tokens_guess}, max_out={safe_max_tokens})"})
        except Exception:
            pass

        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
            async with client.stream(
                "POST",
                f"{OLLAMA_BASE_URL}/api/chat",
                json={
                    "model": payload.model_name,
                    "messages": messages,
                    "stream": True,
                    "think": bool(getattr(payload, "include_thinking", False)),
                    "options": {
                        "temperature": payload.temp,
                        "num_predict": safe_max_tokens,  # Use validated max_tokens
                        "seed": payload.seed if payload.seed is not None else None,
                        "num_ctx": desired_ctx
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

                # Fallback: Take RAPL start snapshot right before reading first chunk if not already taken
                if live_power_enabled and power_monitor and rapl_start_snap is None:
                    try:
                        loop = asyncio.get_running_loop()
                        rapl_start_snap = await loop.run_in_executor(None, power_monitor.take_energy_snapshot)
                        await websocket.send_json({"log": "📍 RAPL snapshot taken: start (fallback)"})
                    except Exception as e:
                        await websocket.send_json({"log": f"⚠️ RAPL snapshot start error: {str(e)}"})

                split_time = None
                # TTFT heartbeat: log periodically while waiting for first token
                heartbeat_task = None
                async def _ttft_heartbeat():
                    try:
                        # Heartbeat while waiting for first token, regardless of RAPL state
                        while rapl_split_snap is None and not cancel_event.is_set():
                            elapsed = asyncio.get_event_loop().time() - start_time
                            try:
                                await websocket.send_json({"log": f"⌛ Prefill in progress… {elapsed:.1f}s (ctx~{estimated_prompt_tokens_guess})"})
                            except Exception:
                                pass
                            await asyncio.sleep(2.0)
                    except Exception:
                        pass

                try:
                    heartbeat_task = asyncio.create_task(_ttft_heartbeat())
                except Exception:
                    heartbeat_task = None

                async for line in response.aiter_lines():
                    if cancel_event.is_set():
                        break

                    if not line:
                        continue

                    try:
                        chunk = json.loads(line)
                        print(f"🔍 Ollama chunk received: {chunk}")
                    except json.JSONDecodeError:
                        continue

                    if "message" in chunk:
                        # Support both regular models (content) and reasoning models (thinking)
                        msg = chunk["message"] or {}
                        content_text = msg.get("content") or ""
                        thinking_text = msg.get("thinking") or ""

                        # On first token (either thinking or content), record split time and
                        # stop the TTFT heartbeat. When live power is enabled, also take a
                        # RAPL split snapshot for prefill/decode separation.
                        if (content_text or thinking_text) and rapl_split_snap is None:
                            # Take RAPL split snapshot only when live power monitoring is active
                            if live_power_enabled and power_monitor:
                                try:
                                    loop = asyncio.get_running_loop()
                                    rapl_split_snap = await loop.run_in_executor(None, power_monitor.take_energy_snapshot)
                                    await websocket.send_json({"log": "📍 RAPL snapshot taken: split (first token)"})
                                except Exception as e:
                                    await websocket.send_json({"log": f"⚠️ RAPL snapshot split error: {str(e)}"})

                            # Always record first-token time for TTFT and stop heartbeat
                            split_time = asyncio.get_event_loop().time()
                            if 'heartbeat_task' in locals() and heartbeat_task is not None:
                                heartbeat_task.cancel()

                        if content_text:
                            print(f"📝 Extracted content: '{content_text}' (length: {len(content_text)})")
                            full_response += content_text
                            content_chars += len(content_text)
                            await websocket.send_json({
                                "token": content_text,
                                "model": payload.model_name,
                                "strategy": payload.strategy_name,
                                "thinking": False,
                                "done": False,
                            })
                        elif thinking_text and getattr(payload, "include_thinking", False):
                            print(f"🧠 Extracted thinking: '{thinking_text}' (length: {len(thinking_text)})")
                            full_response += thinking_text
                            thinking_chars += len(thinking_text)
                            await websocket.send_json({
                                "token": thinking_text,
                                "model": payload.model_name,
                                "strategy": payload.strategy_name,
                                "thinking": True,
                                "done": False,
                            })

                    if chunk.get("done", False):
                        prompt_tokens = chunk.get("prompt_eval_count", 0)
                        completion_tokens = chunk.get("eval_count", 0)
                        prompt_eval_duration_ns = chunk.get("prompt_eval_duration")
                        eval_duration_ns = chunk.get("eval_duration")
                        print(f"✅ Generation complete. Tokens: prompt={prompt_tokens}, completion={completion_tokens}")
                        break
                
                # Always close the response stream properly
                await response.aclose()
                # Ensure heartbeat stopped
                if 'heartbeat_task' in locals() and heartbeat_task is not None:
                    with contextlib.suppress(Exception):
                        heartbeat_task.cancel()

        generation_done = True
        if scaph_sampler_task is not None:
            with contextlib.suppress(Exception):
                await scaph_sampler_task
        if live_power_enabled and power_monitor:
            # Take end snapshot and compute energies
            try:
                loop = asyncio.get_running_loop()
                rapl_end_snap = await loop.run_in_executor(None, power_monitor.take_energy_snapshot)
                await websocket.send_json({"log": "📍 RAPL snapshot taken: end"})
                total_wh = power_monitor.energy_diff_wh(rapl_start_snap or {}, rapl_end_snap or {})
                prefill_wh = power_monitor.energy_diff_wh(rapl_start_snap or {}, rapl_split_snap or (rapl_start_snap or {}))
                decode_wh = power_monitor.energy_diff_wh(rapl_split_snap or (rapl_start_snap or {}), rapl_end_snap or {})
                # Closure error diagnostics
                try:
                    closure_err_wh = (total_wh or 0.0) - ((prefill_wh or 0.0) + (decode_wh or 0.0))
                    closure_err_pct = (closure_err_wh / total_wh * 100.0) if (total_wh and total_wh > 0) else None
                except Exception:
                    closure_err_wh = None
                    closure_err_pct = None
                # Baseline-corrected active energy (subtract idle baseline × time)
                prefill_time = None
                decode_time = None
                active_prefill_wh = None
                active_decode_wh = None
                active_total_wh = None
                try:
                    prefill_time = (rapl_split_snap is not None and 'split_time' in locals() and split_time is not None) and (split_time - start_time) or None
                    decode_time = ('split_time' in locals() and split_time is not None) and (end_time - split_time) or None
                    baseline = power_baseline.total_watts if power_baseline else None
                    if baseline is not None:
                        if prefill_time is not None and prefill_wh is not None:
                            active_prefill_wh = max(0.0, prefill_wh - (baseline * prefill_time / 3600.0))
                        if decode_time is not None and decode_wh is not None:
                            active_decode_wh = max(0.0, decode_wh - (baseline * decode_time / 3600.0))
                        if total_wh is not None and latency is not None:
                            active_total_wh = max(0.0, total_wh - (baseline * latency / 3600.0))
                except Exception:
                    pass
                await websocket.send_json({
                    "log": f"🔬 Prefill energy: {prefill_wh:.6f} Wh",
                })
                await websocket.send_json({
                    "log": f"🔬 Decode energy:  {decode_wh:.6f} Wh"
                })
                await websocket.send_json({
                    "log": f"🎯 Total energy:   {total_wh:.6f} Wh"
                })
                if closure_err_wh is not None:
                    await websocket.send_json({
                        "log": f"🧪 Energy closure: prefill+decode {( (prefill_wh or 0.0)+(decode_wh or 0.0) ):.6f} Wh, total {total_wh:.6f} Wh, Δ={closure_err_wh:.6f} Wh ({(closure_err_pct or 0.0):.2f}%)"
                    })
                if active_total_wh is not None:
                    await websocket.send_json({
                        "log": f"🧮 Active energy (baseline-subtracted): prefill={(active_prefill_wh if active_prefill_wh is not None else 0.0):.6f} Wh, decode={(active_decode_wh if active_decode_wh is not None else 0.0):.6f} Wh, total={active_total_wh:.6f} Wh"
                    })
            except Exception as e:
                await websocket.send_json({"log": f"⚠️ RAPL snapshot end error: {str(e)}"})

        if cancel_event.is_set():
            await websocket.send_json({
                "token": "[CANCELLED]",
                "model": payload.model_name,
                "strategy": payload.strategy_name,
                "done": True,
                "cancelled": True,
            })
            return

    except Exception as e:
        print(f"❌ Error in run_energy_test: {str(e)}")
        had_error = True
        await websocket.send_json({"error": str(e), "done": True})
    finally:
        generation_done = True

        # If an error occurred or the request was cancelled, skip emitting final results
        if had_error or (cancel_event and cancel_event.is_set()):
            return

        # Calculate timing
        end_time = asyncio.get_event_loop().time()
        latency = end_time - start_time
        if live_power_enabled:
            await websocket.send_json({"log": f"✅ Generated {prompt_tokens + completion_tokens} tokens"})
            await websocket.send_json({"log": f"⏱ Duration: {latency:.2f}s"})

        # ===== PHASE 7: FINALIZE RAPL MEASUREMENT USING CUMULATIVE ENERGY (IF ENABLED) =====
        # NOTE: All Wh/1000 calculations here are defined per 1000 *output* tokens (completion tokens).
        measured_wh_per_1000_tokens = None
        # Per-process LLM energy (Scaphandre) and model compute efficiency
        e_llm_wh: Optional[float] = None
        e_llm_wh_per_1000_tokens: Optional[float] = None
        m_eff_wh_per_1000_tokens: Optional[float] = None
        m_eff_deviation_pct: Optional[float] = None
        # Calibration-based metrics (computed when we have total_wh and completion_tokens)
        calib_profile_used = None
        calib_baseline_per_1k_out = None
        calib_expected_decode_wh = None
        calib_prefill_est_wh = None
        calib_delta_per_1k_out = None
        calib_prefill_per_1k_in = None
        if live_power_enabled and total_wh is not None:
            try:
                # Use output tokens only for Wh/1000 calculation (package-level RAPL)
                if completion_tokens > 0:
                    measured_wh_per_1000_tokens = (total_wh / completion_tokens) * 1000

                # Derive E_llm (LLM-only) from Scaphandre if available, else fall back to RAPL package energy
                if scaph_e_llm_wh and scaph_e_llm_wh > 0.0:
                    e_llm_wh = scaph_e_llm_wh
                else:
                    e_llm_wh = total_wh

                if e_llm_wh is not None and completion_tokens > 0:
                    e_llm_wh_per_1000_tokens = (e_llm_wh / completion_tokens) * 1000.0

                # Model compute efficiency (active-time scaled Wh/1K) using Ollama's reported durations
                ollama_active_seconds: Optional[float] = None
                try:
                    pe_ns = float(prompt_eval_duration_ns or 0)
                    de_ns = float(eval_duration_ns or 0)
                    total_ns = pe_ns + de_ns
                    if total_ns > 0.0:
                        ollama_active_seconds = total_ns / 1_000_000_000.0
                except Exception:
                    ollama_active_seconds = None

                if (
                    e_llm_wh_per_1000_tokens is not None
                    and ollama_active_seconds is not None
                    and latency is not None
                    and latency > 0
                ):
                    scale = max(0.0, min(1.0, ollama_active_seconds / latency))
                    m_eff_wh_per_1000_tokens = e_llm_wh_per_1000_tokens * scale
                    if e_llm_wh_per_1000_tokens > 0:
                        m_eff_deviation_pct = abs(m_eff_wh_per_1000_tokens - e_llm_wh_per_1000_tokens) / e_llm_wh_per_1000_tokens * 100.0

                # Log measurement breakdown
                await websocket.send_json({
                    "log": (
                        f"🎯 Measured (RAPL counters): {measured_wh_per_1000_tokens:.4f} Wh/1K tokens"
                        if measured_wh_per_1000_tokens
                        else "🎯 Measured: (insufficient tokens)"
                    ),
                    "measured_wh": total_wh,
                    "measured_wh_per_1000_tokens": measured_wh_per_1000_tokens
                })

                if e_llm_wh_per_1000_tokens is not None:
                    try:
                        src = "Scaphandre (per-process)" if scaph_e_llm_wh and scaph_e_llm_wh > 0.0 else "RAPL package"
                        await websocket.send_json({
                            "log": f"🧮 E_llm [{src}]: {e_llm_wh_per_1000_tokens:.4f} Wh/1K output tokens"
                        })
                    except Exception:
                        pass

                if m_eff_wh_per_1000_tokens is not None:
                    try:
                        await websocket.send_json({
                            "log": f"🧮 M_eff (active-time scaled): {m_eff_wh_per_1000_tokens:.4f} Wh/1K (Δ={m_eff_deviation_pct or 0.0:.1f}% vs E_llm)"
                        })
                    except Exception:
                        pass

                # === Calibration-based prefill from baselines ===
                try:
                    model_key = (payload.model_name or "").replace(":", "_").replace(" ", "_")
                    # Prefer thinking profile if requested and available
                    candidates = []
                    if getattr(payload, "include_thinking", False):
                        candidates.append(f"rapl_calibrated_{model_key}_thinking")
                    candidates.append(f"rapl_calibrated_{model_key}")
                    # Also try currently selected benchmark if it looks calibrated
                    if payload.energy_benchmark and payload.energy_benchmark.startswith("rapl_calibrated_"):
                        candidates.insert(0, payload.energy_benchmark)
                    for name in candidates:
                        if name in ENERGY_BENCHMARKS:
                            calib_profile_used = name
                            calib_b = ENERGY_BENCHMARKS[name]
                            calib_baseline_per_1k_out = float(calib_b.output_wh_per_1000_tokens or calib_b.watt_hours_per_1000_tokens)
                            break
                    # If baseline found, compute expected decode Wh and prefill estimate
                    if calib_baseline_per_1k_out is not None and completion_tokens > 0:
                        calib_expected_decode_wh = calib_baseline_per_1k_out * (completion_tokens / 1000.0)
                        # Prefill (and any extra overhead) attributed to variance
                        calib_prefill_est_wh = max(0.0, float(total_wh) - float(calib_expected_decode_wh))
                        if measured_wh_per_1000_tokens is not None:
                            calib_delta_per_1k_out = max(0.0, float(measured_wh_per_1000_tokens) - float(calib_baseline_per_1k_out))
                        if (prompt_tokens or 0) > 0:
                            calib_prefill_per_1k_in = calib_prefill_est_wh / (prompt_tokens / 1000.0)
                        await websocket.send_json({
                            "log": (
                                f"📏 Calibration baseline: {calib_profile_used} | baseline(out)={calib_baseline_per_1k_out:.4f} Wh/1K; "
                                f"expected decode={calib_expected_decode_wh:.6f} Wh; prefill≈{calib_prefill_est_wh:.6f} Wh; Δ(out)={calib_delta_per_1k_out if calib_delta_per_1k_out is not None else 'n/a'}"
                            ) if calib_profile_used else "ℹ️ No calibration baseline found; skipping calibration-based split"
                        })
                except Exception as e_cal:
                    with contextlib.suppress(Exception):
                        await websocket.send_json({"log": f"⚠️ Calibration-based split failed: {str(e_cal)}"})

                # Record into session-level RAPL tracker using OUTPUT tokens for Wh/1K semantics
                # Using completion tokens aligns with the definition of Wh per 1000 output tokens.
                if (completion_tokens or 0) > 0 and total_wh is not None:
                    energy_tracker.record_rapl_measurement(total_wh, completion_tokens)

                # DYNAMIC BENCHMARK UPDATE
                # If we have a valid measurement, create/update a dynamic benchmark and use it
                # Do NOT auto-switch when the user selected a calibrated baseline (keep baselines stable)
                auto_switch_allowed = not (payload.energy_benchmark and payload.energy_benchmark.startswith("rapl_calibrated_"))
                if auto_switch_allowed and measured_wh_per_1000_tokens is not None and measured_wh_per_1000_tokens > 0:
                    dynamic_benchmark_name = "rapl_live_dynamic"
                    print(f"🔄 Creating dynamic benchmark '{dynamic_benchmark_name}' with {measured_wh_per_1000_tokens:.4f} Wh/1K")
                    
                    # Add/Update the benchmark
                    energy_tracker.add_custom_benchmark(
                        name=dynamic_benchmark_name,
                        description="Live RAPL Measurement (Dynamic)",
                        watt_hours_per_1000_tokens=measured_wh_per_1000_tokens,
                        source="Real-time RAPL Measurement",
                        hardware_specs="Current System (Dynamic)",
                        force_update=True
                    )
                    
                    # Force use of this benchmark
                    payload.energy_benchmark = dynamic_benchmark_name
                    await websocket.send_json({
                        "log": f"✅ Created new RAPL benchmark: {measured_wh_per_1000_tokens:.4f} Wh/1K"
                    })
                    await websocket.send_json({"log": "✅ Measurement Complete!"})

            except Exception as e:
                await websocket.send_json({
                    "log": f"⚠️ Power measurement error: {str(e)}"
                })

        # ===== PHASE 8: CREATE COMPREHENSIVE TOKEN BREAKDOWN (CRITICAL FIX) =====
        # If Ollama did not report prompt_eval_count (prompt_tokens == 0),
        # fall back to our middleware-based estimate for input token counts.
        try:
            prompt_tokens_for_breakdown = int(prompt_tokens or middleware_tokens.get("final_total", 0) or 0)
        except Exception:
            prompt_tokens_for_breakdown = prompt_tokens

        token_breakdown = create_token_breakdown(
            original_messages=original_messages,
            middleware_tracking=middleware_tokens,
            ollama_prompt_tokens=prompt_tokens_for_breakdown,
            ollama_completion_tokens=completion_tokens
        )
        # Estimate split of completion tokens between thinking vs content by proportional characters
        try:
            total_gen_chars = thinking_chars + content_chars
            if total_gen_chars > 0 and completion_tokens >= 0:
                thinking_tokens_est = int(round((completion_tokens * thinking_chars) / max(total_gen_chars, 1)))
                thinking_tokens_est = max(0, min(thinking_tokens_est, completion_tokens))
                content_tokens_est = max(0, completion_tokens - thinking_tokens_est)
            else:
                thinking_tokens_est = 0
                content_tokens_est = completion_tokens
        except Exception:
            thinking_tokens_est = 0
            content_tokens_est = completion_tokens

        if isinstance(token_breakdown.get("generation"), dict):
            token_breakdown["generation"]["thinking_tokens"] = thinking_tokens_est
            token_breakdown["generation"]["content_tokens"] = content_tokens_est
        
        # Log breakdown for debugging
        print(f"✅ Token breakdown created: {token_breakdown['totals']['grand_total']} total tokens")
        print(f"   Accuracy: {token_breakdown['verification']['accuracy_percent']}%")

        # Log likely truncation if estimation >> actual
        try:
            if estimated_prompt_tokens_guess > 0 and prompt_tokens > 0 and estimated_prompt_tokens_guess > (prompt_tokens * 2):
                await websocket.send_json({
                    "log": f"⚠️ Estimated prompt ({estimated_prompt_tokens_guess}) >> actual used ({prompt_tokens}). Context truncation likely; increase num_ctx or reduce injections."
                })
        except Exception:
            pass

        # ===== PHASE 9: RECORD ENERGY CONSUMPTION =====
        # CRITICAL FIX: Set the benchmark before recording
        print(f"🔄 Setting benchmark to: '{payload.energy_benchmark}'")
        try:
            energy_tracker.set_benchmark(payload.energy_benchmark)
            print(f"✅ Benchmark set to: {energy_tracker.benchmark.name} ({energy_tracker.benchmark.watt_hours_per_1000_tokens} Wh/1K)")
        except ValueError as e:
            print(f"⚠️ Invalid benchmark '{payload.energy_benchmark}', using conservative_estimate. Error: {e}")
            energy_tracker.set_benchmark("conservative_estimate")
        
        energy_reading = energy_tracker.record_usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_seconds=latency,
            model_name=payload.model_name,
            strategy_name=payload.strategy_name
        )

        # ===== PHASE 10: SEND COMPREHENSIVE RESULTS (WITH FIXED TOKEN_METRICS) =====
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

            # 🔥 CRITICAL FIX: Add proper token_metrics with breakdown
            "token_metrics": {
                "breakdown": token_breakdown
            },

            # Energy metrics
            # NOTE: energy_efficiency_score is defined per 1000 *output* tokens
            # (completion_tokens) to match UI semantics of "Wh/1000 e-tokens".
            "energy_metrics": {
                "benchmark_used": energy_reading.benchmark_used,
                "watt_hours_consumed": energy_reading.watt_hours_consumed,
                "carbon_grams_co2": energy_reading.carbon_grams_co2,
                # Output-normalized efficiency (per 1000 etokens-o, i.e. total run Wh / output tokens).
                "energy_efficiency_score": energy_reading.watt_hours_consumed / max(completion_tokens / 1000, 0.001),
                "energy_efficiency_wh_per_1000_etokens_o": energy_reading.watt_hours_consumed / max(completion_tokens / 1000, 0.001),
                # Input-normalized efficiency (per 1000 etokens-i, i.e. total run Wh / input tokens).
                "energy_efficiency_wh_per_1000_etokens_i": (
                    energy_reading.watt_hours_consumed / max(prompt_tokens / 1000, 0.001)
                    if (prompt_tokens or 0) > 0
                    else None
                ),
            },

            # Live power monitoring data (if enabled)
            "live_power_metrics": {
                "enabled": live_power_enabled,
                "baseline_watts": power_baseline.total_watts if power_baseline else None,
                "before_watts": power_before.total_watts if power_before else None,
                "after_watts": power_after.total_watts if power_after else None,
                "active_watts": ((power_before.total_watts + power_after.total_watts) / 2 - power_baseline.total_watts) if (power_before and power_after and power_baseline) else None,
                # Per-1000 metrics are expressed per 1000 *effective* tokens.
                # For backward compatibility these remain defined per 1000 output tokens
                # (etokens-o) and we add explicit etokens-i / etokens-o aliases.
                "measured_wh_per_1000_tokens": measured_wh_per_1000_tokens,
                "benchmark_wh_per_1000_tokens": energy_reading.watt_hours_consumed / max(completion_tokens / 1000, 0.001),
                "accuracy_percent": (
                    (
                        measured_wh_per_1000_tokens
                        / (energy_reading.watt_hours_consumed / max(completion_tokens / 1000, 0.001))
                    ) * 100
                ) if measured_wh_per_1000_tokens else None,
                # Explicit aliases for clarity in the UI layer
                "measured_wh_per_1000_etokens_o": measured_wh_per_1000_tokens,
                "benchmark_wh_per_1000_etokens_o": energy_reading.watt_hours_consumed / max(completion_tokens / 1000, 0.001),
                # Whole-run energy normalized by input tokens (etokens-i)
                "measured_wh_per_1000_input_tokens": (
                    (total_wh / max(prompt_tokens / 1000, 0.001))
                    if (total_wh is not None and (prompt_tokens or 0) > 0)
                    else None
                ),
                "measured_wh_per_1000_etokens_i": (
                    (total_wh / max(prompt_tokens / 1000, 0.001))
                    if (total_wh is not None and (prompt_tokens or 0) > 0)
                    else None
                ),
                # Snapshot-based split
                "prefill_wh": prefill_wh,
                "decode_wh": decode_wh,
                "total_wh": total_wh,
                "closure_error_wh": closure_err_wh if 'closure_err_wh' in locals() else None,
                "closure_error_percent": closure_err_pct if 'closure_err_pct' in locals() else None,
                "prefill_wh_per_1000_input_tokens": (prefill_wh / max(prompt_tokens / 1000, 0.001)) if (prefill_wh is not None) else None,
                "decode_wh_per_1000_output_tokens": (decode_wh / max(completion_tokens / 1000, 0.001)) if (decode_wh is not None) else None,
                "energy_weighted_output_wh_per_1000_tokens": measured_wh_per_1000_tokens,
                # Calibration-based split (preferred when available)
                "calib_profile_used": calib_profile_used,
                "calib_baseline_per_1000_output_tokens": calib_baseline_per_1k_out,
                "calib_expected_decode_wh": calib_expected_decode_wh,
                "calib_prefill_est_wh": calib_prefill_est_wh,
                "calib_delta_per_1000_output_tokens": calib_delta_per_1k_out,
                "calib_prefill_per_1000_input_tokens": calib_prefill_per_1k_in,
                # Timing metrics
                "ttft_seconds": (split_time - start_time) if ("split_time" in locals() and split_time is not None) else None,
                "decode_duration_seconds": (end_time - split_time) if ("split_time" in locals() and split_time is not None) else None,
                "run_duration_seconds": latency,
                # Scaphandre per-process LLM energy metrics and active-time efficiency
                "scaphandre_available": bool(SCAPHANDRE_URL),
                "scaph_e_llm_wh": e_llm_wh,
                "scaph_e_llm_wh_per_1000_tokens": e_llm_wh_per_1000_tokens,
                "model_compute_efficiency_wh_per_1000_tokens": m_eff_wh_per_1000_tokens,
                "m_eff_vs_e_llm_deviation_percent": m_eff_deviation_pct,
                "ollama_prompt_eval_duration_seconds": (float(prompt_eval_duration_ns) / 1_000_000_000.0) if prompt_eval_duration_ns else None,
                "ollama_eval_duration_seconds": (float(eval_duration_ns) / 1_000_000_000.0) if eval_duration_ns else None,
                # Baseline-corrected active energy
                "prefill_wh_active": active_prefill_wh if 'active_prefill_wh' in locals() else None,
                "decode_wh_active": active_decode_wh if 'active_decode_wh' in locals() else None,
                "total_wh_active": active_total_wh if 'active_total_wh' in locals() else None,
                "prefill_active_per_1000_input_tokens": ((active_prefill_wh / max(prompt_tokens / 1000, 0.001)) if ('active_prefill_wh' in locals() and active_prefill_wh is not None) else None),
                "decode_active_per_1000_output_tokens": ((active_decode_wh / max(completion_tokens / 1000, 0.001)) if ('active_decode_wh' in locals() and active_decode_wh is not None) else None)
            },

            # Modification tracking (IMPROVED with actual token counts)
            "modification_info": {
                "injection_applied": injection_result.get("injection_metadata", {}),
                "tool_integration_applied": tool_integration_result.get("integration_metadata", {}),
                "original_prompt_length": len(payload.system) + len(payload.user),  # Characters
                "final_prompt_length": sum(len(m.get("content", "")) for m in messages),  # Characters after middleware
                "injection_overhead": middleware_tokens["injection_added"],  # 🔥 FIX: Real token count
                "tool_overhead": middleware_tokens["tools_added"],  # 🔥 FIX: Real token count
                "modification_overhead": middleware_tokens["total_middleware"],  # 🔥 FIX: Not hardcoded to 0!
            },

            # Session summary
            "session_summary": energy_tracker.get_session_summary(),
            # Debug helpers for truncation/tokenization sanity check
            "debug_metrics": {
                "estimated_prompt_tokens": middleware_tokens.get("final_total", 0),
                "num_ctx_used": desired_ctx,
                # Tokenizer-based context validation is currently disabled.
                "context_validation_performed": False
            }
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

@app.get("/api/model-info/{model_name:path}")
async def get_model_information(model_name: str):
    """Get detailed information about a specific model including context length"""
    info = await get_model_info(model_name)
    if info:
        return info
    else:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found or info unavailable")

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

@app.get("/api/model-info/{model_name}")
async def get_model_info_route(model_name: str):
    """Get model information including context length (via shared ollama_client.get_model_info)."""
    try:
        info = await get_model_info(model_name)
        if info and isinstance(info, dict):
            ctx = info.get("context_length") or 40960
            details = info.get("details", {}) or {}
            return {
                "model_name": info.get("name", model_name),
                "context_length": ctx,
                "architecture": details.get("architecture", "unknown"),
                "source": "ollama_client.get_model_info"
            }
        # Fallback if info is None or malformed
        return {
            "model_name": model_name,
            "context_length": 40960,
            "architecture": "unknown",
            "source": "fallback_estimate"
        }
    except Exception as e:
        # Return fallback on any error
        return {
            "model_name": model_name,
            "context_length": 40960,
            "architecture": "unknown",
            "source": "error_fallback",
            "error": str(e)
        }

@app.post("/api/export-session")
async def export_session(filepath: str = "energy_session.json"):
    """Export session data to file"""
    try:
        energy_tracker.export_readings(filepath)
        return {"success": True, "filepath": filepath}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")

@app.post("/api/switch-benchmark")
async def switch_benchmark(request: dict):
    """Switch to a different energy benchmark and recalculate all readings."""
    try:
        benchmark_name = request.get("benchmark_name") if isinstance(request, dict) else request
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
        input_wh = request.get("input_wh_per_1000_tokens", 0.0)
        output_wh = request.get("output_wh_per_1000_tokens", 0.0)
        source = request.get("source", "Custom")
        hardware_specs = request.get("hardware_specs", "User defined")

        if not name or not description or watt_hours is None:
            raise HTTPException(status_code=400, detail="Missing required fields")

        success = energy_tracker.add_custom_benchmark(
            name=name,
            description=description,
            watt_hours_per_1000_tokens=float(watt_hours),
            input_wh_per_1000_tokens=float(input_wh),
            output_wh_per_1000_tokens=float(output_wh),
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
        # Legacy benchmarks used for internal estimation and RAPL calibration
        "benchmarks": get_available_benchmarks(),
        # Hugging Face AI Energy Score benchmarks (H100, text_generation_and_reasoning)
        # Exposed separately so the UI can use wh_per_1000_input_etokens for
        # impact metrics and multi-benchmark comparison.
        "hf_benchmarks": get_hf_benchmarks(),
        "co2_info": {
            "global_average_gco2_per_kwh": 400,
            "description": "Global average carbon intensity for electricity generation",
            "source": "IEA 2023 Global Energy Review",
            "units": "grams of CO2 equivalent per kilowatt-hour",
            "calculation": "Energy consumption (Wh) × Carbon intensity (gCO2/kWh) ÷ 1000"
        },
        # All external benchmark sources discovered under 'benchmark_data/'.
        # Keys correspond to data_source in each JSON (e.g. "hugging face",
        # "jegham_et_al") and expose their models and per-1000 etoken
        # baselines for UI selection.
        "benchmark_sources": get_benchmark_sources(),
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
