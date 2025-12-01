#!/usr/bin/env python3
"""Shared helper for RAPL-based energy measurement.

This module provides a single helper to measure Wh and Wh/1000 tokens for
an Ollama inference run using integrated RAPL sampling, so both the
standalone benchmark script and other components can reuse the same logic.
"""

import time
import asyncio
import contextlib
from dataclasses import dataclass
from typing import Optional

import httpx

from power_monitor import power_monitor
from ollama_client import OLLAMA_BASE_URL, REQUEST_TIMEOUT


@dataclass
class RAPLMeasurementResult:
    wh_total: float
    wh_per_1000_tokens: Optional[float]
    total_tokens: int
    duration_seconds: float
    baseline_watts: Optional[float]


async def measure_wh_per_1000_tokens(
    prompt: str,
    model: str,
    max_tokens: int = 512,
    idle_seconds: float = 0.5,
    sample_seconds: float = 0.5,
    include_thinking: bool = False,
    target_words: int = 120,
) -> Optional[RAPLMeasurementResult]:
    """Measure Wh and Wh/1000 tokens for a single non-streaming Ollama run.

    This uses continuous RAPL sampling during inference, subtracting an
    idle baseline to approximate active inference energy.
    """

    # Check availability
    info = power_monitor.get_info()
    if not info.get("available"):
        return None

    loop = asyncio.get_running_loop()

    # Baseline (idle) power
    baseline = await loop.run_in_executor(None, power_monitor.read_power, idle_seconds)
    baseline_watts = baseline.total_watts if baseline else None

    rapl_measured_wh = 0.0
    sampling_done = False

    async def rapl_sampler():
        nonlocal rapl_measured_wh
        if not baseline:
            return
        try:
            while not sampling_done:
                sample = await loop.run_in_executor(None, power_monitor.read_power, sample_seconds)
                if sample:
                    active_watts = max(sample.total_watts - baseline.total_watts, 0.0)
                    rapl_measured_wh += (active_watts * sample_seconds) / 3600.0
        except Exception:
            # Best-effort sampler; caller will decide whether result is usable
            return

    total_tokens = 0
    start_time = time.time()

    sampler_task = asyncio.create_task(rapl_sampler())
    try:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
            # Use generate API with a concrete prompt; ignore include_thinking for robustness here
            full_prompt = f"Write exactly {max(target_words, 10)} words. Keep it concise.\n\n{prompt}"
            resp = await client.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json={
                    "model": model,
                    "prompt": full_prompt,
                    "stream": False,
                    "options": {"num_predict": max_tokens},
                },
            )
            resp.raise_for_status()
            data = resp.json()
            total_tokens = int(data.get("prompt_eval_count", 0)) + int(data.get("eval_count", 0))
    finally:
        # Stop sampling and wait for sampler to exit
        sampling_done = True
        with contextlib.suppress(Exception):
            await sampler_task

    duration = time.time() - start_time

    if total_tokens <= 0 or rapl_measured_wh <= 0:
        return None

    wh_per_1000 = (rapl_measured_wh / total_tokens) * 1000.0
    return RAPLMeasurementResult(
        wh_total=rapl_measured_wh,
        wh_per_1000_tokens=wh_per_1000,
        total_tokens=total_tokens,
        duration_seconds=duration,
        baseline_watts=baseline_watts,
    )
