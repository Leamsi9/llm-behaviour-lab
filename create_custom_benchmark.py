#!/usr/bin/env python3
"""Create a custom energy benchmark based on real power measurements.

This script now supports running the same prompt many times and exporting
only the per-run electricity values to JSON so we can analyze the
variability and reliability of single-run measurements. It reuses the
integrated RAPL helper from ``rapl_benchmark`` to avoid duplicating power
measurement logic.
"""

import time
import json
from pathlib import Path
import argparse
import asyncio

from energy_tracker import energy_tracker
from rapl_benchmark import measure_wh_per_1000_tokens

OLLAMA_BASE_URL = "http://localhost:11434"

# Fixed prompt/model for variability experiments
TEST_PROMPT = "Explain how transformers work in machine learning in 3 sentences."
TEST_MODEL = "qwen3:0.6b"  # Small fast model for testing


async def measure_inference_power(add_benchmark: bool = True):
    """Measure actual power consumption during LLM inference via shared helper.

    Returns average Wh per 1000 tokens, or ``None`` on failure.
    """

    print("🔋 Creating Custom Benchmark for HP 255 G10 (Ryzen 7 7730U)")
    print("=" * 60)

    test_prompt = TEST_PROMPT
    model = TEST_MODEL

    print("Running test inference with integrated RAPL measurement...")
    print(f"Model: {model}")
    print(f"Prompt: {test_prompt}\n")

    result = await measure_wh_per_1000_tokens(
        prompt=test_prompt,
        model=model,
        max_tokens=512,
    )

    if result is None:
        print("\n❌ Measurement failed (RAPL unavailable, read error, or no tokens)")
        print("Make sure:")
        print("  • RAPL is readable (e.g. correct permissions under /sys/class/powercap)")
        print("  • Ollama is running: ollama serve")
        return None

    wh_per_1000_tokens = result.wh_per_1000_tokens or 0.0

    print(f"\n{'='*60}")
    print("✅ Measurement Complete (integrated RAPL)!")
    print(f"{'='*60}")
    print(f"Duration: {result.duration_seconds:.2f}s")
    print(f"Total tokens: {result.total_tokens}")
    if result.duration_seconds > 0:
        print(f"Tokens/sec: {result.total_tokens/result.duration_seconds:.1f}")
    if result.baseline_watts is not None:
        print(f"\nPower:")
        print(f"  Idle baseline: {result.baseline_watts:.2f} W")
    print(f"\nEnergy:")
    print(f"  Total active energy (RAPL): {result.wh_total:.6f} Wh")
    print(f"  🎯 Wh per 1000 tokens: {wh_per_1000_tokens:.4f}")
    print(f"{'='*60}\n")

    # Optionally add as custom benchmark (for single-measurement mode)
    if add_benchmark:
        benchmark_name = "hp_255_g10_ryzen7_7730u_measured"
        success = energy_tracker.add_custom_benchmark(
            name=benchmark_name,
            description=(
                f"Real measurement on HP 255 G10, Ryzen 7 7730U "
                f"(measured: {wh_per_1000_tokens:.4f} Wh/1K)"
            ),
            watt_hours_per_1000_tokens=wh_per_1000_tokens,
            source="Direct RAPL measurement during inference",
            hardware_specs=(
                f"AMD Ryzen 7 7730U, measured at {result.total_tokens/result.duration_seconds:.0f} tok/s"
                if result.duration_seconds > 0
                else "AMD Ryzen 7 7730U"
            ),
        )

        if success:
            print(f"✅ Added custom benchmark: '{benchmark_name}'")
            print("\nYou can now select this benchmark in the Energy Lab UI!")
        else:
            print(f"⚠️  Benchmark '{benchmark_name}' already exists")

    return wh_per_1000_tokens
if __name__ == "__main__":
    # Run many identical measurements and export only the electricity values
    # (Wh per 1000 tokens) to a JSON file so we can analyze variability.

    parser = argparse.ArgumentParser(
        description=(
            "Measure Wh/1000 tokens for a fixed prompt/model multiple times "
            "and export the per-run values to JSON for variability analysis."
        )
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Number of measurement runs to perform (default: 1)",
    )

    args = parser.parse_args()

    runs = max(args.runs, 1)
    values = []

    print(f"Starting variability experiment: {runs} run(s) of the same prompt")
    print(f"Model: {TEST_MODEL}")
    print(f"Prompt: {TEST_PROMPT}\n")

    for i in range(runs):
        current = i + 1
        print("=" * 60)
        print(f"Run {current}/{runs}")
        print("=" * 60)
        result = asyncio.run(measure_inference_power(add_benchmark=False))

        if result is not None:
            values.append(result)
        else:
            print("⚠️ Measurement failed for this run; skipping in JSON output")

    # Prepare JSON payload with only the electricity values plus a bit of context
    payload = {
        "metric": "wh_per_1000_tokens",
        "model": TEST_MODEL,
        "prompt": TEST_PROMPT,
        "requested_runs": runs,
        "successful_runs": len(values),
        "timestamp": time.time(),
        "values": values,
    }

    output_path = Path("benchmark_variability_runs.json")
    output_path.write_text(json.dumps(payload, indent=2))

    print("\n" + "=" * 60)
    print(f"Saved {len(values)} electricity values to: {output_path.resolve()}")
    print("You can now analyze the per-run variability from this JSON file.")
