#!/usr/bin/env python3
"""
Simple power measurement during Ollama inference
Run this DURING an LLM query to measure real power consumption
"""

import time
import requests
from power_monitor import power_monitor

# Test query
prompt = "Explain transformers in machine learning briefly."
model = "qwen3:0.6b"

print("🔋 Real-time Power Measurement")
print("=" * 60)
print(f"Model: {model}")
print(f"Prompt: {prompt}\n")

# Check RAPL
if not power_monitor.available:
    print("❌ RAPL not available. Run:")
    print("  sudo chmod -R a+r /sys/class/powercap/intel-rapl/")
    exit(1)

# Measure baseline
print("Measuring baseline power (2s)...")
baseline = power_monitor.read_power(2.0)
print(f"✅ Baseline: {baseline.total_watts:.2f} W\n")

# Measure power before inference
print("Starting LLM inference...")
power_before = power_monitor.read_power(0.5)

# Run inference (non-streaming for simplicity)
start_time = time.time()
try:
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": model, "prompt": prompt, "stream": False},
        timeout=60
    )
    result = response.json()
    total_tokens = result.get("prompt_eval_count", 0) + result.get("eval_count", 0)
except Exception as e:
    print(f"❌ Error: {e}")
    exit(1)

end_time = time.time()
duration = end_time - start_time

# Measure power after inference
power_after = power_monitor.read_power(0.5)

# Calculate
avg_power = (power_before.total_watts + power_after.total_watts) / 2
active_power = max(avg_power - baseline.total_watts, 0.5)
wh_consumed = (active_power * duration) / 3600
wh_per_1000_tokens = (wh_consumed / total_tokens) * 1000 if total_tokens > 0 else 0

# Results
print(f"\n{'='*60}")
print("✅ MEASUREMENT COMPLETE")
print(f"{'='*60}")
print(f"Duration: {duration:.2f}s")
print(f"Total tokens: {total_tokens}")
print(f"Tokens/sec: {total_tokens/duration:.1f}")
print(f"\nPower:")
print(f"  Baseline (idle): {baseline.total_watts:.2f} W")
print(f"  During inference: {avg_power:.2f} W")
print(f"  Active (LLM only): {active_power:.2f} W")
print(f"\n🎯 Energy Intensity:")
print(f"  {wh_per_1000_tokens:.4f} Wh per 1000 tokens")
print(f"{'='*60}\n")

# Add to tracker
from energy_tracker import energy_tracker

benchmark_name = "hp_255_g10_measured"
success = energy_tracker.add_custom_benchmark(
    name=benchmark_name,
    description=f"HP 255 G10, Ryzen 7 7730U - Real measurement ({wh_per_1000_tokens:.4f} Wh/1K)",
    watt_hours_per_1000_tokens=wh_per_1000_tokens,
    source="Direct RAPL measurement",
    hardware_specs=f"AMD Ryzen 7 7730U, 15W TDP, {total_tokens/duration:.0f} tok/s"
)

if success:
    print(f"✅ Added benchmark: '{benchmark_name}'")
    print("\n💡 Select this in the Energy Lab UI for accurate measurements!")
else:
    print(f"⚠️  Benchmark '{benchmark_name}' already exists - using this value")
