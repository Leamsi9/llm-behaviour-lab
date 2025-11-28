#!/usr/bin/env python3
"""
Create a custom energy benchmark based on real power measurements
Measures your system during LLM inference and calculates actual Wh/1000 tokens
"""

import time
import asyncio
import httpx
from power_monitor import power_monitor
from energy_tracker import energy_tracker

OLLAMA_BASE_URL = "http://localhost:11434"


async def measure_inference_power():
    """
    Measure actual power consumption during LLM inference
    Returns average Wh per 1000 tokens
    """
    print("🔋 Creating Custom Benchmark for HP 255 G10 (Ryzen 7 7730U)")
    print("=" * 60)
    
    # Check RAPL availability
    info = power_monitor.get_info()
    if not info['available']:
        print("❌ RAPL not available!")
        print("\nPlease run:")
        print("  sudo chmod -R a+r /sys/class/powercap/intel-rapl/")
        return None
    
    print(f"✅ RAPL Available")
    print(f"📊 Power zones: {', '.join(info['zones'])}\n")
    
    # Run a test inference
    test_prompt = "Explain how transformers work in machine learning in 3 sentences."
    model = "qwen3:0.6b"  # Small fast model for testing
    
    print(f"Running test inference...")
    print(f"Model: {model}")
    print(f"Prompt: {test_prompt}\n")
    
    # Measure baseline power (idle)
    print("Measuring idle power...")
    baseline = power_monitor.read_power(2.0)
    if baseline:
        print(f"  Idle power: {baseline.total_watts:.2f} W\n")
    
    # Run inference with power measurement
    print("Running LLM inference with power monitoring...")
    
    # Take power reading before inference
    power_before = power_monitor.read_power(1.0)
    
    start_time = time.time()
    total_tokens = 0
    
    # Run inference
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json={
                    "model": model,
                    "prompt": test_prompt,
                    "stream": False
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                total_tokens = result.get("prompt_eval_count", 0) + result.get("eval_count", 0)
                print(f"✅ Generated {total_tokens} tokens")
            else:
                print(f"❌ Ollama error: {response.status_code}")
                print(f"Response: {response.text}")
                return None
                
    except Exception as e:
        print(f"❌ Error connecting to Ollama: {e}")
        print("\nMake sure Ollama is running:")
        print("  ollama serve")
        return None
    
    end_time = time.time()
    duration = end_time - start_time
    
    # Take power reading after inference
    power_after = power_monitor.read_power(1.0)
    
    
    # Calculate average power during inference
    if power_before and power_after and total_tokens > 0:
        avg_power = (power_before.total_watts + power_after.total_watts) / 2
        idle_power = baseline.total_watts if baseline else 3.0  # Assume 3W idle
        active_power = max(avg_power - idle_power, 0.5)  # At least 0.5W for inference
        
        # Calculate energy consumed
        wh_consumed = (active_power * duration) / 3600
        wh_per_1000_tokens = (wh_consumed / total_tokens) * 1000
        
        print(f"\n{'='*60}")
        print(f"✅ Measurement Complete!")
        print(f"{'='*60}")
        print(f"Duration: {duration:.2f}s")
        print(f"Total tokens: {total_tokens}")
        print(f"Tokens/sec: {total_tokens/duration:.1f}")
        print(f"\nPower Consumption:")
        print(f"  Average during inference: {avg_power:.2f} W")
        print(f"  Idle baseline: {idle_power:.2f} W")
        print(f"  Active (inference only): {active_power:.2f} W")
        print(f"\nEnergy Efficiency:")
        print(f"  Total energy: {wh_consumed:.6f} Wh")
        print(f"  🎯 Wh per 1000 tokens: {wh_per_1000_tokens:.4f}")
        print(f"{'='*60}\n")
        
        # Add as custom benchmark
        benchmark_name = "hp_255_g10_ryzen7_7730u_measured"
        success = energy_tracker.add_custom_benchmark(
            name=benchmark_name,
            description=f"Real measurement on HP 255 G10, Ryzen 7 7730U (measured: {wh_per_1000_tokens:.4f} Wh/1K)",
            watt_hours_per_1000_tokens=wh_per_1000_tokens,
            source="Direct RAPL measurement during inference",
            hardware_specs=f"AMD Ryzen 7 7730U, 15W TDP, measured at {total_tokens/duration:.0f} tok/s"
        )
        
        if success:
            print(f"✅ Added custom benchmark: '{benchmark_name}'")
            print(f"\nYou can now select this benchmark in the Energy Lab UI!")
        else:
            print(f"⚠️  Benchmark '{benchmark_name}' already exists")
        
        return wh_per_1000_tokens
    else:
        print("\n❌ No power readings captured or no tokens generated")
        return None


if __name__ == "__main__":
    result = asyncio.run(measure_inference_power())
    
    if result:
        print(f"\n🎉 Your HP 255 G10 uses {result:.4f} Wh per 1000 tokens!")
        print("\nThis is much more efficient than data center GPUs! 🌱")
