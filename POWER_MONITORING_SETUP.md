# Real-time Power Monitoring Integration Guide

This document explains how to enable and use real-time power monitoring on your HP 255 G10 Notebook PC.

## Your System

**CPU**: AMD Ryzen 7 7730U with Radeon Graphics (15W TDP)
**RAPL Status**: ✅ Available on your system
**Power Zones Detected**: `package-0`, `core`

## Quick Setup

### 1. Enable RAPL Access (One-time setup)

The RAPL energy counters require read permissions. Run this once:

```bash
# Allow reading RAPL energy counters
sudo chmod -R a+r /sys/class/powercap/intel-rapl/
```

To make this permanent across reboots, create a udev rule:

```bash
# Create udev rule
sudo tee /etc/udev/rules.d/99-rapl.rules << EOF
SUBSYSTEM=="powercap", KERNEL=="intel-rapl:*", RUN+="/bin/chmod -R a+r /sys/class/powercap/intel-rapl/"
EOF

# Reload udev
sudo udevadm control --reload-rules
sudo udevadm trigger
```

### 2. Test Real Power Monitoring

```bash
cd /home/ismael/Github/llm-behaviour-lab
python3 power_monitor.py
```

You should see actual wattage readings like:
```
Power Reading:
  Package:  8.50 W
  Cores:    5.20 W
  TOTAL:    13.70 W
```

### 3. Create Your Custom Benchmark

Once power monitoring works, run this to create a benchmark based on YOUR actual hardware:

```bash
# Run an LLM query and measure actual power
python3 create_custom_benchmark.py
```

This will:
- Measure your system's actual power consumption during LLM inference
- Calculate real Wh/1000 tokens for your specific hardware
- Add it as a custom benchmark: "HP_255_G10_Ryzen7_7730U"

## How It Works

### Power Measurement Process

1. **RAPL (Running Average Power Limit)**: Your AMD CPU exposes energy counters
2. **Energy Counters**: Track microjoules consumed by CPU package and cores
3. **Delta Measurement**: Read counter before/after inference, calculate watts
4. **Token Correlation**: Divide energy by tokens generated = Wh/1000 tokens

### What Gets Measured

- **Package Power**: Total CPU package (cores + integrated GPU + memory controller)
- **Core Power**: Just the CPU cores
- **Estimated GPU Power**: Integrated Radeon graphics power (part of package)

For Ollama running on CPU, this captures the real energy cost!

## Integration Options

### Option 1: Manual Benchmark (Simplest)

Run a test manually, then add the measured value:

```python
# After measuring, add custom benchmark in UI:
# Name: HP_255_G10_Measured
# Wh/1000 tokens: <your measured value>
# Description: Real measurement on HP 255 G10, Ryzen 7 7730U
```

### Option 2: Automatic Live Measurement (Advanced)

Modify `app_energy.py` to measure power during each test:

```python
from power_monitor import power_monitor

# Before LLM inference
power_before = power_monitor.read_power(0.1)

# ... run LLM ...

# After LLM inference  
power_after = power_monitor.read_power(0.1)
actual_watts = (power_before.total_watts + power_after.total_watts) / 2
```

### Option 3: Process-Specific Tracking

Track just the Ollama process power:

```python
from power_monitor import process_monitor
import subprocess

# Get Ollama PID
pid = int(subprocess.check_output(['pgrep', '-f', 'ollama']).strip())

# Measure power used by Ollama during inference
estimated_watts = process_monitor.measure_process_power(pid, duration_seconds=2.0)
```

## Expected Readings

For your AMD Ryzen 7 7730U (15W TDP):

- **Idle**: 2-4W total system power
- **Light Inference**: 8-12W during LLM query processing
- **Heavy Inference**: 12-18W during intensive generation

**Estimated benchmarks**:
- CPU-only small models (2B-7B params): 3-6 Wh/1000 tokens
- CPU-only large models (13B+ params): 6-12 Wh/1000 tokens

These are MUCH better than data center GPUs due to your CPU's power efficiency!

## Troubleshooting

### "Permission denied" when reading energy_uj files
→ Run the chmod command from step 1

### All readings show 0.00 W
→ RAPL might be disabled in BIOS, or permissions not set

### Readings seem too low
→ AMD RAPL sometimes underreports, multiply by ~1.5x correction factor

### Want even more accuracy?
→ Use `turbostat` (already installed on your system):
```bash
sudo turbostat --Summary --quiet --show PkgWatt,CorWatt --interval 1
```

## Next Steps

1. Run `sudo chmod -R a+r /sys/class/powercap/intel-rapl/`
2. Test with `python3 power_monitor.py`
3. If readings work, I'll integrate automatic power measurement into the energy lab!
