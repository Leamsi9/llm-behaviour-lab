#!/usr/bin/env python3
"""
Energy Tracking Module for LLM Behavior Lab

Tracks energy consumption based on token usage with configurable benchmarks.
Provides real-time energy monitoring and cost analysis.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
import json
import os


@dataclass
class EnergyBenchmark:
    """Energy consumption benchmark configuration."""
    name: str
    watt_hours_per_1000_tokens: float
    description: str
    source: str = ""
    hardware_specs: Optional[Dict[str, Any]] = None


# Predefined energy benchmarks
ENERGY_BENCHMARKS = {
    "nvidia_a100": EnergyBenchmark(
        name="NVIDIA A100",
        watt_hours_per_1000_tokens=0.8,  # Conservative estimate
        description="NVIDIA A100 GPU (40GB)",
        source="Estimated based on typical LLM inference",
        hardware_specs={"gpu": "A100", "memory": "40GB", "tdp": "400W"}
    ),

    "nvidia_rtx4090": EnergyBenchmark(
        name="NVIDIA RTX 4090",
        watt_hours_per_1000_tokens=1.2,  # Higher due to gaming-optimized design
        description="NVIDIA RTX 4090 GPU",
        source="Measured consumer GPU performance",
        hardware_specs={"gpu": "RTX 4090", "memory": "24GB", "tdp": "450W"}
    ),

    "nvidia_rtx3080": EnergyBenchmark(
        name="NVIDIA RTX 3080",
        watt_hours_per_1000_tokens=1.5,
        description="NVIDIA RTX 3080 GPU",
        source="Typical consumer GPU benchmarks",
        hardware_specs={"gpu": "RTX 3080", "memory": "10GB", "tdp": "320W"}
    ),

    "cpu_baseline": EnergyBenchmark(
        name="CPU Baseline",
        watt_hours_per_1000_tokens=5.0,  # Much higher energy cost
        description="CPU-only inference",
        source="Estimated CPU vs GPU comparison",
        hardware_specs={"cpu": "Intel i9/AMD Ryzen 9", "no_gpu": True}
    ),

    "conservative_estimate": EnergyBenchmark(
        name="Conservative Estimate",
        watt_hours_per_1000_tokens=1.0,
        description="General conservative estimate",
        source="User specified baseline"
    ),

    "hp_255_g10_measured": EnergyBenchmark(
        name="HP 255 G10 (Measured)",
        watt_hours_per_1000_tokens=0.0311,
        description="HP 255 G10, Ryzen 7 7730U - Real RAPL measurement",
        source="Direct RAPL measurement during inference",
        hardware_specs={"cpu": "AMD Ryzen 7 7730U", "tdp": "15W", "measured_tokens_per_sec": "60"}
    )
}


@dataclass
class EnergyReading:
    """Single energy consumption measurement."""
    timestamp: datetime
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    benchmark_used: str
    watt_hours_consumed: float
    carbon_grams_co2: float
    latency_seconds: float
    model_name: str = ""
    strategy_name: str = ""


@dataclass
class EnergyTracker:
    """Tracks energy consumption across multiple readings."""

    benchmark: EnergyBenchmark = field(default_factory=lambda: ENERGY_BENCHMARKS["conservative_estimate"])
    carbon_intensity_gco2_per_kwh: float = 400  # Global average
    readings: List[EnergyReading] = field(default_factory=list)
    session_start: Optional[datetime] = None

    def __post_init__(self):
        if self.session_start is None:
            self.session_start = datetime.now()

    def set_benchmark(self, benchmark_name: str):
        """Set the energy benchmark to use."""
        if benchmark_name in ENERGY_BENCHMARKS:
            self.benchmark = ENERGY_BENCHMARKS[benchmark_name]
        else:
            raise ValueError(f"Unknown benchmark: {benchmark_name}")

    def record_usage(self, prompt_tokens: int, completion_tokens: int,
                    latency_seconds: float, model_name: str = "",
                    strategy_name: str = "") -> EnergyReading:
        """Record a single energy consumption measurement."""

        total_tokens = prompt_tokens + completion_tokens
        watt_hours = (total_tokens / 1000) * self.benchmark.watt_hours_per_1000_tokens
        carbon_grams = watt_hours * (self.carbon_intensity_gco2_per_kwh / 1000)

        reading = EnergyReading(
            timestamp=datetime.now(),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            benchmark_used=self.benchmark.name,
            watt_hours_consumed=watt_hours,
            carbon_grams_co2=carbon_grams,
            latency_seconds=latency_seconds,
            model_name=model_name,
            strategy_name=strategy_name
        )

        self.readings.append(reading)
        return reading

    def get_session_summary(self) -> Dict[str, Any]:
        """Get summary statistics for the current session."""
        if not self.readings:
            return {"error": "No readings recorded"}

        total_tokens = sum(r.total_tokens for r in self.readings)
        total_energy = sum(r.watt_hours_consumed for r in self.readings)
        total_carbon = sum(r.carbon_grams_co2 for r in self.readings)
        total_latency = sum(r.latency_seconds for r in self.readings)

        avg_energy_per_1000_tokens = total_energy / (total_tokens / 1000) if total_tokens > 0 else 0
        avg_latency_per_token = total_latency / total_tokens if total_tokens > 0 else 0

        # Convert readings to dicts with datetime objects converted to strings
        readings_dict = []
        for reading in self.readings[-10:]:  # Last 10 readings
            reading_dict = reading.__dict__.copy()
            if 'timestamp' in reading_dict and isinstance(reading_dict['timestamp'], datetime):
                reading_dict['timestamp'] = reading_dict['timestamp'].isoformat()
            readings_dict.append(reading_dict)

        return {
            "session_duration_seconds": (datetime.now() - self.session_start).total_seconds(),
            "total_readings": len(self.readings),
            "total_tokens": total_tokens,
            "total_energy_wh": round(total_energy, 4),
            "total_carbon_gco2": round(total_carbon, 2),
            "average_energy_per_1000_tokens": round(avg_energy_per_1000_tokens, 4),
            "average_latency_per_token": round(avg_latency_per_token, 6),
            "benchmark_used": self.benchmark.name,
            "readings": readings_dict
        }

    def export_readings(self, filepath: str):
        """Export all readings to a JSON file."""
        # Convert readings to dicts with datetime objects converted to strings
        readings_dict = []
        for reading in self.readings:
            reading_dict = reading.__dict__.copy()
            if 'timestamp' in reading_dict and isinstance(reading_dict['timestamp'], datetime):
                reading_dict['timestamp'] = reading_dict['timestamp'].isoformat()
            readings_dict.append(reading_dict)

        data = {
            "benchmark": self.benchmark.__dict__,
            "session_start": self.session_start.isoformat(),
            "readings": readings_dict
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def recalculate_with_benchmark(self, benchmark_name: str) -> Dict[str, Any]:
        """Recalculate all readings using a different benchmark."""
        if benchmark_name not in ENERGY_BENCHMARKS:
            raise ValueError(f"Unknown benchmark: {benchmark_name}")

        new_benchmark = ENERGY_BENCHMARKS[benchmark_name]

        # Recalculate each reading
        recalculated_readings = []
        for reading in self.readings:
            # Recalculate energy consumption with new benchmark
            watt_hours = (reading.total_tokens / 1000) * new_benchmark.watt_hours_per_1000_tokens
            carbon_grams = watt_hours * (self.carbon_intensity_gco2_per_kwh / 1000)

            recalculated_reading = reading.__dict__.copy()
            recalculated_reading.update({
                "benchmark_used": new_benchmark.name,
                "watt_hours_consumed": watt_hours,
                "carbon_grams_co2": carbon_grams,
            })
            recalculated_readings.append(recalculated_reading)

        # Calculate new session summary
        total_tokens = sum(r["total_tokens"] for r in recalculated_readings)
        total_energy = sum(r["watt_hours_consumed"] for r in recalculated_readings)
        total_carbon = sum(r["carbon_grams_co2"] for r in recalculated_readings)
        total_latency = sum(r["latency_seconds"] for r in recalculated_readings)

        avg_energy_per_1000_tokens = total_energy / (total_tokens / 1000) if total_tokens > 0 else 0
        avg_latency_per_token = total_latency / total_tokens if total_tokens > 0 else 0

        return {
            "benchmark_used": new_benchmark.name,
            "total_readings": len(recalculated_readings),
            "total_tokens": total_tokens,
            "total_energy_wh": round(total_energy, 4),
            "total_carbon_gco2": round(total_carbon, 2),
            "average_energy_per_1000_tokens": round(avg_energy_per_1000_tokens, 4),
            "average_latency_per_token": round(avg_latency_per_token, 6),
            "readings": recalculated_readings[-10:]  # Last 10 recalculated readings
        }

    def add_custom_benchmark(self, name: str, description: str, watt_hours_per_1000_tokens: float,
                           source: str = "Custom", hardware_specs: str = "User defined", force_update: bool = False) -> bool:
        """Add or update a custom energy benchmark."""
        if name in ENERGY_BENCHMARKS and not force_update:
            return False  # Benchmark already exists and no update requested

        ENERGY_BENCHMARKS[name] = EnergyBenchmark(
            name=name,
            description=description,
            watt_hours_per_1000_tokens=watt_hours_per_1000_tokens,
            source=source,
            hardware_specs=hardware_specs
        )
        return True


# Global energy tracker instance
energy_tracker = EnergyTracker()


def estimate_energy_impact(tokens: int, benchmark_name: str = "conservative_estimate") -> Dict[str, float]:
    """
    Estimate energy impact for a given token count.

    Args:
        tokens: Number of tokens to process
        benchmark_name: Name of energy benchmark to use

    Returns:
        Dict with energy consumption estimates
    """
    if benchmark_name not in ENERGY_BENCHMARKS:
        benchmark_name = "conservative_estimate"

    benchmark = ENERGY_BENCHMARKS[benchmark_name]
    watt_hours = (tokens / 1000) * benchmark.watt_hours_per_1000_tokens
    carbon_grams = watt_hours * (400 / 1000)  # Global average carbon intensity

    return {
        "tokens": tokens,
        "benchmark": benchmark_name,
        "watt_hours": round(watt_hours, 4),
        "carbon_grams_co2": round(carbon_grams, 2),
        "energy_per_1000_tokens": benchmark.watt_hours_per_1000_tokens
    }


def get_available_benchmarks() -> List[Dict[str, Any]]:
    """Get list of available energy benchmarks."""
    return [
        {
            "name": name,
            "description": benchmark.description,
            "watt_hours_per_1000_tokens": benchmark.watt_hours_per_1000_tokens,
            "source": benchmark.source,
            "hardware_specs": benchmark.hardware_specs
        }
        for name, benchmark in ENERGY_BENCHMARKS.items()
    ]
