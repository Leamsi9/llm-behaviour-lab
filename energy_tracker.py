#!/usr/bin/env python3
"""
Energy Tracking Module for LLM Behavior Lab

Tracks energy consumption based on token usage with configurable benchmarks.
Provides real-time energy monitoring and cost analysis.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import json
import os


@dataclass
class EnergyBenchmark:
    """Energy consumption benchmark configuration."""
    name: str
    watt_hours_per_1000_tokens: float  # Legacy/Combined metric (for backward compatibility or simple view)
    description: str
    input_wh_per_1000_tokens: float = 0.0  # Energy cost for prefill (prompt processing)
    output_wh_per_1000_tokens: float = 0.0  # Energy cost for decode (generation)
    source: str = ""
    hardware_specs: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        # If split values aren't provided, estimate them from the combined value
        # Typically input processing is cheaper per token than generation, but highly parallel.
        # For simplicity in migration, if 0, we'll derive them.
        # A common heuristic: Input is ~10-20% of Output cost per token in low-batch settings,
        # but let's default to a 50/50 split if unknown, or use the provided total.
        if self.input_wh_per_1000_tokens == 0 and self.output_wh_per_1000_tokens == 0:
             self.output_wh_per_1000_tokens = self.watt_hours_per_1000_tokens
             self.input_wh_per_1000_tokens = self.watt_hours_per_1000_tokens * 0.25 # Estimate input as 25% of output cost


# Predefined energy benchmarks
ENERGY_BENCHMARKS = {
    "nvidia_a100": EnergyBenchmark(
        name="NVIDIA A100",
        watt_hours_per_1000_tokens=0.8,
        input_wh_per_1000_tokens=0.2,
        output_wh_per_1000_tokens=0.8,
        description="NVIDIA A100 GPU (40GB)",
        source="Estimated based on typical LLM inference",
        hardware_specs={"gpu": "A100", "memory": "40GB", "tdp": "400W"}
    ),

    "nvidia_rtx4090": EnergyBenchmark(
        name="NVIDIA RTX 4090",
        watt_hours_per_1000_tokens=1.2,
        input_wh_per_1000_tokens=0.3,
        output_wh_per_1000_tokens=1.2,
        description="NVIDIA RTX 4090 GPU",
        source="Measured consumer GPU performance",
        hardware_specs={"gpu": "RTX 4090", "memory": "24GB", "tdp": "450W"}
    ),

    "nvidia_rtx3080": EnergyBenchmark(
        name="NVIDIA RTX 3080",
        watt_hours_per_1000_tokens=1.5,
        input_wh_per_1000_tokens=0.4,
        output_wh_per_1000_tokens=1.5,
        description="NVIDIA RTX 3080 GPU",
        source="Typical consumer GPU benchmarks",
        hardware_specs={"gpu": "RTX 3080", "memory": "10GB", "tdp": "320W"}
    ),

    "cpu_baseline": EnergyBenchmark(
        name="CPU Baseline",
        watt_hours_per_1000_tokens=5.0,
        input_wh_per_1000_tokens=2.0,
        output_wh_per_1000_tokens=5.0,
        description="CPU-only inference",
        source="Estimated CPU vs GPU comparison",
        hardware_specs={"cpu": "Intel i9/AMD Ryzen 9", "no_gpu": True}
    ),

    "conservative_estimate": EnergyBenchmark(
        name="Conservative Estimate",
        watt_hours_per_1000_tokens=1.0,
        input_wh_per_1000_tokens=0.25,
        output_wh_per_1000_tokens=1.0,
        description="General conservative estimate",
        source="User specified baseline"
    ),

    "hp_255_g10_measured": EnergyBenchmark(
        name="HP 255 G10 (Measured)",
        watt_hours_per_1000_tokens=0.0311,
        input_wh_per_1000_tokens=0.01, # Estimated
        output_wh_per_1000_tokens=0.0311,
        description="HP 255 G10, Ryzen 7 7730U - Real RAPL measurement",
        source="Direct RAPL measurement during inference",
        hardware_specs={"cpu": "AMD Ryzen 7 7730U", "tdp": "15W", "measured_tokens_per_sec": "60"}
    )
}


# Optional: Hugging Face AI Energy Score benchmarks (NVIDIA H100, text_generation_and_reasoning)
_HF_BENCHMARK_PATH = Path("data/benchmark_data/hugging_face.json")
_HF_BENCHMARK_CACHE: Optional[Dict[str, Any]] = None


def _load_hf_benchmark_data() -> Optional[Dict[str, Any]]:
    """Load raw Hugging Face benchmark JSON once and cache it.

    Returns the parsed JSON dict, or None if the file is missing/invalid.
    """
    global _HF_BENCHMARK_CACHE

    if _HF_BENCHMARK_CACHE is not None:
        return _HF_BENCHMARK_CACHE

    try:
        if not _HF_BENCHMARK_PATH.exists():
            _HF_BENCHMARK_CACHE = None
            return None

        with _HF_BENCHMARK_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, dict):
            _HF_BENCHMARK_CACHE = None
            return None

        _HF_BENCHMARK_CACHE = data
        return data
    except Exception:
        _HF_BENCHMARK_CACHE = None
        return None


def get_hf_benchmarks() -> Dict[str, Any]:
    """Return Hugging Face AI Energy Score benchmarks in a UI-friendly format.

    The structure matches the JSON in benchmark_data/hugging_face.json but
    normalises model entries to a minimal, typed subset:

      {
        "task": str | None,
        "input_tokens_per_1000_queries": int | None,
        "wh_per_1000_tokens_factor": float | None,
        "energy_unit": str | None,
        "models": [
          {
            "model_id": str,
            "class": str | None,
            "total_gpu_energy_per_query_wh": float | None,
            "wh_per_1000_queries": float | None,
            "wh_per_1000_input_etokens": float | None,
          },
          ...
        ]
      }

    This is intentionally kept separate from ENERGY_BENCHMARKS because the
    Hugging Face metrics are defined per 1000 *input* e-tokens (etokens-i),
    whereas ENERGY_BENCHMARKS use a split input/output view tied to our
    internal EnergyBenchmark model.
    """
    raw = _load_hf_benchmark_data() or {}

    models: List[Dict[str, Any]] = []
    for m in raw.get("models", []) or []:
        if not isinstance(m, dict):
            continue
        model_id = m.get("model_id")
        if not model_id:
            continue

        models.append({
            "model_id": model_id,
            "class": m.get("class"),
            "total_gpu_energy_per_query_wh": m.get("total_gpu_energy_per_query_wh"),
            "wh_per_1000_queries": m.get("wh_per_1000_queries"),
            # This is the key metric used for impact calculations (e-tokens-i)
            "wh_per_1000_input_etokens": m.get("wh_per_1000_input_etokens"),
        })

    return {
        "task": raw.get("task"),
        "input_tokens_per_1000_queries": raw.get("input_tokens_per_1000_queries"),
        "wh_per_1000_tokens_factor": raw.get("wh_per_1000_tokens_factor"),
        "energy_unit": raw.get("energy_unit"),
        "models": models,
    }


# Keys for legacy built-in baselines that should not appear in the public
# benchmark lists (we want HF e-token baselines + measured/custom profiles
# instead of these fixed hardware presets in the UI).
_LEGACY_BASELINE_KEYS = {
    "nvidia_a100",
    "nvidia_rtx4090",
    "nvidia_rtx3080",
    "cpu_baseline",
    "conservative_estimate",
    "hp_255_g10_measured",
}


def get_benchmark_sources() -> Dict[str, Any]:
    """Discover external benchmark JSON sources under 'data/benchmark_data/'.

    Each JSON file is expected to have at least:

      {
        "data_source": str,              # e.g. "hugging face", "jegham_et_al"
        "task": str | None,
        "energy_unit": str | None,
        "models": [
          {
            "model_id": str,
            "wh_per_1000_input_etokens": float | None,
            "wh_per_1000_output_etokens": float | None,
          },
          ...
        ]
      }

    The return value is a dict keyed by data_source for convenient lookup on the
    frontend.
    """
    sources: Dict[str, Any] = {}

    data_dir = _HF_BENCHMARK_PATH.parent
    if not data_dir.exists():
        return sources

    for path in sorted(data_dir.glob("*.json")):
        try:
            with path.open("r", encoding="utf-8") as f:
                raw = json.load(f)
        except Exception:
            continue

        if not isinstance(raw, dict):
            continue

        src_id = raw.get("data_source") or path.stem
        models_raw = raw.get("models") or []
        if not isinstance(models_raw, list):
            models_raw = []

        models_norm: List[Dict[str, Any]] = []
        for m in models_raw:
            if not isinstance(m, dict):
                continue
            model_id = m.get("model_id")
            if not model_id:
                continue

            models_norm.append({
                "model_id": model_id,
                "wh_per_1000_input_etokens": m.get("wh_per_1000_input_etokens"),
                "wh_per_1000_output_etokens": m.get("wh_per_1000_output_etokens"),
            })

        sources[src_id] = {
            "data_source": src_id,
            "file": str(path.name),
            "task": raw.get("task"),
            "energy_unit": raw.get("energy_unit"),
            "models": models_norm,
        }

    return sources


def ensure_hf_energy_benchmarks_registered() -> None:
    """Materialize Hugging Face benchmarks as ENERGY_BENCHMARKS entries.

    Each model_id becomes a benchmark key. We use wh_per_1000_input_etokens as
    the input-side coefficient and 0 for the output-side cost, so that total
    energy for a run is normalised to input e-tokens (etokens-i):

        E_run ≈ (prompt_tokens / 1000) * wh_per_1000_input_etokens

    This baseline is then used by recalculate_with_benchmark and
    /api/switch-benchmark when the user selects an HF-derived benchmark.
    """
    data = get_hf_benchmarks()
    models = data.get("models") or []

    for m in models:
        if not isinstance(m, dict):
            continue
        model_id = m.get("model_id")
        wh_in = m.get("wh_per_1000_input_etokens")
        if not model_id or not isinstance(wh_in, (int, float)):
            continue

        # Use the raw model_id as the benchmark key so the UI can refer to it
        # directly when switching/recalculating.
        if model_id in ENERGY_BENCHMARKS:
            continue

        ENERGY_BENCHMARKS[model_id] = EnergyBenchmark(
            name=model_id,
            description=(
                "Hugging Face AI Energy Score (H100, text_generation_and_reasoning) "
                "– e-tokens input baseline"
            ),
            watt_hours_per_1000_tokens=wh_in,
            input_wh_per_1000_tokens=wh_in,
            output_wh_per_1000_tokens=0.0,
            # Use a short, stable source ID so the frontend can filter by
            # benchmark source (matches 'data_source' in hugging_face.json).
            source="hugging face",
            hardware_specs={
                "gpu": "NVIDIA H100 80GB",
                "task": data.get("task"),
                "wh_per_1000_queries": m.get("wh_per_1000_queries"),
            },
        )


def ensure_external_energy_benchmarks_registered() -> None:
    """Materialize all external benchmark sources as ENERGY_BENCHMARKS entries.

    - Hugging Face AI Energy Score baselines are registered via
      ensure_hf_energy_benchmarks_registered().
    - Additional sources (e.g. jegham_et_al) are registered from
      get_benchmark_sources(), using wh_per_1000_input_etokens as input-side
      e-token baselines and 0 for output-side cost.
    """
    ensure_hf_energy_benchmarks_registered()

    sources = get_benchmark_sources()
    for source_id, source in sources.items():
        # Hugging Face is already handled by ensure_hf_energy_benchmarks_registered
        if isinstance(source_id, str) and source_id.lower().startswith("hugging"):
            continue

        models = source.get("models") or []
        for m in models:
            if not isinstance(m, dict):
                continue
            model_id = m.get("model_id")
            wh_in = m.get("wh_per_1000_input_etokens")
            if not model_id or not isinstance(wh_in, (int, float)):
                continue
            if model_id in ENERGY_BENCHMARKS:
                continue

            ENERGY_BENCHMARKS[model_id] = EnergyBenchmark(
                name=model_id,
                description=f"{source_id} – e-tokens input baseline",
                watt_hours_per_1000_tokens=wh_in,
                input_wh_per_1000_tokens=wh_in,
                output_wh_per_1000_tokens=0.0,
                source=source_id,
                hardware_specs={
                    "task": source.get("task"),
                    "energy_unit": source.get("energy_unit"),
                },
            )


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
    rapl_measured_wh: float = 0.0
    rapl_measured_tokens: int = 0

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
        
        # Calculate energy using split coefficients: Input Energy + Output Energy
        input_energy = (prompt_tokens / 1000) * self.benchmark.input_wh_per_1000_tokens
        output_energy = (completion_tokens / 1000) * self.benchmark.output_wh_per_1000_tokens
        
        watt_hours = input_energy + output_energy
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
        total_output_tokens = sum(r.completion_tokens for r in self.readings)
        total_input_tokens = sum(r.prompt_tokens for r in self.readings)
        total_energy = sum(r.watt_hours_consumed for r in self.readings)
        total_carbon = sum(r.carbon_grams_co2 for r in self.readings)
        total_latency = sum(r.latency_seconds for r in self.readings)

        # Average energy is reported per 1000 *effective* tokens.
        # For backward compatibility, average_energy_per_1000_tokens remains
        # defined per 1000 output tokens ("etokens-o") and a new
        # average_energy_per_1000_input_tokens field exposes the
        # input-normalized view ("etokens-i").
        avg_energy_per_1000_tokens = (
            total_energy / (total_output_tokens / 1000)
            if total_output_tokens > 0
            else 0
        )
        avg_energy_per_1000_input_tokens = (
            total_energy / (total_input_tokens / 1000)
            if total_input_tokens > 0
            else 0
        )
        avg_latency_per_token = total_latency / total_tokens if total_tokens > 0 else 0

        # Convert readings to dicts with datetime objects converted to strings
        readings_dict = []
        for reading in self.readings[-10:]:  # Last 10 readings
            reading_dict = reading.__dict__.copy()
            if 'timestamp' in reading_dict and isinstance(reading_dict['timestamp'], datetime):
                reading_dict['timestamp'] = reading_dict['timestamp'].isoformat()
            readings_dict.append(reading_dict)

        rapl_avg_wh_per_1k = 0.0
        if self.rapl_measured_tokens > 0:
            rapl_avg_wh_per_1k = (self.rapl_measured_wh / (self.rapl_measured_tokens / 1000))

        return {
            "session_duration_seconds": (datetime.now() - self.session_start).total_seconds(),
            "total_readings": len(self.readings),
            "total_tokens": total_tokens,
            "total_energy_wh": round(total_energy, 4),
            "total_carbon_gco2": round(total_carbon, 2),
            # Output-normalized (per 1000 etokens-o, kept for compatibility)
            "average_energy_per_1000_tokens": round(avg_energy_per_1000_tokens, 4),
            # Input-normalized view (per 1000 etokens-i)
            "average_energy_per_1000_input_tokens": round(avg_energy_per_1000_input_tokens, 4),
            "average_latency_per_token": round(avg_latency_per_token, 6),
            "benchmark_used": self.benchmark.name,
            "readings": readings_dict,
            "rapl_session": {
                "measured_wh": round(self.rapl_measured_wh, 6),
                "measured_tokens": self.rapl_measured_tokens,
                # RAPL session average per 1000 etokens-o (alias keeps original field name)
                "measured_wh_per_1000_tokens": round(rapl_avg_wh_per_1k, 6) if rapl_avg_wh_per_1k else 0.0,
                "measured_wh_per_1000_etokens_o": round(rapl_avg_wh_per_1k, 6) if rapl_avg_wh_per_1k else 0.0,
            },
        }

    def record_rapl_measurement(self, watt_hours: float, total_tokens: int) -> None:
        """Record a RAPL-measured energy value for the session."""
        if watt_hours <= 0 or total_tokens <= 0:
            return
        self.rapl_measured_wh += watt_hours
        self.rapl_measured_tokens += total_tokens

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
            # Recalculate energy consumption with new benchmark using split coefficients
            input_energy = (reading.prompt_tokens / 1000) * new_benchmark.input_wh_per_1000_tokens
            output_energy = (reading.completion_tokens / 1000) * new_benchmark.output_wh_per_1000_tokens
            
            watt_hours = input_energy + output_energy
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
        total_output_tokens = sum(r["completion_tokens"] for r in recalculated_readings)
        total_input_tokens = sum(r["prompt_tokens"] for r in recalculated_readings)
        total_energy = sum(r["watt_hours_consumed"] for r in recalculated_readings)
        total_carbon = sum(r["carbon_grams_co2"] for r in recalculated_readings)
        total_latency = sum(r["latency_seconds"] for r in recalculated_readings)

        avg_energy_per_1000_tokens = (
            total_energy / (total_output_tokens / 1000)
            if total_output_tokens > 0
            else 0
        )
        avg_energy_per_1000_input_tokens = (
            total_energy / (total_input_tokens / 1000)
            if total_input_tokens > 0
            else 0
        )
        avg_latency_per_token = total_latency / total_tokens if total_tokens > 0 else 0

        return {
            "benchmark_used": new_benchmark.name,
            "total_readings": len(recalculated_readings),
            "total_tokens": total_tokens,
            "total_energy_wh": round(total_energy, 4),
            "total_carbon_gco2": round(total_carbon, 2),
            # Output-normalized (per 1000 etokens-o, kept for compatibility)
            "average_energy_per_1000_tokens": round(avg_energy_per_1000_tokens, 4),
            # Input-normalized (per 1000 etokens-i)
            "average_energy_per_1000_input_tokens": round(avg_energy_per_1000_input_tokens, 4),
            "average_latency_per_token": round(avg_latency_per_token, 6),
            "readings": recalculated_readings[-10:]  # Last 10 recalculated readings
        }

    def add_custom_benchmark(self, name: str, description: str, watt_hours_per_1000_tokens: float,
                           input_wh_per_1000_tokens: float = 0.0, output_wh_per_1000_tokens: float = 0.0,
                           source: str = "Custom", hardware_specs: str = "User defined", force_update: bool = False) -> bool:
        """Add or update a custom energy benchmark."""
        if name in ENERGY_BENCHMARKS and not force_update:
            return False  # Benchmark already exists and no update requested

        # Auto-estimate if split not provided
        if input_wh_per_1000_tokens == 0 and output_wh_per_1000_tokens == 0:
            output_wh_per_1000_tokens = watt_hours_per_1000_tokens
            input_wh_per_1000_tokens = watt_hours_per_1000_tokens * 0.25

        ENERGY_BENCHMARKS[name] = EnergyBenchmark(
            name=name,
            description=description,
            watt_hours_per_1000_tokens=watt_hours_per_1000_tokens,
            input_wh_per_1000_tokens=input_wh_per_1000_tokens,
            output_wh_per_1000_tokens=output_wh_per_1000_tokens,
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
    
    # Simple estimation assuming 1:1 input/output ratio if only total tokens provided
    # Or just use output cost as a conservative upper bound for everything?
    # Better: Assume a typical ratio if we don't know the split.
    # But this function takes 'tokens' as a single int.
    # Let's assume these are output tokens for consistency with old behavior, 
    # OR assume a 50/50 split. 
    # Actually, for "impact of X tokens", we should probably use the output cost as it's the dominant factor usually.
    watt_hours = (tokens / 1000) * benchmark.output_wh_per_1000_tokens
    carbon_grams = watt_hours * (400 / 1000)  # Global average carbon intensity

    return {
        "tokens": tokens,
        "benchmark": benchmark_name,
        "watt_hours": round(watt_hours, 4),
        "carbon_grams_co2": round(carbon_grams, 2),
        "energy_per_1000_tokens": benchmark.output_wh_per_1000_tokens
    }


def get_available_benchmarks() -> List[Dict[str, Any]]:
    """Get list of energy benchmarks exposed to the UI.

    Includes:
      - Hugging Face AI Energy Score baselines (model_id-keyed, input e-tokens)
      - RAPL-calibrated benchmarks
      - User custom benchmarks

    Filters out the original fixed hardware presets so that UI baselines are
    expressed directly in terms of e-token (input) or measured profiles.
    """
    ensure_external_energy_benchmarks_registered()

    return [
        {
            "name": name,
            "description": benchmark.description,
            "watt_hours_per_1000_tokens": benchmark.watt_hours_per_1000_tokens,
            "input_wh_per_1000_tokens": benchmark.input_wh_per_1000_tokens,
            "output_wh_per_1000_tokens": benchmark.output_wh_per_1000_tokens,
            "source": benchmark.source,
            "hardware_specs": benchmark.hardware_specs,
        }
        for name, benchmark in ENERGY_BENCHMARKS.items()
        if name not in _LEGACY_BASELINE_KEYS
    ]
