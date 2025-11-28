#!/usr/bin/env python3
"""Simple helper to analyze variability of benchmark runs.

Reads ``benchmark_variability_runs.json`` (or a path you provide) and prints
basic statistics: mean, median, standard deviation, percentiles, and
coefficient of variation.
"""

import json
import math
from pathlib import Path
import argparse


def load_values(path: Path):
    data = json.loads(path.read_text())
    return data, list(map(float, data.get("values", [])))


def summarize(values):
    if not values:
        return {}

    n = len(values)
    mean = sum(values) / n
    var = sum((x - mean) ** 2 for x in values) / n
    std = math.sqrt(var)

    sorted_vals = sorted(values)
    median = sorted_vals[n // 2] if n % 2 == 1 else 0.5 * (
        sorted_vals[n // 2 - 1] + sorted_vals[n // 2]
    )

    def percentile(p):
        if n == 1:
            return sorted_vals[0]
        k = (n - 1) * p / 100.0
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return sorted_vals[int(k)]
        d0 = sorted_vals[int(f)] * (c - k)
        d1 = sorted_vals[int(c)] * (k - f)
        return d0 + d1

    p5 = percentile(5)
    p95 = percentile(95)

    cv = std / mean if mean != 0 else float("inf")

    return {
        "count": n,
        "mean": mean,
        "median": median,
        "std": std,
        "p5": p5,
        "p95": p95,
        "cv": cv,
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Analyze variability of Wh/1000 tokens measurements from "
            "benchmark_variability_runs.json."
        )
    )
    parser.add_argument(
        "path",
        nargs="?",
        default="benchmark_variability_runs.json",
        help="Path to the JSON file (default: benchmark_variability_runs.json)",
    )

    args = parser.parse_args()
    path = Path(args.path)

    if not path.exists():
        print(f"❌ File not found: {path}")
        return

    data, values = load_values(path)
    stats = summarize(values)

    if not stats:
        print("No values found in JSON file.")
        return

    print("File:", path)
    print("Metric:", data.get("metric"))
    print("Model:", data.get("model"))
    print("Prompt:", data.get("prompt"))
    print("Requested runs:", data.get("requested_runs"))
    print("Successful runs:", data.get("successful_runs"))
    print()
    print("Count:", stats["count"])
    print(f"Mean: {stats['mean']:.6f} Wh/1K")
    print(f"Median: {stats['median']:.6f} Wh/1K")
    print(f"Std dev: {stats['std']:.6f} Wh/1K")
    print(f"5th percentile: {stats['p5']:.6f} Wh/1K")
    print(f"95th percentile: {stats['p95']:.6f} Wh/1K")
    print(f"Coeff. of variation: {stats['cv']:.2f}")


if __name__ == "__main__":
    main()
