#!/usr/bin/env python3
"""Plot paired FP16 GEMM benchmark summaries from per-run CSV files."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_run(path):
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    baseline = next(row for row in rows if row["impl"] == "cuBLAS GemmEx FP16")
    streamk = next(row for row in rows if row["impl"] == "Custom FP16 StreamK")
    if baseline["valid"] != "true" or streamk["valid"] != "true":
        raise ValueError(f"validation failed in {path}")
    return float(baseline["median_ms"]) * 1000, float(streamk["median_ms"]) * 1000


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    results = [read_run(path) for path in args.inputs]
    runs = list(range(1, len(results) + 1))
    cublas = [pair[0] for pair in results]
    streamk = [pair[1] for pair in results]
    speedups = [(a / b - 1) * 100 for a, b in results]
    mean_speedup = sum(speedups) / len(speedups)

    plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, (latency_ax, speedup_ax) = plt.subplots(
        1, 2, figsize=(11, 4.7), gridspec_kw={"width_ratios": [1.55, 1]},
        constrained_layout=True,
    )
    latency_ax.plot(runs, cublas, "o-", color="#3366aa", linewidth=2.2,
                    markersize=7, label="cuBLAS GemmEx")
    latency_ax.plot(runs, streamk, "o-", color="#d97732", linewidth=2.2,
                    markersize=7, label="Custom Stream-K")
    latency_ax.set(xlabel="Run", ylabel="Median latency (µs)",
                   title="Latency across paired runs", xticks=runs)
    latency_ax.grid(axis="y", alpha=0.25)
    latency_ax.legend(frameon=False, loc="upper left")

    speedup_ax.bar(runs, speedups, width=0.65, color="#d97732", alpha=0.9)
    speedup_ax.axhline(mean_speedup, color="#444444", linewidth=1.4,
                       linestyle="--", label=f"Mean: {mean_speedup:.2f}%")
    speedup_ax.set(xlabel="Run", ylabel="Stream-K speedup (%)",
                   title="Improvement over cuBLAS", xticks=runs)
    speedup_ax.set_ylim(0, max(speedups) * 1.4)
    speedup_ax.grid(axis="y", alpha=0.25)
    speedup_ax.legend(frameon=False, loc="upper right")

    fig.suptitle("RTX 5090 · FP16 GEMM · 4096 × 4096 × 4096", fontsize=15)
    fig.supxlabel("Each run: 40 alternating samples × 10 launches; FP32 accumulation",
                 fontsize=9, color="#555555")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, facecolor="white")
    fig.savefig(args.output.with_suffix(".svg"), facecolor="white")
    print(f"Wrote {args.output} and {args.output.with_suffix('.svg')}")


if __name__ == "__main__":
    main()
