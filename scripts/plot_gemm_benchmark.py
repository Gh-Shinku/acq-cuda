import argparse
import csv
import sys
from collections import defaultdict

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    print(
        "matplotlib is required. Install it with:\n"
        "  /home/zhaoyutong/miniconda3/bin/conda install -n base matplotlib\n"
        "or run with PYTHON=/path/to/python that already has matplotlib.",
        file=sys.stderr,
    )
    raise


def read_rows(path):
    with open(path, newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))
    if not rows:
        raise RuntimeError(f"no benchmark rows found in {path}")
    return rows


def group_by_kernel(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["impl"]].append(row)
    for values in grouped.values():
        values.sort(key=lambda row: int(row["size"]))
    return grouped


def get_xy(rows, key):
    x = [int(row["size"]) for row in rows]
    y = [float(row[key]) for row in rows]
    return x, y


def plot_metric(ax, grouped, key, ylabel, title, log_y=False):
    for impl in sorted(grouped):
        x, y = get_xy(grouped[impl], key)
        ax.plot(x, y, marker="o", linewidth=1.8, label=impl)
    ax.set_xscale("log", base=2)
    if log_y:
        ax.set_yscale("log")
    ax.set_xlabel("Matrix Size (M=N=K)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rows = read_rows(args.csv)
    grouped = group_by_kernel(rows)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    plot_metric(
        axes[0][0],
        grouped,
        "tflops",
        "TFLOPS",
        "TFLOPS Performance vs Matrix Size",
    )
    plot_metric(
        axes[0][1],
        grouped,
        "bandwidth_gbs",
        "Bandwidth (GB/s)",
        "Memory Bandwidth vs Matrix Size",
    )
    plot_metric(
        axes[1][0],
        grouped,
        "avg_time_ms",
        "Time (ms)",
        "Execution Time vs Matrix Size",
        log_y=True,
    )

    naive_rows = grouped.get("CUDA Naive", [])
    if naive_rows:
        x, y = get_xy(naive_rows, "speedup_vs_cutlass")
        axes[1][1].plot(x, y, marker="o", linewidth=1.8, label="CUDA Naive")
        axes[1][1].axhline(
            y=1.0,
            color="tab:gray",
            linestyle="--",
            linewidth=1.5,
            label="CUTLASS Baseline",
        )
        axes[1][1].set_xscale("log", base=2)
    axes[1][1].set_xlabel("Matrix Size (M=N=K)")
    axes[1][1].set_ylabel("Speedup (x)")
    axes[1][1].set_title("Speedup vs CUTLASS Baseline")
    axes[1][1].grid(True, alpha=0.3)
    axes[1][1].legend()

    fig.suptitle("SGEMM Benchmark", fontsize=14)
    fig.savefig(args.output, dpi=160)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
