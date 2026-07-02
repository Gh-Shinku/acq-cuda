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
        grouped[row["kernel"]].append(row)
    for values in grouped.values():
        values.sort(key=lambda row: int(row["m"]))
    return grouped


def get_xy(rows, key):
    x = [int(row["m"]) for row in rows]
    y = [float(row[key]) for row in rows]
    return x, y


def plot_metric(ax, grouped, key, ylabel, title):
    for kernel in sorted(grouped):
        x, y = get_xy(grouped[kernel], key)
        ax.plot(x, y, marker="o", linewidth=1.8, label=kernel)
    ax.set_xlabel("Square matrix size")
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
        "time_ms",
        "ms",
        "Execution Time",
    )
    plot_metric(
        axes[0][1],
        grouped,
        "tflops",
        "TFLOPS",
        "FP32 Throughput",
    )
    plot_metric(
        axes[1][0],
        grouped,
        "effective_bandwidth_gbps",
        "GB/s",
        "Effective Memory Bandwidth",
    )

    cutlass_rows = grouped.get("cutlass", [])
    if cutlass_rows:
        x, y = get_xy(cutlass_rows, "speedup_vs_naive")
        axes[1][1].plot(x, y, marker="o", linewidth=1.8, color="tab:green")
    axes[1][1].set_xlabel("Square matrix size")
    axes[1][1].set_ylabel("x")
    axes[1][1].set_title("CUTLASS Speedup vs Naive")
    axes[1][1].grid(True, alpha=0.3)

    fig.suptitle("SGEMM Benchmark", fontsize=14)
    fig.savefig(args.output, dpi=160)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
