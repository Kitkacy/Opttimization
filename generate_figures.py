#!/usr/bin/env python3
"""
Reads results/summary.csv and writes three PNG chart files to results/:
  fig1_relative.png   – relative improvement over baseline (all 9 instances)
  fig2_primary.png    – per-instance comparison with error bars (3 primary instances)
  fig3_timing.png     – wall-clock time overhead per heuristic

Run this script after run_tests.py whenever test results change.
"""

import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

RESULTS_DIR = Path(__file__).parent / "results"
CSV_PATH    = RESULTS_DIR / "summary.csv"

PRIMARY_INSTANCES = ["pm1d_100.9.txt", "w09_100.9.txt", "pw09_100.9.txt"]

ALL_INSTANCES = [
    "g05_60.1.txt", "g05_80.1.txt",
    "pm1s_100.1.txt", "pm1d_100.9.txt",
    "w01_100.1.txt", "w09_100.9.txt", "pw09_100.9.txt",
    "t2g10_5555.txt", "t2g10_6666.txt",
]

INSTANCE_SHORT = {
    "g05_60.1.txt":   "g05-60",
    "g05_80.1.txt":   "g05-80",
    "pm1s_100.1.txt": "pm1s",
    "pm1d_100.9.txt": "pm1d",
    "w01_100.1.txt":  "w01",
    "w09_100.9.txt":  "w09",
    "pw09_100.9.txt": "pw09",
    "t2g10_5555.txt": "t2g-5555",
    "t2g10_6666.txt": "t2g-6666",
}

H_LABEL = {
    "none":       "Baseline",
    "greedy":     "Greedy",
    "ranked_mut": "Ranked Mut.",
    "balancer":   "Balancer",
    "all":        "All Three",
}

COLORS = {
    "none":       "#5b9bd5",
    "greedy":     "#ed7d31",
    "ranked_mut": "#70ad47",
    "balancer":   "#ffc000",
    "all":        "#7030a0",
}

plt.rcParams.update({
    "font.family":  "serif",
    "font.size":    9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi":   150,
    "axes.grid":    True,
    "grid.linestyle": "--",
    "grid.alpha":   0.4,
    "axes.axisbelow": True,
})


def pct(new_val, base_val):
    return (new_val - base_val) / base_val * 100.0


def load_csv(path):
    data = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            key = (row["instance"], row["heuristic"])
            data[key] = {
                "cut_mean":     float(row["cut_mean"]),
                "cut_stdev":    float(row["cut_stdev"]),
                "cut_best":     float(row["cut_best"]),
                "time_mean_ms": float(row["time_mean_ms"]),
            }
    return data


# ── Figure 1: Relative improvement over baseline (all instances) ──────────────
def fig_relative_improvement(data, out_path):
    single     = ["greedy", "ranked_mut", "balancer"]
    labels     = [INSTANCE_SHORT[i] for i in ALL_INSTANCES]
    x          = np.arange(len(labels))
    n          = len(single)
    width      = 0.22
    offsets    = np.linspace(-(n - 1) / 2, (n - 1) / 2, n) * width

    fig, ax = plt.subplots(figsize=(11, 4.5))

    for offset, h in zip(offsets, single):
        vals = []
        for inst in ALL_INSTANCES:
            base = data.get((inst, "none"))
            row  = data.get((inst, h))
            vals.append(pct(row["cut_mean"], base["cut_mean"]) if base and row else 0.0)
        ax.bar(x + offset, vals, width, label=H_LABEL[h],
               color=COLORS[h], edgecolor="white", linewidth=0.5)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("% change in mean cut value vs. baseline")
    ax.set_title("Relative improvement over baseline GA across all benchmark instances")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Written: {out_path}")


# ── Figure 2: Primary instances — all five configs with error bars ────────────
def fig_primary(data, out_path):
    configs    = ["none", "greedy", "ranked_mut", "balancer", "all"]
    labels     = [H_LABEL[c] for c in configs]
    x          = np.arange(len(configs))
    width      = 0.55

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5), sharey=True)

    for ax, inst in zip(axes, PRIMARY_INSTANCES):
        base_mean = data[(inst, "none")]["cut_mean"]
        means, errs = [], []
        for cfg in configs:
            row = data.get((inst, cfg))
            if row:
                means.append(pct(row["cut_mean"], base_mean))
                errs.append(row["cut_stdev"] / base_mean * 100.0)
            else:
                means.append(0.0)
                errs.append(0.0)

        bars = ax.bar(x, means, width,
                      color=[COLORS[c] for c in configs],
                      edgecolor="white", linewidth=0.5,
                      yerr=errs, capsize=3,
                      error_kw={"elinewidth": 1.0, "ecolor": "#444444"})
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.set_title(f"{INSTANCE_SHORT[inst]}", family="monospace")
        ax.set_ylabel("% vs. baseline" if ax is axes[0] else "")

    fig.suptitle(
        "Mean cut value relative to baseline for the three primary instances\n"
        "(error bars = ±1 std dev; All Three = 10 runs, others = 5 runs)",
        fontsize=9
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Written: {out_path}")


# ── Figure 3: Wall-clock time overhead ────────────────────────────────────────
def fig_timing(data, out_path):
    single  = ["greedy", "ranked_mut", "balancer"]
    labels  = [INSTANCE_SHORT[i] for i in ALL_INSTANCES]
    x       = np.arange(len(labels))
    n       = len(single)
    width   = 0.22
    offsets = np.linspace(-(n - 1) / 2, (n - 1) / 2, n) * width

    fig, ax = plt.subplots(figsize=(11, 3.8))

    for offset, h in zip(offsets, single):
        vals = []
        for inst in ALL_INSTANCES:
            base = data.get((inst, "none"))
            row  = data.get((inst, h))
            vals.append(pct(row["time_mean_ms"], base["time_mean_ms"]) if base and row else 0.0)
        ax.bar(x + offset, vals, width, label=H_LABEL[h],
               color=COLORS[h], edgecolor="white", linewidth=0.5)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("% change vs. baseline wall-clock time")
    ax.set_title("Wall-clock execution time overhead relative to baseline")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Written: {out_path}")


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    data = load_csv(CSV_PATH)

    fig_relative_improvement(data, RESULTS_DIR / "fig1_relative.png")
    fig_primary(data,              RESULTS_DIR / "fig2_primary.png")
    fig_timing(data,               RESULTS_DIR / "fig3_timing.png")


if __name__ == "__main__":
    main()
