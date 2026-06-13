#!/usr/bin/env python3
"""
GA benchmark runner and report generator.

Test matrix
-----------
- Every (instance, heuristic-config) pair: 5 runs each.
- Primary instances (pm1d_100.9, w09_100.9, pw09_100.9) with all-three heuristics on: 10 runs each.

Outputs
-------
- results/raw.jsonl          one JSON object per run (full solution vector included)
- results/summary.csv        per-(instance, config) aggregated statistics
- results/report.txt         human-readable report
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from statistics import mean, stdev

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
BINARY = BASE_DIR / "ga_main"
INSTANCES_DIR = BASE_DIR / "Instances"
CONFIGS_DIR = BASE_DIR / "configs"
RESULTS_DIR = BASE_DIR / "results"

ALL_INSTANCES = [
    "g05_60.1.txt",
    "g05_80.1.txt",
    "pm1s_100.1.txt",
    "pm1d_100.9.txt",
    "w01_100.1.txt",
    "w09_100.9.txt",
    "pw09_100.9.txt",
    "t2g10_5555.txt",
    "t2g10_6666.txt",
]

PRIMARY_INSTANCES = {
    "pm1d_100.9.txt",
    "w09_100.9.txt",
    "pw09_100.9.txt",
}

SINGLE_HEURISTIC_CONFIGS = [
    "baseline.txt",
    "greedy.txt",
    "ranked_mutation.txt",
    "balancer.txt",
]

ALL_HEURISTICS_CONFIG = "all.txt"

STANDARD_RUNS = 5
PRIMARY_RUNS = 10

# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_once(instance_file: str, config_file: str) -> dict:
    """Run ga_main once with --json and return the parsed result dict."""
    cmd = [
        str(BINARY),
        str(INSTANCES_DIR / instance_file),
        str(CONFIGS_DIR / config_file),
        "--json",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"ga_main failed (exit {proc.returncode}) for "
            f"instance={instance_file} config={config_file}:\n{proc.stderr.strip()}"
        )
    line = proc.stdout.strip()
    if not line:
        raise RuntimeError(
            f"ga_main produced no output for instance={instance_file} config={config_file}"
        )
    result = json.loads(line)
    result["instance_file"] = instance_file
    result["config_file"] = config_file
    return result


def build_test_matrix() -> list[tuple[str, str, int]]:
    """Return list of (instance_file, config_file, n_runs) tuples."""
    matrix = []

    for instance in ALL_INSTANCES:
        for config in SINGLE_HEURISTIC_CONFIGS:
            matrix.append((instance, config, STANDARD_RUNS))

    for instance in PRIMARY_INSTANCES:
        matrix.append((instance, ALL_HEURISTICS_CONFIG, PRIMARY_RUNS))

    return matrix


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def heuristic_label(config_file: str) -> str:
    stem = Path(config_file).stem
    labels = {
        "baseline": "none",
        "greedy": "greedy",
        "ranked_mutation": "ranked_mut",
        "balancer": "balancer",
        "all": "all",
    }
    return labels.get(stem, stem)


def aggregate(runs: list[dict]) -> dict:
    cuts = [r["cut_value"] for r in runs]
    times = [r["elapsed_ms"] for r in runs]
    nfes = [r["nfe"] for r in runs]
    gens = [r["generations"] for r in runs]
    return {
        "n_runs": len(runs),
        "cut_best": max(cuts),
        "cut_worst": min(cuts),
        "cut_mean": mean(cuts),
        "cut_stdev": stdev(cuts) if len(cuts) > 1 else 0.0,
        "time_mean_ms": mean(times),
        "time_stdev_ms": stdev(times) if len(times) > 1 else 0.0,
        "nfe_mean": mean(nfes),
        "generations_mean": mean(gens),
        "all_cuts": cuts,
        "all_solutions": [r["solution"] for r in runs],
    }


# ---------------------------------------------------------------------------
# Report writers
# ---------------------------------------------------------------------------

def write_raw_jsonl(all_runs: list[dict], path: Path) -> None:
    with open(path, "w") as f:
        for run in all_runs:
            f.write(json.dumps(run) + "\n")


def write_csv(agg_table: dict, path: Path) -> None:
    header = (
        "instance,heuristic,n_runs,"
        "cut_best,cut_worst,cut_mean,cut_stdev,"
        "time_mean_ms,time_stdev_ms,nfe_mean,generations_mean"
    )
    rows = []
    for (instance, config), stats in sorted(agg_table.items()):
        row = ",".join([
            instance,
            heuristic_label(config),
            str(stats["n_runs"]),
            str(stats["cut_best"]),
            str(stats["cut_worst"]),
            f"{stats['cut_mean']:.2f}",
            f"{stats['cut_stdev']:.2f}",
            f"{stats['time_mean_ms']:.3f}",
            f"{stats['time_stdev_ms']:.3f}",
            f"{stats['nfe_mean']:.1f}",
            f"{stats['generations_mean']:.1f}",
        ])
        rows.append(row)
    with open(path, "w") as f:
        f.write(header + "\n")
        f.write("\n".join(rows) + "\n")


def write_report(agg_table: dict, path: Path, total_runs: int, elapsed_wall: float) -> None:
    lines = []
    lines.append("=" * 72)
    lines.append("  GA BENCHMARK REPORT")
    lines.append("=" * 72)
    lines.append(f"  Total runs executed : {total_runs}")
    lines.append(f"  Wall-clock time     : {elapsed_wall:.1f} s")
    lines.append("")

    # Group by instance
    by_instance: dict[str, dict] = {}
    for (instance, config), stats in agg_table.items():
        by_instance.setdefault(instance, {})[config] = stats

    for instance in ALL_INSTANCES:
        if instance not in by_instance:
            continue
        configs_stats = by_instance[instance]
        lines.append("-" * 72)
        lines.append(f"  Instance: {instance}")
        lines.append("-" * 72)

        col_w = 14
        header = (
            f"  {'Heuristic':<14} {'Runs':>4}  "
            f"{'Best':>10}  {'Worst':>10}  {'Mean':>12}  {'StdDev':>10}  {'Time(ms)':>10}"
        )
        lines.append(header)
        lines.append("  " + "-" * 68)

        for config in SINGLE_HEURISTIC_CONFIGS + ([ALL_HEURISTICS_CONFIG] if instance.replace(".txt","") in {p.replace(".txt","") for p in PRIMARY_INSTANCES} else []):
            if config not in configs_stats:
                continue
            s = configs_stats[config]
            label = heuristic_label(config)
            row = (
                f"  {label:<14} {s['n_runs']:>4}  "
                f"{s['cut_best']:>10}  {s['cut_worst']:>10}  "
                f"{s['cut_mean']:>12.2f}  {s['cut_stdev']:>10.2f}  "
                f"{s['time_mean_ms']:>10.3f}"
            )
            lines.append(row)

            # Per-run cut values as a compact vector
            cuts_str = "  runs: [" + ", ".join(str(c) for c in s["all_cuts"]) + "]"
            lines.append(cuts_str)

        lines.append("")

    lines.append("=" * 72)
    lines.append("  All individual solutions are stored in results/raw.jsonl")
    lines.append("  Aggregated statistics are stored in results/summary.csv")
    lines.append("=" * 72)

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not BINARY.exists():
        print(f"Error: binary not found at {BINARY}. Run 'make' first.", file=sys.stderr)
        sys.exit(1)

    RESULTS_DIR.mkdir(exist_ok=True)

    matrix = build_test_matrix()
    total_runs = sum(n for _, _, n in matrix)

    print(f"Test matrix: {len(matrix)} configurations, {total_runs} total runs")
    print(f"Results will be written to: {RESULTS_DIR}/")
    print()

    all_runs: list[dict] = []
    agg_table: dict[tuple, dict] = {}

    wall_start = time.monotonic()
    completed = 0

    for instance, config, n_runs in matrix:
        label = f"{instance} / {heuristic_label(config)} x{n_runs}"
        print(f"  Running {label} ...", end="", flush=True)
        t0 = time.monotonic()

        runs_this = []
        for _ in range(n_runs):
            result = run_once(instance, config)
            all_runs.append(result)
            runs_this.append(result)
            completed += 1

        elapsed = time.monotonic() - t0
        agg = aggregate(runs_this)
        agg_table[(instance, config)] = agg

        best = agg["cut_best"]
        mean_cut = agg["cut_mean"]
        print(f"  done in {elapsed:.1f}s  best={best}  mean={mean_cut:.1f}")

    wall_elapsed = time.monotonic() - wall_start

    print()
    print(f"All {completed} runs complete in {wall_elapsed:.1f}s. Writing results...")

    raw_path = RESULTS_DIR / "raw.jsonl"
    csv_path = RESULTS_DIR / "summary.csv"
    report_path = RESULTS_DIR / "report.txt"

    write_raw_jsonl(all_runs, raw_path)
    write_csv(agg_table, csv_path)
    write_report(agg_table, report_path, completed, wall_elapsed)

    print(f"  {raw_path}  ({len(all_runs)} records)")
    print(f"  {csv_path}")
    print(f"  {report_path}")
    print()

    # Quick summary to stdout
    print("Quick summary (mean cut by instance / heuristic):")
    print(f"  {'Instance':<20}  {'Heuristic':<14}  {'Mean cut':>12}  {'Best':>10}")
    print("  " + "-" * 62)
    for (instance, config), stats in sorted(agg_table.items()):
        print(
            f"  {instance:<20}  {heuristic_label(config):<14}  "
            f"{stats['cut_mean']:>12.1f}  {stats['cut_best']:>10}"
        )


if __name__ == "__main__":
    main()
