"""
Loads all JSON run logs and produces summary tables for dissertation analysis.

Usage:
    python analyse_results.py           # summarise all runs
    python analyse_results.py --group A # summarise one group only
"""

import json
import os
import argparse
import glob

RESULTS_DIR = os.path.join("results", "runs")


def load_runs(group=None):
    pattern = os.path.join(RESULTS_DIR, "*.json")
    files = sorted(glob.glob(pattern))
    runs = []
    for f in files:
        with open(f) as fh:
            data = json.load(fh)
        if group and data.get("config_params", {}).get("group") != group:
            continue
        runs.append(data)
    return runs


def print_summary_table(runs):
    if not runs:
        print("No runs found.")
        return

    header = (
        f"{'Run ID':<25} {'Group':<6} {'Spec':>5} {'Seeds':>5} "
        f"{'Pop':>5} {'Gens':>5} {'Peak':>8} {'Final':>8} "
        f"+Seeds  Nodes  Plateau  FirstPos"
    )
    print(header)
    print("-" * len(header))

    for r in runs:
        cp = r.get("config_params", {})
        cm = r.get("curve_metrics", {})
        fg = r.get("final_genome", {})
        print(
            f"{r['run_id']:<25} "
            f"{cp.get('group','?'):<6} "
            f"{cp.get('speciation_threshold','-'):>5} "
            f"{cp.get('seed_pool_size','-'):>5} "
            f"{cp.get('population_size','-'):>5} "
            f"{cp.get('num_generations','-'):>5} "
            f"{cm.get('peak_fitness','-'):>8} "
            f"{cm.get('final_best_fitness','-'):>8} "
            f"{r.get('positive_seeds','-'):>6}  "
            f"{fg.get('nodes','-'):>5}  "
            f"{cm.get('longest_plateau_gens','-'):>7}  "
            f"{cm.get('first_positive_generation','-')}"
        )


def print_cross_seed_table(runs):
    """Print cross-seed scores for all runs side by side."""
    seeds = [42, 7, 99, 123, 2024, 17, 256, 1001, 555, 88]
    print(f"\n{'Run ID':<25}", end="")
    for s in seeds:
        print(f"  {s:>5}", end="")
    print(f"  {'Pos':>4}")
    print("-" * (25 + len(seeds) * 7 + 8))

    for r in runs:
        cs = r.get("cross_seed_scores", {})
        print(f"{r['run_id']:<25}", end="")
        for s in seeds:
            val = cs.get(str(s), cs.get(s, "?"))
            print(f"  {val:>5.0f}" if isinstance(val, float) else f"  {'?':>5}", end="")
        print(f"  {r.get('positive_seeds','?'):>4}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", default=None, help="Filter by group A/B/C/D")
    args = parser.parse_args()

    runs = load_runs(args.group)
    label = f"Group {args.group}" if args.group else "All runs"
    print(f"\n=== {label} — {len(runs)} run(s) ===\n")
    print_summary_table(runs)
    print()
    print_cross_seed_table(runs)