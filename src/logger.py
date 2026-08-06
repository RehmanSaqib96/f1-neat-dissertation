"""
Automatic per-run logging for systematic NEAT experiments.

Every call to run_experiment.py produces one JSON file in results/runs/
containing everything needed for later comparative analysis
"""

import json
import os
import time
import neat
import numpy as np
from datetime import datetime
from src.neat_runner import eval_genome, TRAINING_SEED_POOL


RESULTS_DIR = os.path.join("results", "runs")


def ensure_dir():
    os.makedirs(RESULTS_DIR, exist_ok=True)


def cross_seed_eval(genome, config):
    """Re-evaluate genome on every seed in the pool individually."""
    import src.neat_runner as nr
    original_pool = nr.TRAINING_SEED_POOL[:]
    scores = {}
    for seed in TRAINING_SEED_POOL:
        nr.TRAINING_SEED_POOL = [seed]
        score = eval_genome(genome, config)
        scores[seed] = round(score, 4)
    nr.TRAINING_SEED_POOL = original_pool
    return scores


def curve_metrics(fitness_history):
    """Derive summary metrics from the generation-by-generation fitness list."""
    if not fitness_history:
        return {}

    best_vals = [g["best_fitness"] for g in fitness_history]
    n = len(best_vals)

    # First generation with a positive best fitness
    first_positive = next((i + 1 for i, v in enumerate(best_vals) if v > 0), None)

    # Longest plateau (consecutive gens with same best fitness)
    max_plateau = 1
    current_plateau = 1
    for i in range(1, n):
        if abs(best_vals[i] - best_vals[i - 1]) < 0.01:
            current_plateau += 1
            max_plateau = max(max_plateau, current_plateau)
        else:
            current_plateau = 1

    return {
        "final_best_fitness": round(best_vals[-1], 4),
        "peak_fitness": round(max(best_vals), 4),
        "peak_generation": int(np.argmax(best_vals)) + 1,
        "first_positive_generation": first_positive,
        "longest_plateau_gens": max_plateau,
        "mean_fitness_last_50": round(float(np.mean(best_vals[-50:])), 4),
    }


class RunLogger:
    """Attach to a NEAT training run and collect data as it progresses."""

    def __init__(self, run_id, config_params):
        """
        run_id: short string identifying this run, e.g. "A1_spec1.5_rep1"
        config_params: dict of the parameters being varied, e.g.
                       {"group": "A", "speciation_threshold": 1.5,
                        "seed_pool_size": 10, "population_size": 100}
        """
        ensure_dir()
        self.run_id = run_id
        self.config_params = config_params
        self.fitness_history = []
        self.start_time = time.time()
        self.start_dt = datetime.now().isoformat()
        print(f"[Logger] Run '{run_id}' started at {self.start_dt}")

    def on_generation(self, generation, population, best_genome):
        """Call this at the end of every generation."""
        fitnesses = [g.fitness for g in population.values() if g.fitness is not None]
        entry = {
            "generation": generation,
            "best_fitness": round(best_genome.fitness, 4),
            "mean_fitness": round(float(np.mean(fitnesses)), 4) if fitnesses else 0.0,
            "num_species": len(set(
                getattr(g, "species_id", 0)
                for g in population.values()
            )),
            "best_nodes": len(best_genome.nodes),
            "best_connections": len(best_genome.connections),
        }
        self.fitness_history.append(entry)

    def save(self, best_genome, config):
        """Call this once training is complete."""
        elapsed = round(time.time() - self.start_time, 1)

        print(f"[Logger] Running cross-seed evaluation for '{self.run_id}'...")
        cross_seed_scores = cross_seed_eval(best_genome, config)
        positive_seeds = sum(1 for v in cross_seed_scores.values() if v > 0)

        result = {
            "run_id": self.run_id,
            "started_at": self.start_dt,
            "elapsed_seconds": elapsed,
            "config_params": self.config_params,
            "final_genome": {
                "nodes": len(best_genome.nodes),
                "connections": len(best_genome.connections),
                "stored_fitness": round(best_genome.fitness, 4),
            },
            "cross_seed_scores": cross_seed_scores,
            "positive_seeds": positive_seeds,
            "curve_metrics": curve_metrics(self.fitness_history),
            "fitness_history": self.fitness_history,
        }

        path = os.path.join(RESULTS_DIR, f"{self.run_id}.json")
        with open(path, "w") as f:
            json.dump(result, f, indent=2)

        print(f"[Logger] Saved → {path}")
        print(f"[Logger] Positive seeds: {positive_seeds}/10  |  "
              f"Peak fitness: {result['curve_metrics']['peak_fitness']}  |  "
              f"Elapsed: {elapsed/60:.1f} min")
        return path