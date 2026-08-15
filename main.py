"""
main.py
=======
Entry point for the F1 NEAT dissertation project.

Run this to start evolution:
    python main.py

What happens:
    1. Load NEAT config from config/neat_config.txt
    2. Create initial population of 100 random genomes
    3. For each generation:
        a. Evaluate all 100 genomes in CarRacing-v3
        b. Print generation statistics
        c. Save checkpoint every 5 generations
        d. NEAT evolves the next generation
    4. Save the best genome ever found
    5. Print fitness history summary
"""

import os
import neat
import pickle
import time

from src.neat_runner    import eval_genomes
from src.parallel_eval  import ParallelEvaluator

# ── Paths ────────────────────────────────────────────────────────────────────
CONFIG_PATH      = os.path.join("config", "neat_config.txt")
CHECKPOINT_DIR   = "checkpoints"
RESULTS_DIR      = "results"
BEST_GENOME_PATH = os.path.join(RESULTS_DIR, "best_genome.pkl")
STATS_PATH       = os.path.join(RESULTS_DIR, "fitness_history.txt")

# ── Training settings ────────────────────────────────────────────────────────
# Start with 10 generations for your first run —
# just enough to see the car attempt to drive and
# confirm the full pipeline works end to end.
# Phase 3 increases this to 300+.
NUM_GENERATIONS     = 300
CHECKPOINT_INTERVAL = 5


def run_neat():
    """Main training loop."""

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR,    exist_ok=True)

    # ── Load config ───────────────────────────────────────────────────────
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        CONFIG_PATH
    )

    # ── Create population ─────────────────────────────────────────────────
    population = neat.Population(config)

    # ── Add reporters ─────────────────────────────────────────────────────

    # Prints generation stats to terminal
    population.add_reporter(neat.StdOutReporter(True))

    # Tracks fitness history — we use this for dissertation charts
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)

    # Saves full population state every N generations
    # Critical risk mitigation — if the run crashes at gen 147,
    # you resume from gen 145, not from zero
    population.add_reporter(
        neat.Checkpointer(
            generation_interval = CHECKPOINT_INTERVAL,
            filename_prefix     = os.path.join(CHECKPOINT_DIR, "neat-checkpoint-")
        )
    )

    # ── Print run summary ─────────────────────────────────────────────────
    print()
    print("=" * 62)
    print("  F1 NEAT DISSERTATION — Evolution Starting")
    print("=" * 62)
    print(f"  Generations  : {NUM_GENERATIONS}")
    print(f"  Population   : {config.pop_size}")
    print(f"  Inputs       : {config.genome_config.num_inputs} (24x24 pixels)")
    print(f"  Outputs      : {config.genome_config.num_outputs} (steer/gas/brake)")
    print(f"  Config       : {CONFIG_PATH}")
    print(f"  Checkpoints  : every {CHECKPOINT_INTERVAL} generations")
    print("=" * 62)
    print()

    start_time = time.time()

    # ── Run evolution ─────────────────────────────────────────────────────
    # Use parallel evaluation for full training runs.
    # Switch to eval_genomes (sequential) only for debugging.
    evaluator   = ParallelEvaluator()
    best_genome = population.run(evaluator.evaluate, NUM_GENERATIONS)

    elapsed = time.time() - start_time

    # ── Save best genome ──────────────────────────────────────────────────
    with open(BEST_GENOME_PATH, "wb") as f:
        pickle.dump(best_genome, f)

    # ── Save fitness history ──────────────────────────────────────────────
    with open(STATS_PATH, "w") as f:
        f.write("generation,best_fitness,avg_fitness\n")
        best_scores = stats.get_fitness_stat(max)
        avg_scores  = stats.get_fitness_mean()
        for i, (b, a) in enumerate(zip(best_scores, avg_scores)):
            f.write(f"{i+1},{b:.4f},{a:.4f}\n")

    # ── Final summary ─────────────────────────────────────────────────────
    print()
    print("=" * 62)
    print("  Evolution Complete")
    print("=" * 62)
    print(f"  Total time        : {elapsed/60:.1f} minutes")
    print(f"  Best fitness      : {best_genome.fitness:.2f}")
    print(f"  Best genome nodes : {len(best_genome.nodes)}")
    print(f"  Best genome conns : {len(best_genome.connections)}")
    print(f"  Saved to          : {BEST_GENOME_PATH}")
    print("=" * 62)
    print()

    print("Generation-by-generation best fitness:")
    for i, score in enumerate(stats.get_fitness_stat(max)):
        bar_len = int((score + 100) / 20)
        bar     = '█' * max(0, min(bar_len, 40))
        print(f"  Gen {i+1:3d} | {score:8.2f} | {bar}")

    return best_genome, stats


if __name__ == "__main__":
    run_neat()