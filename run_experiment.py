"""
Replaces main.py for systematic experimental runs.

Usage:
    python run_experiment.py --group A --spec 1.5 --rep 1
    python run_experiment.py --group B --seeds 5 --rep 2
    python run_experiment.py --group C --pop 200 --rep 1
    python run_experiment.py --group D --gens 1000 --rep 1

Arguments:
    --group   A / B / C / D  (which experimental group)
    --spec    speciation threshold  (Group A, default 3.0)
    --seeds   seed pool size        (Group B, default 10)
    --pop     population size       (Group C, default 100)
    --gens    number of generations (Group D, default 300)
    --rep     repetition number     (for reproducibility)
"""

import argparse
import neat
import os
import pickle

from src.logger import RunLogger
from src.neat_runner import eval_genome, TRAINING_SEED_POOL
from src.neat_runner import eval_genomes


# ── Defaults (Phase 8 validated config) ───────────────────────────────────────
DEFAULT_SPEC  = 3.0
DEFAULT_SEEDS = 10
DEFAULT_POP   = 100
DEFAULT_GENS  = 300

ALL_SEEDS = [42, 7, 99, 123, 2024, 17, 256, 1001, 555, 88]


def build_run_id(group, spec, seeds, pop, gens, rep):
    if group == "A":
        return f"A_spec{spec}_rep{rep}"
    elif group == "B":
        return f"B_seeds{seeds}_rep{rep}"
    elif group == "C":
        return f"C_pop{pop}_rep{rep}"
    elif group == "D":
        return f"D_spec{spec}_gens{gens}_rep{rep}"
    return f"run_{group}_rep{rep}"


def get_config_params(group, spec, seeds, pop, gens, rep):
    return {
        "group": group,
        "speciation_threshold": spec,
        "seed_pool_size": seeds,
        "population_size": pop,
        "num_generations": gens,
        "repetition": rep,
    }


def run(group, spec, seeds, pop, gens, rep):
    import src.neat_runner as nr

    # Apply seed pool for this run
    seed_pool = ALL_SEEDS[:seeds]
    nr.TRAINING_SEED_POOL = seed_pool

    run_id = build_run_id(group, spec, seeds, pop, gens, rep)
    config_params = get_config_params(group, spec, seeds, pop, gens, rep)
    logger = RunLogger(run_id, config_params)

    # Load NEAT config and override parameters
    config = neat.Config(
        neat.DefaultGenome, neat.DefaultReproduction,
        neat.DefaultSpeciesSet, neat.DefaultStagnation,
        "config/neat_config.txt"
    )
    config.pop_size = pop
    config.species_set_config.compatibility_threshold = spec

    pop_obj = neat.Population(config)
    pop_obj.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    pop_obj.add_reporter(stats)

    best_genome = None
    for generation in range(1, gens + 1):
        pop_obj.run(eval_genomes, 1)
        current_best = max(pop_obj.population.values(),
                          key=lambda g: g.fitness if g.fitness is not None else -999)
        logger.on_generation(generation, pop_obj.population, current_best)
        if best_genome is None or current_best.fitness > (best_genome.fitness or -999):
            best_genome = current_best

    # Save genome
    genome_path = os.path.join("results", "runs", f"{run_id}_genome.pkl")
    with open(genome_path, "wb") as f:
        pickle.dump(best_genome, f)

    # Save full log with cross-seed evaluation
    logger.save(best_genome, config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", required=True, choices=["A", "B", "C", "D"])
    parser.add_argument("--spec",  type=float, default=DEFAULT_SPEC)
    parser.add_argument("--seeds", type=int,   default=DEFAULT_SEEDS)
    parser.add_argument("--pop",   type=int,   default=DEFAULT_POP)
    parser.add_argument("--gens",  type=int,   default=DEFAULT_GENS)
    parser.add_argument("--rep",   type=int,   default=1)
    args = parser.parse_args()

    run(args.group, args.spec, args.seeds, args.pop, args.gens, args.rep)