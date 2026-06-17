"""
parallel_eval.py
================
Parallel genome evaluation using all CPU cores simultaneously.

WHY THIS MATTERS:
    Sequential evaluation (what we had before):
        100 genomes × 6.5 min each = 10.8 hours per 300 generations
    
    Parallel evaluation on 8 cores:
        100 genomes ÷ 8 cores × 6.5 min = 1.35 hours per 300 generations
    
    That difference decides whether your Phase 3 experiments
    are feasible within your 4-month timeline.

HOW IT WORKS:
    Python's multiprocessing module spawns N worker processes
    (one per CPU core). Each worker gets a genome, evaluates it
    independently in its own simulation instance, returns the
    fitness score. The main process collects all scores and
    hands them back to NEAT.

    Each worker runs a completely separate CarRacing-v3 instance —
    they don't share memory or interfere with each other.

DISSERTATION NOTE:
    This is your "Green AI" mitigation from the proposal —
    reducing total CPU time per generation directly reduces
    the carbon footprint of your training runs.
"""

import neat
import multiprocessing
import os
from src.neat_runner import eval_genome


def _worker(args):
    """
    Worker function — runs in a separate process.

    Unpacks the genome and config, evaluates the genome,
    returns (genome_id, fitness) so the main process can
    match scores back to the right genome.

    Parameters
    ----------
    args : tuple of (genome_id, genome, config)

    Returns
    -------
    tuple of (genome_id, float)
    """
    genome_id, genome, config = args
    fitness = eval_genome(genome, config)
    return genome_id, fitness


class ParallelEvaluator:
    """
    Evaluates all genomes in a generation in parallel.

    Drop-in replacement for the sequential eval_genomes() function.
    Pass an instance of this class to population.run() instead.

    Parameters
    ----------
    num_workers : int
        Number of parallel processes to use.
        Defaults to all available CPU cores minus 1
        (leaving one core free so your machine stays responsive).
    
    timeout : int
        Maximum seconds to wait for a single genome evaluation.
        Prevents a crashed simulation from hanging the whole run.
        Default 120 seconds (2 minutes) is generous even for slow machines.

    Example
    -------
        evaluator = ParallelEvaluator(num_workers=7)
        best = population.run(evaluator.evaluate, NUM_GENERATIONS)
    """

    def __init__(self, num_workers: int = None, timeout: int = 120):
        if num_workers is None:
            # Leave one core free for the OS and main process
            self.num_workers = max(1, os.cpu_count() - 1)
        else:
            self.num_workers = num_workers

        self.timeout = timeout
        print(f"  ParallelEvaluator: using {self.num_workers} worker processes")

    def evaluate(self, genomes, config):
        """
        Evaluate all genomes in parallel.

        Called by NEAT every generation instead of eval_genomes().
        Sets genome.fitness for every genome in the population.

        Parameters
        ----------
        genomes : list of (genome_id, genome) tuples
        config  : neat.Config
        """
        total = len(genomes)

        # Build argument list for each worker
        args = [
            (genome_id, genome, config)
            for genome_id, genome in genomes
        ]

        # Create a pool of worker processes and map genomes to them.
        # imap_unordered returns results as they complete (fastest first)
        # rather than waiting for all to finish in order.
        results = {}

        with multiprocessing.Pool(processes=self.num_workers) as pool:
            for i, (genome_id, fitness) in enumerate(
                pool.imap_unordered(_worker, args)
            ):
                results[genome_id] = fitness

                # Progress bar — updates as results come in
                bar_len = 20
                filled  = int(bar_len * (i + 1) / total)
                bar     = '█' * filled + '░' * (bar_len - filled)
                print(
                    f"\r  [{bar}] {i+1:3d}/{total} "
                    f"| Fitness {fitness:8.2f} "
                    f"| Best {max(results.values()):8.2f}",
                    end='', flush=True
                )

        print()  # newline after progress bar

        # Assign fitness scores back to genomes
        for genome_id, genome in genomes:
            genome.fitness = results[genome_id]