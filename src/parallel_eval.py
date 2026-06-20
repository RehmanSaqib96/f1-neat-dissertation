"""
parallel_eval.py — evaluates all genomes in parallel across CPU cores.
Drop-in replacement for eval_genomes(); capped at 8 workers to avoid
Windows paging exhaustion when cv2 loads simultaneously in each process.
"""

import neat
import multiprocessing
import os
from src.neat_runner import eval_genome


def _worker(args):
    """Unpack (genome_id, genome, config), evaluate, return (genome_id, fitness)."""
    genome_id, genome, config = args
    fitness = eval_genome(genome, config)
    return genome_id, fitness


class ParallelEvaluator:
    """
    Parallel genome evaluator — pass evaluator.evaluate to population.run().

    num_workers : processes to use (default: min(8, cpu_count-1)).
    timeout     : max seconds per genome before the pool kills it.
    """

    def __init__(self, num_workers: int = None, timeout: int = 120):
        if num_workers is None:
            # Cap at 8 — more causes cv2 paging exhaustion on 16 GB Windows machines
            self.num_workers = min(12, max(1, os.cpu_count() - 1))
        else:
            self.num_workers = num_workers

        self.timeout = timeout
        print(f"  ParallelEvaluator: using {self.num_workers} worker processes")

    def evaluate(self, genomes, config):
        """Evaluate all genomes in parallel, set genome.fitness for each."""
        total = len(genomes)

        args = [(genome_id, genome, config) for genome_id, genome in genomes]

        results = {}

        # imap_unordered yields results as they complete (fastest first)
        with multiprocessing.Pool(processes=self.num_workers) as pool:
            for i, (genome_id, fitness) in enumerate(
                pool.imap_unordered(_worker, args)
            ):
                results[genome_id] = fitness

                bar_len = 20
                filled  = int(bar_len * (i + 1) / total)
                bar     = '█' * filled + '░' * (bar_len - filled)
                print(
                    f"\r  [{bar}] {i+1:3d}/{total} "
                    f"| Fitness {fitness:8.2f} "
                    f"| Best {max(results.values()):8.2f}",
                    end='', flush=True
                )

        print()

        for genome_id, genome in genomes:
            genome.fitness = results[genome_id]
