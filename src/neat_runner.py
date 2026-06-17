"""
neat_runner.py
==============
The core evaluation loop — connects NEAT genomes to CarRacing-v3.

This is the most important file in the project. NEAT calls
eval_genomes() every generation. For each genome it:

    1. Builds a neural network from the genome's genes
    2. Runs it in the simulation for up to MAX_FRAMES frames
    3. Scores it using our fitness function
    4. Returns the score so NEAT knows who to keep and breed

Everything else in the project feeds into or out of this file.
"""

import neat
import numpy as np
from src.environment import RacingEnv
from src.fitness     import compute_fitness, detect_grass

# Windows multiprocessing fix — prevents worker processes from
# trying to spawn their own children when the module is imported
import multiprocessing
multiprocessing.freeze_support()

# ── Evaluation constants ────────────────────────────────────────────────────

# Maximum frames per episode.
# At ~50fps, 1000 frames ≈ 20 seconds of driving.
# Long enough to show real behaviour, short enough to keep
# training fast. We increase this in Phase 3.
MAX_FRAMES = 1000

# If total reward drops below this, kill the episode early.
# Stops hopeless genomes wasting time spinning in circles
# or sitting completely still.
EARLY_STOP_THRESHOLD = -15.0

# Fixed track seed for Weeks 1-10.
# Same seed = same track layout every evaluation.
# This is essential for fair comparison between genomes —
# they all face identical conditions.
# In Phase 3 we test on multiple seeds to check generalisation.
TRAINING_SEED = 42


# ── Action processing ───────────────────────────────────────────────────────

def process_action(outputs: list) -> np.ndarray:
    """
    Convert raw neural network output values into valid actions.

    The network's output nodes use tanh activation → range [-1, 1].
    CarRacing-v3 expects:
        steering : [-1.0,  1.0]  — tanh range is perfect, use directly
        gas      : [ 0.0,  1.0]  — clip negative values to 0
        brake    : [ 0.0,  1.0]  — clip negative values to 0

    We also apply a small gas minimum so the car always moves
    forward slightly — prevents the agent from learning to
    sit still as a strategy (sitting still = no negative reward
    from going off-track, but also no progress).

    Parameters
    ----------
    outputs : list of 3 floats from the neural network output nodes

    Returns
    -------
    np.ndarray  [steering, gas, brake]
    """
    steering = float(np.clip(outputs[0], -1.0,  1.0))
    gas      = float(np.clip(outputs[1],  0.0,  1.0))
    brake    = float(np.clip(outputs[2],  0.0,  1.0))

    # Minimum gas — prevents the "sit still" strategy
    # 0.1 is enough to keep the car moving without overwhelming steering
    gas = max(gas, 0.1)

    return np.array([steering, gas, brake], dtype=np.float32)


# ── Single genome evaluation ────────────────────────────────────────────────

def eval_genome(genome, config) -> float:
    """
    Evaluate one genome by letting it drive in the simulation.

    Called by eval_genomes() for every genome every generation.
    Must return a single float — the genome's fitness score.

    Parameters
    ----------
    genome : neat.DefaultGenome
        One individual from the population.
        Contains the network's connection genes and node genes
        (topology + weights). NEAT built this from mutations
        and crossover of previous generation survivors.

    config : neat.Config
        The loaded configuration from neat_config.txt.
        Needed to reconstruct the neural network from the genome.

    Returns
    -------
    float
        Fitness score. Higher = better driving.
    """

    # ── Step 1: Build neural network from genome ──────────────────────────
    # This reads the genome's gene list and builds an actual callable
    # network. We feed it pixel values and get steering/gas/brake back.
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    # ── Step 2: Set up environment ────────────────────────────────────────
    env = RacingEnv(seed=TRAINING_SEED, render=False)
    obs = env.reset()

    # ── Step 3: Run the episode ───────────────────────────────────────────
    total_reward  = 0.0
    total_speed   = 0.0
    grass_frames  = 0
    frame_count   = 0
    raw_obs_cache = None  # stores last raw obs for grass detection

    for _ in range(MAX_FRAMES):

        # Forward pass — feed 576 pixel values into the network
        # activate() runs the computation and returns 3 output values
        outputs = net.activate(obs)

        # Convert outputs to a valid driving action
        action = process_action(outputs)

        # Step the simulation one frame forward
        obs, reward, done = env.step(action)

        # Accumulate stats for fitness calculation
        total_reward += reward
        frame_count  += 1

        # Estimate speed from reward signal.
        # CarRacing-v3 gives reward proportional to track progress.
        # A positive reward this frame = moving forward on the track.
        # We use this as a speed proxy since we don't have direct
        # access to the physics engine's velocity value here.
        if reward > 0:
            total_speed += reward

        # Early stopping — don't waste time on hopeless genomes
        if total_reward < EARLY_STOP_THRESHOLD:
            break

        if done:
            break

    env.close()

    # ── Step 4: Calculate fitness ─────────────────────────────────────────
    avg_speed = total_speed / max(frame_count, 1)

    fitness = compute_fitness(
        total_env_reward = total_reward,
        frame_count      = frame_count,
        avg_speed        = avg_speed,
        grass_frames     = grass_frames
    )

    return fitness


# ── Population evaluation ───────────────────────────────────────────────────

def eval_genomes(genomes, config):
    """
    Evaluate every genome in one generation.

    NEAT calls this function once per generation, passing all
    genomes. We must set genome.fitness for each one.

    This is the SEQUENTIAL version — one genome at a time.
    Fine for early development and debugging.
    In Phase 2 we replace this with parallel evaluation
    (src/parallel_eval.py) so all 100 genomes run simultaneously
    across CPU cores — roughly 8-16x faster on a modern machine.

    Parameters
    ----------
    genomes : list of (genome_id, genome) tuples
    config  : neat.Config
    """
    total    = len(genomes)
    best_gen = float('-inf')

    for i, (genome_id, genome) in enumerate(genomes):
        genome.fitness = eval_genome(genome, config)
        best_gen       = max(best_gen, genome.fitness)

        # Progress indicator — shows which genome is being evaluated
        # and the running best so you can watch improvement in real time
        bar_len   = 20
        filled    = int(bar_len * (i + 1) / total)
        bar       = '█' * filled + '░' * (bar_len - filled)
        print(
            f"\r  [{bar}] {i+1:3d}/{total} "
            f"| Genome {genome_id:4d} "
            f"| Fitness {genome.fitness:8.2f} "
            f"| Best {best_gen:8.2f}",
            end='', flush=True
        )

    print()  # newline after progress bar