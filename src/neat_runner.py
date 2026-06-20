"""
neat_runner.py — core eval loop. NEAT calls eval_genomes() each generation;
each genome drives in simulation, receives a fitness score, and is ranked.
"""

import neat
import numpy as np
from src.environment import RacingEnv
from src.fitness     import compute_fitness, detect_grass

import multiprocessing
multiprocessing.freeze_support()  # Windows: prevent worker re-spawning on import

# ── Evaluation constants ────────────────────────────────────────────────────

MAX_FRAMES           = 1000   # ~20 s at 50 fps; increase in later phases
EARLY_STOP_THRESHOLD = -15.0  # kill hopeless genomes early to save time
TRAINING_SEED        = 42     # fixed seed → same track every eval (fair comparison)


# ── Action processing ───────────────────────────────────────────────────────

def process_action(outputs: list) -> np.ndarray:
    """
    Clip tanh network outputs into valid CarRacing-v3 action ranges.
    Enforces gas >= 0.1 to prevent the "sit still" no-op strategy.
    Returns [steering[-1,1], gas[0,1], brake[0,1]].
    """
    steering = float(np.clip(outputs[0], -1.0,  1.0))
    gas      = float(np.clip(outputs[1],  0.0,  1.0))
    brake    = float(np.clip(outputs[2],  0.0,  1.0))

    gas = max(gas, 0.1)  # minimum forward thrust

    return np.array([steering, gas, brake], dtype=np.float32)


# ── Single genome evaluation ────────────────────────────────────────────────

def eval_genome(genome, config) -> float:
    """
    Evaluate one genome — v5, tiles-per-second metric.

    This is the CURRENT, CORRECT version. Verify after saving by running:
        python -c "import inspect, src.neat_runner as nr; print(inspect.getsource(nr.eval_genome))"
    It must show 'tiles_per_second' in the output, NOT 'compute_fitness'.
    """

    net  = neat.nn.FeedForwardNetwork.create(genome, config)
    env  = RacingEnv(seed=TRAINING_SEED, render=False)
    obs  = env.reset()

    frame_count       = 0
    tiles_visited     = 0
    total_env_reward  = 0.0
    steer_lock_frames = 0
    off_track_frames  = 0

    frames_since_tile  = 0
    MAX_FRAMES_NO_TILE = 150

    for _ in range(MAX_FRAMES):

        outputs = net.activate(obs)

        steer = float(np.clip(outputs[0], -1.0,  1.0))
        gas   = float(np.clip(outputs[1],  0.0,  1.0))   # NO gas floor
        brake = float(np.clip(outputs[2],  0.0,  1.0))

        action = np.array([steer, gas, brake], dtype=np.float32)

        obs, env_reward, done = env.step(action)

        frame_count      += 1
        total_env_reward += env_reward

        if abs(steer) > 0.95:
            steer_lock_frames += 1

        if env_reward > 1.0:
            tiles_visited     += 1
            frames_since_tile  = 0
        else:
            frames_since_tile += 1

        if env_reward < -50:
            off_track_frames += 10

        if frame_count > 50 and frames_since_tile > MAX_FRAMES_NO_TILE:
            break

        if total_env_reward < -20:
            break

        if done:
            break

    env.close()

    seconds           = max(frame_count / 50.0, 0.1)
    tiles_per_second   = tiles_visited / seconds
    base_fitness       = tiles_per_second * 50.0
    completion_bonus   = tiles_visited * 0.5
    lock_penalty       = steer_lock_frames * 0.05
    off_track_penalty  = off_track_frames  * 2.0

    fitness = base_fitness + completion_bonus - lock_penalty - off_track_penalty

    if tiles_visited == 0:
        return -50.0

    return max(fitness, -100.0)


# ── Population evaluation ───────────────────────────────────────────────────

def eval_genomes(genomes, config):
    """Sequential eval — sets genome.fitness for every genome in the generation."""
    total    = len(genomes)
    best_gen = float('-inf')

    for i, (genome_id, genome) in enumerate(genomes):
        genome.fitness = eval_genome(genome, config)
        best_gen       = max(best_gen, genome.fitness)

        bar_len = 20
        filled  = int(bar_len * (i + 1) / total)
        bar     = '█' * filled + '░' * (bar_len - filled)
        print(
            f"\r  [{bar}] {i+1:3d}/{total} "
            f"| Genome {genome_id:4d} "
            f"| Fitness {genome.fitness:8.2f} "
            f"| Best {best_gen:8.2f}",
            end='', flush=True
        )

    print()
