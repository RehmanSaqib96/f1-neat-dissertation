"""
neat_runner.py — core eval loop. NEAT calls eval_genomes() each generation;
each genome drives in simulation, receives a fitness score, and is ranked.
"""

import neat
import numpy as np
from src.environment import RacingEnv
from src.track_distance import get_track_points, corridor_fraction, corridor_penalty
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
    Evaluate one genome — v6, adds continuous corridor-distance reward.

    NEW IN v6:
        Every frame, compute how close the car is to the track edge
        (corridor_fraction) and apply a penalty if it's getting close
        (corridor_penalty). This is SEPARATE from the discrete tile
        reward — it gives steering a continuous, immediate benefit
        rather than only mattering at the exact moment of a corner.

        Verified in isolation: penalty stays at 0.0 during normal
        driving and starts rising ~70-90 frames before an actual
        off-track crash, giving the network early warning.
    """

    net  = neat.nn.FeedForwardNetwork.create(genome, config)
    env  = RacingEnv(seed=TRAINING_SEED, render=False)
    obs  = env.reset()

    # Get track centreline points once per episode (track layout
    # is fixed for a given seed, so this doesn't change frame to frame)
    track_points = get_track_points(env.env)

    frame_count        = 0
    tiles_visited       = 0
    total_env_reward    = 0.0
    steer_lock_frames   = 0
    off_track_frames    = 0
    total_corridor_pen  = 0.0

    frames_since_tile   = 0
    MAX_FRAMES_NO_TILE  = 150
    crashed_off_track    = False

    for _ in range(MAX_FRAMES):

        outputs = net.activate(obs)

        steer = float(np.clip(outputs[0], -1.0,  1.0))
        gas   = float(np.clip(outputs[1],  0.0,  1.0))
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

        # ── Corridor penalty — NEW continuous signal ───────────────────────
        car_pos = env.env.unwrapped.car.hull.position
        frac    = corridor_fraction(car_pos, track_points)
        pen     = corridor_penalty(frac)
        total_corridor_pen += pen

        if env_reward < -50:
            off_track_frames += 10
            crashed_off_track = True
            break

        if frame_count > 50 and frames_since_tile > MAX_FRAMES_NO_TILE:
            break

        if total_env_reward < -20:
            break

        if done:
            break

    env.close()

    seconds            = max(frame_count / 50.0, 0.1)
    tiles_per_second    = tiles_visited / seconds
    base_fitness        = tiles_per_second * 50.0
    completion_bonus    = tiles_visited * 0.5
    lock_penalty        = steer_lock_frames * 0.05

    # Average the corridor penalty per frame so longer episodes aren't
    # unfairly punished just for accumulating more penalty-frames
    avg_corridor_penalty = total_corridor_pen / max(frame_count, 1)
    corridor_fitness_hit = avg_corridor_penalty * 20.0

    fitness = base_fitness + completion_bonus - lock_penalty - corridor_fitness_hit

    if tiles_visited == 0:
        return -50.0

    if crashed_off_track:
        fitness = fitness * 0.15 - 30.0

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
