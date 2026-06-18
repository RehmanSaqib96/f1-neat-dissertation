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
    Evaluate one genome — v3 with structural exploit prevention.

    Key changes from v2:
    - Episode killed if car is off-track for more than 60 consecutive frames
    - Episode killed if speed is below minimum for more than 100 frames
    - Reward capped per frame — prevents tile-grinding exploit
    - Forward progress check — car must be moving forward to score
    """

    net = neat.nn.FeedForwardNetwork.create(genome, config)
    env = RacingEnv(seed=TRAINING_SEED, render=False)
    obs = env.reset()

    total_reward      = 0.0
    total_speed       = 0.0
    grass_frames      = 0
    steer_lock_frames = 0
    frame_count       = 0

    # ── Exploit prevention counters ───────────────────────────────────────
    # ── Exploit prevention counters ───────────────────────────────────────
    consecutive_grass     = 0
    MAX_CONSECUTIVE_GRASS = 120  # raised — 60 was too tight
    WARMUP_FRAMES         = 100  # ignore checks during zoom-in animation

    for _ in range(MAX_FRAMES):

        outputs = net.activate(obs)

        steer    = float(np.clip(outputs[0], -1.0,  1.0))
        gas      = float(np.clip(outputs[1],  0.0,  1.0))
        brake    = float(np.clip(outputs[2],  0.0,  1.0))
        gas      = max(gas, 0.1)
        action   = np.array([steer, gas, brake], dtype=np.float32)

        obs, reward, done = env.step(action)

        total_reward += reward
        frame_count  += 1

        # Track speed proxy
        if reward > 0:
            total_speed += reward

        # Track steer lock
        if abs(steer) > 0.95:
            steer_lock_frames += 1

        # ── Exploit prevention checks ─────────────────────────────────────

        # Check if on grass (only after warmup — first 100 frames
        # are the zoom-in animation where reward is always near zero)
        if frame_count > WARMUP_FRAMES:
            if reward < 0.1:
                consecutive_grass += 1
                grass_frames      += 1
            else:
                consecutive_grass  = 0

            # Kill episode if stuck off track continuously
            if consecutive_grass > MAX_CONSECUTIVE_GRASS:
                total_reward -= 20.0
                break

        # Standard termination
        if total_reward < EARLY_STOP_THRESHOLD:
            break

        if done:
            break

    env.close()

    avg_speed = total_speed / max(frame_count, 1)

    from src.fitness import compute_fitness
    fitness = compute_fitness(
        total_env_reward  = total_reward,
        frame_count       = frame_count,
        avg_speed         = avg_speed,
        grass_frames      = grass_frames,
        steer_lock_frames = steer_lock_frames
    )

    return fitness


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
