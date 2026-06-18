"""
fitness.py — multi-objective fitness: progress + speed - time penalty - grass penalty.
Weights below are the primary experimental variable; wrong weights → degenerate behaviour.
"""

import numpy as np

# ── Fitness weights v3 ─────────────────────────────────────────────────────
# Structural exploit prevention now handles the main cheat strategies.
# These weights fine-tune quality of genuine driving behaviour.
W_PROGRESS    = 1.0    # base environment reward
W_SPEED       = 0.03   # speed bonus
W_TIME        = 0.05   # time penalty
W_GRASS       = 5.0    # grass penalty — extremely costly
W_STEER_LOCK  = 0.15   # steer lock penalty

# Pixels with G > threshold and G > R*1.3 are classified as grass.
GRASS_GREEN_THRESHOLD = 150


def compute_fitness(
    total_env_reward: float,
    frame_count: int,
    avg_speed: float,
    grass_frames: int,
    steer_lock_frames: int = 0
) -> float:
    """
    Compute fitness score for one genome's episode.

    v2 changes:
    - Grass penalty tripled (3.0 vs 0.5)
    - Speed bonus quadrupled (0.02 vs 0.005)
    - Time penalty increased (0.05 vs 0.03)
    - Steer lock penalty added — discourages always-right strategy
    - Speed efficiency bonus — rewards fast tile completion

    """
    # Base environment reward
    base = total_env_reward

    # Speed bonus — rewards aggressive throttle and fast lap times
    speed_bonus = W_SPEED * avg_speed * frame_count

    # Grass penalty — heavily discourages off-track driving
    grass_penalty = W_GRASS * grass_frames

    # Steer lock penalty — discourages the always-steer-right cheat
    steer_penalty = W_STEER_LOCK * steer_lock_frames

    # Speed efficiency bonus — rewards completing tiles quickly
    # Divides progress by time, so slow crawling scores worse
    if frame_count > 0 and total_env_reward > 0:
        efficiency_bonus = (total_env_reward / frame_count) * 50.0
    else:
        efficiency_bonus = 0.0

    fitness = base + speed_bonus + efficiency_bonus - grass_penalty - steer_penalty

    return max(fitness, -100.0)


def detect_grass(obs_flat: 'np.ndarray') -> bool:
    """
    True if the car is on grass. Uses raw 96x96x3 RGB frame (not preprocessed).
    Checks the centre-bottom strip [60:80, 30:65] for green-dominant pixels.
    """
    region = obs_flat[60:80, 30:65]

    green_channel = region[:, :, 1]  # G
    red_channel   = region[:, :, 0]  # R

    green_pixels = np.sum(
        (green_channel > GRASS_GREEN_THRESHOLD) &
        (green_channel > red_channel * 1.3)
    )

    total_pixels = region.shape[0] * region.shape[1]
    return (green_pixels / total_pixels) > 0.20
