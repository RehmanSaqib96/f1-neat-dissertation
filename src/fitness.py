"""
fitness.py — multi-objective fitness: progress + speed - time penalty - grass penalty.
Weights below are the primary experimental variable; wrong weights → degenerate behaviour.
"""

import numpy as np

# ── Fitness weights ────────────────────────────────────────────────────────
W_PROGRESS   = 1.0    # reward per track tile visited
W_SPEED      = 0.005  # reward per unit of speed
W_TIME       = 0.03   # penalty per frame
W_GRASS      = 0.5    # penalty per grass frame

# Pixels with G > threshold and G > R*1.3 are classified as grass.
GRASS_GREEN_THRESHOLD = 150


def compute_fitness(
    total_env_reward: float,
    frame_count: int,
    avg_speed: float,
    grass_frames: int
) -> float:
    """Return fitness score (higher = better). Floored at -100 so NEAT always has a signal."""

    base          = total_env_reward
    speed_bonus   = W_SPEED * avg_speed * frame_count
    grass_penalty = W_GRASS * grass_frames

    return max(base + speed_bonus - grass_penalty, -100.0)


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
