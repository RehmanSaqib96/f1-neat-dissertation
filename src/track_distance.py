"""
track_distance.py
==================
Computes the car's lateral distance from the track centreline,
expressed as a fraction of the track's half-width.

WHY THIS EXISTS:
    Reward hacking investigation (see Progress Report v3) found that
    the agent learned to brake-and-idle rather than steer, because the
    fitness function only rewarded discrete tile crossings — steering
    had no benefit except at the exact moment of a corner.

    This module provides a CONTINUOUS, per-frame signal: how close is
    the car to running off the track edge. This gives steering an
    immediate benefit on every single frame, not just at corners.

DESIGN DECISION:
    We do NOT reward "distance to centre" directly, because that would
    teach the car to hug the centreline everywhere — which is not how
    real racing lines work (cars use the full track width, including
    apexing near the inside edge of corners).

    Instead we return a CORRIDOR fraction: 0.0 at the centre, 1.0 at
    the very edge of the track. The fitness function (built separately)
    will only penalise once this fraction gets large — comfortably
    inside the track costs nothing, approaching the edge costs more
    and more as the car gets closer to going off.

CarRacing-v3 internals used:
    env.unwrapped.car.hull.position  — car's (x, y) in world space
    env.unwrapped.track               — list of (alpha, beta, x, y) tuples
                                         representing the centreline,
                                         evenly spaced, ~3-4 units apart
    TRACK_WIDTH = 6.667                — half-width of the track corridor
"""

import numpy as np

TRACK_WIDTH = 6.666666666666667  # confirmed from gymnasium.envs.box2d.car_racing


def get_track_points(env) -> np.ndarray:
    """
    Extract the centreline as a flat (N, 2) array of (x, y) points.

    Parameters
    ----------
    env : the unwrapped CarRacing-v3 environment
          (call env.unwrapped if using a wrapped/gym.make environment)

    Returns
    -------
    np.ndarray  shape (N, 2)
    """
    track = env.unwrapped.track
    points = np.array([(pt[2], pt[3]) for pt in track], dtype=np.float64)
    return points


def corridor_fraction(car_pos: tuple, track_points: np.ndarray) -> float:
    """
    How far the car is from the centreline, as a fraction of half-width.

    Parameters
    ----------
    car_pos       : (x, y) tuple — car.hull.position
    track_points  : (N, 2) array from get_track_points()

    Returns
    -------
    float
        0.0   = exactly on the centreline
        1.0   = exactly at the track edge
        >1.0  = off the track entirely (in the grass)
    """
    cx, cy = car_pos[0], car_pos[1]

    # Squared distance to every track point — avoids sqrt until the end,
    # cheap enough for 283 points every frame (no spatial indexing needed)
    dx = track_points[:, 0] - cx
    dy = track_points[:, 1] - cy
    dist_sq = dx * dx + dy * dy

    nearest_dist = np.sqrt(np.min(dist_sq))

    return nearest_dist / TRACK_WIDTH


def corridor_penalty(fraction: float, threshold: float = 0.6) -> float:
    """
    Convert a corridor fraction into a penalty value.

    Design: zero penalty while comfortably inside the track
    (fraction < threshold), then penalty increases sharply as the
    car approaches and crosses the edge. This avoids punishing
    normal racing-line use of the track width while still giving
    a strong, continuous incentive to steer away from the edge
    before a crash happens.

    Parameters
    ----------
    fraction  : output of corridor_fraction()
    threshold : fraction below which there is no penalty at all
                (default 0.6 — car can use 60% of the half-width
                freely, only the outer 40% incurs cost)

    Returns
    -------
    float  penalty value, 0.0 or positive. Subtract this from fitness.
    """
    if fraction <= threshold:
        return 0.0

    # Quadratic ramp from threshold (0 penalty) to 1.0 (off track edge)
    # and beyond (fully in the grass) — squared so it's gentle near
    # the threshold and sharp near/past the edge.
    overshoot = fraction - threshold
    return (overshoot ** 2) * 10.0