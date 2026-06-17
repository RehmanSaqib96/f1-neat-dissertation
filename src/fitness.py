"""
fitness.py
==========
The multi-objective fitness function — the most important design
decision in the entire project.

This is the signal that tells NEAT what "good driving" means.
Get it wrong and NEAT finds ways to cheat it (e.g. driving in
circles scores tiles without ever completing a lap).

Our function balances four things:
    1. Progress     — how many track tiles were visited
    2. Speed        — reward going fast
    3. Efficiency   — penalise wasting time
    4. Track limits — penalise going on the grass

WHY THIS MATTERS FOR YOUR DISSERTATION:
The weighting of these four components is a key experimental
variable. Chapter 4 of your dissertation will analyse how
changing these weights affects the evolved driving behaviour.
"""

# ── Fitness weights ────────────────────────────────────────────────────────
# These are your starting values. We will tune them in Phase 2.
# Documented here so every experiment is reproducible.

W_PROGRESS   = 1.0    # reward per track tile visited
W_SPEED      = 0.005  # reward per unit of speed
W_TIME       = 0.03   # penalty per frame (encourages faster completion)
W_GRASS      = 0.5    # penalty per frame spent on grass (green pixels)

# Grass detection threshold — pixels brighter than this in the
# green channel are likely grass, not tarmac.
GRASS_GREEN_THRESHOLD = 150


def compute_fitness(
    total_env_reward: float,
    frame_count: int,
    avg_speed: float,
    grass_frames: int
) -> float:
    """
    Compute the final fitness score for one genome's episode.

    Parameters
    ----------
    total_env_reward : float
        Cumulative reward from CarRacing-v3.
        The environment gives +1000/N per tile visited (N = total tiles)
        and -0.1 every frame (built-in time penalty).

    frame_count : int
        Total frames the genome survived. Max is MAX_FRAMES in neat_runner.py.

    avg_speed : float
        Mean speed across all frames of the episode.
        Extracted from the environment's internal state.

    grass_frames : int
        Number of frames the car spent on grass/off-track.
        Detected via pixel colour analysis.

    Returns
    -------
    float
        Fitness score. Higher = better driver.
    """

    # Base score — the environment's own reward signal
    # (already encodes tile progress and a time penalty)
    base = total_env_reward

    # Speed bonus — encourages aggressive throttle application
    speed_bonus = W_SPEED * avg_speed * frame_count

    # Grass penalty — discourages cutting corners or going off-track
    grass_penalty = W_GRASS * grass_frames

    fitness = base + speed_bonus - grass_penalty

    # Never return negative infinity — NEAT needs a numeric signal
    # even from terrible genomes so it knows direction of improvement
    return max(fitness, -100.0)


def detect_grass(obs_flat: 'np.ndarray') -> bool:
    """
    Detect whether the car is currently on grass using pixel analysis.

    The 24x24 preprocessed observation is grayscale so we can't use
    colour directly. Instead we check the RAW observation in neat_runner.py
    before preprocessing — this function takes the centre-bottom region
    of the raw frame (where the car bonnet is visible) and checks for
    green-dominant pixels.

    This is called in neat_runner.py with the raw observation, not the
    preprocessed one.

    Parameters
    ----------
    obs_flat : np.ndarray  shape (96, 96, 3)  raw RGB observation

    Returns
    -------
    bool  True if grass detected under/around the car
    """
    # Centre-bottom strip of the frame — where the track should be
    # if the car is on it. Rows 60-80, columns 30-65.
    region = obs_flat[60:80, 30:65]

    # In CarRacing-v3, grass is bright green (high G channel, lower R and B)
    green_channel = region[:, :, 1]  # G
    red_channel   = region[:, :, 0]  # R

    green_pixels = np.sum(
        (green_channel > GRASS_GREEN_THRESHOLD) &
        (green_channel > red_channel * 1.3)
    )

    # If more than 20% of the region is green, we're on grass
    total_pixels = region.shape[0] * region.shape[1]
    return (green_pixels / total_pixels) > 0.20