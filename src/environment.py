import gymnasium as gym
import numpy as np
import cv2

# ── Constants ──────────────────────────────────────────────────────────────
PROCESSED_WIDTH  = 24
PROCESSED_HEIGHT = 24
NUM_INPUTS       = PROCESSED_WIDTH * PROCESSED_HEIGHT  # 576 — must match neat_config.txt

# ── Preprocessing ──────────────────────────────────────────────────────────

def preprocess_observation(obs: np.ndarray) -> np.ndarray:
    """
    Convert a raw 96x96x3 RGB frame into a flat 576-element array.

    Steps:
        1. Grayscale  — 3 colour channels → 1 intensity channel
        2. Resize     — 96x96 → 24x24 (reduces input count by 94%)
        3. Normalise  — pixel values from [0,255] → [0.0, 1.0]
        4. Flatten    — 2D grid → 1D array for NEAT input nodes

    Parameters
    ----------
    obs : np.ndarray  shape (96, 96, 3)

    Returns
    -------
    np.ndarray  shape (576,)  dtype float32  values in [0.0, 1.0]
    """
    gray      = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
    resized   = cv2.resize(gray, (PROCESSED_WIDTH, PROCESSED_HEIGHT),
                           interpolation=cv2.INTER_AREA)
    normalised = resized / 255.0
    return normalised.flatten().astype(np.float32)


# ── Environment wrapper ─────────────────────────────────────────────────────

class RacingEnv:
    """
    Clean wrapper around CarRacing-v3 for NEAT evaluation.

    Parameters
    ----------
    seed   : int   Controls which track layout is generated.
                   Same seed = same track every time (reproducibility).
                   Different seeds = different layouts (generalisation testing).
    render : bool  True opens a visible window. ALWAYS False during training —
                   rendering costs ~40% extra CPU time per genome.
    """

    def __init__(self, seed: int = 42, render: bool = False):
        render_mode  = "human" if render else None
        self.env     = gym.make(
            "CarRacing-v3",
            render_mode=render_mode,
            continuous=True   # action space = [steering, gas, brake] as floats
        )
        self.seed = seed

    def reset(self) -> np.ndarray:
        """
        Reset to start of a new episode.
        Returns preprocessed flat observation (576 elements).
        """
        obs, _ = self.env.reset(seed=self.seed)
        return preprocess_observation(obs)

    def step(self, action: np.ndarray):
        """
        Advance simulation by one frame.

        Parameters
        ----------
        action : array-like  [steering, gas, brake]
            steering : float  [-1.0,  1.0]   left=negative  right=positive
            gas      : float  [ 0.0,  1.0]
            brake    : float  [ 0.0,  1.0]

        Returns
        -------
        obs    : np.ndarray  preprocessed frame (576,)
        reward : float       raw environment reward
        done   : bool        True if episode ended
        """
        obs, reward, terminated, truncated, _ = self.env.step(action)
        done = terminated or truncated
        return preprocess_observation(obs), reward, done

    def close(self):
        self.env.close()