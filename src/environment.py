import gymnasium as gym
import numpy as np
import cv2

# ── Constants ──────────────────────────────────────────────────────────────
PROCESSED_WIDTH  = 24
PROCESSED_HEIGHT = 24
NUM_INPUTS       = PROCESSED_WIDTH * PROCESSED_HEIGHT  # 576 — must match neat_config.txt

# ── Preprocessing ──────────────────────────────────────────────────────────

def preprocess_observation(obs: np.ndarray) -> np.ndarray:
    """Grayscale → resize 96x96→24x24 → normalise [0,1] → flatten. Returns (576,) float32."""
    gray      = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
    resized   = cv2.resize(gray, (PROCESSED_WIDTH, PROCESSED_HEIGHT),
                           interpolation=cv2.INTER_AREA)
    normalised = resized / 255.0
    return normalised.flatten().astype(np.float32)


# ── Environment wrapper ─────────────────────────────────────────────────────

class RacingEnv:
    """
    Thin wrapper around CarRacing-v3 for NEAT evaluation.

    seed   : controls track layout (same seed = same track).
    render : keep False during training (~40% CPU overhead if True).
    """

    def __init__(self, seed: int = 42, render: bool = False):
        render_mode  = "human" if render else None
        self.env     = gym.make(
            "CarRacing-v3",
            render_mode=render_mode,
            continuous=True   # action space = [steering, gas, brake]
        )
        self.seed = seed

    def reset(self) -> np.ndarray:
        """Reset episode, return preprocessed obs (576,)."""
        obs, _ = self.env.reset(seed=self.seed)
        return preprocess_observation(obs)

    def step(self, action: np.ndarray):
        """
        Step one frame. action = [steering[-1,1], gas[0,1], brake[0,1]].
        Returns (obs, reward, done).
        """
        obs, reward, terminated, truncated, _ = self.env.step(action)
        done = terminated or truncated
        return preprocess_observation(obs), reward, done

    def close(self):
        self.env.close()
