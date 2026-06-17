"""
verify_setup.py
===============
Run this to confirm your entire environment is working correctly
before we start training.

Every block should print OK.
Any ERROR = fix that before proceeding.
"""

import sys
print(f"Python version: {sys.version}\n")

# ── 1. Core libraries ───────────────────────────────────────────────────────
print("── Checking libraries ──────────────────────────────")

try:
    import neat
    print(f"✅  neat-python     {neat.__version__}")
except Exception as e:
    print(f"❌  neat-python     FAILED: {e}")

try:
    import gymnasium as gym
    print(f"✅  gymnasium       {gym.__version__}")
except Exception as e:
    print(f"❌  gymnasium       FAILED: {e}")

try:
    import numpy as np
    print(f"✅  numpy           {np.__version__}")
except Exception as e:
    print(f"❌  numpy           FAILED: {e}")

try:
    import cv2
    print(f"✅  opencv          {cv2.__version__}")
except Exception as e:
    print(f"❌  opencv          FAILED: {e}")

try:
    import fastf1
    print(f"✅  fastf1          {fastf1.__version__}")
except Exception as e:
    print(f"❌  fastf1          FAILED: {e}")

try:
    import pandas as pd
    print(f"✅  pandas          {pd.__version__}")
except Exception as e:
    print(f"❌  pandas          FAILED: {e}")

try:
    import matplotlib
    print(f"✅  matplotlib      {matplotlib.__version__}")
except Exception as e:
    print(f"❌  matplotlib      FAILED: {e}")

try:
    import seaborn as sns
    print(f"✅  seaborn         {sns.__version__}")
except Exception as e:
    print(f"❌  seaborn         FAILED: {e}")

try:
    import shapely
    print(f"✅  shapely         {shapely.__version__}")
except Exception as e:
    print(f"❌  shapely         FAILED: {e}")

try:
    import pygame
    print(f"✅  pygame          {pygame.__version__}")
except Exception as e:
    print(f"❌  pygame          FAILED: {e}")

try:
    import plotly
    print(f"✅  plotly          {plotly.__version__}")
except Exception as e:
    print(f"❌  plotly          FAILED: {e}")

# ── 2. CarRacing-v3 environment ─────────────────────────────────────────────
print("\n── Checking CarRacing-v3 ───────────────────────────")
try:
    import gymnasium as gym
    import numpy as np
    env = gym.make("CarRacing-v3", continuous=True)
    obs, _ = env.reset(seed=42)
    assert obs.shape == (96, 96, 3), f"Unexpected shape: {obs.shape}"
    action = env.action_space.sample()
    obs2, reward, terminated, truncated, info = env.step(action)
    env.close()
    print(f"✅  CarRacing-v3    OK — obs shape {obs.shape}, reward {reward:.4f}")
except Exception as e:
    print(f"❌  CarRacing-v3    FAILED: {e}")

# ── 3. Preprocessing pipeline ───────────────────────────────────────────────
print("\n── Checking preprocessing ──────────────────────────")
try:
    import cv2
    import numpy as np
    dummy = np.random.randint(0, 255, (96, 96, 3), dtype=np.uint8)
    gray = cv2.cvtColor(dummy, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (24, 24), interpolation=cv2.INTER_AREA)
    flat = (resized / 255.0).flatten().astype(np.float32)
    assert flat.shape == (576,), f"Wrong shape: {flat.shape}"
    assert flat.min() >= 0.0 and flat.max() <= 1.0
    print(f"✅  Preprocessing   OK — output shape {flat.shape}")
except Exception as e:
    print(f"❌  Preprocessing   FAILED: {e}")

# ── 4. NEAT config loading ───────────────────────────────────────────────────
print("\n── Checking NEAT config ────────────────────────────")
try:
    import neat, os
    config_path = os.path.join("config", "neat_config.txt")
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path
    )
    pop = neat.Population(config)
    print(f"✅  NEAT config     OK — population size {config.pop_size}")
except Exception as e:
    print(f"❌  NEAT config     FAILED: {e}")

# ── 5. FastF1 ───────────────────────────────────────────────────────────────
print("\n── Checking FastF1 ─────────────────────────────────")
try:
    import fastf1, os
    cache_dir = os.path.join("data", "f1_telemetry", "cache")
    os.makedirs(cache_dir, exist_ok=True)
    fastf1.Cache.enable_cache(cache_dir)
    print(f"✅  FastF1          OK — cache ready at {cache_dir}")
except Exception as e:
    print(f"❌  FastF1          FAILED: {e}")

# ── Summary ─────────────────────────────────────────────────────────────────
print("\n────────────────────────────────────────────────────")
print("Verification complete.")
print("All ✅ = ready to build.")
print("Any ❌ = paste the error here before proceeding.")