"""
Loads the best evolved genome and replays it in a visible window.
"""

import pickle
import neat
import numpy as np
import os
from src.environment import RacingEnv, preprocess_observation

# ── Config ──────────────────────────────────────────────────────────────────
CONFIG_PATH      = os.path.join("config", "neat_config.txt")
BEST_GENOME_PATH = os.path.join("results", "best_genome.pkl")

# How many laps / episodes to replay
NUM_EPISODES = 3

# Max frames per episode
MAX_FRAMES = 2000


def process_action(outputs):
    steering = float(np.clip(outputs[0], -1.0,  1.0))
    gas      = float(np.clip(outputs[1],  0.0,  1.0))
    brake    = float(np.clip(outputs[2],  0.0,  1.0))
    gas      = max(gas, 0.1)
    return np.array([steering, gas, brake], dtype=np.float32)


def replay():
    # ── Load config ────────────────────────────────────────────────────────
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        CONFIG_PATH
    )

    # ── Load best genome ───────────────────────────────────────────────────
    if not os.path.exists(BEST_GENOME_PATH):
        print(f"ERROR: No best genome found at {BEST_GENOME_PATH}")
        print("Run 'python main.py' first to train the agent.")
        return

    with open(BEST_GENOME_PATH, "rb") as f:
        genome = pickle.load(f)

    print()
    print("=" * 50)
    print("  Replaying Best Evolved Genome")
    print("=" * 50)
    print(f"  Fitness    : {genome.fitness:.2f}")
    print(f"  Nodes      : {len(genome.nodes)}")
    print(f"  Connections: {len(genome.connections)}")
    print("=" * 50)
    print()
    print("  A window will open showing the car driving.")
    print("  Close the window or press Ctrl+C to stop.")
    print()

    # ── Build network ──────────────────────────────────────────────────────
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    # ── Run episodes ───────────────────────────────────────────────────────
    for episode in range(NUM_EPISODES):
        print(f"  Episode {episode + 1}/{NUM_EPISODES} starting...")

        # render=True opens the visible window
        env = RacingEnv(seed=42, render=True)
        obs = env.reset()

        total_reward = 0.0
        frame_count  = 0

        try:
            for _ in range(MAX_FRAMES):
                outputs = net.activate(obs)
                action  = process_action(outputs)
                obs, reward, done = env.step(action)
                total_reward += reward
                frame_count  += 1

                if done:
                    break

        except KeyboardInterrupt:
            print("\n  Stopped by user.")
            env.close()
            return

        env.close()
        print(f"  Episode {episode + 1} complete — "
              f"Reward: {total_reward:.2f} — "
              f"Frames: {frame_count}")

    print()
    print("  Replay complete.")


if __name__ == "__main__":
    replay()