"""
replay.py
=========
Watches the best evolved genome drive.
Uses Gymnasium's built-in renderer cleanly —
no conflicting pygame windows.
"""

import pickle
import neat
import numpy as np
import os
import time

CONFIG_PATH      = os.path.join("config", "neat_config.txt")
BEST_GENOME_PATH = os.path.join("results", "best_genome.pkl")
NUM_EPISODES     = 3
MAX_FRAMES       = 2000


def process_action(outputs):
    steering = float(np.clip(outputs[0], -1.0,  1.0))
    gas      = float(np.clip(outputs[1],  0.0,  1.0))
    brake    = float(np.clip(outputs[2],  0.0,  1.0))
    gas      = max(gas, 0.1)
    return np.array([steering, gas, brake], dtype=np.float32)


def replay():
    import gymnasium as gym
    from src.environment import preprocess_observation

    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        CONFIG_PATH
    )

    with open(BEST_GENOME_PATH, "rb") as f:
        genome = pickle.load(f)

    net = neat.nn.FeedForwardNetwork.create(genome, config)

    print()
    print("=" * 55)
    print("  Best Genome Replay")
    print(f"  Fitness: {genome.fitness:.2f} | "
          f"Nodes: {len(genome.nodes)} | "
          f"Connections: {len(genome.connections)}")
    print("=" * 55)
    print()

    for episode in range(1, NUM_EPISODES + 1):
        print(f"  Episode {episode}/{NUM_EPISODES}")

        env = gym.make("CarRacing-v3", render_mode="human", continuous=True)
        obs, _ = env.reset(seed=42)

        total_reward      = 0.0
        steer_lock_frames = 0
        frame             = 0
        start             = time.time()

        while frame < MAX_FRAMES:
            processed = preprocess_observation(obs)
            outputs   = net.activate(processed)
            action    = process_action(outputs)

            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            frame        += 1

            if abs(action[0]) > 0.95:
                steer_lock_frames += 1

            # Print live telemetry to terminal every 100 frames
            if frame % 100 == 0:
                lock_pct = steer_lock_frames / frame * 100
                print(f"    Frame {frame:4d} | "
                      f"Reward: {total_reward:7.2f} | "
                      f"Steer: {action[0]:+.2f} | "
                      f"Gas: {action[1]*100:.0f}% | "
                      f"Lock%: {lock_pct:.0f}%")

            if terminated or truncated or total_reward < -15:
                break

        env.close()
        elapsed = time.time() - start
        lock_pct = steer_lock_frames / max(frame, 1) * 100
        print(f"  Result: Reward {total_reward:.2f} | "
              f"Frames {frame} | "
              f"Time {elapsed:.1f}s | "
              f"Steer lock {lock_pct:.0f}% of frames")
        print()

    print("  Replay complete.")
    print()
    print("  KEY FINDING: If steer lock % is high (>50%),")
    print("  the genome exploited the fitness function.")
    print("  This is fixed in fitness.py v2 — rerun training.")


if __name__ == "__main__":
    replay()