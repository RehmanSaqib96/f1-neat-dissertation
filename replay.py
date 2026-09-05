"""
replay.py
=========
Replays a saved genome visually in the CarRacing-v3 window.
Run from the project root:
    python replay.py --run A_spec5.0_rep1 --seed 42
    python replay.py --run D_spec1.5_gens600_rep2 --seed 555
"""

import argparse
import pickle
import neat
import numpy as np
from src.environment import RacingEnv


def replay(run_id, seed, slow=False):
    config = neat.Config(
        neat.DefaultGenome, neat.DefaultReproduction,
        neat.DefaultSpeciesSet, neat.DefaultStagnation,
        "config/neat_config.txt"
    )

    genome_path = f"results/runs/{run_id}_genome.pkl"
    with open(genome_path, "rb") as f:
        genome = pickle.load(f)

    net = neat.nn.FeedForwardNetwork.create(genome, config)

    print(f"Replaying: {run_id}  |  seed: {seed}")
    print(f"Genome: {len(genome.nodes)} nodes, {len(genome.connections)} connections")
    print(f"Stored fitness: {genome.fitness:.2f}")
    print("Close the window to exit.\n")

    env = RacingEnv(seed=seed, render=True)
    obs = env.reset()

    total_reward = 0
    tiles = 0
    frame = 0
    steer_vals = []

    for _ in range(1000):
        outputs = net.activate(obs)
        steer = float(np.clip(outputs[0], -1.0, 1.0))
        gas   = float(np.clip(outputs[1],  0.0, 1.0))
        brake = float(np.clip(outputs[2],  0.0, 1.0))

        obs, reward, done = env.step(np.array([steer, gas, brake]))
        total_reward += reward
        frame += 1
        steer_vals.append(steer)
        if reward > 1.0:
            tiles += 1

        if frame % 50 == 0:
            print(f"Frame {frame:4d} | reward {total_reward:7.2f} | "
                  f"steer {steer:+.4f} | gas {gas:.2f} | brake {brake:.2f} | tiles {tiles}")

        if slow:
            import time
            time.sleep(0.02)

        if done:
            print(f"\nEpisode ended at frame {frame}")
            break

    env.close()

    print(f"\nFinal reward: {total_reward:.2f}")
    print(f"Tiles visited: {tiles}")
    print(f"Steer range: min={min(steer_vals):.4f}  max={max(steer_vals):.4f}  "
          f"unique={len(set(round(s, 3) for s in steer_vals))}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replay a saved NEAT genome in CarRacing-v3")
    parser.add_argument("--run", default="A_spec5.0_rep1",
                        help="Run ID, e.g. A_spec5.0_rep1 or D_spec1.5_gens600_rep2")
    parser.add_argument("--seed", type=int, default=42,
                        help="Track seed to replay on (default: 42)")
    parser.add_argument("--slow", action="store_true",
                        help="Slow down to roughly 50fps for easier viewing or recording")
    args = parser.parse_args()
    replay(args.run, args.seed, args.slow)
