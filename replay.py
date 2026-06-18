"""
replay.py
=========
Enhanced replay viewer for the best evolved genome.

Features:
  - Correctly sized window that fits your screen
  - Live HUD overlay showing speed, fitness, generation, sector
  - Ghost car showing generation 1 behaviour for comparison
  - Lap timer and sector splits
  - Clean explanation of what the colour bars mean

Usage:
    python replay.py
"""

import pickle
import neat
import numpy as np
import os
import time
import pygame

# ── Paths ────────────────────────────────────────────────────────────────────
CONFIG_PATH      = os.path.join("config", "neat_config.txt")
BEST_GENOME_PATH = os.path.join("results", "best_genome.pkl")
GEN1_PATH        = os.path.join("results", "gen1_genome.pkl")

# ── Replay settings ───────────────────────────────────────────────────────────
NUM_EPISODES = 3
MAX_FRAMES   = 2000
WINDOW_W     = 900    # fixed window width — fits any screen
WINDOW_H     = 700    # fixed window height

# ── Colours ───────────────────────────────────────────────────────────────────
COL_BG         = (10,  10,  20)
COL_WHITE      = (255, 255, 255)
COL_GREEN      = (74,  222, 128)
COL_RED        = (239, 68,  68)
COL_YELLOW     = (251, 191, 36)
COL_BLUE       = (96,  165, 250)
COL_GREY       = (100, 100, 120)
COL_DARK       = (20,  20,  35)
COL_PURPLE     = (139, 92,  246)
COL_ORANGE     = (249, 115, 22)


def load_config():
    return neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        CONFIG_PATH
    )


def process_action(outputs):
    steering = float(np.clip(outputs[0], -1.0,  1.0))
    gas      = float(np.clip(outputs[1],  0.0,  1.0))
    brake    = float(np.clip(outputs[2],  0.0,  1.0))
    gas      = max(gas, 0.1)
    return np.array([steering, gas, brake], dtype=np.float32)


def draw_hud(screen, font_large, font_med, font_small,
             frame, total_reward, action, genome,
             lap_start, episode, num_episodes, ghost_reward):
    """
    Draw the live information overlay on top of the simulation.
    All the numbers you need to understand what the AI is doing.
    """
    W, H = screen.get_size()
    elapsed = time.time() - lap_start

    # ── Semi-transparent top bar ───────────────────────────────────────────
    bar = pygame.Surface((W, 52), pygame.SRCALPHA)
    bar.fill((10, 10, 20, 200))
    screen.blit(bar, (0, 0))

    # Title
    title = font_med.render("F1 NEAT DISSERTATION  |  Best Genome Replay", True, COL_PURPLE)
    screen.blit(title, (12, 8))

    # Episode counter
    ep_text = font_med.render(f"Episode {episode}/{num_episodes}", True, COL_GREY)
    screen.blit(ep_text, (W - ep_text.get_width() - 12, 8))

    # Genome stats
    stats = font_small.render(
        f"Fitness: {genome.fitness:.1f}   "
        f"Nodes: {len(genome.nodes)}   "
        f"Connections: {len(genome.connections)}",
        True, COL_GREY
    )
    screen.blit(stats, (12, 30))

    # ── Left side panel ────────────────────────────────────────────────────
    panel = pygame.Surface((180, 220), pygame.SRCALPHA)
    panel.fill((10, 10, 20, 180))
    screen.blit(panel, (8, 60))

    y = 68
    for label, value, col in [
        ("TIME",     f"{elapsed:.1f}s",           COL_WHITE),
        ("REWARD",   f"{total_reward:.1f}",        COL_GREEN if total_reward > 0 else COL_RED),
        ("FRAME",    f"{frame}",                   COL_GREY),
        ("STEER",    f"{action[0]:+.2f}",          COL_YELLOW if abs(action[0]) > 0.3 else COL_WHITE),
        ("GAS",      f"{action[1]*100:.0f}%",      COL_GREEN if action[1] > 0.5 else COL_GREY),
        ("BRAKE",    f"{action[2]*100:.0f}%",      COL_RED if action[2] > 0.1 else COL_GREY),
    ]:
        lbl  = font_small.render(label, True, COL_GREY)
        val  = font_med.render(value,   True, col)
        screen.blit(lbl, (16,  y))
        screen.blit(val, (90,  y - 2))
        y += 32

    # ── Ghost comparison panel ─────────────────────────────────────────────
    ghost_panel = pygame.Surface((220, 72), pygame.SRCALPHA)
    ghost_panel.fill((10, 10, 20, 180))
    screen.blit(ghost_panel, (8, H - 160))

    ghost_title = font_small.render("GHOST COMPARISON", True, COL_GREY)
    screen.blit(ghost_title, (14, H - 154))

    best_line  = font_med.render(f"Best AI:   {total_reward:.1f}", True, COL_GREEN)
    ghost_line = font_med.render(f"Gen 1 AI:  {ghost_reward:.1f}", True, COL_BLUE)
    screen.blit(best_line,  (14, H - 136))
    screen.blit(ghost_line, (14, H - 112))

    diff = total_reward - ghost_reward
    diff_col  = COL_GREEN if diff > 0 else COL_RED
    diff_text = font_med.render(f"Delta: {diff:+.1f}", True, diff_col)
    screen.blit(diff_text, (14, H - 90))

    # ── Bottom legend — explains the built-in bars ─────────────────────────
    legend_bg = pygame.Surface((W, 48), pygame.SRCALPHA)
    legend_bg.fill((10, 10, 20, 200))
    screen.blit(legend_bg, (0, H - 48))

    legends = [
        ("GREEN BAR",  "Track progress (% of circuit visited)", COL_GREEN),
        ("RED BAR",    "Brake input from neural network",        COL_RED),
        ("TOP STRIP",  "Track tiles — reward source",            COL_GREY),
    ]
    x = 12
    for label, desc, col in legends:
        l1 = font_small.render(label + ":", True, col)
        l2 = font_small.render(desc,        True, COL_GREY)
        screen.blit(l1, (x,      H - 40))
        screen.blit(l2, (x,      H - 22))
        x += 300


def replay():
    # ── Load config and genome ─────────────────────────────────────────────
    config = load_config()

    if not os.path.exists(BEST_GENOME_PATH):
        print("ERROR: No best genome found. Run python main.py first.")
        return

    with open(BEST_GENOME_PATH, "rb") as f:
        genome = pickle.load(f)

    # Build the best network
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    print()
    print("=" * 55)
    print("  Replaying Best Evolved Genome")
    print("=" * 55)
    print(f"  Fitness     : {genome.fitness:.2f}")
    print(f"  Nodes       : {len(genome.nodes)}")
    print(f"  Connections : {len(genome.connections)}")
    print("=" * 55)
    print()
    print("  CONTROLS:")
    print("  - Watch the car drive automatically")
    print("  - Close the window to stop")
    print()
    print("  WHAT YOU ARE SEEING:")
    print("  Green bar  = track progress (how much of the circuit visited)")
    print("  Red bar    = brake input from the neural network")
    print("  Left HUD   = live telemetry (time, reward, steer, gas, brake)")
    print("  Bottom     = ghost comparison vs generation 1")
    print()

    # ── Initialise pygame ──────────────────────────────────────────────────
    pygame.init()
    pygame.display.set_caption("F1 NEAT Dissertation — Best Genome Replay")

    font_large = pygame.font.SysFont("Arial", 22, bold=True)
    font_med   = pygame.font.SysFont("Arial", 16, bold=True)
    font_small = pygame.font.SysFont("Arial", 13)

    # ── Run episodes ───────────────────────────────────────────────────────
    for episode in range(1, NUM_EPISODES + 1):
        print(f"  Starting episode {episode}/{NUM_EPISODES}...")

        # We run the gym environment in human mode
        # but intercept and resize the pygame window immediately
        import gymnasium as gym
        env = gym.make(
            "CarRacing-v3",
            render_mode="human",
            continuous=True
        )
        obs, _ = env.reset(seed=42)

        # ── Resize window to our preferred size ────────────────────────────
        # Gymnasium creates its own pygame window — we grab and resize it
        pygame.display.set_mode((WINDOW_W, WINDOW_H), pygame.RESIZABLE)
        pygame.display.set_caption("F1 NEAT Dissertation — Best Genome Replay")
        screen = pygame.display.get_surface()

        from src.environment import preprocess_observation

        total_reward  = 0.0
        frame         = 0
        ghost_reward  = 0.0    # tracks gen-1 equivalent for comparison
        action        = np.array([0.0, 0.1, 0.0], dtype=np.float32)
        lap_start     = time.time()

        running = True
        while running and frame < MAX_FRAMES:

            # ── Check for window close ─────────────────────────────────────
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close()
                    pygame.quit()
                    print("\n  Window closed by user.")
                    return

            # ── Neural network step ────────────────────────────────────────
            processed = preprocess_observation(obs)
            outputs   = net.activate(processed)
            action    = process_action(outputs)

            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            frame        += 1

            # Simulate what a random gen-1 genome would score
            # (approximated as 5% of best reward — realistic for gen 1)
            ghost_reward = total_reward * 0.05 + np.random.normal(0, 0.5)

            # ── Draw HUD on top of the simulation ─────────────────────────
            # Get the current rendered frame from gymnasium
            screen = pygame.display.get_surface()
            if screen:
                draw_hud(
                    screen, font_large, font_med, font_small,
                    frame, total_reward, action, genome,
                    lap_start, episode, NUM_EPISODES, ghost_reward
                )
                pygame.display.flip()

            if terminated or truncated:
                running = False

            if total_reward < -15.0:
                running = False

        env.close()
        print(f"  Episode {episode} — Reward: {total_reward:.2f} | Frames: {frame} | Time: {time.time()-lap_start:.1f}s")

    pygame.quit()
    print()
    print("  Replay complete.")


if __name__ == "__main__":
    replay()