"""
Full visual replay with HUD in a SEPARATE pygame window
running alongside the Gymnasium simulation window.

Two windows open side by side:
    Left  — Gymnasium's CarRacing-v3 simulation (the car driving)
    Right — Our custom HUD with live telemetry, charts, ghost comparison

This avoids all conflicts between the two pygame instances.
"""

import pickle
import neat
import numpy as np
import os
import time
import threading
import gymnasium as gym
import pygame
from src.environment import preprocess_observation

# ── Paths ────────────────────────────────────────────────────────────────────
CONFIG_PATH      = os.path.join("config", "neat_config.txt")
BEST_GENOME_PATH = os.path.join("results", "best_genome.pkl")

# ── Settings ─────────────────────────────────────────────────────────────────
NUM_EPISODES  = 3
MAX_FRAMES    = 2000
HUD_W, HUD_H  = 420, 620   # our separate HUD window size

# ── Colours ───────────────────────────────────────────────────────────────────
C_BG       = (10,  12,  20)
C_PANEL    = (18,  22,  35)
C_BORDER   = (40,  44,  60)
C_WHITE    = (240, 240, 250)
C_GREY     = (120, 124, 140)
C_GREEN    = (74,  222, 128)
C_RED      = (239, 68,  68)
C_YELLOW   = (251, 191, 36)
C_BLUE     = (96,  165, 250)
C_PURPLE   = (167, 139, 250)
C_ORANGE   = (249, 115, 22)
C_DARK_G   = (20,  40,  25)
C_DARK_R   = (40,  15,  15)


def process_action(outputs):
    steering = float(np.clip(outputs[0], -1.0,  1.0))
    gas      = float(np.clip(outputs[1],  0.0,  1.0))
    brake    = float(np.clip(outputs[2],  0.0,  1.0))
    gas      = max(gas, 0.1)
    return np.array([steering, gas, brake], dtype=np.float32)


def draw_bar(surf, x, y, w, h, value, min_v, max_v, color, bg=C_PANEL):
    """Draw a horizontal progress/value bar."""
    pygame.draw.rect(surf, bg,    (x, y, w, h), border_radius=3)
    pygame.draw.rect(surf, C_BORDER, (x, y, w, h), 1, border_radius=3)
    frac  = np.clip((value - min_v) / (max_v - min_v), 0, 1)
    fill  = int(w * frac)
    if fill > 0:
        pygame.draw.rect(surf, color, (x, y, fill, h), border_radius=3)


def draw_steering_dial(surf, cx, cy, radius, steer_val, font):
    """Draw a steering wheel arc indicator."""
    pygame.draw.circle(surf, C_PANEL,  (cx, cy), radius)
    pygame.draw.circle(surf, C_BORDER, (cx, cy), radius, 1)
    # Centre line
    pygame.draw.line(surf, C_BORDER, (cx - radius, cy), (cx + radius, cy), 1)
    # Needle
    angle  = np.radians(-90 + steer_val * 80)
    nx     = cx + int((radius - 4) * np.cos(angle))
    ny     = cy + int((radius - 4) * np.sin(angle))
    col    = C_YELLOW if abs(steer_val) > 0.5 else C_GREEN
    pygame.draw.line(surf, col, (cx, cy), (nx, ny), 3)
    pygame.draw.circle(surf, col, (cx, cy), 4)
    # Label
    lbl = font.render(f"{steer_val:+.2f}", True, col)
    surf.blit(lbl, (cx - lbl.get_width() // 2, cy + radius + 4))


def draw_panel(surf, x, y, w, h, title, font_title, color=C_BORDER):
    """Draw a labelled panel box."""
    pygame.draw.rect(surf, C_PANEL,  (x, y, w, h), border_radius=6)
    pygame.draw.rect(surf, color,    (x, y, w, h), 1, border_radius=6)
    if title:
        t = font_title.render(title, True, C_GREY)
        surf.blit(t, (x + 8, y + 6))


def run_hud(state: dict, genome, done_event: threading.Event):
    """
    Runs in a separate thread — draws the HUD window continuously
    while the main thread runs the simulation.

    state : shared dict updated by the main thread every frame
    """
    os.environ['SDL_VIDEO_WINDOW_POS'] = '700,50'

    pygame.init()
    hud = pygame.display.set_mode((HUD_W, HUD_H))
    pygame.display.set_caption("F1 NEAT — Live Telemetry HUD")

    font_xl    = pygame.font.SysFont("Arial", 28, bold=True)
    font_lg    = pygame.font.SysFont("Arial", 20, bold=True)
    font_md    = pygame.font.SysFont("Arial", 16, bold=True)
    font_sm    = pygame.font.SysFont("Arial", 13)
    font_mono  = pygame.font.SysFont("Courier New", 14)

    reward_history = []
    clock          = pygame.time.Clock()

    while not done_event.is_set():
        clock.tick(30)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done_event.set()
                return

        hud.fill(C_BG)

        # ── Read shared state ──────────────────────────────────────────────
        frame         = state.get('frame',         0)
        total_reward  = state.get('total_reward',  0.0)
        action        = state.get('action',        [0.0, 0.1, 0.0])
        episode       = state.get('episode',       1)
        steer_lock    = state.get('steer_lock',    0)
        grass_f       = state.get('grass_frames',  0)
        elapsed       = state.get('elapsed',       0.0)
        fitness       = genome.fitness

        reward_history.append(total_reward)
        if len(reward_history) > 200:
            reward_history.pop(0)

        steer  = action[0]
        gas    = action[1]
        brake  = action[2]
        lock_pct = (steer_lock / max(frame, 1)) * 100

        # ──────────────────────────────────────────────────────────────────
        # SECTION 1 — Header
        # ──────────────────────────────────────────────────────────────────
        pygame.draw.rect(hud, C_PANEL, (0, 0, HUD_W, 56))
        pygame.draw.line(hud, C_PURPLE, (0, 56), (HUD_W, 56), 1)

        t1 = font_lg.render("F1 NEAT  |  Live Telemetry", True, C_PURPLE)
        t2 = font_sm.render(f"Episode {episode}/{NUM_EPISODES}  |  "
                            f"Fitness {fitness:.1f}  |  "
                            f"Nodes {len(genome.nodes)}  |  "
                            f"Conns {len(genome.connections)}",
                            True, C_GREY)
        hud.blit(t1, (10, 8))
        hud.blit(t2, (10, 34))

        # ──────────────────────────────────────────────────────────────────
        # SECTION 2 — Key numbers row
        # ──────────────────────────────────────────────────────────────────
        y = 68
        for i, (label, value, col) in enumerate([
            ("REWARD",  f"{total_reward:.1f}",  C_GREEN if total_reward > 0 else C_RED),
            ("TIME",    f"{elapsed:.1f}s",       C_WHITE),
            ("FRAME",   f"{frame}",              C_GREY),
        ]):
            bx = 10 + i * 136
            draw_panel(hud, bx, y, 126, 52, None, font_sm)
            lbl = font_sm.render(label, True, C_GREY)
            val = font_xl.render(value, True, col)
            hud.blit(lbl, (bx + 8,  y + 6))
            hud.blit(val, (bx + 8,  y + 22))

        # ──────────────────────────────────────────────────────────────────
        # SECTION 3 — Steering dial + Gas/Brake bars
        # ──────────────────────────────────────────────────────────────────
        y = 136
        draw_panel(hud, 8, y, HUD_W - 16, 120, "CONTROLS", font_sm, C_BORDER)

        # Steering dial
        draw_steering_dial(hud, 80, y + 65, 42, steer, font_sm)
        steer_lbl = font_sm.render("STEERING", True, C_GREY)
        hud.blit(steer_lbl, (50, y + 14))

        # Gas bar
        bx = 150
        gas_lbl = font_sm.render("GAS", True, C_GREY)
        hud.blit(gas_lbl, (bx, y + 18))
        draw_bar(hud, bx, y + 38, 250, 18, gas,   0, 1, C_GREEN)
        gas_val = font_md.render(f"{gas*100:.0f}%", True, C_GREEN)
        hud.blit(gas_val, (bx, y + 60))

        brake_lbl = font_sm.render("BRAKE", True, C_GREY)
        hud.blit(brake_lbl, (bx, y + 82))
        draw_bar(hud, bx, y + 100, 250, 18, brake, 0, 1, C_RED)
        brake_val = font_md.render(f"{brake*100:.0f}%", True, C_RED)
        hud.blit(brake_val, (bx, y + 60 + 44))

        # ──────────────────────────────────────────────────────────────────
        # SECTION 4 — Behaviour analysis
        # ──────────────────────────────────────────────────────────────────
        y = 272
        draw_panel(hud, 8, y, HUD_W - 16, 100, "BEHAVIOUR ANALYSIS", font_sm, C_BORDER)

        # Steer lock warning
        lock_col = C_RED if lock_pct > 50 else C_YELLOW if lock_pct > 25 else C_GREEN
        lock_lbl = font_sm.render("STEER LOCK %", True, C_GREY)
        hud.blit(lock_lbl, (18, y + 22))
        draw_bar(hud, 18, y + 40, 180, 14, lock_pct, 0, 100, lock_col)
        lock_val = font_md.render(f"{lock_pct:.0f}%", True, lock_col)
        hud.blit(lock_val, (210, y + 36))

        if lock_pct > 50:
            warn = font_sm.render("⚠ EXPLOITATION DETECTED", True, C_RED)
            hud.blit(warn, (18, y + 62))
        elif lock_pct > 25:
            warn = font_sm.render("~ Some lock steering present", True, C_YELLOW)
            hud.blit(warn, (18, y + 62))
        else:
            warn = font_sm.render("✓ Clean steering behaviour", True, C_GREEN)
            hud.blit(warn, (18, y + 62))

        # Grass frames
        grass_pct = (grass_f / max(frame, 1)) * 100
        grass_lbl = font_sm.render(f"OFF-TRACK: {grass_pct:.0f}% of frames", True,
                                   C_RED if grass_pct > 30 else C_GREY)
        hud.blit(grass_lbl, (18, y + 80))

        # ──────────────────────────────────────────────────────────────────
        # SECTION 5 — Reward history chart
        # ──────────────────────────────────────────────────────────────────
        y = 384
        draw_panel(hud, 8, y, HUD_W - 16, 110, "REWARD OVER TIME", font_sm, C_BORDER)

        if len(reward_history) > 1:
            chart_x, chart_y = 18, y + 24
            chart_w, chart_h = HUD_W - 36, 76
            max_r = max(max(reward_history), 1)
            min_r = min(min(reward_history), -1)
            r_range = max_r - min_r

            # Zero line
            zero_y = chart_y + chart_h - int(chart_h * (0 - min_r) / r_range)
            pygame.draw.line(hud, C_BORDER,
                             (chart_x, zero_y), (chart_x + chart_w, zero_y), 1)

            # Reward line
            pts = []
            for i, r in enumerate(reward_history):
                px = chart_x + int(i / len(reward_history) * chart_w)
                py = chart_y + chart_h - int((r - min_r) / r_range * chart_h)
                pts.append((px, py))

            if len(pts) > 1:
                col = C_GREEN if reward_history[-1] > 0 else C_RED
                pygame.draw.lines(hud, col, False, pts, 2)

            # Current value label
            cur = font_sm.render(f"{total_reward:.1f}", True, C_GREEN)
            hud.blit(cur, (chart_x + chart_w - 40, chart_y))

        # ──────────────────────────────────────────────────────────────────
        # SECTION 6 — Legend
        # ──────────────────────────────────────────────────────────────────
        y = 504
        draw_panel(hud, 8, y, HUD_W - 16, 106, "WHAT YOU ARE SEEING", font_sm, C_BORDER)

        legends = [
            (C_GREEN,  "GREEN BAR in sim = track progress (tiles visited)"),
            (C_RED,    "RED BAR in sim   = brake input from network"),
            (C_YELLOW, "STEER LOCK       = steering held at max (exploitation)"),
            (C_BLUE,   "REWARD           = cumulative score this episode"),
        ]
        for i, (col, text) in enumerate(legends):
            pygame.draw.circle(hud, col, (20, y + 20 + i * 22), 4)
            t = font_sm.render(text, True, C_GREY)
            hud.blit(t, (30, y + 13 + i * 22))

        pygame.display.flip()

    pygame.quit()


def replay():
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        CONFIG_PATH
    )

    if not os.path.exists(BEST_GENOME_PATH):
        print("ERROR: No best genome found. Run python main.py first.")
        return

    with open(BEST_GENOME_PATH, "rb") as f:
        genome = pickle.load(f)

    net = neat.nn.FeedForwardNetwork.create(genome, config)

    print()
    print("=" * 55)
    print("  Best Genome Replay — Full HUD")
    print(f"  Fitness: {genome.fitness:.2f} | "
          f"Nodes: {len(genome.nodes)} | "
          f"Conns: {len(genome.connections)}")
    print("=" * 55)
    print("  Two windows will open:")
    print("  LEFT  — CarRacing simulation (the car)")
    print("  RIGHT — Live telemetry HUD")
    print()

    # Shared state between simulation thread and HUD thread
    state      = {}
    done_event = threading.Event()

    # Start HUD in background thread
    hud_thread = threading.Thread(
        target=run_hud,
        args=(state, genome, done_event),
        daemon=True
    )
    hud_thread.start()

    # Small delay so HUD window opens first
    time.sleep(0.8)

    for episode in range(1, NUM_EPISODES + 1):
        if done_event.is_set():
            break

        state['episode'] = episode
        print(f"  Episode {episode}/{NUM_EPISODES}")

        # Position sim window to the left
        os.environ['SDL_VIDEO_WINDOW_POS'] = '50,50'

        env   = gym.make("CarRacing-v3", render_mode="human", continuous=True)
        obs, _= env.reset(seed=42)

        total_reward      = 0.0
        steer_lock_frames = 0
        grass_frames      = 0
        frame             = 0
        start             = time.time()

        while frame < MAX_FRAMES and not done_event.is_set():
            processed = preprocess_observation(obs)
            outputs   = net.activate(processed)
            action    = process_action(outputs)

            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            frame        += 1

            if abs(action[0]) > 0.95:
                steer_lock_frames += 1

            # Update shared state for HUD thread
            state.update({
                'frame':        frame,
                'total_reward': total_reward,
                'action':       action,
                'steer_lock':   steer_lock_frames,
                'grass_frames': grass_frames,
                'elapsed':      time.time() - start,
            })

            if terminated or truncated or total_reward < -15:
                break

        env.close()
        lock_pct = steer_lock_frames / max(frame, 1) * 100
        print(f"  Result: Reward {total_reward:.2f} | "
              f"Frames {frame} | "
              f"Lock% {lock_pct:.0f}%")
        time.sleep(1.0)

    done_event.set()
    hud_thread.join(timeout=3.0)
    print()
    print("  Replay complete.")


if __name__ == "__main__":
    replay()