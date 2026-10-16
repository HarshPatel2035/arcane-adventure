"""Render docs/demo.gif from a scripted, camera-free play session.

Run: ./.venv/bin/python tools/make_demo_gif.py
Uses the real game classes headless (no window, audio or webcam). The wizard
weaves between lanes and casts like a player would; output is deterministic.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pygame  # noqa: E402
from PIL import Image  # noqa: E402

from arcane_adventure import AdventureGame, AdventureView  # noqa: E402

FPS, SECONDS, EVERY = 30, 9, 3  # simulate 30 fps, keep every 3rd frame -> 10 fps GIF
SIZE = (640, 360)


def capture(screen):
    raw = pygame.image.tobytes(screen, "RGB")
    return Image.frombytes("RGB", screen.get_size(), raw).resize(SIZE, Image.LANCZOS)


def main(output=ROOT / "docs" / "demo.gif"):
    pygame.init()
    screen = pygame.display.set_mode((1280, 720))
    game = AdventureGame(seed=7, companion="dragon")
    view = AdventureView(screen)
    snapshot = (None, 0, [], None, "Keyboard-only mode")
    frames = []
    for _ in range(12):  # camp screen
        view.draw(game, snapshot, FPS, True, True)
    frames += [capture(screen)] * 12
    game.reset()
    game.state = "playing"
    plan = {45: "Telekinesis", 70: "Fireball", 120: "Shield", 170: "Lightning", 230: "Fireball"}
    for frame in range(FPS * SECONDS):
        t = frame / FPS
        game.steer(0.5 + 0.32 * math.sin(t * 1.1))
        if frame in plan:
            game.cast(plan[frame])
        elif game.enemies and frame % 14 == 0:
            game.cast("Fireball")
        if game.mana >= 100 and frame > 200:
            game.cast("Ultimate")
        game.health = max(game.health, 60)  # keep the demo going
        game.update(1 / FPS)
        view.draw(game, snapshot, FPS, True, True)
        if frame % EVERY == 0:
            frames.append(capture(screen))
    pygame.quit()
    output.parent.mkdir(parents=True, exist_ok=True)
    palette = frames[len(frames) // 2].quantize(colors=128, method=Image.Quantize.MEDIANCUT)
    frames = [f.quantize(palette=palette, dither=Image.Dither.NONE) for f in frames]
    frames[0].save(
        output, save_all=True, append_images=frames[1:], duration=1000 // (FPS // EVERY),
        loop=0, optimize=True,
    )
    print(f"Wrote {output} ({output.stat().st_size / 1e6:.1f} MB, {len(frames)} frames)")


if __name__ == "__main__":
    main()
