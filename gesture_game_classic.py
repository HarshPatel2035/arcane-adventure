"""Gesture-Controlled Spellcasting Game: CPU webcam inference + Pygame.

Run: ./.venv/bin/python gesture_game.py
Keyboard-only: ./.venv/bin/python gesture_game.py --no-camera
Space starts; P pauses; R restarts after game over; 1-5 cast; Esc quits.
Graphics and sound are generated locally; no external art/audio is required.
"""

from __future__ import annotations

import argparse
import base64
import io
from collections import deque
from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
import sys
import subprocess
from types import SimpleNamespace
import threading
import time

import numpy as np
import pygame

ROOT = Path(__file__).resolve().parent
WIDTH, HEIGHT, ARENA_WIDTH = 1280, 720, 870
PLAYER = pygame.Vector2(ARENA_WIDTH / 2, 610)
SPELLS = ("Fireball", "Shield", "Lightning", "Telekinesis", "Ultimate")
COSTS = dict(zip(SPELLS, (10, 20, 30, 15, 100)))
COOLDOWNS = dict(zip(SPELLS, (0.45, 2.0, 1.2, 0.8, 3.0)))
COLORS = {
    "Fireball": (255, 139, 69),
    "Shield": (89, 183, 255),
    "Lightning": (253, 223, 100),
    "Telekinesis": (87, 232, 192),
    "Ultimate": (192, 141, 255),
}
INK = (12, 15, 29)
TEXT = (226, 234, 252)
MUTED = (137, 152, 184)


@dataclass
class Enemy:
    x: float
    y: float
    speed: float
    radius: float = 18
    phase: float = 0

    @property
    def position(self):
        return pygame.Vector2(self.x, self.y)


@dataclass
class Projectile:
    position: pygame.Vector2
    velocity: pygame.Vector2
    radius: float = 10


@dataclass
class Particle:
    position: pygame.Vector2
    velocity: pygame.Vector2
    life: float
    total_life: float
    color: tuple
    size: float


class SoundBank:
    """Short synthesized stereo sounds, matching the actual mixer format."""

    def __init__(self, muted=False):
        self.muted = muted
        self.sounds = {}
        if pygame.mixer.get_init() is None:
            return
        rate, sample_format, channels = pygame.mixer.get_init()
        if sample_format != -16:
            return
        rng = np.random.default_rng(12)
        settings = {
            "Fireball": (220, 0.22),
            "Shield": (420, 0.35),
            "Lightning": (1100, 0.18),
            "Telekinesis": (180, 0.28),
            "Ultimate": (100, 0.65),
            "hit": (90, 0.16),
            "ui": (620, 0.10),
        }
        for name, (frequency, duration) in settings.items():
            t = np.arange(int(rate * duration)) / rate
            envelope = np.minimum(t * 45, 1) * np.exp(-t * 6 / duration)
            wave = np.sin(2 * np.pi * (frequency * t + frequency * t * t * 0.6))
            if name in ("Fireball", "Lightning", "hit", "Ultimate"):
                wave = 0.55 * wave + 0.45 * rng.uniform(-1, 1, len(t))
            samples = (wave * envelope * 14000).astype(np.int16)
            if channels > 1:
                samples = np.repeat(samples[:, None], channels, axis=1)
            self.sounds[name] = pygame.sndarray.make_sound(
                np.ascontiguousarray(samples)
            )
        pygame.mixer.set_num_channels(16)

    def play(self, name):
        if not self.muted and name in self.sounds:
            self.sounds[name].play()


class CameraWorker:
    """Run the camera in a separate process; keep only its newest frame.

    Reset generations drop in-flight actions at start, pause, resume and restart.
    The rendering loop never waits for detection or grows a video-frame queue.
    """

    def __init__(self, camera=0):
        self.camera = camera
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(
            target=self._run, name="gesture-camera", daemon=True
        )
        self.prediction = None
        self.updated_at = 0.0
        self.events = deque(maxlen=8)
        self.error = None
        self.status = "Starting camera and gesture model..."
        self.generation = 0
        self.process = None

    def start(self):
        self.thread.start()

    def reset(self):
        with self.lock:
            self.generation += 1
            self.events.clear()
            if self.process is not None and self.process.poll() is None:
                try:
                    self.process.stdin.write(
                        json.dumps({"command": "reset", "generation": self.generation})
                        + "\n"
                    )
                    self.process.stdin.flush()
                except (OSError, BrokenPipeError):
                    pass

    def snapshot(self):
        with self.lock:
            actions = list(self.events)
            self.events.clear()
            return self.prediction, self.updated_at, actions, self.error, self.status

    def _run(self):
        process = None
        try:
            if self.stop_event.is_set():
                return
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-u",
                    str(ROOT / "game_camera.py"),
                    "--camera",
                    str(self.camera),
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                text=True,
                bufsize=1,
                cwd=ROOT,
            )
            with self.lock:
                self.process = process
                process.stdin.write(
                    json.dumps({"command": "reset", "generation": self.generation})
                    + "\n"
                )
                process.stdin.flush()
            for line in process.stdout:
                if self.stop_event.is_set():
                    break
                packet = json.loads(line)
                with self.lock:
                    if "error" in packet:
                        self.error = packet["error"]
                        self.events.clear()
                        break
                    if packet["generation"] != self.generation:
                        continue
                    image = base64.b64decode(packet.pop("image"), validate=True)
                    self.prediction = SimpleNamespace(**packet, preview_image=image)
                    self.updated_at = packet["observed_at"]
                    self.status = packet["status"]
                    if packet["action"] is not None:
                        self.events.append(
                            (packet["action"], self.updated_at, packet["generation"])
                        )
            if not self.stop_event.is_set():
                with self.lock:
                    if self.error is None:
                        self.error = (
                            "Camera helper stopped. Keyboard controls remain available."
                        )
        except Exception as exc:
            with self.lock:
                self.error = str(exc)
                self.events.clear()
        finally:
            if process is not None:
                if process.poll() is None:
                    process.terminate()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                process.stdin.close()
                process.stdout.close()

    def close(self):
        self.stop_event.set()
        with self.lock:
            process = self.process
            if process is not None and process.poll() is None:
                try:
                    process.stdin.write('{"command":"stop"}\n')
                    process.stdin.flush()
                except (OSError, BrokenPipeError):
                    pass
        if self.thread.is_alive():
            self.thread.join(timeout=3)
        if process is not None and process.poll() is None:
            process.terminate()
            self.thread.join(timeout=2)


class SpellGame:
    def __init__(self, sound=None, seed=None):
        self.rng = random.Random(seed)
        self.sound = sound
        self.best_score = 0
        self.reset()
        self.state = "start"

    def reset(self):
        self.state = "playing"
        self.health, self.mana, self.score = 100, 100.0, 0
        self.elapsed, self.level, self.spawn_timer = 0.0, 1, 0.8
        self.shield_until, self.flash, self.damage_flash = 0.0, 0.0, 0.0
        self.enemies, self.projectiles, self.particles, self.bolts, self.rings = (
            [],
            [],
            [],
            [],
            [],
        )
        self.last_cast = dict.fromkeys(SPELLS, float("-inf"))
        self.message, self.message_until = (
            "Protect the arena. Show a gesture to cast.",
            3.0,
        )
        self.cast_counts = dict.fromkeys(SPELLS, 0)

    def play_sound(self, name):
        if self.sound:
            self.sound.play(name)

    def notify(self, message):
        self.message, self.message_until = message, self.elapsed + 1.8

    def burst(self, position, color, count=24, speed=130):
        for _ in range(count):
            angle, life = self.rng.uniform(0, math.tau), self.rng.uniform(0.25, 0.75)
            velocity = pygame.Vector2(
                math.cos(angle), math.sin(angle)
            ) * self.rng.uniform(speed * 0.2, speed)
            self.particles.append(
                Particle(
                    pygame.Vector2(position),
                    velocity,
                    life,
                    life,
                    color,
                    self.rng.uniform(2, 5),
                )
            )
        self.particles = self.particles[-650:]

    def defeat(self, enemy, color):
        if enemy in self.enemies:
            self.enemies.remove(enemy)
            self.score += 10
            self.best_score = max(self.best_score, self.score)
            self.burst(enemy.position, color)

    def cast(self, spell):
        if self.state != "playing" or spell not in COSTS:
            return False
        if self.elapsed - self.last_cast[spell] < COOLDOWNS[spell]:
            self.notify(f"{spell} is cooling down")
            return False
        if self.mana + 1e-6 < COSTS[spell]:
            self.notify(f"Need {COSTS[spell]} mana for {spell}")
            return False
        self.mana = max(0.0, self.mana - COSTS[spell])
        self.last_cast[spell] = self.elapsed
        self.cast_counts[spell] += 1
        self.play_sound(spell)
        self.notify(f"{spell} cast!")
        color = COLORS[spell]
        if spell == "Fireball":
            origin = PLAYER + pygame.Vector2(0, -30)
            targets = [enemy for enemy in self.enemies if enemy.y < origin.y]
            direction = (
                min(targets, key=lambda e: e.position.distance_to(origin)).position
                - origin
                if targets
                else pygame.Vector2(0, -1)
            )
            if direction.length_squared() < 1:
                direction = pygame.Vector2(0, -1)
            self.projectiles.append(Projectile(origin, direction.normalize() * 500))
            self.burst(origin, color, 12, 70)
        elif spell == "Shield":
            self.shield_until = self.elapsed + 2
            self.rings.append([PLAYER.copy(), 28.0, 1.0, color])
        elif spell == "Lightning":
            origin = PLAYER.copy()
            for _ in range(min(3, len(self.enemies))):
                enemy = min(self.enemies, key=lambda e: e.position.distance_to(origin))
                target = enemy.position
                points = [origin.copy()]
                for index in range(1, 6):
                    point = origin.lerp(target, index / 6)
                    point.x += self.rng.uniform(-16, 16)
                    points.append(point)
                points.append(target.copy())
                self.bolts.append([points, 0.3])
                self.defeat(enemy, color)
                origin = target
        elif spell == "Telekinesis":
            for enemy in self.enemies:
                enemy.y = max(-65, enemy.y - 100)
                self.burst(enemy.position, color, 10, 80)
            self.rings.append([PLAYER.copy(), 40.0, 0.65, color])
        elif spell == "Ultimate":
            for enemy in self.enemies[:]:
                self.defeat(enemy, color)
            self.flash = 0.45
            self.rings.append([PLAYER.copy(), 20.0, 1.2, color])
            self.burst(PLAYER, color, 100, 340)
        return True

    @staticmethod
    def segment_hit(start, end, enemy, radius):
        delta = end - start
        if delta.length_squared() == 0:
            return start.distance_to(enemy.position) <= radius + enemy.radius
        fraction = max(
            0.0, min(1.0, (enemy.position - start).dot(delta) / delta.length_squared())
        )
        return (start + delta * fraction).distance_to(
            enemy.position
        ) <= radius + enemy.radius

    def update(self, dt):
        if self.state != "playing":
            return
        dt = max(0.0, min(dt, 0.1))
        self.elapsed += dt
        self.level = 1 + int(self.elapsed // 30)
        self.mana = min(100.0, self.mana + 5 * dt)
        self.flash = max(0.0, self.flash - dt)
        self.damage_flash = max(0.0, self.damage_flash - dt)
        self.spawn_timer -= dt
        if self.spawn_timer <= 0:
            self.enemies.append(
                Enemy(
                    self.rng.uniform(55, ARENA_WIDTH - 55),
                    -25,
                    46 + (self.level - 1) * 11,
                    phase=self.rng.uniform(0, math.tau),
                )
            )
            self.spawn_timer = max(0.38, 1.65 * 0.86 ** (self.level - 1))
        for enemy in self.enemies:
            enemy.y += enemy.speed * dt
            enemy.x += (PLAYER.x - enemy.x) * 0.22 * dt
        for projectile in self.projectiles[:]:
            old = projectile.position.copy()
            projectile.position += projectile.velocity * dt
            candidates = sorted(self.enemies, key=lambda e: e.position.distance_to(old))
            hit = next(
                (
                    enemy
                    for enemy in candidates
                    if self.segment_hit(
                        old, projectile.position, enemy, projectile.radius
                    )
                ),
                None,
            )
            if hit:
                self.defeat(hit, COLORS["Fireball"])
                self.projectiles.remove(projectile)
            elif not (
                -60 <= projectile.position.x <= ARENA_WIDTH + 60
                and -60 <= projectile.position.y <= HEIGHT
            ):
                self.projectiles.remove(projectile)
            else:
                self.burst(projectile.position, COLORS["Fireball"], 2, 45)
        for enemy in self.enemies[:]:
            if enemy.y >= PLAYER.y - 24:
                if self.elapsed < self.shield_until:
                    self.defeat(enemy, COLORS["Shield"])
                else:
                    self.enemies.remove(enemy)
                    self.health = max(0, self.health - 20)
                    self.damage_flash = 0.28
                    self.burst(PLAYER, (255, 84, 118), 25)
                    self.play_sound("hit")
                    if self.health == 0:
                        self.state = "gameover"
                        self.best_score = max(self.best_score, self.score)
                        break
        for particle in self.particles:
            particle.life -= dt
            particle.position += particle.velocity * dt
            particle.velocity *= max(0, 1 - 1.4 * dt)
        self.particles = [p for p in self.particles if p.life > 0]
        for bolt in self.bolts:
            bolt[1] -= dt
        self.bolts = [b for b in self.bolts if b[1] > 0]
        for ring in self.rings:
            ring[1] += 440 * dt
            ring[2] -= dt
        self.rings = [r for r in self.rings if r[2] > 0]


class GameView:
    def __init__(self, screen):
        self.screen = screen
        self.fonts = {
            size: pygame.font.SysFont("Arial", size, bold=size >= 24)
            for size in (14, 16, 18, 20, 24, 30, 42, 60)
        }
        self.background = pygame.Surface((WIDTH, HEIGHT))
        for y in range(HEIGHT):
            t = y / HEIGHT
            pygame.draw.line(
                self.background,
                (int(13 + 8 * t), int(18 + 5 * t), int(37 + 17 * t)),
                (0, y),
                (WIDTH, y),
            )
        rng = random.Random(17)
        for _ in range(125):
            x, y = rng.randrange(30, ARENA_WIDTH - 30), rng.randrange(105, 665)
            pygame.draw.circle(
                self.background, (38, 48, 77), (x, y), rng.choice((1, 1, 2))
            )
        for y in range(150, 650, 65):
            pygame.draw.line(
                self.background, (27, 34, 56), (28, y), (ARENA_WIDTH - 28, y)
            )
        self.camera_surface = None
        self.camera_timestamp = 0
        self.buttons = {}

    def text(self, value, x, y, size=18, color=TEXT, center=False):
        surface = self.fonts[size].render(str(value), True, color)
        rect = (
            surface.get_rect(center=(x, y))
            if center
            else surface.get_rect(topleft=(x, y))
        )
        self.screen.blit(surface, rect)
        return rect

    def wrapped(self, value, x, y, width, size=16, color=MUTED):
        line = ""
        for word in value.split():
            candidate = f"{line} {word}".strip()
            if line and self.fonts[size].size(candidate)[0] > width:
                self.text(line, x, y, size, color)
                y += size + 5
                line = word
            else:
                line = candidate
        self.text(line, x, y, size, color)
        return y + size + 5

    def button(self, key, caption, rect, color=(74, 90, 150)):
        self.buttons[key] = pygame.Rect(rect)
        pygame.draw.rect(self.screen, color, rect, border_radius=10)
        self.text(
            caption,
            self.buttons[key].centerx,
            self.buttons[key].centery,
            18,
            TEXT,
            True,
        )

    def bar(self, caption, value, x, y, color):
        self.text(f"{caption} {int(value)}/100", x, y, 14, MUTED)
        pygame.draw.rect(
            self.screen, (36, 43, 64), (x, y + 22, 180, 10), border_radius=5
        )
        if value > 0:
            pygame.draw.rect(
                self.screen,
                color,
                (x, y + 22, int(180 * value / 100), 10),
                border_radius=5,
            )

    def wizard(self, game):
        x, y = int(PLAYER.x), int(PLAYER.y)
        pygame.draw.ellipse(self.screen, (19, 24, 43), (x - 45, y + 25, 90, 20))
        pygame.draw.polygon(
            self.screen,
            (117, 86, 205),
            [(x, y - 30), (x - 28, y + 30), (x + 27, y + 30)],
        )
        pygame.draw.polygon(
            self.screen, (65, 51, 123), [(x, y - 25), (x - 28, y + 30), (x, y + 30)]
        )
        pygame.draw.circle(self.screen, (239, 201, 171), (x, y - 33), 13)
        pygame.draw.polygon(
            self.screen,
            (144, 117, 245),
            [(x - 25, y - 42), (x + 22, y - 42), (x - 5, y - 88)],
        )
        pygame.draw.line(
            self.screen, (205, 183, 252), (x - 24, y - 42), (x + 24, y - 42), 4
        )
        pygame.draw.circle(self.screen, (30, 28, 52), (x - 4, y - 34), 2)
        pygame.draw.circle(self.screen, (30, 28, 52), (x + 5, y - 34), 2)
        pygame.draw.line(
            self.screen, (181, 143, 98), (x + 28, y + 24), (x + 36, y - 43), 5
        )
        pygame.draw.circle(self.screen, (76, 239, 214), (x + 36, y - 48), 8)
        if game.elapsed < game.shield_until:
            bubble = pygame.Surface((140, 160), pygame.SRCALPHA)
            pygame.draw.ellipse(bubble, (85, 184, 255, 45), (4, 4, 132, 152))
            pygame.draw.ellipse(bubble, (110, 207, 255, 210), (4, 4, 132, 152), 3)
            self.screen.blit(bubble, (x - 70, y - 105))

    def camera(self, worker_snapshot, no_camera):
        x = ARENA_WIDTH + 24
        prediction, timestamp, actions, error, status = worker_snapshot
        self.text("GESTURE CAMERA", x, 28, 20)
        rect = pygame.Rect(x, 64, 362, 272)
        pygame.draw.rect(self.screen, (10, 13, 23), rect, border_radius=12)
        if prediction is not None and timestamp != self.camera_timestamp:
            self.camera_surface = pygame.image.load(
                io.BytesIO(prediction.preview_image), "camera.jpg"
            ).convert()
            self.camera_surface = pygame.transform.smoothscale(
                self.camera_surface, rect.size
            )
            self.camera_timestamp = timestamp
        if self.camera_surface:
            self.screen.blit(self.camera_surface, rect)
        else:
            self.text(
                "Keyboard mode" if no_camera else "Connecting camera...",
                rect.centerx,
                rect.centery,
                20,
                MUTED,
                True,
            )
        stale = prediction is not None and time.monotonic() - timestamp > 0.7
        active = prediction is not None and not stale and error is None
        gesture = (
            prediction.gesture_name
            if active
            else ("Keyboard controls active" if no_camera else "Waiting for gesture")
        )
        confidence = prediction.confidence if active else 0
        self.text(gesture, x, 352, 24)
        self.text(f"Confidence {confidence:.0%}", x, 389, 16, (93, 224, 190))
        self.wrapped(
            error or ("Camera feed delayed" if stale else status), x, 418, 362, 14
        )
        return active

    def draw(self, game, snapshot, fps, no_camera, muted):
        self.buttons.clear()
        self.screen.blit(self.background, (0, 0))
        pygame.draw.rect(
            self.screen, (15, 19, 33), (ARENA_WIDTH, 0, WIDTH - ARENA_WIDTH, HEIGHT)
        )
        pygame.draw.line(
            self.screen, (47, 56, 83), (ARENA_WIDTH, 0), (ARENA_WIDTH, HEIGHT), 2
        )
        self.text("ARCANE ARENA", 28, 22, 30)
        self.text("Gesture-controlled spellcasting", 30, 62, 14, MUTED)
        self.bar("HEALTH", game.health, 295, 24, (242, 95, 130))
        self.bar("MANA", game.mana, 505, 24, (78, 203, 222))
        self.text(f"{game.score:05d}", 726, 20, 30)
        self.text(f"LEVEL {game.level}  |  {int(game.elapsed)}s", 725, 58, 14, MUTED)
        pygame.draw.line(self.screen, (57, 68, 99), (28, 99), (ARENA_WIDTH - 28, 99))
        # A glowing summoning circle behind the wizard.
        pygame.draw.ellipse(
            self.screen, (49, 42, 87), (PLAYER.x - 110, PLAYER.y + 6, 220, 57), 2
        )
        pygame.draw.ellipse(
            self.screen, (74, 57, 121), (PLAYER.x - 75, PLAYER.y + 16, 150, 33), 1
        )
        for enemy in game.enemies:
            x, y = int(enemy.x), int(enemy.y)
            pygame.draw.ellipse(self.screen, (8, 11, 22), (x - 20, y + 14, 40, 12))
            pygame.draw.circle(
                self.screen, (100, 42, 90), (x, y), int(enemy.radius) + 4
            )
            pygame.draw.circle(self.screen, (202, 70, 123), (x, y), int(enemy.radius))
            pygame.draw.polygon(
                self.screen,
                (230, 129, 160),
                [(x - 17, y - 9), (x - 23, y - 25), (x - 7, y - 15)],
            )
            pygame.draw.polygon(
                self.screen,
                (230, 129, 160),
                [(x + 17, y - 9), (x + 23, y - 25), (x + 7, y - 15)],
            )
            for offset in (-6, 6):
                pygame.draw.circle(self.screen, (255, 230, 184), (x + offset, y - 3), 3)
        for projectile in game.projectiles:
            point = (int(projectile.position.x), int(projectile.position.y))
            pygame.draw.circle(self.screen, (167, 66, 52), point, 16)
            pygame.draw.circle(self.screen, COLORS["Fireball"], point, 10)
            pygame.draw.circle(self.screen, (255, 243, 173), point, 5)
        for particle in game.particles:
            fade = max(0, particle.life / particle.total_life)
            color = tuple(int(component * fade) for component in particle.color)
            pygame.draw.circle(
                self.screen, color, particle.position, max(1, int(particle.size * fade))
            )
        for points, life in game.bolts:
            pygame.draw.lines(self.screen, (237, 211, 72), False, points, 5)
            pygame.draw.lines(self.screen, (255, 252, 218), False, points, 2)
        for position, radius, life, color in game.rings:
            pygame.draw.circle(self.screen, color, position, int(radius), 2)
        self.wizard(game)
        if game.flash or game.damage_flash:
            overlay = pygame.Surface((ARENA_WIDTH, HEIGHT), pygame.SRCALPHA)
            color = (
                (190, 140, 255, int(game.flash * 230))
                if game.flash
                else (255, 55, 90, int(game.damage_flash * 240))
            )
            overlay.fill(color)
            self.screen.blit(overlay, (0, 0))
        if game.elapsed < game.message_until:
            self.text(game.message, ARENA_WIDTH / 2, 126, 18, (208, 220, 251), True)
        self.camera(snapshot, no_camera)
        x = ARENA_WIDTH + 24
        self.text("YOUR SPELLBOOK", x, 486, 18)
        gestures = (
            "Open palm",
            "Closed fist",
            "Peace sign",
            "Index pointing",
            "Both palms open",
        )
        for index, (spell, gesture) in enumerate(zip(SPELLS, gestures)):
            y = 520 + index * 31
            self.text(f"{index+1}", x, y, 16, COLORS[spell])
            self.text(f"{gesture}  /  {spell}", x + 24, y, 14, TEXT)
            self.text(f"{COSTS[spell]}", WIDTH - 49, y, 14, MUTED)
        self.text("Mana cost", WIDTH - 110, 487, 14, MUTED)
        for index, spell in enumerate(SPELLS):
            rect = (28 + index * 164, 674, 154, 32)
            self.button(
                "cast_" + spell,
                f"{index+1}  {spell}",
                rect,
                (45, 65, 72) if game.mana >= COSTS[spell] else (35, 37, 49),
            )
        self.text(f"FPS {fps:.0f}", 28, 639, 14, MUTED)
        self.text("Mana regenerates 5 / sec", 145, 650, 14, MUTED)
        self.text("SPACE start  |  P pause  |  R restart", x, 681, 14, MUTED)
        self.button(
            "mute", "UNMUTE" if muted else "MUTE", (740, 630, 104, 30), (41, 48, 70)
        )
        if game.state in ("start", "paused", "gameover"):
            shade = pygame.Surface((ARENA_WIDTH, HEIGHT), pygame.SRCALPHA)
            shade.fill((8, 10, 22, 220))
            self.screen.blit(shade, (0, 0))
            self.text("ARCANE ARENA", ARENA_WIDTH / 2, 225, 60, (192, 165, 255), True)
            if game.state == "start":
                self.text(
                    "Your hands are your spellbook",
                    ARENA_WIDTH / 2,
                    299,
                    24,
                    TEXT,
                    True,
                )
                self.text(
                    "Defeat incoming spirits. Protect your health. Watch your mana.",
                    ARENA_WIDTH / 2,
                    352,
                    18,
                    MUTED,
                    True,
                )
                self.text(
                    "Use gestures, keys 1-5, or the spell buttons.",
                    ARENA_WIDTH / 2,
                    385,
                    18,
                    MUTED,
                    True,
                )
                caption = "START GAME  /  SPACE"
            elif game.state == "paused":
                self.text("PAUSED", ARENA_WIDTH / 2, 320, 30, TEXT, True)
                caption = "RESUME  /  P"
            else:
                self.text("GAME OVER", ARENA_WIDTH / 2, 305, 30, TEXT, True)
                self.text(
                    f"Score {game.score}   |   Level {game.level}   |   Best {game.best_score}",
                    ARENA_WIDTH / 2,
                    361,
                    24,
                    MUTED,
                    True,
                )
                caption = "PLAY AGAIN  /  R"
            self.button(
                "primary",
                caption,
                (ARENA_WIDTH / 2 - 165, 436, 330, 54),
                (102, 78, 169),
            )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--no-camera", action="store_true")
    parser.add_argument("--mute", action="store_true")
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Exit after N frames; 0 means unlimited",
    )
    args = parser.parse_args(argv)
    if args.camera < 0 or args.max_frames < 0:
        parser.error("Camera and max-frames must be nonnegative.")
    return args


def run(args):
    pygame.mixer.pre_init(44100, -16, 2, 512)
    pygame.init()
    worker = None
    game = None
    frames = 0
    frame_times = []
    try:
        screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Arcane Arena - Gesture Spellcasting")
        sound = SoundBank(args.mute)
        game = SpellGame(sound, args.seed)
        view = GameView(screen)
        clock = pygame.time.Clock()
        if not args.no_camera:
            worker = CameraWorker(args.camera)
            worker.start()

        def reset_camera():
            if worker:
                worker.reset()

        def primary():
            if game.state == "paused":
                game.state = "playing"
            elif game.state in ("start", "gameover"):
                game.reset()
                sound.play("ui")
            reset_camera()

        running = True
        while running:
            frame_seconds = clock.tick(30) / 1000
            dt = min(frame_seconds, 0.1)
            if len(frame_times) < 18000:
                frame_times.append(frame_seconds)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_ESCAPE, pygame.K_q):
                        running = False
                    elif event.key == pygame.K_SPACE and game.state == "start":
                        primary()
                    elif event.key == pygame.K_r and game.state == "gameover":
                        primary()
                    elif event.key == pygame.K_p and game.state in (
                        "playing",
                        "paused",
                    ):
                        if game.state == "playing":
                            game.state = "paused"
                            reset_camera()
                        else:
                            primary()
                    elif event.key == pygame.K_m:
                        sound.muted = not sound.muted
                    elif pygame.K_1 <= event.key <= pygame.K_5:
                        game.cast(SPELLS[event.key - pygame.K_1])
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    for key, rect in view.buttons.items():
                        if not rect.collidepoint(event.pos):
                            continue
                        if key == "primary":
                            primary()
                        elif key == "mute":
                            sound.muted = not sound.muted
                        elif key.startswith("cast_"):
                            game.cast(key[5:])
                        break
            snapshot = (None, 0, [], None, "Keyboard-only mode")
            if worker:
                snapshot = worker.snapshot()
                if game.state == "playing" and snapshot[3] is None:
                    now = time.monotonic()
                    for spell, observed_at, generation in snapshot[2]:
                        if now - observed_at <= 0.5:
                            game.cast(spell)
            before = game.state
            game.update(dt)
            if before != game.state:
                reset_camera()
            view.draw(game, snapshot, clock.get_fps(), args.no_camera, sound.muted)
            pygame.display.flip()
            frames += 1
            if args.max_frames and frames >= args.max_frames:
                running = False
    finally:
        if worker:
            worker.close()
        if game:
            try:
                directory = ROOT / "reports"
                directory.mkdir(exist_ok=True)
                report = {
                    "score": game.score,
                    "best_score_this_session": game.best_score,
                    "level": game.level,
                    "survival_seconds": round(game.elapsed, 2),
                    "rendered_frames": frames,
                    "average_game_fps": (
                        round(len(frame_times) / sum(frame_times), 2)
                        if sum(frame_times)
                        else 0
                    ),
                    "spell_counts": game.cast_counts,
                    "camera_enabled": not args.no_camera,
                    "camera_error": worker.error if worker else None,
                }
                (directory / "game_last_run.json").write_text(
                    json.dumps(report, indent=2) + "\n"
                )
            except OSError as exc:
                print(f"Could not save game metrics: {exc}", file=sys.stderr)
        pygame.quit()


def main():
    args = parse_args()
    try:
        run(args)
    except KeyboardInterrupt:
        print("\nGame stopped.")
    except Exception as exc:
        print(f"Game failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
