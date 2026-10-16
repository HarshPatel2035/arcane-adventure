"""Shared spellcasting engine: combat rules, camera worker, sound and base view.

Not a launcher. Run the game with: ./.venv/bin/python arcane_adventure.py
arcane_adventure.py extends SpellGame/GameView with encounters, upgrades and props.
Includes an illustrated arena in assets/images; characters and sound are generated locally.
"""

from __future__ import annotations

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
PLAYER = pygame.Vector2(ARENA_WIDTH / 2, 560)
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
    shooter: bool = False
    shot_timer: float = 3.0
    charge: float = 0.0
    aim: object = None

    @property
    def position(self):
        return pygame.Vector2(self.x, self.y)


@dataclass
class Projectile:
    position: pygame.Vector2
    velocity: pygame.Vector2
    radius: float = 10
    damage: int = 1
    splash: float = 0


@dataclass
class HostileOrb:
    position: pygame.Vector2
    velocity: pygame.Vector2
    radius: float = 9


@dataclass
class Boss:
    x: float
    y: float
    hp: int
    max_hp: int
    radius: float = 48
    shot_timer: float = 1.8
    charge: float = 0
    aim: object = None

    @property
    def position(self):
        return pygame.Vector2(self.x, self.y)


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

    def __init__(self, camera=0, motion=False):
        self.motion = motion
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
            self.prediction = None
            self.updated_at = 0
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
                ]
                + (["--motion"] if self.motion else []),
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
                    if packet.get("motion"):
                        self.events.append(
                            (
                                "motion:" + packet["motion"],
                                self.updated_at,
                                packet["generation"],
                            )
                        )
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
    def __init__(self, sound=None, seed=None, boss_delay=45):
        self.rng = random.Random(seed)
        self.sound = sound
        self.best_score = 0
        self.boss_delay = boss_delay
        self.reset()
        self.state = "start"

    def reset(self):
        self.state = "playing"
        self.player = PLAYER.copy()
        self.move_target = None
        self.dodges = 0
        self.hostile_orbs = []
        self.boss = None
        self.bosses_defeated = 0
        self.next_boss_at = self.boss_delay
        self.spawn_count = 0
        self.perfect_until = 0
        self.hurt_until = 0
        self.parries = 0
        self.combo_window = 3
        self.combo_count = 0
        self.combo_spell = None
        self.combo_at = float("-inf")
        self.combo_name = ""
        self.combo_until = 0
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
            "Move your hand left or right to dodge. Show a pose to cast.",
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
            self.mana = min(100, self.mana + 8)
            self.best_score = max(self.best_score, self.score)
            self.burst(enemy.position, color)

    def hurt(self, damage):
        if self.elapsed < self.hurt_until or self.state != "playing":
            return
        self.health = max(0, self.health - damage)
        self.hurt_until = self.elapsed + 0.65
        self.damage_flash = 0.28
        self.burst(self.player, (255, 84, 118), 25)
        self.play_sound("hit")
        if self.health == 0:
            self.state = "gameover"

    def damage_boss(self, damage):
        if self.boss is None:
            return
        self.boss.hp -= damage
        self.burst(self.boss.position, COLORS["Lightning"], 14)
        if self.boss.hp <= 0:
            self.burst(self.boss.position, COLORS["Ultimate"], 90, 250)
            self.boss = None
            self.bosses_defeated += 1
            self.score += 150
            self.best_score = max(self.best_score, self.score)
            self.mana = min(100, self.mana + 35)
            self.health = min(100, self.health + 20)
            self.hostile_orbs.clear()
            self.next_boss_at = self.elapsed + 60
            self.notify("Guardian defeated! +150 score, health and mana restored")

    def reflect(self, orb):
        targets = [e for e in self.enemies if e.y < self.player.y - 60]
        if self.boss:
            targets.append(self.boss)
        direction = (
            (
                min(
                    targets, key=lambda e: e.position.distance_to(orb.position)
                ).position
                - orb.position
            )
            if targets
            else pygame.Vector2(0, -1)
        )
        if direction.length_squared() < 1:
            direction = pygame.Vector2(0, -1)
        self.projectiles.append(
            Projectile(orb.position.copy(), direction.normalize() * 550, damage=2)
        )
        self.parries += 1
        self.score += 15
        self.best_score = max(self.best_score, self.score)
        self.mana = min(100, self.mana + 12)
        self.notify("Perfect counter! Spell reflected / +15 score / +12 mana")
        self.play_sound("Shield")

    def shoot(self, source, count=1):
        direction = source.aim or pygame.Vector2(0, 1)
        angles = (
            [0] if count == 1 else [(-26 + 52 * i / (count - 1)) for i in range(count)]
        )
        for angle in angles:
            self.hostile_orbs.append(
                HostileOrb(
                    source.position.copy(),
                    direction.rotate(angle) * (190 + min(self.level, 8) * 8),
                )
            )

    def update_threats(self, dt):
        if self.boss is None and self.elapsed >= self.next_boss_at:
            hp = 12 + self.bosses_defeated * 4
            self.boss = Boss(ARENA_WIDTH / 2, 265, hp, hp)
            self.notify("The Grove Guardian arrives! Dodge, counter and combine spells")
        sources = [e for e in self.enemies if e.shooter and 90 <= e.y <= 400]
        if self.boss:
            self.boss.x = ARENA_WIDTH / 2 + math.sin(self.elapsed * 0.7) * 235
            sources.append(self.boss)
        for source in sources:
            if source.charge > 0:
                source.charge -= dt
                if source.charge <= 0:
                    self.shoot(
                        source,
                        (
                            5
                            if isinstance(source, Boss)
                            and source.hp <= source.max_hp / 2
                            else 3 if isinstance(source, Boss) else 1
                        ),
                    )
                    source.shot_timer = 1.6 if isinstance(source, Boss) else 3.5
            else:
                source.shot_timer -= dt
                if source.shot_timer <= 0:
                    direction = self.player + pygame.Vector2(0, -30) - source.position
                    source.aim = (
                        direction.normalize()
                        if direction.length_squared()
                        else pygame.Vector2(0, 1)
                    )
                    source.charge = 0.85
        for orb in self.hostile_orbs[:]:
            old = orb.position.copy()
            orb.position += orb.velocity * dt
            player_target = Enemy(self.player.x, self.player.y - 30, 0, radius=24)
            if self.segment_hit(old, orb.position, player_target, orb.radius):
                self.hostile_orbs.remove(orb)
                if self.elapsed < self.perfect_until:
                    self.reflect(orb)
                elif self.elapsed < self.shield_until:
                    self.burst(orb.position, COLORS["Shield"], 10)
                else:
                    self.hurt(12)
            elif not (
                -40 <= orb.position.x <= ARENA_WIDTH + 40
                and -40 <= orb.position.y <= 655
            ):
                self.hostile_orbs.remove(orb)
                if orb.position.y > 655:
                    self.dodges += 1
                    self.mana = min(100, self.mana + 3)

    def damage_enemy(self, enemy, damage=1, kind="Fireball"):
        self.defeat(enemy, COLORS.get(kind, COLORS["Fireball"]))

    def enemy_target(self, enemy):
        return self.player

    def aim_targets(self, origin):
        targets = [enemy for enemy in self.enemies if enemy.y < origin.y]
        return targets + ([self.boss] if self.boss else [])

    def spawn_enemy(self):
        self.spawn_count += 1
        self.enemies.append(
            Enemy(
                self.rng.uniform(55, ARENA_WIDTH - 55),
                -25,
                46 + (self.level - 1) * 11,
                phase=self.rng.uniform(0, math.tau),
                shooter=self.spawn_count % 3 == 0,
            )
        )

    def enemy_contact(self, enemy):
        if (
            enemy.position.distance_to(self.player + pygame.Vector2(0, -30))
            <= enemy.radius + 24
        ):
            if self.elapsed < self.shield_until:
                self.defeat(enemy, COLORS["Shield"])
            else:
                self.enemies.remove(enemy)
                self.hurt(20)
        elif enemy.y > 655:
            self.enemies.remove(enemy)
            self.dodges += 1
            self.mana = min(100, self.mana + 3)

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
        combo = None
        if self.elapsed - self.combo_at <= self.combo_window:
            combo = {
                ("Telekinesis", "Fireball"): "Meteor Burst",
                ("Shield", "Lightning"): "Storm Counter",
            }.get((self.combo_spell, spell))
        # Repeated held fireballs cannot overwrite a pending combo or trigger one twice.
        if combo:
            self.combo_count += 1
            self.combo_name, self.combo_until = combo, self.elapsed + 2
            self.combo_spell = None
            self.notify(combo + "!")
        elif (
            spell != self.combo_spell
            or self.elapsed - self.combo_at > self.combo_window
        ):
            self.combo_spell, self.combo_at = spell, self.elapsed
        if spell == "Fireball":
            origin = self.player + pygame.Vector2(0, -30)
            targets = self.aim_targets(origin)
            direction = pygame.Vector2(0, -1)
            if targets:
                target = min(targets, key=lambda e: e.position.distance_to(origin))
                aim = target.position
                # Lead the moving guardian so ordinary aimed shots remain useful.
                if target is self.boss:
                    for _ in range(3):
                        travel = origin.distance_to(aim) / 500
                        aim.x = (
                            ARENA_WIDTH / 2
                            + math.sin((self.elapsed + travel) * 0.7) * 235
                        )
                direction = aim - origin
            if direction.length_squared() < 1:
                direction = pygame.Vector2(0, -1)
            self.projectiles.append(
                Projectile(
                    origin,
                    direction.normalize() * 500,
                    damage=3 if combo else 1,
                    splash=110 if combo else 0,
                )
            )
            self.burst(origin, color, 12, 70)
        elif spell == "Shield":
            self.shield_until = self.elapsed + 2
            self.perfect_until = self.elapsed + 0.65
            self.notify("Shield up! Gold ring: reflect incoming shots now")
            self.rings.append([self.player.copy(), 28.0, 1.0, color])
        elif spell == "Lightning":
            origin = self.player.copy()
            visited = []
            self.lightning_visited = visited
            for _ in range(min(5 if combo else 3, len(self.enemies))):
                candidates = [enemy for enemy in self.enemies if enemy not in visited]
                if not candidates:
                    break
                enemy = min(candidates, key=lambda e: e.position.distance_to(origin))
                visited.append(enemy)
                target = enemy.position
                points = [origin.copy()]
                for index in range(1, 6):
                    point = origin.lerp(target, index / 6)
                    point.x += self.rng.uniform(-16, 16)
                    points.append(point)
                points.append(target.copy())
                self.bolts.append([points, 0.3])
                self.damage_enemy(enemy, 3, "Lightning")
                origin = target
            self.damage_boss(5 if combo else 3)
            if combo:
                self.hostile_orbs.clear()
                self.rings.append([self.player.copy(), 30, 0.7, color])
        elif spell == "Telekinesis":
            for enemy in self.enemies:
                enemy.y = max(-65, enemy.y - 100)
                self.burst(enemy.position, color, 10, 80)
            for orb in self.hostile_orbs:
                orb.velocity.y = -abs(orb.velocity.y)
            if self.boss:
                self.boss.shot_timer = max(self.boss.shot_timer, 2)
                self.boss.charge = 0
            self.rings.append([self.player.copy(), 40.0, 0.65, color])
        elif spell == "Ultimate":
            for enemy in self.enemies[:]:
                self.damage_enemy(enemy, 8, "Ultimate")
            self.damage_boss(8)
            self.hostile_orbs.clear()
            self.flash = 0.45
            self.rings.append([self.player.copy(), 20.0, 1.2, color])
            self.burst(self.player, color, 100, 340)
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

    def steer(self, hand_x):
        """Use mirrored palm position, independently of the spell classifier."""
        if hand_x is None or not math.isfinite(hand_x):
            self.move_target = None
            return
        fraction = max(0.0, min(1.0, (hand_x - 0.15) / 0.7))
        self.move_target = 72 + fraction * (ARENA_WIDTH - 144)

    def keyboard_move(self, axis, dt):
        if self.state == "playing" and axis:
            self.player.x = max(
                72, min(ARENA_WIDTH - 72, self.player.x + axis * 480 * min(dt, 0.1))
            )
            self.move_target = None

    def update(self, dt):
        if self.state != "playing":
            return
        dt = max(0.0, min(dt, 0.1))
        if self.move_target is not None:
            delta = (self.move_target - self.player.x) * (1 - math.exp(-10 * dt))
            self.player.x += max(-480 * dt, min(480 * dt, delta))
        self.elapsed += dt
        self.level = 1 + int(self.elapsed // 30)
        self.mana = min(100.0, self.mana + 10 * dt)
        self.flash = max(0.0, self.flash - dt)
        self.damage_flash = max(0.0, self.damage_flash - dt)
        self.spawn_timer -= dt
        if self.spawn_timer <= 0:
            self.spawn_enemy()
            self.spawn_timer = max(0.55, 1.65 * 0.86 ** (min(self.level, 8) - 1)) * (
                1.6 if self.boss else 1
            )
        for enemy in self.enemies:
            enemy.y += enemy.speed * dt
            enemy.x += (self.enemy_target(enemy).x - enemy.x) * 0.22 * dt
        self.update_threats(dt)
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
                impact = hit.position.copy()
                self.damage_enemy(
                    hit, projectile.damage, getattr(projectile, "kind", "Fireball")
                )
                if projectile.splash:
                    for enemy in self.enemies[:]:
                        if enemy.position.distance_to(impact) <= projectile.splash:
                            self.damage_enemy(
                                enemy,
                                projectile.damage,
                                getattr(projectile, "kind", "Fireball"),
                            )
                    if (
                        self.boss
                        and self.boss.position.distance_to(impact)
                        <= projectile.splash + self.boss.radius
                    ):
                        self.damage_boss(
                            projectile.damage,
                            **(
                                {"kind": getattr(projectile, "kind", "Fireball")}
                                if hasattr(self, "stage")
                                else {}
                            ),
                        )
                    self.rings.append([impact, 20, 0.3, COLORS["Fireball"]])
                self.projectiles.remove(projectile)
            elif self.boss and self.segment_hit(
                old, projectile.position, self.boss, projectile.radius
            ):
                impact = self.boss.position.copy()
                self.damage_boss(
                    projectile.damage,
                    **(
                        {"kind": getattr(projectile, "kind", "Fireball")}
                        if hasattr(self, "stage")
                        else {}
                    ),
                )
                if projectile.splash:
                    for enemy in self.enemies[:]:
                        if (
                            enemy.position.distance_to(impact)
                            <= projectile.splash + enemy.radius
                        ):
                            self.damage_enemy(
                                enemy,
                                projectile.damage,
                                getattr(projectile, "kind", "Fireball"),
                            )
                self.projectiles.remove(projectile)
            elif not (
                -60 <= projectile.position.x <= ARENA_WIDTH + 60
                and -60 <= projectile.position.y <= HEIGHT
            ):
                self.projectiles.remove(projectile)
            else:
                self.burst(projectile.position, COLORS["Fireball"], 2, 45)
        for enemy in self.enemies[:]:
            self.enemy_contact(enemy)
            if self.state == "gameover":
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
    """Illustrated adventure arena with a parchment spellbook and clear feedback."""

    def __init__(self, screen):
        self.screen = screen
        self.fonts = {
            size: pygame.font.SysFont("Avenir Next,Helvetica Neue,Arial", size)
            for size in (11, 12, 13, 14, 15, 16, 18, 20, 22, 24, 28, 30, 34, 42, 52, 60)
        }
        self.titles = {
            size: pygame.font.SysFont("Georgia", size, bold=True)
            for size in (24, 28, 34, 42, 52)
        }
        self.dark = (33, 57, 57)
        self.quiet = (105, 123, 118)
        self.cream = (252, 247, 233)
        self.gold = (204, 157, 68)
        self.background = self.make_background()
        self.boss_sprite = pygame.transform.smoothscale(
            self.make_enemy((176, 127, 220)), (128, 144)
        )
        self.wizard_sprite = self.make_wizard()
        self.enemy_sprites = [
            self.make_enemy(color)
            for color in ((232, 130, 165), (166, 142, 232), (103, 201, 198))
        ]
        self.icons = {spell: self.make_icon(spell) for spell in SPELLS}
        self.glows = {}
        self.camera_surface = None
        self.camera_timestamp = 0
        self.buttons = {}
        self.stats = {}

    def make_background(self):
        result = pygame.Surface((WIDTH, HEIGHT))
        result.fill((238, 233, 217))
        path = ROOT / "assets/images/enchanted_arena.png"
        if path.exists():
            art = pygame.image.load(str(path)).convert()
            # Scale to cover the arena, preserving the artwork's proportions.
            ratio = max(ARENA_WIDTH / art.get_width(), HEIGHT / art.get_height())
            art = pygame.transform.smoothscale(
                art, (int(art.get_width() * ratio), int(art.get_height() * ratio))
            )
            result.blit(
                art,
                (
                    (ARENA_WIDTH - art.get_width()) // 2,
                    (HEIGHT - art.get_height()) // 2,
                ),
            )
        else:
            for y in range(HEIGHT):
                t = y / HEIGHT
                pygame.draw.line(
                    result,
                    (int(38 + 35 * t), int(91 + 45 * t), int(92 + 15 * t)),
                    (0, y),
                    (ARENA_WIDTH, y),
                )
            rng = random.Random(5)
            for _ in range(90):
                x = rng.choice((rng.randrange(0, 140), rng.randrange(730, 870)))
                y = rng.randrange(HEIGHT)
                pygame.draw.circle(
                    result,
                    rng.choice(((35, 105, 94), (65, 133, 108), (125, 153, 105))),
                    (x, y),
                    rng.randrange(15, 45),
                )
            pygame.draw.ellipse(result, (145, 174, 152), (120, 110, 630, 600))
        # Fade the edges softly without covering the playable field.
        veil = pygame.Surface((ARENA_WIDTH, HEIGHT), pygame.SRCALPHA)
        for y in range(HEIGHT):
            edge = max(0, 1 - y / 150) * 90 + max(0, (y - 560) / 160) * 100
            pygame.draw.line(veil, (12, 40, 42, int(edge)), (0, y), (ARENA_WIDTH, y))
        result.blit(veil, (0, 0))
        pygame.draw.rect(
            result, self.cream, (ARENA_WIDTH, 0, WIDTH - ARENA_WIDTH, HEIGHT)
        )
        pygame.draw.line(
            result, (214, 211, 191), (ARENA_WIDTH, 0), (ARENA_WIDTH, HEIGHT), 2
        )
        return result

    @staticmethod
    def make_wizard():
        # Draw at twice display size for smooth character outlines.
        s = pygame.Surface((224, 264), pygame.SRCALPHA)
        s.fill((0, 0, 0, 0))
        pygame.draw.ellipse(s, (24, 48, 46, 75), (36, 238, 157, 20))
        pygame.draw.ellipse(s, (47, 53, 86), (69, 222, 38, 24))
        pygame.draw.ellipse(s, (47, 53, 86), (114, 222, 38, 24))
        pygame.draw.polygon(
            s,
            (38, 89, 105),
            [(110, 100), (51, 224), (76, 240), (159, 235), (177, 220), (142, 110)],
        )
        pygame.draw.polygon(
            s,
            (71, 160, 175),
            [(107, 105), (58, 222), (103, 235), (146, 231), (153, 197), (136, 107)],
        )
        pygame.draw.polygon(
            s, (111, 201, 199), [(104, 108), (83, 207), (105, 217), (114, 109)]
        )
        pygame.draw.lines(
            s, (235, 194, 102), False, [(60, 223), (104, 239), (158, 230)], 5
        )
        pygame.draw.ellipse(s, (40, 98, 112), (43, 121, 39, 86))
        pygame.draw.ellipse(s, (68, 148, 164), (145, 121, 36, 77))
        pygame.draw.circle(s, (245, 186, 145), (58, 196), 13)
        pygame.draw.circle(s, (245, 186, 145), (164, 177), 12)
        pygame.draw.line(s, (107, 75, 56), (181, 237), (192, 83), 9)
        pygame.draw.line(s, (202, 161, 92), (184, 233), (194, 87), 3)
        pygame.draw.circle(s, (82, 189, 180, 65), (194, 77), 27)
        pygame.draw.circle(s, (90, 224, 207), (194, 77), 14)
        pygame.draw.circle(s, (212, 255, 230), (190, 72), 6)
        pygame.draw.ellipse(s, (131, 91, 74), (78, 81, 67, 62))
        pygame.draw.ellipse(s, (254, 210, 168), (84, 86, 58, 62))
        pygame.draw.circle(s, (242, 171, 145), (87, 122), 8)
        pygame.draw.circle(s, (242, 171, 145), (137, 122), 8)
        pygame.draw.ellipse(s, (32, 60, 69), (98, 111, 7, 10))
        pygame.draw.ellipse(s, (32, 60, 69), (121, 111, 7, 10))
        pygame.draw.arc(s, (168, 99, 79), (105, 122, 18, 13), math.pi, math.tau, 2)
        pygame.draw.polygon(
            s, (56, 80, 118), [(70, 94), (146, 97), (151, 66), (113, 11), (80, 68)]
        )
        pygame.draw.polygon(
            s, (97, 125, 177), [(81, 81), (142, 83), (137, 57), (113, 11), (107, 63)]
        )
        pygame.draw.polygon(s, (140, 166, 211), [(113, 11), (107, 63), (137, 57)])
        pygame.draw.ellipse(s, (53, 77, 115), (48, 75, 125, 29))
        pygame.draw.arc(s, (159, 187, 225), (48, 75, 125, 29), 0, math.pi, 3)
        pygame.draw.line(s, (238, 194, 91), (78, 78), (148, 81), 7)
        pygame.draw.circle(s, (255, 223, 121), (117, 61), 6)
        pygame.draw.circle(s, (255, 239, 175), (115, 59), 2)
        return pygame.transform.smoothscale(
            s.premul_alpha(), (112, 132)
        ).convert_alpha()

    @staticmethod
    def make_enemy(color):
        s = pygame.Surface((128, 144), pygame.SRCALPHA)
        s.fill((0, 0, 0, 0))
        pygame.draw.ellipse(s, (15, 46, 47, 55), (23, 119, 85, 15))
        darker = tuple(max(0, c - 55) for c in color)
        pygame.draw.polygon(s, darker, [(31, 65), (10, 55), (17, 88), (43, 88)])
        pygame.draw.polygon(s, darker, [(95, 65), (119, 52), (109, 86), (84, 86)])
        pygame.draw.ellipse(s, darker, (27, 27, 77, 87))
        pygame.draw.ellipse(s, color, (24, 21, 77, 86))
        pygame.draw.polygon(
            s,
            color,
            [(27, 74), (24, 119), (45, 103), (62, 127), (77, 102), (98, 116), (98, 74)],
        )
        pygame.draw.ellipse(s, tuple(min(255, c + 28) for c in color), (36, 31, 46, 19))
        for x in (45, 74):
            pygame.draw.ellipse(s, (255, 248, 223), (x, 57, 17, 21))
            pygame.draw.ellipse(s, (47, 59, 70), (x + 6, 63, 7, 12))
            pygame.draw.circle(s, (255, 255, 250), (x + 8, 64), 2)
        pygame.draw.arc(s, (84, 71, 91), (54, 77, 19, 15), math.pi, math.tau, 3)
        return pygame.transform.smoothscale(s.premul_alpha(), (64, 72)).convert_alpha()

    @staticmethod
    def make_icon(spell):
        s = pygame.Surface((80, 80), pygame.SRCALPHA)
        if spell == "Fireball":
            pygame.draw.polygon(
                s,
                (249, 139, 63),
                [
                    (40, 7),
                    (56, 32),
                    (66, 46),
                    (61, 64),
                    (45, 73),
                    (25, 65),
                    (15, 50),
                    (22, 30),
                    (28, 40),
                ],
            )
            pygame.draw.polygon(
                s, (255, 222, 103), [(39, 34), (51, 48), (48, 65), (33, 66), (28, 54)]
            )
        elif spell == "Shield":
            pygame.draw.polygon(
                s,
                (62, 143, 195),
                [(40, 9), (67, 22), (63, 52), (40, 73), (17, 52), (13, 22)],
            )
            pygame.draw.polygon(
                s,
                (114, 210, 235),
                [(40, 18), (58, 27), (54, 49), (40, 62), (26, 49), (22, 27)],
            )
            pygame.draw.line(s, (239, 250, 233), (40, 25), (40, 53), 4)
        elif spell == "Lightning":
            pygame.draw.polygon(
                s,
                (235, 171, 37),
                [(43, 5), (18, 43), (36, 43), (28, 76), (65, 32), (45, 32), (55, 5)],
            )
            pygame.draw.polygon(
                s,
                (255, 225, 112),
                [(44, 10), (25, 39), (43, 37), (36, 60), (57, 35), (40, 36)],
            )
        elif spell == "Telekinesis":
            pygame.draw.arc(s, (37, 170, 132), (13, 13, 54, 54), 0.35, 5.65, 7)
            pygame.draw.polygon(s, (74, 210, 167), [(59, 10), (70, 29), (47, 25)])
            pygame.draw.arc(s, (98, 216, 177), (27, 27, 26, 26), 0.5, 5.5, 5)
        else:
            points = []
            for j in range(10):
                angle = -math.pi / 2 + j * math.pi / 5
                r = 34 if j % 2 == 0 else 15
                points.append((40 + math.cos(angle) * r, 40 + math.sin(angle) * r))
            pygame.draw.polygon(s, (146, 101, 202), points)
            pygame.draw.circle(s, (229, 204, 250), (40, 40), 10)
            pygame.draw.circle(s, (255, 247, 228), (40, 40), 4)
        return pygame.transform.smoothscale(s, (36, 36))

    def glow(self, position, color, radius=35):
        key = (color, radius)
        if key not in self.glows:
            s = pygame.Surface((radius * 2 + 2, radius * 2 + 2), pygame.SRCALPHA)
            for r in range(radius, 0, -2):
                alpha = int(3 + 24 * (1 - r / radius))
                pygame.draw.circle(s, (*color, alpha), (radius + 1, radius + 1), r)
            self.glows[key] = s
        self.screen.blit(
            self.glows[key], (position[0] - radius - 1, position[1] - radius - 1)
        )

    def text(self, value, x, y, size=18, color=TEXT, center=False, title=False):
        font = self.titles.get(size, self.fonts[size]) if title else self.fonts[size]
        s = font.render(str(value), True, color)
        rect = (
            s.get_rect(center=(int(x), int(y)))
            if center
            else s.get_rect(topleft=(int(x), int(y)))
        )
        self.screen.blit(s, rect)
        return rect

    def panel(self, rect, color=None, border=(219, 214, 195), radius=16):
        rect = pygame.Rect(rect)
        shadow = pygame.Surface((rect.width + 12, rect.height + 14), pygame.SRCALPHA)
        pygame.draw.rect(
            shadow,
            (13, 37, 39, 45),
            (6, 8, rect.width, rect.height),
            border_radius=radius,
        )
        self.screen.blit(shadow, (rect.x - 6, rect.y - 3))
        pygame.draw.rect(self.screen, color or self.cream, rect, border_radius=radius)
        pygame.draw.rect(self.screen, border, rect, 1, border_radius=radius)

    def wrapped(self, value, x, y, width, size=13, color=None, max_lines=2):
        lines = []
        line = ""
        for word in value.split():
            candidate = (line + " " + word).strip()
            if line and self.fonts[size].size(candidate)[0] > width:
                lines.append(line)
                line = word
            else:
                line = candidate
        if line:
            lines.append(line)
        if len(lines) > max_lines:
            lines = lines[:max_lines]
            while self.fonts[size].size(lines[-1] + "...")[0] > width:
                lines[-1] = lines[-1][:-1]
            lines[-1] += "..."
        for index, line in enumerate(lines):
            self.text(line, x, y + index * (size + 4), size, color or self.quiet)
        return y + len(lines) * (size + 4)

    def button(self, key, caption, rect, color=(48, 117, 105), size=16):
        rect = pygame.Rect(rect)
        self.buttons[key] = rect
        hovered = rect.collidepoint(pygame.mouse.get_pos())
        fill = tuple(min(255, c + 14) for c in color) if hovered else color
        self.panel(rect, fill, tuple(min(255, c + 20) for c in color), 12)
        self.text(caption, rect.centerx, rect.centery, size, (255, 249, 229), True)

    def meter(self, label, value, x, y, color, width=142):
        self.text(label, x, y, 11, (184, 213, 206))
        self.text(str(int(value)), x + width - 20, y - 1, 13, (249, 245, 220))
        pygame.draw.rect(
            self.screen, (36, 72, 72), (x, y + 21, width, 7), border_radius=4
        )
        if value > 0:
            pygame.draw.rect(
                self.screen,
                color,
                (x, y + 21, max(1, int(width * value / 100)), 7),
                border_radius=4,
            )

    def wizard(self, game):
        pulse = 1 + math.sin(game.elapsed * 3) * 0.04
        pygame.draw.ellipse(
            self.screen,
            (61, 193, 165),
            (game.player.x - 50 * pulse, game.player.y + 18, 100 * pulse, 25),
            2,
        )
        self.glow((game.player.x, game.player.y + 20), (64, 230, 191), 45)
        self.screen.blit(
            self.wizard_sprite,
            (game.player.x - 56, game.player.y - 104),
            special_flags=pygame.BLEND_PREMULTIPLIED,
        )
        if game.elapsed < game.shield_until:
            b = pygame.Surface((160, 178), pygame.SRCALPHA)
            pygame.draw.ellipse(b, (119, 228, 252, 43), (7, 6, 144, 166))
            pygame.draw.ellipse(
                b,
                (
                    (255, 228, 133, 240)
                    if game.elapsed < game.perfect_until
                    else (186, 255, 246, 200)
                ),
                (7, 6, 144, 166),
                3,
            )
            pygame.draw.arc(b, (248, 255, 249, 220), (19, 17, 116, 141), 0.8, 2.4, 4)
            self.screen.blit(b, (game.player.x - 80, game.player.y - 127))

    def draw_arena(self, game):
        old_clip = self.screen.get_clip()
        self.screen.set_clip((0, 86, ARENA_WIDTH, 535))
        sources = list(game.enemies) + ([game.boss] if game.boss else [])
        for source in sources:
            if source.charge > 0 and source.aim is not None:
                endpoint = source.position + source.aim * 500
                pygame.draw.line(
                    self.screen, (255, 202, 132), source.position, endpoint, 2
                )
                pygame.draw.circle(
                    self.screen,
                    (255, 220, 126),
                    source.position,
                    int(source.radius + 8 + 8 * source.charge),
                    2,
                )
        if game.boss:
            boss = game.boss
            self.glow(boss.position, (209, 141, 243), 75)
            self.screen.blit(
                self.boss_sprite,
                (boss.x - 64, boss.y - 67),
                special_flags=pygame.BLEND_PREMULTIPLIED,
            )
            pygame.draw.polygon(
                self.screen,
                (255, 221, 128),
                [
                    (boss.x - 25, boss.y - 49),
                    (boss.x - 31, boss.y - 72),
                    (boss.x - 12, boss.y - 63),
                    (boss.x, boss.y - 83),
                    (boss.x + 12, boss.y - 63),
                    (boss.x + 31, boss.y - 72),
                    (boss.x + 25, boss.y - 49),
                ],
            )
        for enemy in game.enemies:
            if getattr(enemy, "kind", "") == "boss_crystal":
                continue
            variant = 0 if enemy.shooter else int(enemy.phase * 10) % 3
            bob = math.sin(game.elapsed * 4 + enemy.phase) * 3
            self.screen.blit(
                self.enemy_sprites[variant],
                (enemy.x - 32, enemy.y - 34 + bob),
                special_flags=pygame.BLEND_PREMULTIPLIED,
            )
            if enemy.shooter:
                pygame.draw.circle(
                    self.screen, (255, 221, 130), (enemy.x, enemy.y - 36), 5
                )
        for orb in game.hostile_orbs:
            self.glow(orb.position, (255, 103, 155), 22)
            pygame.draw.circle(self.screen, (182, 54, 112), orb.position, 10)
            pygame.draw.circle(self.screen, (255, 152, 181), orb.position, 7)
            pygame.draw.circle(self.screen, (255, 240, 210), orb.position, 3)
        for projectile in game.projectiles:
            position = projectile.position
            self.glow(position, (255, 168, 67), 30)
            pygame.draw.circle(self.screen, (226, 89, 49), position, 12)
            pygame.draw.circle(self.screen, (255, 183, 79), position, 9)
            pygame.draw.circle(
                self.screen, (255, 246, 182), position + pygame.Vector2(-2, -2), 4
            )
        for particle in game.particles:
            fade = max(0, particle.life / particle.total_life)
            pygame.draw.circle(
                self.screen,
                particle.color,
                particle.position,
                max(1, int(particle.size * fade)),
            )
        for points, life in game.bolts:
            pygame.draw.lines(self.screen, (241, 177, 45), False, points, 7)
            pygame.draw.lines(self.screen, (255, 251, 201), False, points, 3)
        for position, radius, life, color in game.rings:
            pygame.draw.circle(self.screen, color, position, int(radius), 3)
        self.wizard(game)
        if game.flash or game.damage_flash:
            veil = pygame.Surface((ARENA_WIDTH, HEIGHT), pygame.SRCALPHA)
            veil.fill(
                (252, 220, 255, int(game.flash * 220))
                if game.flash
                else (255, 111, 99, int(game.damage_flash * 180))
            )
            self.screen.blit(veil, (0, 0))
        self.screen.set_clip(old_clip)

    def camera(self, snapshot, no_camera):
        prediction, timestamp, actions, error, status = snapshot
        x = 892
        self.text("Your magic", x, 23, 28, self.dark, title=True)
        self.text("Combos: point then palm / fist then peace", x, 59, 13, self.quiet)
        rect = pygame.Rect(x, 87, 366, 246)
        self.panel(rect, (221, 233, 219), (210, 219, 204), 18)
        if prediction is not None and timestamp != self.camera_timestamp:
            decoded = pygame.image.load(
                io.BytesIO(prediction.preview_image), "camera.jpg"
            ).convert()
            ratio = max(
                rect.width / decoded.get_width(), rect.height / decoded.get_height()
            )
            decoded = pygame.transform.smoothscale(
                decoded,
                (int(decoded.get_width() * ratio), int(decoded.get_height() * ratio)),
            )
            cropped = pygame.Surface(rect.size)
            cropped.blit(
                decoded,
                (
                    (rect.width - decoded.get_width()) // 2,
                    (rect.height - decoded.get_height()) // 2,
                ),
            )
            mask = pygame.Surface(rect.size, pygame.SRCALPHA)
            pygame.draw.rect(
                mask, (255, 255, 255, 255), mask.get_rect(), border_radius=18
            )
            rounded = pygame.Surface(rect.size, pygame.SRCALPHA)
            rounded.blit(cropped, (0, 0))
            rounded.blit(mask, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
            self.camera_surface = rounded
            self.camera_timestamp = timestamp
        if self.camera_surface:
            self.screen.blit(self.camera_surface, rect)
        else:
            pygame.draw.circle(self.screen, (194, 217, 198), rect.center, 37)
            # A clean camera glyph in place of the former large black placeholder.
            pygame.draw.rect(
                self.screen,
                (78, 122, 107),
                (rect.centerx - 22, rect.centery - 16, 43, 30),
                3,
                border_radius=6,
            )
            pygame.draw.circle(self.screen, (78, 122, 107), rect.center, 9, 3)
            self.text(
                "Keyboard adventure" if no_camera else "Waking up your camera...",
                rect.centerx,
                rect.centery + 58,
                14,
                self.quiet,
                True,
            )
        stale = prediction is not None and time.monotonic() - timestamp > 0.7
        active = prediction is not None and not stale and error is None
        accepted = active and prediction.gesture_id is not None
        label = (
            prediction.gesture_name
            if accepted
            else ("Keys 1-5 are ready" if no_camera else "Show your hand")
        )
        confidence = prediction.confidence if accepted else 0.0
        self.panel((x, 346, 366, 86), (239, 241, 220), (218, 224, 201), 12)
        pygame.draw.circle(
            self.screen,
            (52, 161, 115) if active or no_camera else (221, 173, 72),
            (x + 18, 367),
            4,
        )
        self.text(label, x + 31, 356, 20, self.dark)
        if accepted:
            self.text(f"{confidence:.0%}", x + 312, 358, 18, (47, 128, 105))
        friendly = (
            "Camera unavailable. Use keys 1-5."
            if error
            else (
                "Camera paused. Use keys 1-5."
                if stale
                else (
                    "Hold a pose, then lower your hands to cast again."
                    if accepted
                    else (
                        "One hand for most spells. Both palms for your ultimate."
                        if not no_camera
                        else "Cast with your keyboard or click a spell below."
                    )
                )
            )
        )
        self.wrapped(friendly, x + 15, 388, 337, 12, max_lines=2)
        return active

    def gesture_guide(self, game, snapshot):
        x = 892
        self.text("SPELLBOOK", x, 452, 11, self.quiet)
        self.text("Gesture", x + 177, 452, 11, self.quiet)
        self.text("Mana", x + 322, 452, 11, self.quiet)
        gestures = ("Open palm", "Closed fist", "Peace sign", "Pointing", "Both palms")
        prediction, timestamp, *_ = snapshot
        active_id = (
            prediction.gesture_id
            if prediction is not None and time.monotonic() - timestamp <= 0.7
            else None
        )
        for index, (spell, gesture) in enumerate(zip(SPELLS, gestures)):
            y = 476 + index * 40
            if active_id == index:
                pygame.draw.rect(
                    self.screen,
                    (231, 239, 220),
                    (x - 4, y - 4, 374, 38),
                    border_radius=8,
                )
            self.screen.blit(self.icons[spell], (x, y - 3))
            self.text(spell, x + 45, y + 2, 15, self.dark)
            self.text(gesture, x + 177, y + 3, 12, self.quiet)
            self.text(str(COSTS[spell]), x + 330, y + 3, 13, self.dark)
        pygame.draw.line(self.screen, (221, 216, 198), (x, 685), (1258, 685))
        self.text(
            "Move hand to steer. Lower hands to stop casting.", x, 697, 11, self.quiet
        )

    def spell_cards(self, game):
        for index, spell in enumerate(SPELLS):
            rect = pygame.Rect(18 + index * 167, 634, 157, 68)
            remaining = max(
                0, COOLDOWNS[spell] - (game.elapsed - game.last_cast[spell])
            )
            ready = (
                game.mana >= COSTS[spell] and remaining <= 0 and game.state == "playing"
            )
            self.buttons["cast_" + spell] = rect
            hover = rect.collidepoint(pygame.mouse.get_pos())
            fill = (255, 250, 232) if ready else (221, 226, 211)
            border = (222, 172, 77) if ready and hover else (202, 213, 195)
            self.panel(rect, fill, border, 12)
            self.screen.blit(self.icons[spell], (rect.x + 8, rect.y + 9))
            self.text(spell, rect.x + 49, rect.y + 11, 13, self.dark)
            feedback = (
                f"{remaining:.1f}s cooldown"
                if remaining > 0
                else (
                    f"{COSTS[spell]} mana"
                    if game.mana >= COSTS[spell]
                    else "Need more mana"
                )
            )
            self.text(feedback, rect.x + 49, rect.y + 34, 11, self.quiet)
            pygame.draw.rect(
                self.screen,
                (229, 232, 218),
                (rect.x + 8, rect.y + 51, 19, 13),
                border_radius=3,
            )
            self.text(str(index + 1), rect.x + 17, rect.y + 57, 11, self.dark, True)
            if remaining > 0:
                pygame.draw.rect(
                    self.screen,
                    (152, 181, 166),
                    (rect.x + 38, rect.y + 56, 108, 4),
                    border_radius=2,
                )
                pygame.draw.rect(
                    self.screen,
                    (65, 139, 116),
                    (
                        rect.x + 38,
                        rect.y + 56,
                        max(1, int(108 * (1 - remaining / COOLDOWNS[spell]))),
                        4,
                    ),
                    border_radius=2,
                )

    def overlay(self, game):
        # Block clicks through to spells whenever the game is not playing.
        self.buttons = {
            key: rect
            for key, rect in self.buttons.items()
            if key == "mute" or (key == "pause" and game.state == "paused")
        }
        veil = pygame.Surface((ARENA_WIDTH, HEIGHT), pygame.SRCALPHA)
        veil.fill((24, 67, 62, 75))
        self.screen.blit(veil, (0, 0))
        self.panel((129, 124, 612, 474), (253, 248, 229), (224, 200, 139), 25)
        self.text(
            "BOSS PRACTICE" if game.boss_delay == 5 else "COLORFUL ADVENTURE EDITION",
            435,
            163,
            11,
            (160, 120, 54),
            True,
        )
        if game.state == "start":
            self.text("Arcane Arena", 435, 220, 52, self.dark, True, True)
            self.text("Your hands. Your magic.", 435, 270, 20, (78, 113, 103), True)
            sprite = pygame.transform.smoothscale(self.wizard_sprite, (95, 112))
            self.screen.blit(
                sprite, (388, 296), special_flags=pygame.BLEND_PREMULTIPLIED
            )
            self.text(
                "Move your hand left or right to dodge the spirits.",
                435,
                432,
                16,
                self.dark,
                True,
            )
            self.text(
                "Fist counters shots. Point then palm makes a Meteor Burst.",
                435,
                461,
                14,
                self.quiet,
                True,
            )
            caption = "Begin adventure    /    SPACE"
        elif game.state == "paused":
            self.text("Take a breather", 435, 234, 42, self.dark, True, True)
            self.text(
                "Your adventure will be right here.", 435, 292, 18, self.quiet, True
            )
            self.screen.blit(
                self.wizard_sprite, (379, 317), special_flags=pygame.BLEND_PREMULTIPLIED
            )
            caption = "Back to the magic    /    P"
        else:
            self.text("A spell well spent", 435, 232, 42, self.dark, True, True)
            self.text(
                "The spirits won this round. Try again?", 435, 291, 18, self.quiet, True
            )
            for x, label, value in (
                (260, "SCORE", game.score),
                (435, "LEVEL", game.level),
                (610, "BEST", game.best_score),
            ):
                self.text(str(value), x, 368, 34, self.dark, True, True)
                self.text(label, x, 405, 11, self.quiet, True)
            self.text(
                f"{game.dodges} dodges  /  {game.parries} counters  /  {game.combo_count} combos  /  {game.bosses_defeated} guardians",
                435,
                452,
                14,
                self.quiet,
                True,
            )
            caption = "Another adventure    /    R"
        self.button("primary", caption, (239, 510, 392, 52), (45, 122, 103), 16)

    def draw(self, game, snapshot, fps, no_camera, muted):
        self.buttons.clear()
        self.screen.blit(self.background, (0, 0))
        self.draw_arena(game)
        self.panel((18, 18, 834, 65), (25, 67, 64), (83, 122, 99), 16)
        self.text("Arcane Arena", 35, 31, 24, (255, 242, 195), title=True)
        self.text(
            f"LEVEL {game.level}  /  {int(game.elapsed)//60:02d}:{int(game.elapsed)%60:02d}",
            36,
            62,
            11,
            (171, 210, 189),
        )
        self.meter("HEALTH", game.health, 262, 33, (243, 129, 126))
        self.meter("MANA", game.mana, 430, 33, (103, 220, 205))
        self.text("SCORE", 599, 31, 11, (174, 211, 190))
        self.text(f"{game.score:04d}", 599, 45, 24, (255, 242, 195))
        self.button(
            "pause",
            "II" if game.state != "paused" else ">",
            (709, 33, 40, 33),
            (46, 99, 87),
            15,
        )
        self.button(
            "mute", "OFF" if muted else "SFX", (761, 33, 68, 33), (46, 99, 87), 13
        )
        if game.elapsed < game.message_until and game.state == "playing":
            message = game.message
            width = min(770, self.fonts[14].size(message)[0] + 38)
            self.panel(
                (435 - width / 2, 99, width, 32), (253, 244, 211), (220, 198, 147), 10
            )
            self.text(message, 435, 115, 14, self.dark, True)
        if game.boss:
            self.panel((233, 144, 404, 40), (59, 51, 91), (202, 169, 221), 10)
            self.text(
                "GROVE GUARDIAN"
                + (" / ENRAGED" if game.boss.hp <= game.boss.max_hp / 2 else ""),
                435,
                157,
                11,
                (255, 238, 202),
                True,
            )
            pygame.draw.rect(
                self.screen, (104, 79, 121), (248, 170, 374, 6), border_radius=3
            )
            pygame.draw.rect(
                self.screen,
                (232, 164, 220),
                (248, 170, max(1, int(374 * game.boss.hp / game.boss.max_hp)), 6),
                border_radius=3,
            )
        elif game.state == "playing" and math.isfinite(game.next_boss_at):
            self.text(
                f"Guardian arrives in {max(0, math.ceil(game.next_boss_at-game.elapsed))}s",
                435,
                153,
                13,
                self.dark,
                True,
            )
        if game.elapsed < game.combo_until:
            self.panel((18, 144, 199, 40), (253, 244, 211), (220, 198, 147), 10)
            self.text(game.combo_name, 117, 164, 16, self.dark, True)
        elif (
            game.state == "playing"
            and game.combo_spell in ("Telekinesis", "Shield")
            and game.elapsed - game.combo_at <= 3
        ):
            hint = (
                "Palm -> Meteor"
                if game.combo_spell == "Telekinesis"
                else "Peace -> Storm"
            )
            self.panel((18, 144, 199, 40), (253, 244, 211), (220, 198, 147), 10)
            self.text(
                hint + f" / {3 - (game.elapsed - game.combo_at):.1f}s",
                117,
                164,
                11,
                self.dark,
                True,
            )
        self.spell_cards(game)
        self.panel((18, 600, 834, 24), (25, 67, 64), (83, 122, 99), 8)
        self.text(
            f"{fps:.0f} FPS  /  DODGED {game.dodges}", 22, 612, 11, (226, 244, 223)
        )
        self.text(
            "Move hand left / right   •   A / D or arrow keys",
            435,
            612,
            11,
            (255, 249, 225),
            True,
        )
        self.text(f"COUNTERS {game.parries}", 740, 612, 11, (226, 244, 223))
        # Keep footer information inside the arena and away from the sidebar.
        self.camera(snapshot, no_camera)
        self.gesture_guide(game, snapshot)
        if game.state in ("start", "paused", "gameover"):
            self.overlay(game)
        self.stats = {
            "enemy_count": len(game.enemies),
            "particle_count": len(game.particles),
        }


def apply_camera_input(game, snapshot, now=None):
    """Consume fresh events; holding an accepted open palm continues fireballs."""
    now = time.monotonic() if now is None else now
    prediction, timestamp, events, error, status = snapshot
    if game.state != "playing" or error is not None or not 0 <= now - timestamp <= 0.7:
        game.move_target = None
        return
    game.steer(getattr(prediction, "hand_x", None))
    for spell, observed_at, generation in events:
        if 0 <= now - observed_at <= 0.5:
            game.cast(spell)
    if (
        prediction is not None
        and prediction.gesture_id == 0
        and prediction.confidence > 0.8
        and game.elapsed - game.last_cast["Fireball"] >= COOLDOWNS["Fireball"]
        and game.mana >= COSTS["Fireball"]
    ):
        game.cast("Fireball")
