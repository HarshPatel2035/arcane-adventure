"""Arcane Adventure - The Enchanted Grove: webcam-gesture spellcasting game.

Run: ./.venv/bin/python arcane_adventure.py
Keyboard-only: ./.venv/bin/python arcane_adventure.py --no-camera
Move your hand horizontally to steer; arrow keys or A/D also move.
Space starts; P pauses; R restarts after game over; 1-5 cast; Esc quits.
Built on the shared combat engine in spell_engine.py.
"""

from __future__ import annotations
import argparse
from dataclasses import dataclass
from pathlib import Path
import json
import math
import sys
import time
import pygame
import spell_engine as base

UPGRADES = {
    "Twin Flame": ("Two weaker fireballs instead of one.", "Fireball"),
    "Mirror Shield": ("Perfect counters return three shots.", "Shield"),
    "Thunder Chain": ("Lightning reaches two more enemies.", "Lightning"),
    "Gravity Well": ("Pointing groups enemies for splash attacks.", "Telekinesis"),
    "Vital Bloom": ("Restore 30 health now and 5 after each encounter.", "Shield"),
    "Mana Spring": ("Recover 5 extra mana / sec while standing still.", "Ultimate"),
}
STAGES = (
    "Forest Entrance",
    "Swarm Hollow",
    "Protect the Heart Tree",
    "Seal the Portals",
    "Grove Guardian",
)
COMPANIONS = {
    "owl": ("Willow the Owl", "Collects mana drops for you."),
    "dragon": ("Ember the Dragon", "Blocks one hit every 12 seconds."),
    "spirit": ("Luma the Spirit", "Marks priority targets; combos last 4 seconds."),
}
SKINS = {"mint": (71, 160, 175), "gold": (220, 173, 76), "rose": (205, 111, 156)}


@dataclass
class ManaDrop:
    position: pygame.Vector2
    life: float = 9


@dataclass
class MagicObject:
    position: pygame.Vector2
    kind: str
    velocity: object = None
    life: float = 3
    radius: float = 19


@dataclass
class Portal:
    position: pygame.Vector2
    progress: float = 0


def load_profile(path):
    defaults = {"wins": 0, "guardians": 0, "best": 0, "skin": "mint"}
    if path is None:
        return defaults
    try:
        data = json.loads(Path(path).read_text())
        for key in ("wins", "guardians", "best"):
            defaults[key] = max(0, int(data.get(key, 0)))
        if data.get("skin") in SKINS:
            defaults["skin"] = data["skin"]
    except (OSError, ValueError, TypeError, AttributeError):
        pass
    return defaults


class AdventureGame(base.SpellGame):
    def __init__(
        self,
        sound=None,
        seed=None,
        mode="adventure",
        companion="owl",
        motion=False,
        practice=False,
        profile_path=None,
    ):
        self.mode = mode
        self.companion = companion
        self.motion_enabled = motion
        self.practice = practice
        self.profile_path = profile_path
        self.profile = load_profile(profile_path)
        self.skin = (
            self.profile["skin"]
            if self.profile["skin"] in self.unlocked_skins()
            else "mint"
        )
        super().__init__(sound, seed, boss_delay=5 if practice else 45)
        self.best_score = self.profile["best"]

    def unlocked_skins(self):
        return (
            ["mint"]
            + (["gold"] if self.profile["wins"] >= 1 else [])
            + (["rose"] if self.profile["guardians"] >= 3 else [])
        )

    def save_profile(self):
        self.profile["best"] = max(self.profile["best"], self.score)
        self.profile["skin"] = self.skin
        if self.profile_path is None:
            return
        try:
            path = Path(self.profile_path)
            path.parent.mkdir(exist_ok=True, parents=True)
            temp = path.with_suffix(".tmp")
            temp.write_text(json.dumps(self.profile, indent=2) + "\n")
            temp.replace(path)
        except OSError as exc:
            self.save_error = str(exc)

    def reset(self):
        super().reset()
        self.stage = 4 if self.practice and self.mode == "adventure" else 0
        self.stage_started = 0
        self.stage_spawned = 0
        self.stage_quota = 8
        self.tree_health = 100
        self.portals = []
        self.upgrades = set()
        self.upgrade_options = []
        self.upgrade_index = 0
        self.menu_hover_since = None
        self.menu_confirm_ready = False
        self.combo_window = 4 if self.companion == "spirit" else 3
        self.dragon_ready = 0
        self.marked = None
        self.mana_drops = []
        self.objects = []
        self.held_object = None
        self.held_by_hand = False
        self.cursor = None
        self.cursor_live = False
        self.last_player_x = self.player.x
        self.object_respawn_at = 0
        self.boss_phase = "crystal"
        self.boss_exposed_until = 0
        self.phase_notice = ""
        self.storm_at = 0
        self.safe_x = None
        self.storm_strike_at = 0
        self.runes = []
        self.motion_last = dict.fromkeys(("sweep", "circle", "charge"), float("-inf"))
        self.feedback = "Show a pose to cast; move your hand to steer."
        self.save_error = None
        self.next_boss_at = (
            5 if self.practice else float("inf") if self.mode == "adventure" else 45
        )
        self.spawn_timer = 0.8
        self.prepare_objects()

    def prepare_objects(self):
        self.objects = [
            MagicObject(pygame.Vector2(x, y), kind)
            for x, y, kind in (
                (140, 405, "rock"),
                (435, 390, "seed"),
                (730, 405, "crystal"),
            )
        ]
        self.held_object = None
        self.held_by_hand = False

    def enter_stage(self, stage):
        self.stage = stage
        self.stage_started = self.elapsed
        self.stage_spawned = 0
        self.stage_quota = (8, 12, 999, 999, 999)[stage]
        self.enemies.clear()
        self.projectiles.clear()
        self.hostile_orbs.clear()
        self.mana_drops.clear()
        self.runes.clear()
        self.combo_spell = None
        self.move_target = None
        self.prepare_objects()
        self.spawn_timer = 0.8
        self.portals = (
            [Portal(pygame.Vector2(x, 325)) for x in (180, 435, 690)]
            if stage == 3
            else []
        )
        if stage == 3:
            for obj in self.objects:
                obj.position.y = 435
        if "Vital Bloom" in self.upgrades:
            self.health = min(100, self.health + 5)
        self.next_boss_at = self.elapsed + 1 if stage == 4 else float("inf")
        self.notify(STAGES[stage])

    def finish_stage(self):
        if self.state != "playing":
            return
        if self.stage == 4 and self.mode == "adventure":
            self.state = "victory"
            self.profile["wins"] += 1
            self.save_profile()
            return
        self.state = "upgrade"
        available = [name for name in UPGRADES if name not in self.upgrades]
        if not available:
            self.state = "playing"
            return
        self.upgrade_options = self.rng.sample(available, min(3, len(available)))
        self.upgrade_index = 0
        self.menu_confirm_ready = False
        self.menu_hover_since = None
        self.move_target = None
        self.held_object = None

    def choose_upgrade(self, index):
        if self.state != "upgrade" or not 0 <= index < len(self.upgrade_options):
            return False
        name = self.upgrade_options[index]
        self.upgrades.add(name)
        if name == "Vital Bloom":
            self.health = min(100, self.health + 30)
        self.state = "playing"
        if self.mode == "adventure":
            self.enter_stage(self.stage + 1)
        self.notify(name + " learned!")
        return True

    def spawn_enemy(self):
        if (
            self.mode == "adventure"
            and self.stage < 2
            and self.stage_spawned >= self.stage_quota
        ):
            return
        self.stage_spawned += 1
        self.spawn_count += 1
        pool = (
            ["spirit", "archer"]
            if self.stage == 0
            else ["swarm", "swarm", "golem", "archer", "healer"]
        )
        kind = pool[(self.stage_spawned - 1) % len(pool)]
        if self.mode == "endless":
            kind = ["spirit", "archer", "swarm", "golem", "healer"][
                self.spawn_count % 5
            ]
        enemy = base.Enemy(
            self.rng.uniform(70, 800),
            -25,
            85 if kind == "swarm" else 42 if kind == "golem" else 54,
            phase=self.rng.uniform(0, math.tau),
            shooter=kind == "archer",
        )
        enemy.kind = kind
        enemy.hp = enemy.max_hp = (
            4 if kind == "golem" else 2 if kind in ("healer", "archer") else 1
        )
        enemy.exposed_until = 0
        enemy.heal_at = self.elapsed + 3
        self.enemies.append(enemy)
        if kind == "swarm":
            for offset in (-26, 26):
                small = base.Enemy(
                    max(45, min(825, enemy.x + offset)),
                    -35,
                    82,
                    radius=13,
                    phase=enemy.phase,
                )
                small.kind = "swarm"
                small.hp = small.max_hp = 1
                small.exposed_until = 0
                self.enemies.append(small)

    def enemy_target(self, enemy):
        if getattr(enemy, "kind", "") == "boss_crystal":
            return enemy.position
        if self.mode == "adventure" and self.stage == 2:
            return pygame.Vector2(435, 450)
        return self.player

    def enemy_contact(self, enemy):
        if getattr(enemy, "kind", "") == "boss_crystal":
            return
        if self.mode == "adventure" and self.stage == 2 and enemy.y >= 430:
            self.enemies.remove(enemy)
            self.tree_health = max(0, self.tree_health - 12)
            self.notify("Heart Tree hit! Push enemies away or clear them")
            if self.tree_health == 0:
                self.state = "gameover"
            return
        super().enemy_contact(enemy)

    def damage_enemy(self, enemy, damage=1, kind="Fireball"):
        if enemy not in self.enemies:
            return
        if (
            getattr(enemy, "kind", "") == "golem"
            and self.elapsed >= enemy.exposed_until
            and kind == "Fireball"
        ):
            self.notify("Armored! Point to expose the golem, then attack")
            return
        if enemy is self.marked:
            damage += 1
        enemy.hp = getattr(enemy, "hp", 1) - damage
        self.burst(enemy.position, base.COLORS.get(kind, base.COLORS["Fireball"]), 8)
        if enemy.hp <= 0:
            self.defeat(enemy, base.COLORS.get(kind, base.COLORS["Fireball"]))

    def defeat(self, enemy, color):
        if enemy not in self.enemies:
            return
        position = enemy.position.copy()
        super().defeat(enemy, color)
        if getattr(enemy, "kind", "") != "boss_crystal":
            self.mana_drops.append(ManaDrop(position))

    def hurt(self, damage):
        if self.state != "playing":
            return
        if (
            self.companion == "dragon"
            and self.elapsed >= self.dragon_ready
            and self.elapsed >= self.hurt_until
        ):
            self.dragon_ready = self.elapsed + 12
            self.burst(self.player, (255, 185, 93), 20)
            self.notify("Ember blocked a hit!")
            return
        super().hurt(damage)

    def cast(self, spell):
        start = len(self.projectiles)
        chain_before = list(self.enemies)
        held = self.held_object
        combo_before = self.combo_count
        crystal_positions = [
            (e, e.position.copy())
            for e in self.enemies
            if getattr(e, "kind", "") == "boss_crystal"
        ]
        if not super().cast(spell):
            self.feedback = self.message
            return False
        self.feedback = spell + " cast"
        if spell == "Fireball":
            shots = self.projectiles[start:]
            if held:
                del self.projectiles[start:]
                self.launch_object(held, charged=self.combo_count > combo_before)
            else:
                for shot in shots:
                    if shot.damage == 1:
                        shot.damage = 2
                    if "Twin Flame" in self.upgrades:
                        shot.damage = max(1, shot.damage / 2)
                        other = base.Projectile(
                            shot.position.copy(),
                            shot.velocity.rotate(9),
                            damage=shot.damage,
                            splash=shot.splash,
                        )
                        shot.velocity = shot.velocity.rotate(-9)
                        self.projectiles.append(other)
        elif spell == "Lightning" and "Thunder Chain" in self.upgrades:
            remaining = [
                e
                for e in chain_before
                if e in self.enemies and e not in self.lightning_visited
            ]
            for enemy in sorted(
                remaining, key=lambda e: e.position.distance_to(self.player)
            )[:2]:
                self.bolts.append([[self.player.copy(), enemy.position.copy()], 0.3])
                self.damage_enemy(enemy, 3, "Lightning")
        elif spell == "Telekinesis":
            for crystal, position in crystal_positions:
                crystal.x, crystal.y = position
            for enemy in self.enemies:
                if getattr(enemy, "kind", "") == "golem":
                    enemy.exposed_until = self.elapsed + 4
                if (
                    "Gravity Well" in self.upgrades
                    and getattr(enemy, "kind", "") != "boss_crystal"
                ):
                    enemy.x += (435 - enemy.x) * 0.75
                    enemy.y += (260 - enemy.y) * 0.5
            choices = [obj for obj in self.objects if obj.velocity is None]
            aim = (
                self.cursor
                if self.cursor_live and self.cursor is not None
                else self.player + pygame.Vector2(0, -130)
            )
            if choices and self.held_object is None:
                obj = min(choices, key=lambda item: item.position.distance_to(aim))
                if obj.position.distance_to(aim) <= 160:
                    self.held_object = obj
                    self.held_by_hand = self.cursor_live
                    self.move_target = None
                    self.notify("Object lifted! Move hand to drag; palm to throw")
        return True

    def reflect(self, orb):
        start = len(self.projectiles)
        super().reflect(orb)
        shot = self.projectiles[start]
        shot.kind = "Counter"
        if self.boss and self.boss_phase == "mirror":
            aim = self.boss.position
            for _ in range(3):
                travel = shot.position.distance_to(aim) / 550
                aim.x = (
                    base.ARENA_WIDTH / 2 + math.sin((self.elapsed + travel) * 0.7) * 235
                )
            direction = aim - shot.position
            if direction.length_squared():
                shot.velocity = direction.normalize() * 550
        if "Mirror Shield" in self.upgrades:
            for angle in (-12, 12):
                extra = base.Projectile(
                    shot.position.copy(),
                    shot.velocity.rotate(angle),
                    damage=shot.damage,
                )
                extra.kind = "Counter"
                self.projectiles.append(extra)

    def feedback_for(self, prediction):
        if prediction is None or prediction.gesture_id is None:
            return "No accepted pose. Move hand to steer."
        spell = base.SPELLS[prediction.gesture_id]
        remaining = max(
            0, base.COOLDOWNS[spell] - (self.elapsed - self.last_cast[spell])
        )
        if remaining:
            return f"{spell}: cooldown {remaining:.1f}s"
        if self.mana < base.COSTS[spell]:
            return f"{spell}: need {base.COSTS[spell]} mana"
        return (
            spell
            + ": ready / "
            + (
                "hold palm to fire"
                if spell == "Fireball"
                else "switch pose or lower hand to rearm"
            )
        )

    def damage_boss(self, damage, kind="Lightning"):
        if self.boss is None:
            return
        if self.boss_phase == "crystal" and any(
            getattr(e, "kind", "") == "boss_crystal" for e in self.enemies
        ):
            self.notify("Crystal armor: destroy the two glowing crystals first")
            return
        if self.boss_phase == "mirror" and self.elapsed >= self.boss_exposed_until:
            if kind != "Counter":
                self.notify("Mirror shield: reflect a pink shot to stun the guardian")
                return
            self.boss_exposed_until = self.elapsed + 4
            self.boss.charge = 0
            self.notify("Mirror shattered! Attack during the 4-second opening")
        previous = self.boss
        super().damage_boss(damage)
        if self.boss is None:
            self.safe_x = None
            self.profile["guardians"] += 1
            self.save_profile()
            self.finish_stage()
        elif previous.hp <= previous.max_hp / 2 and self.boss_phase != "storm":
            self.boss_phase = "storm"
            self.storm_at = self.elapsed + 3
            self.notify("Storm phase! Stand inside the green safe lane")

    def update_threats(self, dt):
        previous = self.boss
        if (
            self.boss
            and self.boss_phase == "mirror"
            and self.elapsed < self.boss_exposed_until
        ):
            self.boss.charge = 0
            self.boss.shot_timer = max(1, self.boss.shot_timer)
        # Cover and drawn rune wards intercept projectiles before player collision.
        for orb in self.hostile_orbs[:]:
            end = orb.position + orb.velocity * dt
            barriers = [
                o for o in self.objects if o.kind == "rock" and o.velocity is None
            ]
            blocked = any(
                base.SpellGame.segment_hit(
                    orb.position,
                    end,
                    base.Enemy(o.position.x, o.position.y, 0, radius=o.radius),
                    orb.radius,
                )
                for o in barriers
            )
            blocked = blocked or any(
                base.SpellGame.segment_hit(
                    orb.position,
                    end,
                    base.Enemy(pos.x, pos.y, 0, radius=70),
                    orb.radius,
                )
                for pos, until in self.runes
                if until > self.elapsed
            )
            if blocked:
                self.hostile_orbs.remove(orb)
                self.burst(orb.position, base.COLORS["Shield"], 6)
        super().update_threats(dt)
        if self.boss is not None and previous is None:
            self.boss.hp = self.boss.max_hp = 18 + 4 * self.bosses_defeated
            self.boss_phase = "crystal"
            self.safe_x = None
            self.boss_exposed_until = 0
            for x in (185, 685):
                crystal = base.Enemy(x, 235, 0, radius=22)
                crystal.kind = "boss_crystal"
                crystal.hp = crystal.max_hp = 3
                self.enemies.append(crystal)
            self.notify("Guardian / Crystal phase: break the two glowing crystals")
        if (
            self.boss
            and self.boss_phase == "crystal"
            and not any(getattr(e, "kind", "") == "boss_crystal" for e in self.enemies)
        ):
            self.boss_phase = "mirror"
            self.notify("Guardian / Mirror phase: counter a pink shot")
        if self.boss and self.boss_phase == "storm":
            if self.safe_x is None and self.elapsed >= self.storm_at:
                self.safe_x = self.rng.choice((150, 435, 720))
                self.storm_strike_at = self.elapsed + 2
                self.notify("Storm incoming! Move into the green safe lane")
            elif self.safe_x is not None and self.elapsed >= self.storm_strike_at:
                if abs(self.player.x - self.safe_x) > 80:
                    if self.elapsed >= self.shield_until:
                        self.hurt(18)
                else:
                    self.score += 10
                    self.mana = min(100, self.mana + 5)
                self.rings.append(
                    [
                        pygame.Vector2(self.safe_x, 530),
                        60,
                        0.3,
                        base.COLORS["Lightning"],
                    ]
                )
                self.safe_x = None
                self.storm_at = self.elapsed + 5

    def set_hand(self, x, y=None):
        if x is None or not math.isfinite(x):
            self.cursor_live = False
            self.move_target = None
            if self.held_object and self.held_by_hand:
                self.notify("Object placed. Show a point to lift it again")
                self.held_object = None
            return
        self.cursor_live = True
        if y is not None and not math.isfinite(y):
            y = None
        self.cursor = pygame.Vector2(
            max(60, min(810, x * base.ARENA_WIDTH)),
            max(185, min(450, (y if y is not None else 0.55) * 600)),
        )
        if self.held_object:
            self.held_by_hand = True
            self.move_target = None
        else:
            super().steer(x)

    def keyboard_move(self, axis, dt):
        if self.held_object and self.state == "playing":
            self.held_object.position.x = max(
                50, min(820, self.held_object.position.x + axis * 350 * min(dt, 0.1))
            )
            self.move_target = None
            self.cursor_live = False
        else:
            super().keyboard_move(axis, dt)

    def launch_object(self, obj, charged=False):
        obj.powered = charged
        targets = self.aim_targets(obj.position)
        direction = (
            (
                min(
                    targets, key=lambda e: e.position.distance_to(obj.position)
                ).position
                - obj.position
            )
            if targets
            else pygame.Vector2(0, -1)
        )
        obj.velocity = (
            direction.normalize() * 430
            if direction.length_squared()
            else pygame.Vector2(0, -430)
        )
        obj.life = 2.5
        self.held_object = None
        self.held_by_hand = False
        self.move_target = None
        self.notify(("Meteor " if charged else "") + obj.kind.title() + " thrown!")

    def update_objects(self, dt):
        if self.held_object and self.cursor_live and self.cursor:
            self.held_object.position = self.held_object.position.lerp(
                self.cursor, 1 - math.exp(-12 * dt)
            )
        for obj in self.objects[:]:
            if obj.velocity is None:
                continue
            old = obj.position.copy()
            obj.position += obj.velocity * dt
            obj.life -= dt
            targets = list(self.enemies) + ([self.boss] if self.boss else [])
            hit = next(
                (
                    e
                    for e in targets
                    if self.segment_hit(old, obj.position, e, obj.radius)
                ),
                None,
            )
            if hit:
                splash = (
                    150 if obj.kind == "seed" else 85 if obj.kind == "crystal" else 0
                )
                if getattr(obj, "powered", False):
                    splash = max(120, splash)
                point = hit.position.copy()
                affected = [
                    e
                    for e in targets
                    if e is hit
                    or (splash and e.position.distance_to(point) <= splash + e.radius)
                ]
                for enemy in affected:
                    if enemy is self.boss:
                        self.damage_boss(
                            5 if getattr(obj, "powered", False) else 3, kind="Object"
                        )
                    else:
                        self.damage_enemy(
                            enemy, 6 if getattr(obj, "powered", False) else 4, "Object"
                        )
                self.rings.append(
                    [
                        point,
                        20,
                        0.3,
                        (
                            base.COLORS["Fireball"]
                            if obj.kind == "seed"
                            else base.COLORS["Telekinesis"]
                        ),
                    ]
                )
            if (
                hit
                or obj.life <= 0
                or not (-60 <= obj.position.x <= 930 and -60 <= obj.position.y <= 655)
            ):
                self.objects.remove(obj)
                self.object_respawn_at = self.elapsed + 8
        if not self.objects and self.elapsed >= self.object_respawn_at:
            self.prepare_objects()

    def motion_cast(self, name):
        if not self.motion_enabled or self.state != "playing" or self.held_object:
            return False
        name = name.split(":")[0]
        cost = {"sweep": 15, "circle": 20, "charge": 10}.get(name)
        if cost is None:
            return False
        label = {
            "sweep": "Wind Slash",
            "circle": "Rune Ward",
            "charge": "Charged Ward",
        }[name]
        if self.elapsed - self.motion_last[name] < 3:
            self.notify(label + " is cooling down")
            return False
        if self.mana < cost:
            self.notify(f"{label}: need {cost} mana")
            return False
        self.motion_last[name] = self.elapsed
        self.mana -= cost
        if name == "sweep":
            for enemy in self.enemies[:]:
                if enemy.y > 200:
                    self.damage_enemy(enemy, 2, "Wind")
                    enemy.y -= 35
            self.hostile_orbs = [
                orb for orb in self.hostile_orbs if orb.position.y < 300
            ]
            self.rings.append([self.player.copy(), 30, 0.5, base.COLORS["Telekinesis"]])
            self.notify("Wind Slash! / -15 mana")
        elif name == "circle":
            self.runes.append([self.player + pygame.Vector2(0, -30), self.elapsed + 3])
            self.notify("Rune Ward drawn! / 3-second safe bubble")
        else:
            self.shield_until = max(self.shield_until, self.elapsed + 2)
            self.perfect_until = self.elapsed + 0.65
            self.hostile_orbs = [
                orb
                for orb in self.hostile_orbs
                if orb.position.distance_to(self.player) > 140
            ]
            self.notify("Charged Ward! Close shots dispelled")
        return True

    def update(self, dt):
        if self.state != "playing":
            return
        dt = max(0, min(dt, 0.1))
        old_x = self.player.x
        self.marked = (
            next((e for e in self.enemies if getattr(e, "kind", "") == "healer"), None)
            if self.companion == "spirit"
            else None
        )
        if self.companion == "spirit" and self.marked is None:
            self.marked = next((e for e in self.enemies if e.shooter), None)
        super().update(dt)
        if self.state != "playing":
            return
        if "Mana Spring" in self.upgrades and abs(self.player.x - old_x) < 0.5:
            self.mana = min(100, self.mana + 5 * dt)
        self.update_objects(dt)
        if self.state != "playing":
            return
        self.runes = [r for r in self.runes if r[1] > self.elapsed]
        for enemy in self.enemies:
            if getattr(enemy, "kind", "") == "healer" and self.elapsed >= enemy.heal_at:
                enemy.heal_at = self.elapsed + 3
                for other in self.enemies:
                    if (
                        other is not enemy
                        and other.position.distance_to(enemy.position) < 170
                    ):
                        other.hp = min(other.max_hp, other.hp + 1)
                self.rings.append([enemy.position.copy(), 25, 0.25, (118, 220, 136)])
        for drop in self.mana_drops[:]:
            drop.life -= dt
            if self.companion == "owl":
                delta = self.player + pygame.Vector2(0, -30) - drop.position
                drop.position += delta * (1 - math.exp(-3 * dt))
            else:
                drop.position.y += 100 * dt
            if drop.position.distance_to(self.player + pygame.Vector2(0, -30)) < 45:
                self.mana = min(100, self.mana + 4)
                self.mana_drops.remove(drop)
            elif drop.life <= 0 or drop.position.y > 640:
                self.mana_drops.remove(drop)
        if self.mode == "adventure":
            if (
                self.stage < 2
                and self.stage_spawned >= self.stage_quota
                and not self.enemies
            ):
                self.finish_stage()
            elif (
                self.stage == 2
                and self.elapsed - self.stage_started >= 35
                and self.tree_health > 0
            ):
                self.finish_stage()
            elif self.stage == 3:
                for portal in self.portals:
                    if (
                        abs(self.player.x - portal.position.x) < 65
                        and self.elapsed < self.shield_until
                    ):
                        portal.progress = min(1, portal.progress + dt / 0.9)
                if all(p.progress >= 1 for p in self.portals):
                    self.finish_stage()
        self.last_player_x = self.player.x


class AdventureView(base.GameView):
    def __init__(self, screen):
        super().__init__(screen)
        self.skins = {"mint": self.wizard_sprite.copy()}
        for name, tint in SKINS.items():
            if name == "mint":
                continue
            sprite = self.wizard_sprite.copy()
            for x in range(sprite.get_width()):
                for y in range(52, sprite.get_height()):
                    c = sprite.get_at((x, y))
                    if c.a and c.g > c.r * 1.35 and c.b > c.r * 1.3:
                        strength = c.g / 160
                        sprite.set_at(
                            (x, y), (*[min(c.a, int(v * strength)) for v in tint], c.a)
                        )
            self.skins[name] = sprite
        self.current_game = None

    def camera(self, snapshot, no_camera):
        pred, stamp, events, error, status = snapshot
        if error or pred is None or not 0 <= time.monotonic() - stamp <= 0.7:
            self.camera_surface = None
        return super().camera(snapshot, no_camera)

    def draw_arena(self, game):
        self.wizard_sprite = self.skins[game.skin]
        super().draw_arena(game)
        old = self.screen.get_clip()
        self.screen.set_clip((0, 86, 870, 514))
        if game.mode == "adventure" and game.stage == 2:
            pos = (435, 440)
            pygame.draw.rect(
                self.screen, (124, 84, 60), (425, 417, 20, 53), border_radius=5
            )
            for x, y, r in ((413, 414, 26), (440, 392, 31), (459, 416, 25)):
                pygame.draw.circle(self.screen, (70, 160, 113), (x, y), r)
                pygame.draw.circle(self.screen, (115, 200, 136), (x - 5, y - 5), r - 9)
            self.text("HEART TREE", 435, 477, 11, (255, 251, 225), True)
            pygame.draw.rect(
                self.screen, (65, 81, 67), (385, 488, 100, 6), border_radius=3
            )
            pygame.draw.rect(
                self.screen,
                (128, 232, 151),
                (385, 488, max(1, game.tree_health), 6),
                border_radius=3,
            )
        for portal in game.portals:
            pos = portal.position
            color = (92, 199, 152) if portal.progress >= 1 else (171, 113, 231)
            self.glow(pos, color, 50)
            pygame.draw.ellipse(self.screen, color, (pos.x - 29, pos.y - 42, 58, 84), 4)
            pygame.draw.ellipse(
                self.screen, (247, 227, 165), (pos.x - 23, pos.y - 35, 46, 70), 2
            )
            self.text(
                "SEALED" if portal.progress >= 1 else "FIST TO SEAL",
                pos.x,
                pos.y + 53,
                11,
                (37, 61, 57),
                True,
            )
            pygame.draw.rect(
                self.screen,
                (89, 72, 114),
                (pos.x - 26, pos.y + 67, 52, 5),
                border_radius=2,
            )
            if portal.progress:
                pygame.draw.rect(
                    self.screen,
                    (110, 211, 163),
                    (pos.x - 26, pos.y + 67, 52 * portal.progress, 5),
                    border_radius=2,
                )
        if game.boss:
            boss = game.boss
            if (
                game.boss_phase in ("crystal", "mirror")
                and game.elapsed >= game.boss_exposed_until
            ):
                bubble = pygame.Surface((150, 160), pygame.SRCALPHA)
                color = (
                    (181, 136, 240) if game.boss_phase == "crystal" else (138, 225, 241)
                )
                pygame.draw.ellipse(bubble, (*color, 35), (4, 4, 142, 152))
                pygame.draw.ellipse(bubble, (*color, 210), (4, 4, 142, 152), 3)
                self.screen.blit(bubble, (boss.x - 75, boss.y - 80))
            elif game.boss_phase == "mirror":
                self.text(
                    "STUNNED / ATTACK NOW", boss.x, boss.y + 82, 11, (62, 82, 67), True
                )
        for enemy in game.enemies:
            kind = getattr(enemy, "kind", "spirit")
            x, y = enemy.x, enemy.y
            if kind == "golem":
                armor = (
                    (233, 199, 117)
                    if game.elapsed < getattr(enemy, "exposed_until", 0)
                    else (117, 139, 157)
                )
                pygame.draw.polygon(
                    self.screen,
                    armor,
                    [
                        (x - 22, y - 8),
                        (x, y - 20),
                        (x + 22, y - 8),
                        (x + 18, y + 22),
                        (x, y + 33),
                        (x - 18, y + 22),
                    ],
                )
                pygame.draw.polygon(
                    self.screen,
                    (237, 233, 208),
                    [(x - 9, y - 5), (x, y - 12), (x + 9, y - 5), (x, y + 18)],
                    2,
                )
            elif kind == "healer":
                pygame.draw.circle(self.screen, (91, 173, 107), (x, y - 39), 11)
                pygame.draw.line(
                    self.screen, (245, 252, 219), (x - 6, y - 39), (x + 6, y - 39), 3
                )
                pygame.draw.line(
                    self.screen, (245, 252, 219), (x, y - 45), (x, y - 33), 3
                )
            elif kind == "swarm":
                pygame.draw.circle(self.screen, (250, 236, 181), (x - 30, y), 4)
                pygame.draw.circle(self.screen, (250, 236, 181), (x + 30, y), 4)
            elif kind == "boss_crystal":
                pygame.draw.polygon(
                    self.screen,
                    (172, 115, 235),
                    [(x, y - 34), (x + 22, y), (x, y + 34), (x - 22, y)],
                )
                pygame.draw.polygon(
                    self.screen,
                    (241, 211, 255),
                    [(x, y - 28), (x + 9, y - 2), (x, y + 25), (x - 7, y)],
                    2,
                )
            if getattr(enemy, "hp", 1) > 1:
                self.text(str(math.ceil(enemy.hp)), x, y + 40, 11, (35, 57, 61), True)
            if enemy is game.marked:
                pygame.draw.polygon(
                    self.screen,
                    (254, 219, 114),
                    [(x, y - 62), (x + 7, y - 54), (x, y - 46), (x - 7, y - 54)],
                )
        for obj in game.objects:
            x, y = obj.position
            if obj is game.held_object:
                self.glow(obj.position, (113, 227, 202), 33)
                pygame.draw.circle(self.screen, (185, 255, 229), obj.position, 29, 2)
            if obj.kind == "rock":
                pygame.draw.polygon(
                    self.screen,
                    (92, 121, 125),
                    [
                        (x - 22, y + 15),
                        (x - 25, y - 4),
                        (x - 8, y - 20),
                        (x + 15, y - 13),
                        (x + 24, y + 13),
                    ],
                )
                pygame.draw.lines(
                    self.screen,
                    (177, 206, 190),
                    False,
                    [(x - 18, y - 3), (x - 7, y - 13), (x + 12, y - 8)],
                    3,
                )
            elif obj.kind == "seed":
                pygame.draw.circle(self.screen, (226, 136, 78), obj.position, 18)
                pygame.draw.circle(self.screen, (255, 211, 109), (x - 5, y - 5), 7)
                pygame.draw.line(
                    self.screen, (83, 136, 83), (x, y - 17), (x + 7, y - 27), 4
                )
            else:
                pygame.draw.polygon(
                    self.screen,
                    (72, 192, 180),
                    [(x, y - 23), (x + 16, y), (x, y + 23), (x - 16, y)],
                )
                pygame.draw.line(
                    self.screen, (215, 255, 220), (x - 4, y - 14), (x + 5, y + 10), 3
                )
        if game.cursor_live and game.cursor and game.held_object is None:
            pygame.draw.circle(self.screen, (244, 239, 190), game.cursor, 11, 2)
            pygame.draw.line(
                self.screen,
                (41, 118, 100),
                game.cursor + (-15, 0),
                game.cursor + (15, 0),
                1,
            )
        for drop in game.mana_drops:
            pygame.draw.circle(self.screen, (55, 177, 191), drop.position, 6)
            pygame.draw.circle(self.screen, (204, 255, 223), drop.position, 3)
        for pos, until in game.runes:
            pygame.draw.circle(self.screen, (116, 232, 215), pos, 70, 3)
            for a in range(6):
                angle = a * math.tau / 6
                pygame.draw.circle(
                    self.screen,
                    (253, 224, 145),
                    pos + pygame.Vector2(math.cos(angle), math.sin(angle)) * 70,
                    4,
                )
        if game.safe_x is not None:
            zone = pygame.Surface((160, 155), pygame.SRCALPHA)
            zone.fill((103, 220, 154, 90))
            self.screen.blit(zone, (game.safe_x - 80, 438))
            pygame.draw.rect(
                self.screen,
                (116, 244, 171),
                (game.safe_x - 80, 438, 160, 155),
                3,
                border_radius=10,
            )
            self.text(
                f"SAFE / {max(0,game.storm_strike_at-game.elapsed):.1f}s",
                game.safe_x,
                457,
                13,
                (31, 81, 63),
                True,
            )
        self.companion(game)
        self.screen.set_clip(old)

    def companion(self, game):
        x, y = game.player.x + 66, game.player.y - 38 + math.sin(game.elapsed * 3) * 5
        if x > 820:
            x = game.player.x - 66
        if game.companion == "owl":
            pygame.draw.ellipse(self.screen, (123, 95, 82), (x - 15, y - 17, 30, 36))
            pygame.draw.ellipse(self.screen, (238, 218, 175), (x - 11, y - 12, 22, 25))
            for offset in (-6, 6):
                pygame.draw.circle(self.screen, (255, 249, 220), (x + offset, y - 4), 6)
                pygame.draw.circle(self.screen, (47, 62, 64), (x + offset, y - 4), 3)
            pygame.draw.polygon(
                self.screen,
                (222, 165, 65),
                [(x - 3, y + 2), (x + 3, y + 2), (x, y + 8)],
            )
        elif game.companion == "dragon":
            color = (
                (231, 147, 77) if game.elapsed >= game.dragon_ready else (148, 134, 117)
            )
            pygame.draw.polygon(
                self.screen,
                color,
                [(x - 9, y + 4), (x - 29, y - 12), (x - 25, y + 13), (x - 9, y + 19)],
            )
            pygame.draw.polygon(
                self.screen,
                color,
                [(x + 9, y + 4), (x + 29, y - 12), (x + 25, y + 13), (x + 9, y + 19)],
            )
            pygame.draw.ellipse(self.screen, color, (x - 11, y - 15, 22, 36))
            pygame.draw.circle(self.screen, (255, 226, 129), (x + 4, y - 5), 4)
        else:
            self.glow((x, y), (250, 222, 142), 24)
            pygame.draw.circle(self.screen, (251, 232, 164), (x, y), 13)
            for offset in (-4, 4):
                pygame.draw.circle(self.screen, (67, 111, 103), (x + offset, y), 2)

    def gesture_guide(self, game, snapshot):
        super().gesture_guide(game, snapshot)
        pygame.draw.rect(self.screen, self.cream, (892, 689, 366, 31))
        if game.motion_enabled:
            self.text(
                "Motion: palm sweep / peace circle / hold fist",
                892,
                692,
                11,
                self.quiet,
            )
            self.text(
                "15 / 20 / 10 mana. Keys Z / X / V also work.", 892, 708, 11, self.quiet
            )
        else:
            self.text(
                "Point near a prop; palm throws. P shows help.",
                892,
                697,
                11,
                self.quiet,
            )

    def overlay(self, game):
        self.buttons = {k: v for k, v in self.buttons.items() if k == "mute"}
        veil = pygame.Surface((870, 720), pygame.SRCALPHA)
        veil.fill((24, 67, 62, 110))
        self.screen.blit(veil, (0, 0))
        self.panel((105, 110, 660, 490), (253, 248, 229), (224, 200, 139), 25)
        if game.state == "start":
            self.text("Arcane Adventure", 435, 156, 42, self.dark, True, True)
            self.text(
                "Choose your companion. Build your magic. Save the grove.",
                435,
                210,
                16,
                self.quiet,
                True,
            )
            for i, mode in enumerate(("adventure", "endless")):
                self.button(
                    "mode_" + mode,
                    mode.title(),
                    (149 + i * 290, 248, 242, 38),
                    (45, 122, 103) if game.mode == mode else (127, 149, 112),
                    14,
                )
            for i, key in enumerate(COMPANIONS):
                self.button(
                    "pet_" + key,
                    COMPANIONS[key][0],
                    (137 + i * 200, 316, 190, 40),
                    (45, 122, 103) if game.companion == key else (127, 149, 112),
                    13,
                )
            self.text(COMPANIONS[game.companion][1], 435, 378, 14, self.quiet, True)
            for i, name in enumerate(SKINS):
                label = (
                    name.title()
                    if name in game.unlocked_skins()
                    else name.title() + " locked"
                )
                self.button(
                    "skin_" + name,
                    label,
                    (239 + i * 132, 406, 124, 34),
                    SKINS[name],
                    12,
                )
            self.button(
                "motion",
                "Motion spells: " + ("ON" if game.motion_enabled else "OFF") + "  /  O",
                (239, 455, 392, 34),
                (93, 132, 110),
                13,
            )
            self.button(
                "primary",
                "Begin adventure  /  SPACE",
                (239, 518, 392, 52),
                (45, 122, 103),
                16,
            )
            self.text(
                "Gold robe: win a run. Rose robe: defeat 3 guardians.",
                435,
                586,
                11,
                self.quiet,
                True,
            )
        elif game.state == "upgrade":
            self.text("Choose your next magic", 435, 166, 34, self.dark, True, True)
            self.text(
                (
                    (STAGES[game.stage] + " complete! Combat is paused.")
                    if game.mode == "adventure"
                    else "Guardian defeated! Combat is paused."
                ),
                435,
                217,
                16,
                self.quiet,
                True,
            )
            for i, name in enumerate(game.upgrade_options):
                rect = pygame.Rect(149 + i * 194, 265, 180, 168)
                self.button(
                    "upgrade_" + str(i),
                    "",
                    rect,
                    (226, 236, 209) if i == game.upgrade_index else (242, 235, 213),
                    14,
                )
                if i == game.upgrade_index:
                    pygame.draw.rect(
                        self.screen, (204, 157, 68), rect, 3, border_radius=12
                    )
                self.screen.blit(
                    self.icons[UPGRADES[name][1]], (rect.centerx - 20, rect.y + 15)
                )
                self.text(name, rect.centerx, rect.y + 68, 16, self.dark, True)
                self.wrapped(
                    UPGRADES[name][0], rect.x + 12, rect.y + 98, 156, 12, max_lines=3
                )
                self.text(str(i + 1), rect.x + 12, rect.y + 12, 12, self.quiet)
            self.text(
                "Hand left/right selects. Hold steady, then make a fist.",
                435,
                468,
                13,
                self.quiet,
                True,
            )
            self.button(
                "primary",
                "Learn selected magic  /  SPACE",
                (239, 518, 392, 52),
                (45, 122, 103),
                15,
            )
        elif game.state == "paused":
            self.text("Rest your hands", 435, 195, 42, self.dark, True, True)
            self.text(
                "Combat, objectives and motion timers are paused.",
                435,
                262,
                16,
                self.quiet,
                True,
            )
            self.wrapped(
                "Point near a rock, seed or crystal to lift it. Move your hand to drag it, then open your palm to throw. The wizard stays still while you drag.",
                177,
                316,
                520,
                16,
                max_lines=4,
            )
            self.text(
                "Optional: palm sweep / peace circle / fist held 1.3s",
                435,
                424,
                13,
                self.quiet,
                True,
            )
            self.button(
                "motion",
                "Motion spells: " + ("ON" if game.motion_enabled else "OFF") + "  /  O",
                (239, 455, 392, 34),
                (93, 132, 110),
                13,
            )
            self.button(
                "primary",
                "Back to the grove  /  P",
                (239, 518, 392, 52),
                (45, 122, 103),
                16,
            )
        else:
            win = game.state == "victory"
            self.text(
                "The grove is safe!" if win else "Another adventure awaits",
                435,
                190,
                34,
                self.dark,
                True,
                True,
            )
            self.text(
                (
                    "You defeated the guardian and completed the route."
                    if win
                    else (
                        "The Heart Tree fell. Hold enemies back with telekinesis."
                        if game.tree_health == 0
                        else "Try a different upgrade build or companion."
                    )
                ),
                435,
                250,
                15,
                self.quiet,
                True,
            )
            for x, label, value in (
                (260, "SCORE", game.score),
                (435, "GUARDIANS", game.bosses_defeated),
                (610, "BEST", max(game.best_score, game.profile["best"])),
            ):
                self.text(str(value), x, 326, 34, self.dark, True, True)
                self.text(label, x, 365, 11, self.quiet, True)
            self.text(
                f"{game.dodges} dodges / {game.parries} counters / {game.combo_count} combos",
                435,
                410,
                14,
                self.quiet,
                True,
            )
            self.text(
                "Learned: " + (", ".join(sorted(game.upgrades)) or "No upgrades yet"),
                435,
                445,
                11,
                self.quiet,
                True,
            )
            if win:
                self.text(
                    "Gold robe unlocked! Return to camp to equip it.",
                    435,
                    480,
                    13,
                    (147, 111, 46),
                    True,
                )
            self.button(
                "primary", "Play again  /  R", (169, 518, 255, 52), (45, 122, 103), 15
            )
            self.button(
                "camp", "Return to camp", (446, 518, 255, 52), (103, 130, 110), 15
            )

    def draw(self, game, snapshot, fps, no_camera, muted):
        self.current_game = game
        # Base HUD keeps familiar spell cards and the camera preview.
        super().draw(game, snapshot, fps, no_camera, muted)
        if game.state == "playing":
            self.panel((650, 144, 202, 55), (253, 244, 211), (220, 198, 147), 10)
            self.text(
                (
                    "ENDLESS GROVE"
                    if game.mode == "endless"
                    else f"{game.stage+1}/5  {STAGES[game.stage]}"
                ),
                660,
                155,
                11,
                self.dark,
            )
            elapsed = game.elapsed - game.stage_started
            objective = (
                f"Clear spirits: {game.stage_spawned}/{game.stage_quota} spawned"
                if game.stage < 2
                else (
                    f"Protect tree: {max(0,math.ceil(35-elapsed))}s / HP {game.tree_health}"
                    if game.stage == 2
                    else (
                        f"Sealed: {sum(p.progress>=1 for p in game.portals)}/3 / fist near a portal"
                        if game.stage == 3
                        else "Phase: " + game.boss_phase.title()
                    )
                )
            )
            if game.mode == "endless":
                objective = f"Guardians defeated: {game.bosses_defeated}"
            self.wrapped(objective, 660, 173, 182, 11, max_lines=2)
            # Replace the generic ready message with the actual spell gate.
            self.panel((892, 346, 366, 86), (239, 241, 220), (218, 224, 201), 12)
            pred, stamp, _, error, status = snapshot
            fresh = (
                pred is not None and 0 <= time.monotonic() - stamp <= 0.7 and not error
            )
            self.text(
                (
                    "Keyboard magic"
                    if no_camera
                    else (
                        base.SPELLS[pred.gesture_id]
                        if fresh and pred.gesture_id is not None
                        else "Show your hand"
                    )
                ),
                907,
                355,
                18,
                self.dark,
            )
            if fresh and pred.gesture_id is not None:
                self.text(f"{pred.confidence:.0%}", 1210, 356, 14, self.quiet)
            message = (
                game.feedback_for(pred)
                if fresh
                else (
                    "Camera unavailable: use keys 1-5"
                    if error
                    else (
                        "Keys 1-5 cast. Point (4) lifts; palm (1) throws."
                        if no_camera
                        else "Show your hand / keys 1-5 also cast"
                    )
                )
            )
            self.wrapped(message, 907, 386, 335, 12, max_lines=2)
            self.panel((18, 600, 834, 24), (25, 67, 64), (83, 122, 99), 8)
            self.text(
                (
                    "DRAG OBJECT / palm throws / lower hand places it"
                    if game.held_object
                    else f"{COMPANIONS[game.companion][0]} / "
                    + (
                        "Motion spells ON"
                        if game.motion_enabled
                        else "Move hand or A/D to steer"
                    )
                ),
                30,
                612,
                11,
                (255, 246, 215),
            )
            self.text(
                f"DODGES {game.dodges} / COUNTERS {game.parries}",
                610,
                612,
                11,
                (226, 244, 223),
            )
        else:
            self.panel((892, 346, 366, 86), (239, 241, 220), (218, 224, 201), 12)
            self.text(
                (
                    "Combat paused"
                    if game.state in ("paused", "upgrade", "start")
                    else (
                        "Adventure complete"
                        if game.state == "victory"
                        else "Try another build"
                    )
                ),
                907,
                355,
                18,
                self.dark,
            )
            note = (
                "Hand position selects a card; release pose, then fist confirms."
                if game.state == "upgrade"
                else (
                    "Choose your settings, then press Space."
                    if game.state == "start"
                    else (
                        "P resumes. Gestures cannot cast while paused."
                        if game.state == "paused"
                        else "R plays again. Camp lets you change companions."
                    )
                )
            )
            self.wrapped(note, 907, 386, 335, 12, max_lines=2)
        if game.state in ("upgrade", "victory"):
            # These states are new to the base view; draw their blocking overlay here.
            self.overlay(game)


def apply_input(game, snapshot, now=None):
    now = time.monotonic() if now is None else now
    pred, stamp, events, error, status = snapshot
    fresh = pred is not None and error is None and 0 <= now - stamp <= 0.7
    if game.state == "upgrade":
        game.move_target = None
        if not fresh:
            return
        x = getattr(pred, "hand_x", None)
        if pred.gesture_id != 1:
            game.menu_confirm_ready = True
        if x is None:
            return
        index = max(
            0,
            min(
                len(game.upgrade_options) - 1,
                int(max(0, min(0.999, (x - 0.15) / 0.7)) * len(game.upgrade_options)),
            ),
        )
        if game.menu_hover_since is None or index != game.upgrade_index:
            game.upgrade_index = index
            game.menu_hover_since = now
        if (
            game.menu_confirm_ready
            and pred.gesture_id == 1
            and pred.confidence >= 0.8
            and now - game.menu_hover_since >= 0.6
        ):
            game.choose_upgrade(index)
        return
    if game.state != "playing":
        game.move_target = None
        return
    if not fresh:
        game.set_hand(None)
        return
    game.set_hand(getattr(pred, "hand_x", None), getattr(pred, "hand_y", None))
    for spell, observed, generation in events:
        if not 0 <= now - observed <= 0.5:
            continue
        if spell.startswith("motion:"):
            game.motion_cast(spell[7:])
        else:
            game.cast(spell)
        if game.state != "playing":
            return
    if (
        pred.gesture_id == 0
        and pred.confidence > 0.8
        and game.elapsed - game.last_cast["Fireball"] >= base.COOLDOWNS["Fireball"]
        and game.mana >= base.COSTS["Fireball"]
    ):
        game.cast("Fireball")


def handle_button(game, key):
    if key == "primary":
        if game.state == "upgrade":
            game.choose_upgrade(game.upgrade_index)
        elif game.state == "paused":
            game.state = "playing"
        elif game.state in ("start", "gameover", "victory"):
            game.reset()
    elif key == "camp" and game.state in ("gameover", "victory"):
        game.state = "start"
    elif key == "pause" and game.state in ("playing", "paused"):
        game.state = "paused" if game.state == "playing" else "playing"
    elif key == "motion" and game.state in ("start", "paused"):
        game.motion_enabled = not game.motion_enabled
    elif key.startswith("upgrade_") and game.state == "upgrade":
        game.choose_upgrade(int(key[8:]))
    elif key.startswith("mode_") and game.state == "start":
        game.mode = key[5:]
    elif key.startswith("pet_") and game.state == "start":
        game.companion = key[4:]
    elif key.startswith("skin_") and game.state == "start":
        skin = key[5:]
        if skin in game.unlocked_skins():
            game.skin = skin
            game.save_profile()
    elif key.startswith("cast_"):
        game.cast(key[5:])


def run_adventure(args):
    pygame.mixer.pre_init(44100, -16, 2, 512)
    pygame.init()
    worker = None
    game = None
    frames = 0
    seconds = 0
    try:
        screen = pygame.display.set_mode((base.WIDTH, base.HEIGHT))
        pygame.display.set_caption("Arcane Adventure - The Enchanted Grove")
        sound = base.SoundBank(args.mute)
        game = AdventureGame(
            sound,
            args.seed,
            mode=getattr(args, "mode", "adventure"),
            companion=getattr(args, "companion", "owl"),
            motion=getattr(args, "motion", False),
            practice=args.boss_practice,
            profile_path=base.ROOT / "reports" / "adventure_profile.json",
        )
        view = AdventureView(screen)
        clock = pygame.time.Clock()
        if not args.no_camera:
            worker = base.CameraWorker(args.camera, motion=True)
            worker.start()
        running = True
        while running:
            dt = min(0.1, clock.tick(30) / 1000)
            seconds += dt
            state_before = game.state
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    key = event.key
                    if key in (pygame.K_ESCAPE, pygame.K_q):
                        running = False
                    elif key == pygame.K_SPACE and game.state in ("start", "upgrade"):
                        handle_button(game, "primary")
                    elif key == pygame.K_r and game.state in ("gameover", "victory"):
                        handle_button(game, "primary")
                    elif key == pygame.K_p:
                        handle_button(game, "pause")
                    elif key == pygame.K_o:
                        handle_button(game, "motion")
                    elif key == pygame.K_m:
                        sound.muted = not sound.muted
                    elif game.state == "upgrade":
                        if pygame.K_1 <= key <= pygame.K_3:
                            game.choose_upgrade(key - pygame.K_1)
                        elif key in (pygame.K_LEFT, pygame.K_RIGHT):
                            game.upgrade_index = (
                                game.upgrade_index
                                + (1 if key == pygame.K_RIGHT else -1)
                            ) % len(game.upgrade_options)
                    elif game.state == "playing":
                        if pygame.K_1 <= key <= pygame.K_5:
                            game.cast(base.SPELLS[key - pygame.K_1])
                        elif key in (pygame.K_z, pygame.K_x, pygame.K_v):
                            game.motion_cast(
                                {
                                    pygame.K_z: "sweep",
                                    pygame.K_x: "circle",
                                    pygame.K_v: "charge",
                                }[key]
                            )
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    for key, rect in list(view.buttons.items()):
                        if rect.collidepoint(event.pos):
                            if key == "mute":
                                sound.muted = not sound.muted
                            else:
                                handle_button(game, key)
                            break
            if worker and game.state != state_before:
                worker.reset()
            snapshot = (
                worker.snapshot()
                if worker
                else (None, 0, [], None, "Keyboard-only mode")
            )
            before = game.state
            if worker:
                apply_input(game, snapshot)
            keys = pygame.key.get_pressed()
            axis = int(keys[pygame.K_RIGHT] or keys[pygame.K_d]) - int(
                keys[pygame.K_LEFT] or keys[pygame.K_a]
            )
            game.keyboard_move(axis, dt)
            if game.held_object and game.state == "playing":
                vertical = int(keys[pygame.K_DOWN]) - int(keys[pygame.K_UP])
                game.held_object.position.y = max(
                    180, min(450, game.held_object.position.y + vertical * 300 * dt)
                )
            game.update(dt)
            if game.state != before:
                if worker:
                    worker.reset()
                if game.state in ("gameover", "victory"):
                    game.save_profile()
            view.draw(game, snapshot, clock.get_fps(), args.no_camera, sound.muted)
            pygame.display.flip()
            frames += 1
            if args.max_frames and frames >= args.max_frames:
                running = False
    finally:
        if worker:
            worker.close()
        if game:
            game.save_profile()
            try:
                report = {
                    "mode": game.mode,
                    "state": game.state,
                    "stage": game.stage + 1,
                    "score": game.score,
                    "upgrades": sorted(game.upgrades),
                    "companion": game.companion,
                    "motion_enabled": game.motion_enabled,
                    "bosses_defeated": game.bosses_defeated,
                    "perfect_counters": game.parries,
                    "combos": game.combo_count,
                    "survival_seconds": round(game.elapsed, 2),
                    "rendered_frames": frames,
                    "average_game_fps": round(frames / seconds, 1) if seconds else 0,
                    "camera_error": worker.error if worker else None,
                }
                (base.ROOT / "reports" / "game_last_run.json").write_text(
                    json.dumps(report, indent=2) + "\n"
                )
            except OSError as exc:
                print("Could not save run report:", exc)
        pygame.quit()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--no-camera", action="store_true")
    parser.add_argument("--mute", action="store_true")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--mode", choices=("adventure", "endless"), default="adventure")
    parser.add_argument(
        "--companion", choices=("owl", "dragon", "spirit"), default="owl"
    )
    parser.add_argument(
        "--motion",
        action="store_true",
        help="Enable optional sweep, circle and charged ward controls",
    )
    parser.add_argument(
        "--boss-practice",
        action="store_true",
        help="First boss appears after 5 seconds",
    )
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


def main():
    args = parse_args()
    try:
        run_adventure(args)
    except KeyboardInterrupt:
        print("\nGame stopped.")
    except Exception as exc:
        print(f"Game failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
