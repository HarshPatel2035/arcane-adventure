from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pygame
from arcane_adventure import AdventureGame
from gesture_game_adventure import COOLDOWNS

# Play through using only game actions; refill health for a stress run, not a difficulty verdict.
a = AdventureGame(seed=12, companion="owl")
a.reset()
seen = set()
phases = set()
statechanges = []
for frame in range(18000):
    seen.add(a.stage)
    if a.boss:
        phases.add(a.boss_phase)
    if a.state == "upgrade":
        statechanges.append((a.stage, a.elapsed))
        order = [
            "Thunder Chain",
            "Mirror Shield",
            "Mana Spring",
            "Gravity Well",
            "Twin Flame",
            "Vital Bloom",
        ]
        pick = min(
            range(len(a.upgrade_options)),
            key=lambda i: order.index(a.upgrade_options[i]),
        )
        a.choose_upgrade(pick)
    if a.state == "victory":
        break
    assert a.state != "gameover", (a.stage, a.elapsed, a.tree_health)
    a.health = 100
    if a.stage == 3:
        portal = next((p for p in a.portals if p.progress < 1), None)
        if portal:
            a.player.x = portal.position.x
            if a.elapsed - a.last_cast["Shield"] >= 2 and a.mana >= 20:
                a.cast("Shield")
    elif a.boss and a.boss_phase == "mirror" and a.elapsed >= a.boss_exposed_until:
        # Time a shield to the approaching shots; the actual reflected projectile must hit.
        close = any(
            orb.velocity.y > 0
            and 0 < (a.player.y - 30) - orb.position.y < 115
            and abs(orb.position.x - a.player.x) < 70
            for orb in a.hostile_orbs
        )
        if close and a.elapsed - a.last_cast["Shield"] >= 2 and a.mana >= 20:
            a.cast("Shield")
    elif a.elapsed - a.last_cast["Shield"] >= 4 and a.mana > 60:
        a.cast("Shield")
    if (
        a.elapsed - a.last_cast["Lightning"] >= 1.3
        and a.mana >= 30
        and (
            len(a.enemies) >= 2
            or (
                a.boss and (a.boss_phase == "storm" or a.elapsed < a.boss_exposed_until)
            )
        )
    ):
        a.cast("Lightning")
    if a.elapsed - a.last_cast["Fireball"] >= 0.46 and a.mana >= 10:
        a.cast("Fireball")
    a.update(1 / 30)
    assert len(a.hostile_orbs) < 120 and len(a.particles) <= 650 and 0 <= a.mana <= 100
assert a.state == "victory", (
    a.stage,
    a.elapsed,
    a.boss_phase,
    a.boss.hp if a.boss else None,
    a.parries,
)
assert seen == set(range(5)) and phases == {"crystal", "mirror", "storm"}
assert len(statechanges) == 4 and a.bosses_defeated == 1
print(
    "Passed action-driven full route:",
    round(a.elapsed, 1),
    "seconds; stages",
    sorted(seen),
    "boss phases",
    sorted(phases),
    "counters",
    a.parries,
    "upgrades",
    sorted(a.upgrades),
)
