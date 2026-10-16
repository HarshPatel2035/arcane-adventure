from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pygame
import gesture_game_adventure as g

pygame.init()
screen = pygame.display.set_mode((g.WIDTH, g.HEIGHT))


def fresh():
    game = g.SpellGame(seed=5)
    game.reset()
    game.spawn_timer = 999
    return game


def incoming(game):
    return g.HostileOrb(game.player + pygame.Vector2(0, -50), pygame.Vector2(0, 220))


# Timed counters reflect while late shields block; uncovered hits have grace.
a = fresh()
a.cast("Shield")
a.hostile_orbs = [incoming(a)]
a.update(0.1)
assert (
    a.parries == 1
    and a.health == 100
    and a.score == 15
    and a.projectiles[0].damage == 2
)
b = fresh()
b.cast("Shield")
b.elapsed = 1
b.hostile_orbs = [incoming(b)]
b.update(0.1)
assert b.parries == 0 and b.health == 100 and not b.hostile_orbs
c = fresh()
c.hostile_orbs = [incoming(c), incoming(c)]
c.update(0.1)
assert c.health == 88
c.elapsed = 1
c.hostile_orbs = [incoming(c)]
c.update(0.1)
assert c.health == 76
# Shooter warns before its single shot; half-health guardian spreads more orbs.
a = fresh()
enemy = g.Enemy(300, 200, 0, shooter=True, shot_timer=0)
a.enemies = [enemy]
a.update(0.1)
assert enemy.charge > 0 and not a.hostile_orbs
for _ in range(9):
    a.update(0.1)
assert len(a.hostile_orbs) == 1
a = fresh()
a.next_boss_at = 0
a.update(0.1)
assert a.boss and a.boss.hp == 12
boss = a.boss
boss.shot_timer = 0
a.update(0.1)
for _ in range(9):
    a.update(0.1)
assert len(a.hostile_orbs) == 3
a.hostile_orbs.clear()
boss.hp = 6
boss.shot_timer = 0
a.update(0.1)
for _ in range(9):
    a.update(0.1)
assert len(a.hostile_orbs) == 5
# Point -> palm produces a single splash shot; held palm cannot repeat the combo.
a = fresh()
a.cast("Telekinesis")
a.elapsed = 0.3
a.cast("Fireball")
assert a.combo_name == "Meteor Burst" and a.projectiles[-1].splash == 110
assert a.projectiles[-1].damage == 3
a.elapsed = 1
a.cast("Fireball")
assert a.combo_count == 1 and a.projectiles[-1].splash == 0
b = fresh()
b.cast("Telekinesis")
b.elapsed = 3.1
b.cast("Fireball")
assert b.combo_count == 0
# Splash actually clears a group rather than only altering the label.
a = fresh()
a.enemies = [g.Enemy(435, 350, 0), g.Enemy(475, 355, 0)]
a.cast("Telekinesis")
a.cast("Fireball")
for _ in range(8):
    a.update(0.1)
assert not a.enemies and a.score == 20
# Shield -> lightning clears incoming orbs and does extra boss damage.
a = fresh()
a.boss = g.Boss(435, 250, 12, 12)
a.hostile_orbs = [incoming(a)]
a.cast("Shield")
a.cast("Lightning")
assert a.boss.hp == 7 and not a.hostile_orbs and a.combo_name == "Storm Counter"
# Ordinary fireball and reflected projectiles hit the boss; ultimate is a big hit.
a = fresh()
a.boss = g.Boss(435, 250, 12, 12)
a.cast("Fireball")
for _ in range(7):
    a.update(0.1)
assert a.boss.hp == 11
a.mana = 100
a.cast("Ultimate")
assert a.boss.hp == 3
old = a.elapsed
a.health = 50
a.damage_boss(3)
assert (
    a.boss is None
    and a.bosses_defeated == 1
    and a.health == 70
    and a.next_boss_at == old + 60
)
assert a.score == 150 and a.mana == 35
# Telekinesis turns shots upward and interrupts the boss charge.
a = fresh()
a.boss = g.Boss(435, 250, 12, 12, charge=0.5)
a.hostile_orbs = [incoming(a)]
a.cast("Telekinesis")
assert a.hostile_orbs[0].velocity.y < 0 and a.boss.charge == 0
# Pause freezes warnings, projectiles, boss timers and combo window.
a = fresh()
a.boss = g.Boss(435, 250, 12, 12)
a.hostile_orbs = [incoming(a)]
a.state = "paused"
position = a.hostile_orbs[0].position.copy()
a.update(0.1)
assert (
    a.elapsed == 0
    and a.hostile_orbs[0].position == position
    and a.boss.shot_timer == 1.8
)
# Restart clears threats and pending combo/counter state.
a.reset()
assert (
    not a.hostile_orbs
    and a.boss is None
    and a.parries == a.combo_count == 0
    and a.combo_spell is None
)
view = g.GameView(screen)
a.next_boss_at = 0
a.update(0.1)
a.boss.hp = 10
a.boss.shot_timer = 0
a.update(0.1)
a.enemies = [
    g.Enemy(190, 320, 0, shooter=True, charge=0.5, aim=pygame.Vector2(0.4, 0.9))
]
a.hostile_orbs = [
    g.HostileOrb(pygame.Vector2(430, 405), pygame.Vector2(0, 200)),
    g.HostileOrb(pygame.Vector2(540, 370), pygame.Vector2(0, 200)),
]
a.cast("Shield")
a.cast("Lightning")
# Put visible incoming shots back after Storm Counter clears them.
a.hostile_orbs = [g.HostileOrb(pygame.Vector2(410, 420), pygame.Vector2(0, 200))]
for state in ("playing", "paused", "gameover", "start"):
    a.state = state
    view.draw(a, (None, 0, [], None, ""), 30, True, True)
    pygame.image.save(screen, f"/private/tmp/combat-{state}.png")
# Long deterministic survival simulation catches spawn/cleanup interleavings.
a = fresh()
a.spawn_timer = 0
a.next_boss_at = 5
for frame in range(3600):
    if a.state == "gameover":
        a.reset()
        a.next_boss_at = 5
    a.health = 100
    a.steer((frame % 120) / 120)
    if frame % 20 == 0:
        a.cast("Fireball")
    if frame % 90 == 0:
        a.cast("Shield")
    if frame % 140 == 0:
        a.cast("Lightning")
    if frame % 180 == 0:
        a.cast("Telekinesis")
    a.update(1 / 30)
    assert len(a.hostile_orbs) < 100 and len(a.particles) <= 650
    assert 0 <= a.mana <= 100
pygame.quit()
print(
    "Passed perfect/late shield timing, reflection, damage grace, enemy telegraphs, boss volleys/enrage, both combos, splash collision, boss damage/rewards, telekinesis interruption, pause/restart, rendering and 120-second combat simulation."
)
