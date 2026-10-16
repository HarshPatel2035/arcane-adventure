from pathlib import Path
import sys, math

sys.path.insert(0, str(Path(__file__).resolve().parent))
from types import SimpleNamespace as NS
import pygame
import arcane_adventure as a
import gesture_game_adventure as b
from motion_controls import MotionRecognizer

pygame.init()
screen = pygame.display.set_mode((1280, 720))


def fresh(**kw):
    game = a.AdventureGame(**kw)
    game.reset()
    game.spawn_timer = 999
    return game


# Keyboard-held props must survive camera failure; hand-held props must be placed.
g = fresh()
g.cast("Telekinesis")
assert g.held_object and not g.held_by_hand
held = g.held_object
a.apply_input(g, (None, 0, [], "camera error", ""), 20)
assert g.held_object is held
# Charged props must do splash damage, not just display the combo banner.
g = fresh()
obj = g.objects[1]
g.cast("Telekinesis")
g.cast("Fireball")
assert obj.powered
obj.position = pygame.Vector2(400, 280)
obj.velocity = pygame.Vector2(0, -100)
g.enemies = [b.Enemy(400, 250, 0), b.Enemy(475, 250, 0)]
g.update(0.1)
assert not g.enemies and obj not in g.objects
# Endless should offer upgrades, resume without advancing the route, and schedule the next boss.
g = fresh(mode="endless")
g.boss = b.Boss(435, 265, 1, 1)
g.boss_phase = "storm"
g.safe_x = 150
g.damage_boss(2)
assert g.state == "upgrade" and g.safe_x is None and g.next_boss_at == g.elapsed + 60
g.choose_upgrade(0)
assert g.state == "playing" and g.stage == 0
g.elapsed = g.next_boss_at
g.update(0.01)
assert g.boss and g.boss_phase == "crystal"
# Damage grace and state transitions never activate companion protection outside combat.
g = fresh(companion="dragon")
g.state = "paused"
g.hurt(20)
assert g.health == 100 and g.dragon_ready == 0
# Confidence and time gaps reject noisy motion.
r = MotionRecognizer()
assert all(
    r.process(0.2 + i * 0.03, 0.5, 0, 0.5, 1, i * 0.03) is None for i in range(20)
)
r = MotionRecognizer()
assert all(r.process(0.5, 0.5, 1, 0.99, 1, i * 0.4) is None for i in range(20))
# Render tint variants against a colorful background without solid padding.
view = a.AdventureView(screen)
for name, sprite in view.skins.items():
    assert sprite.get_at((0, 0)).a == 0
    tile = pygame.Surface(sprite.get_size())
    tile.fill((240, 200, 150))
    tile.blit(sprite, (0, 0), special_flags=pygame.BLEND_PREMULTIPLIED)
    assert tile.get_at((0, 0))[:3] == (240, 200, 150)
pygame.quit()
print(
    "Passed camera-failure keyboard props, Meteor object splash, endless upgrades/respawn, pause companion behavior, noisy-motion rejection and cosmetic transparency."
)
