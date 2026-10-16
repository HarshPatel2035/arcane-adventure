from pathlib import Path
import math, tempfile

from types import SimpleNamespace as NS
import pygame
from arcane_adventure import (
    AdventureGame,
    AdventureView,
    apply_input,
)
import spell_engine as b
from motion_controls import MotionRecognizer


def test_adventure_features():
    """Checks adventure route, upgrades, menu isolation, tactical enemies, objectives, boss
    phases, object controls/cover, companions, motion traces, persistence and all screen
    rendering checks.
    """
    pygame.init()
    screen = pygame.display.set_mode((1280, 720))


    def fresh(**kwargs):
        a = AdventureGame(seed=5, **kwargs)
        a.reset()
        a.spawn_timer = 999
        return a


    # Stage quota ends only after remaining enemies have resolved.
    a = fresh()
    a.stage_spawned = 8
    a.enemies = [b.Enemy(180, 200, 0)]
    a.update(0.1)
    assert a.state == "playing"
    a.enemies.clear()
    a.update(0.1)
    assert a.state == "upgrade" and len(a.upgrade_options) == 3
    elapsed = a.elapsed
    mana = a.mana
    a.update(0.1)
    assert a.elapsed == elapsed and a.mana == mana
    assert (
        a.choose_upgrade(0)
        and a.stage == 1
        and len(a.upgrades) == 1
        and a.state == "playing"
    )
    assert not a.choose_upgrade(0)
    # Gesture upgrade confirmation is isolated from combat and requires releasing entry fist.
    a = fresh()
    a.finish_stage()


    def packet(pose, x=0.5):
        return (NS(gesture_id=pose, confidence=0.98, hand_x=x), 10, [], None, "")


    apply_input(a, packet(1), 10)
    apply_input(a, packet(1), 10.65)
    assert a.state == "upgrade"
    apply_input(a, packet(0), 10.1)
    apply_input(
        a, (NS(gesture_id=1, confidence=0.98, hand_x=0.5), 10.8, [], None, ""), 10.8
    )
    assert a.state == "playing" and a.stage == 1 and a.cast_counts["Shield"] == 0
    # Enemy types use armor, exposure, healing, and a distinct swarm.
    a = fresh()
    a.stage = 1
    for _ in range(5):
        a.spawn_enemy()
    kinds = {e.kind for e in a.enemies}
    assert {"swarm", "golem", "archer", "healer"} <= kinds
    arm = next(e for e in a.enemies if e.kind == "golem")
    hp = arm.hp
    a.damage_enemy(arm, 2, "Fireball")
    assert arm.hp == hp
    a.objects.clear()
    a.cast("Telekinesis")
    a.damage_enemy(arm, 2, "Fireball")
    assert arm.hp == hp - 2
    healer = next(e for e in a.enemies if e.kind == "healer")
    healer.position  # property sanity
    arm.x = healer.x
    arm.y = healer.y
    healer.heal_at = 0
    a.update(0.1)
    assert arm.hp == hp - 1
    # All upgrade effects are active, not only displayed on cards.
    a = fresh()
    a.upgrades.add("Twin Flame")
    a.cast("Fireball")
    assert len(a.projectiles) == 2 and all(p.damage == 1 for p in a.projectiles)
    a = fresh()
    a.upgrades.add("Mirror Shield")
    a.reflect(b.HostileOrb(a.player.copy(), pygame.Vector2(0, 100)))
    assert len(a.projectiles) == 3 and all(p.kind == "Counter" for p in a.projectiles)
    a = fresh()
    a.upgrades.add("Thunder Chain")
    a.enemies = [b.Enemy(100 + i * 80, 200, 0) for i in range(5)]
    a.cast("Lightning")
    assert not a.enemies
    a = fresh()
    a.upgrades.add("Gravity Well")
    a.enemies = [b.Enemy(100, 200, 0)]
    a.objects.clear()
    a.cast("Telekinesis")
    assert a.enemies[0].x > 300
    a = fresh()
    a.upgrades.add("Mana Spring")
    a.mana = 30
    a.update(0.1)
    assert abs(a.mana - 31.5) < 0.001
    a = fresh()
    a.health = 40
    a.finish_stage()
    a.upgrade_options = ["Vital Bloom"]
    a.choose_upgrade(0)
    assert a.health == 75
    # Tree challenge has failure and success conditions; portal seals require shield + proximity.
    a = fresh()
    a.enter_stage(2)
    a.spawn_timer = 999
    a.enemies = [b.Enemy(435, 432, 0)]
    a.update(0.1)
    assert a.tree_health == 88
    a.tree_health = 12
    a.enemies = [b.Enemy(435, 432, 0)]
    a.update(0.1)
    assert a.state == "gameover"
    a = fresh()
    a.enter_stage(2)
    a.spawn_timer = 999
    a.elapsed = a.stage_started + 35
    a.update(0.1)
    assert a.state == "upgrade"
    a = fresh()
    a.enter_stage(3)
    a.spawn_timer = 999
    for p in a.portals:
        a.player.x = p.position.x
        a.shield_until = a.elapsed + 2
        for _ in range(10):
            a.update(0.1)
    assert a.state == "upgrade" and all(p.progress == 1 for p in a.portals)
    # Guardian phase progression, immunity, counters, storm damage and victory.
    a = fresh()
    a.enter_stage(4)
    a.spawn_timer = 999
    a.next_boss_at = 0
    a.update(0.1)
    assert a.boss and len([e for e in a.enemies if e.kind == "boss_crystal"]) == 2
    hp = a.boss.hp
    a.damage_boss(8)
    assert a.boss.hp == hp
    for e in a.enemies[:]:
        a.damage_enemy(e, 3, "Lightning")
    a.update(0.1)
    assert a.boss_phase == "mirror"
    a.damage_boss(3)
    assert a.boss.hp == hp
    a.damage_boss(2, "Counter")
    assert a.boss.hp == hp - 2 and a.boss_exposed_until > a.elapsed
    a.damage_boss(8)
    assert a.boss_phase == "storm"
    a.safe_x = 150
    a.player.x = 700
    a.storm_strike_at = a.elapsed
    a.update(0.1)
    assert a.health == 82
    a.safe_x = 150
    a.player.x = 150
    a.storm_strike_at = a.elapsed
    a.update(0.1)
    assert a.health == 82
    a.damage_boss(99)
    assert a.state == "victory" and a.profile["wins"] == 1 and "gold" in a.unlocked_skins()
    # Interactive props: drag uses hand position and leaves wizard stationary; throw hits targets.
    a = fresh()
    obj = a.objects[1]
    a.cursor_live = True
    a.cursor = obj.position.copy()
    a.cast("Telekinesis")
    assert a.held_object is obj
    old = a.player.copy()
    a.set_hand(0.8, 0.4)
    a.update(0.1)
    assert a.player == old and obj.position.x > 435
    assert a.cast("Fireball") and a.held_object is None and obj.velocity is not None
    # Cover blocks hostile shots without consuming the player's health.
    a = fresh()
    rock = a.objects[0]
    a.hostile_orbs = [
        b.HostileOrb(rock.position + pygame.Vector2(0, -30), pygame.Vector2(0, 200))
    ]
    a.update(0.1)
    assert not a.hostile_orbs
    # Losing tracking places a held object; stale events do not cast or move.
    a = fresh()
    a.held_object = a.objects[0]
    a.held_by_hand = True
    apply_input(a, (None, 0, [("Ultimate", 0, 0)], None, ""), 10)
    assert a.held_object is None and a.cast_counts["Ultimate"] == 0
    # Companion effects and opt-in motion resource limits.
    a = fresh(companion="dragon")
    a.hurt(20)
    assert a.health == 100
    a.elapsed = 1
    a.hurt(20)
    assert a.health == 80
    a = fresh(companion="spirit")
    assert a.combo_window == 4
    a = fresh(companion="owl")
    a.enemies = [b.Enemy(435, 200, 0)]
    a.defeat(a.enemies[0], (200, 100, 80))
    a.mana = 30
    for _ in range(15):
        a.update(0.1)
    assert not a.mana_drops and a.mana > 45
    for kind, cost in [("sweep", 15), ("circle", 20), ("charge", 10)]:
        a = fresh()
        assert not a.motion_cast(kind)
        a.motion_enabled = True
        assert a.motion_cast(kind) and a.mana == 100 - cost
        assert not a.motion_cast(kind)
    # Motion trajectories: positives, accidental movement negatives and reset.
    r = MotionRecognizer()
    events = []
    for i in range(20):
        event = r.process(0.2 + i * 0.025, 0.5, 0, 0.99, 1, i * 0.03)
        if event:
            events.append(event)
    assert events == ["sweep"]
    r = MotionRecognizer()
    events = []
    for i in range(36):
        angle = i * math.tau / 35
        event = r.process(
            0.5 + 0.12 * math.cos(angle),
            0.5 + 0.12 * math.sin(angle),
            2,
            0.99,
            1,
            i * 0.035,
        )
        if event:
            events.append(event)
    assert events == ["circle"], events
    r = MotionRecognizer()
    events = []
    for i in range(60):
        event = r.process(0.5, 0.5, 1, 0.99, 1, i * 0.04)
        if event:
            events.append(event)
    assert events == ["charge"]
    r.reset()
    assert not r.path and not r.charged
    r = MotionRecognizer()
    assert all(
        r.process(0.5 + i * 0.005, 0.5, 0, 0.99, 1, i * 0.03) is None for i in range(30)
    )
    r = MotionRecognizer()
    assert all(
        r.process(0.2 + i * 0.03, 0.5, 0, 0.99, 2, i * 0.03) is None for i in range(20)
    )
    # Profile round-trip, unlock conditions, and malformed input recovery.
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "profile.json"
        a = fresh(profile_path=path)
        a.profile.update(wins=1, guardians=3)
        a.skin = "rose"
        a.save_profile()
        c = fresh(profile_path=path)
        assert c.skin == "rose" and "gold" in c.unlocked_skins()
        path.write_text("bad json")
        c = fresh(profile_path=path)
        assert c.profile["wins"] == 0
    # Render every screen and each encounter, verify all interactive rectangles fit.
    view = AdventureView(screen)
    for stage in range(5):
        a = fresh()
        a.enter_stage(stage)
        a.spawn_timer = 999
        if stage == 4:
            a.next_boss_at = 0
            a.update(0.1)
        view.draw(a, (None, 0, [], None, ""), 30, True, True)
        pygame.image.save(screen, f"/private/tmp/route-{stage}.png")
    for state in ("start", "upgrade", "paused", "gameover", "victory"):
        a = fresh()
        if state == "upgrade":
            a.finish_stage()
        else:
            a.state = state
        view.draw(a, (None, 0, [], None, ""), 30, True, True)
        assert all(screen.get_rect().contains(rect) for rect in view.buttons.values())
        if state in ("upgrade", "paused", "victory", "gameover"):
            assert not any(k.startswith("cast_") for k in view.buttons)
        pygame.image.save(screen, f"/private/tmp/route-{state}.png")
    pygame.quit()
