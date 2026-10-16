"""A held pose whose one-shot cast failed (mana/cooldown) fires once it becomes affordable."""

from types import SimpleNamespace as NS

from arcane_adventure import AdventureGame, apply_input

ULTIMATE, PALM, FIST, IDLE = 4, 0, 1, 5


def fresh():
    game = AdventureGame(seed=9)
    game.reset()
    game.state = "playing"
    game.spawn_timer = 999
    return game


def frame(gesture, now, event=None, confidence=0.95):
    pred = NS(gesture_id=gesture, confidence=confidence, hand_x=0.5, hand_y=0.5)
    events = [(event, now, 1)] if event else []
    return (pred, now, events, None, "")


def hold_until_full(game, gesture, start, **kwargs):
    """Keep showing a pose (no new events) while mana regenerates at 10/s."""
    now = start
    for _ in range(100):  # 10 seconds at 10 frames/second
        now += 0.1
        game.update(0.1)
        apply_input(game, frame(gesture, now, **kwargs), now)
    return now


def test_holding_ultimate_casts_when_mana_reaches_100():
    game = fresh()
    game.mana = 40
    apply_input(game, frame(ULTIMATE, 10, event="Ultimate"), 10)
    assert game.cast_counts["Ultimate"] == 0 and game.pending_cast == "Ultimate"
    assert "keep holding" in game.feedback_for(frame(ULTIMATE, 10)[0])
    hold_until_full(game, ULTIMATE, 10)
    assert game.cast_counts["Ultimate"] == 1  # fired once, not repeatedly
    assert game.pending_cast is None


def test_switching_pose_or_idle_cancels_the_waiting_spell():
    for other in (FIST, IDLE):
        game = fresh()
        game.mana = 40
        apply_input(game, frame(ULTIMATE, 10, event="Ultimate"), 10)
        apply_input(game, frame(other, 10.1), 10.1)
        assert game.pending_cast is None
        hold_until_full(game, ULTIMATE, 10.1)
        assert game.cast_counts["Ultimate"] == 0  # needs a fresh pose event now


def test_brief_tracking_gaps_and_low_confidence_keep_waiting():
    game = fresh()
    game.mana = 40
    apply_input(game, frame(ULTIMATE, 10, event="Ultimate"), 10)
    apply_input(game, frame(None, 10.1), 10.1)  # stabilizing / hand-count jitter
    apply_input(game, frame(ULTIMATE, 10.2, confidence=0.6), 10.2)
    assert game.pending_cast == "Ultimate"
    hold_until_full(game, ULTIMATE, 10.2)
    assert game.cast_counts["Ultimate"] == 1


def test_cooldown_retry_and_palm_is_unaffected():
    game = fresh()
    game.cast("Shield")
    apply_input(game, frame(FIST, 10, event="Shield"), 10)  # still cooling down
    assert game.pending_cast == "Shield"
    game.mana = 100
    for step in range(25):  # 2.5 s > 2 s shield cooldown
        game.update(0.1)
        apply_input(game, frame(FIST, 10 + step * 0.1), 10 + step * 0.1)
    assert game.cast_counts["Shield"] == 2
    game = fresh()
    game.mana = 0
    apply_input(game, frame(PALM, 10, event="Fireball"), 10)
    assert game.pending_cast is None  # held palm already auto-repeats fireballs
