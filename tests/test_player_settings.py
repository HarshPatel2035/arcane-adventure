"""Difficulty, hand-range calibration, saved top scores and spellbook pose drawings."""

import json
from types import SimpleNamespace as NS

import pygame
import pytest

import spell_engine as b
from arcane_adventure import (
    CALIBRATION_SECONDS,
    DIFFICULTIES,
    SCOREBOARD_SIZE,
    AdventureGame,
    AdventureView,
    apply_input,
    handle_button,
    load_profile,
)


def fresh(**kwargs):
    game = AdventureGame(seed=4, **kwargs)
    game.reset()
    game.state = "playing"
    game.spawn_timer = 999
    return game


@pytest.mark.parametrize("name", list(DIFFICULTIES))
def test_difficulty_scales_damage_speed_and_spawns(name):
    game = fresh(difficulty=name, companion="owl")
    settings = DIFFICULTIES[name]
    game.hurt(20)
    assert game.health == 100 - round(20 * settings["damage"])
    assert game.spawn_scale == settings["spawn"]
    game.stage, game.stage_spawned = 1, 0
    game.spawn_enemy()  # stage 1 opens with a swarm of three
    assert [e.speed for e in game.enemies] == pytest.approx(
        [85 * settings["speed"], 82 * settings["speed"], 82 * settings["speed"]]
    )


def test_hard_spawns_faster_than_easy():
    counts = {}
    for name in ("easy", "hard"):
        game = fresh(difficulty=name)
        game.spawn_timer = 0
        game.update(0.01)
        counts[name] = game.spawn_timer
    assert counts["hard"] < counts["easy"]


def test_difficulty_cycles_at_camp_and_persists(tmp_path):
    path = tmp_path / "profile.json"
    game = AdventureGame(profile_path=path)
    assert game.difficulty == "normal"
    handle_button(game, "difficulty")
    assert game.difficulty == "hard"
    handle_button(game, "difficulty")
    assert game.difficulty == "easy"
    assert AdventureGame(profile_path=path).difficulty == "easy"


def hand(x, now):
    return (NS(hand_x=x, gesture_id=0, confidence=0.95), now, [], None, "")


def test_calibration_maps_comfortable_reach_to_full_arena(tmp_path):
    path = tmp_path / "profile.json"
    game = AdventureGame(profile_path=path)
    handle_button(game, "calibrate")
    assert game.state == "calibrate"
    steps = 60
    for i in range(steps + 1):
        now = 100 + i * CALIBRATION_SECONDS / steps
        apply_input(game, hand(0.3 + 0.4 * (i % 20) / 19, now), now)
    assert game.state == "start"
    low, high = game.calibration
    assert 0.29 <= low < 0.33 and 0.67 < high <= 0.71
    assert game.map_hand(low) == 0 and game.map_hand(high) == 1
    assert game.map_hand(0.5) == pytest.approx(0.5, abs=0.03)
    assert game.map_hand(0.05) == 0 and game.map_hand(None) is None
    assert AdventureGame(profile_path=path).calibration == game.calibration
    # Playing uses the mapped position: a hand at the calibrated right edge reaches the wall.
    game.reset()
    game.state = "playing"
    apply_input(game, hand(high, 200), 200)
    assert game.move_target >= 790


def test_calibration_rejects_tiny_sweeps_and_can_be_cancelled():
    game = AdventureGame()
    game.calibration = None
    game.begin_calibration()
    for i in range(40):
        apply_input(game, hand(0.5 + 0.001 * i, 10 + i * 0.11), 10 + i * 0.11)
    assert game.state == "start" and game.calibration is None
    assert "Not enough movement" in game.calibration_message
    game.begin_calibration()
    handle_button(game, "calibrate")
    assert game.state == "start"


def test_calibration_without_camera_explains_instead_of_waiting():
    game = AdventureGame()
    game.camera_available = False
    handle_button(game, "calibrate")
    assert game.state == "start" and "needs the camera" in game.calibration_message


def finish(game, score, state="gameover"):
    game.reset()
    game.score = score
    game.state = state
    return game.record_run()


def test_top_scores_are_sorted_capped_recorded_once_and_saved(tmp_path):
    path = tmp_path / "profile.json"
    game = AdventureGame(profile_path=path, difficulty="hard")
    for score in (300, 900, 100, 500, 700, 200):
        finish(game, score)
    board = [entry["score"] for entry in game.profile["scores"]]
    assert board == [900, 700, 500, 300, 200] and len(board) == SCOREBOARD_SIZE
    assert game.record_run() == game.profile["scores"]  # same run is not added twice
    assert not game.new_record
    finish(game, 1500, "victory")
    assert game.new_record and game.profile["scores"][0]["won"]
    saved = AdventureGame(profile_path=path).profile["scores"]
    assert [entry["score"] for entry in saved] == [1500, 900, 700, 500, 300]
    assert saved[0]["difficulty"] == "hard"


def test_corrupt_profile_values_fall_back_safely(tmp_path):
    path = tmp_path / "profile.json"
    path.write_text(json.dumps({
        "difficulty": "nightmare",
        "calibration": [0.9, 0.1],
        "scores": [{"score": "x"}, {"score": 50, "mode": "adventure", "difficulty": "easy"}],
    }))
    profile = load_profile(path)
    assert profile["difficulty"] == "normal" and profile["calibration"] is None
    assert [entry["score"] for entry in profile["scores"]] == [50]


def test_new_screens_render_with_pose_drawings():
    pygame.init()
    try:
        screen = pygame.display.set_mode((1280, 720))
        view = AdventureView(screen)
        game = AdventureGame()
        game.begin_calibration()
        game.add_calibration_sample(0.4, 1.0)
        view.draw(game, (None, 0, [], None, ""), 30, False, True)
        finish(game, 120)
        view.draw(game, (None, 0, [], None, ""), 30, True, True)
        for pose in range(5):
            glyph = b.pose_glyph(pose)
            assert glyph.get_at((0, 0)).a == 0  # transparent corners
            assert any(glyph.get_at((x, y)).a > 200 for x in range(28) for y in range(30))
    finally:
        pygame.quit()
