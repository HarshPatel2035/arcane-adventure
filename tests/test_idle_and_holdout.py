import io
from types import SimpleNamespace as NS

import numpy as np
import pygame
import pytest

import spell_engine as b
from arcane_adventure import AdventureGame, AdventureView, apply_input
from collect_data import HEADER
from train_model import load_data


def test_idle_gesture_never_casts_or_crashes():
    """A trained idle class (label 5) must show feedback, render and cast nothing."""
    assert b.spell_for(5) is None and b.spell_for(None) is None
    assert b.spell_for(0) == "Fireball" and b.spell_for(4) == "Ultimate"
    pygame.init()
    try:
        screen = pygame.display.set_mode((1280, 720))
        game = AdventureGame(seed=3)
        game.reset()
        game.spawn_timer = 999
        jpeg = io.BytesIO()
        pygame.image.save(pygame.Surface((362, 272)), jpeg, "preview.jpg")
        idle = NS(gesture_id=5, gesture_name="Idle", confidence=0.97, hand_x=0.5,
                  hand_y=0.5, preview_image=jpeg.getvalue())
        packet = (idle, 10, [], None, "")
        mana = game.mana
        apply_input(game, packet, 10)
        assert "Idle" in game.feedback_for(idle)
        assert game.mana == mana and not game.projectiles
        AdventureView(screen).draw(game, packet, 30, False, True)
    finally:
        pygame.quit()


def write_csv(path, labels):
    rows = np.random.default_rng(1).normal(size=(len(labels), 126)).astype(np.float32)
    lines = [",".join(HEADER)]
    lines += [",".join([str(label), *map(str, row)]) for label, row in zip(labels, rows)]
    path.write_text("\n".join(lines) + "\n")


def test_holdout_files_may_cover_only_some_classes(tmp_path):
    partial = tmp_path / "session2.csv"
    write_csv(partial, [0, 0, 3])
    features, labels = load_data(partial, holdout=True)
    assert features.shape == (3, 126) and labels.tolist() == [0, 0, 3]
    with pytest.raises(ValueError, match="Missing core gesture labels"):
        load_data(partial)
