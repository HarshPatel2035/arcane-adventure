
from types import SimpleNamespace as NS
import pygame
import spell_engine as g
from game_camera import palm_position


def test_game_movement(tmp_path):
    """Checks palm mapping, speed limit, bounds, no-hand/stale/error/pause stopping,
    keyboard movement, dodge/contact, moving spell origins, independent resets and
    sprite transparency.
    """
    pygame.init()
    screen = pygame.display.set_mode((g.WIDTH, g.HEIGHT))
    game = g.SpellGame(seed=2)
    game.reset()
    game.spawn_timer = 999
    hand = lambda x: [NS(x=x) for _ in range(21)]
    assert palm_position([]) is None
    assert palm_position([hand(0.2), hand(0.8)]) == 0.5
    assert palm_position([hand(float("nan"))]) is None
    p = NS(hand_x=0.85, gesture_id=None, confidence=0)
    g.apply_camera_input(game, (p, 10, [], None, ""), 10)
    old = game.player.x
    game.update(0.1)
    assert 0 < game.player.x - old <= 48
    for _ in range(50):
        game.update(0.1)
    assert 797 < game.player.x <= 798
    # No hands, stale data, camera failure and pause all stop steering.
    for packet in [
        (NS(hand_x=None, gesture_id=None, confidence=0), 10, [], None, ""),
        (p, 8, [], None, ""),
        (p, 10, [], "failure", ""),
    ]:
        game.steer(0.15)
        g.apply_camera_input(game, packet, 10)
        old = game.player.copy()
        game.update(0.1)
        assert game.player == old
    old = game.player.copy()
    game.state = "paused"
    game.steer(0.15)
    game.update(0.1)
    assert game.player == old
    game.reset()
    game.spawn_timer = 999
    game.keyboard_move(-1, 0.1)
    assert game.player.x == g.PLAYER.x - 48
    for _ in range(30):
        game.keyboard_move(-1, 0.1)
    assert game.player.x == 72
    # A spirit crossing another lane cannot damage the wizard.
    game.enemies = [g.Enemy(700, 530, 100)]
    for _ in range(20):
        game.update(0.1)
    assert game.health == 100 and game.dodges == 1 and not game.enemies
    # Contact follows the moved wizard; spell origins follow it too.
    game.enemies = [g.Enemy(game.player.x, 530, 0)]
    game.update(0.01)
    assert game.health == 80
    game.cast("Fireball")
    assert game.projectiles[0].position.x == 72
    other = g.SpellGame()
    assert other.player == g.PLAYER
    view = g.GameView(screen)
    for sprite in [view.wizard_sprite, *view.enemy_sprites]:
        assert sprite.get_at((0, 0)).a == 0
        # Completely transparent pixels must leave a colorful background untouched.
        tile = pygame.Surface(sprite.get_size())
        tile.fill((240, 198, 129))
        tile.blit(sprite, (0, 0), special_flags=pygame.BLEND_PREMULTIPLIED)
        for x in range(sprite.get_width()):
            for y in range(sprite.get_height()):
                if sprite.get_at((x, y)).a == 0:
                    assert tile.get_at((x, y))[:3] == (240, 198, 129)
    game.reset()
    game.player.x = 245
    game.enemies = [g.Enemy(180, 240, 0), g.Enemy(620, 420, 0)]
    view.draw(game, (None, 0, [], None, ""), 30, True, True)
    pygame.image.save(screen, str(tmp_path / "movement-preview.png"))
    pygame.quit()
