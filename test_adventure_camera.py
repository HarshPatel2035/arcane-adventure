from pathlib import Path
import sys, time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pygame
import gesture_game_adventure as b
from arcane_adventure import AdventureGame, AdventureView, apply_input

original = b.subprocess.Popen


def synthetic(arguments, **kwargs):
    return original(arguments + ["--synthetic"], **kwargs)


b.subprocess.Popen = synthetic
worker = b.CameraWorker(motion=True)
try:
    worker.start()
    worker.reset()
    deadline = time.monotonic() + 45
    while time.monotonic() < deadline:
        packet = worker.snapshot()
        if packet[0] is not None or packet[3]:
            break
        time.sleep(0.1)
    assert packet[3] is None, packet[3]
    pred = packet[0]
    assert (
        pred
        and pred.hand_x is None
        and pred.hand_y is None
        and pred.motion is None
        and pred.generation == 1
    )
    assert "cv2" not in sys.modules and "mediapipe" not in sys.modules
    pygame.init()
    screen = pygame.display.set_mode((1280, 720))
    game = AdventureGame()
    game.reset()
    apply_input(game, packet)
    view = AdventureView(screen)
    view.draw(game, packet, 30, False, True)
    worker.reset()
    assert worker.snapshot()[0] is None and worker.snapshot()[1] == 0
finally:
    worker.close()
    b.subprocess.Popen = original
    pygame.quit()
assert not worker.thread.is_alive() and worker.process.poll() is not None
print(
    "Passed real-model synthetic camera transport with motion enabled, hand x/y fields, generation reset, current adventure UI, library isolation and helper shutdown."
)
