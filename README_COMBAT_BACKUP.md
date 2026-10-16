# Gesture-Controlled Spellcasting Game

The game is ready to run in the project's existing `.venv`. Close the collector and live
inference windows first so they release the webcam.

```bash
cd "$HOME/Desktop/Hand Gesture Game "
./.venv/bin/python gesture_game_adventure.py
```

Or double-click **Play Adventure Edition.command**. Click START GAME or press Space to begin.
Show the gesture steadily until it is recognized. Hold an open palm to keep firing fireballs.
For other spells, lower your hands out of view or switch to another stable pose to cast again. Mana regenerates
10 points per second; cooldowns and insufficient mana can block a recognized spell.

## Controls

Move your hand left/right in the mirrored webcam preview to move the wizard and dodge spirits.
Arrow keys or A/D also move. Movement is smoothed and bounded within the arena; losing hand
tracking stops movement. With two hands visible, their average palm position controls movement.
Only spirits touching your wizard cause damage. Spirits that pass you increase the dodge counter.


| Pose | Spell | Cost | Effect |
|---|---|---:|---|
| One open palm | Fireball | 10 | Aimed projectile hits the first enemy in its path |
| One closed fist | Shield | 20 | Blocks for 2 seconds; reflects shots during the first 0.65 seconds |
| Index and middle extended | Lightning | 30 | Chains through up to 3 enemies |
| Only index extended | Telekinesis | 15 | Pushes enemies back, reverses shots, interrupts boss charge |
| Both palms open | Ultimate | 100 | Clears ordinary enemies and shots; deals 8 boss damage |

Keyboard 1–5 follows this order. These are **game keys**, not the dataset label IDs 0–4.
You can also click spell buttons. P or the top pause button pauses/resumes, R restarts after game over, M toggles
sound, and Q/Esc quits. START/RESUME/PLAY AGAIN and MUTE buttons support mouse use.

The wizard starts at the arena's bottom center and moves with your hand. Enemies approach from above and converge
on the wizard. Every defeat earns 10 points; an unprotected enemy contact costs 20 HP.
Health starts at 100. Every 30 seconds the level increases, making enemies faster and more
frequent. The best score is retained across restarts during the current app session.

## Features and files

- `gesture_game.py`: 1280×720 arena, wizard/enemies, spells, HUD, particles, synthesized sound,
  start/pause/game-over screens, keyboard/mouse controls and 30 FPS rendering target.
- `game_camera.py`: supporting helper, launched automatically by the game. Uses the working
  `GestureRecognizer` from `inference.py` and streams webcam previews, confidence and spell
  events to the game. OpenCV runs in a separate process to avoid its SDL library conflict
  with Pygame on macOS. Keep this helper beside the game file.
- `collect_data.py`: manual or guided collection; see its --help for collection options.
- `train_model.py`: CPU training; see `TRAINING.md`.
- `inference.py`: standalone live recognition; see `INFERENCE.md`.
- `models/`: trained weights/metadata and MediaPipe detector.
- `data/gestures.csv`: your 1,000 labeled samples.
- `requirements.txt`: installed dependency versions for the complete current project.
- `reports/game_last_run.json`: last run's game FPS, score, level, survival time and spell counts.

The colorful adventure design includes `assets/images/enchanted_arena.png`, an illustrated
forest arena made with built-in ImageGen. Characters and spell icons are drawn in code;
short spell, hit and UI sounds are synthesized locally using Pygame's mixer.
Artwork provenance and the complete generation prompt are in `assets/images/ART.md`.
All artwork is bundled locally; no runtime downloads or paid assets are required. If an audio device is
unavailable, the game continues silently. Use `--mute` to start without playback.
Pygame documentation: https://www.pygame.org/docs/ref/mixer.html.

## Testing and troubleshooting

Keyboard-only mode:

```bash
./.venv/bin/python gesture_game_adventure.py --no-camera
```

Use `--camera 1` for another camera, `--mute` to silence effects, and `--seed 42` for a
repeatable enemy sequence. `--max-frames N` provides a bounded startup/shutdown check.
No GPU or network is required after dependency/model setup. Recreate `.venv` with a compatible
Python interpreter and install `requirements.txt` if needed; use the existing `.venv/bin/python`
to avoid the earlier Anaconda native-library conflict.

If the webcam fails, close other camera apps and enable Terminal camera access in macOS
Privacy & Security. The game reports camera errors in its right panel and remains playable
with keys/mouse. Starting/resuming/restarting resets gesture history so old camera events
cannot cast into a new game state. Outdated camera events are discarded.

The model currently has five classes without idle. Unrelated visible hand poses may still
be accepted as spells. Removing hands produces no spell; add idle examples and retrain to
improve rejection of other poses. Test new angles and lighting in live inference if a gesture
is unreliable. The right panel shows the accepted pose and confidence; the main loop targets
30 FPS, while webcam recognition runs independently at the rate your CPU supports.

Verified checks cover all five spells, mana costs, cooldowns, shield expiry, collision damage,
scoring, level progression, pause, game over/restart, sound synthesis and screen rendering.
The real model/detector was tested through the isolated helper with a synthetic camera,
including preview transport, stale-event resets and process cleanup. Live webcam gameplay
and physical speaker playback still need a play-through on your laptop.

## Adventure Edition launcher

The redesigned game lives in `gesture_game_adventure.py`. The launchers execute this
file directly. `gesture_game.py` forwards to it for compatibility, while
`gesture_game_classic.py` preserves the original game. The window title and start screen
identify the Adventure Edition. Use the direct filename if an editor restores an older
copy of `gesture_game.py`.

## Expanded combat

Every third spirit is a shooter with a gold marker. It warns with an aiming line for
0.85 seconds before launching a pink orb. Boss volleys use the same warning.
Move out of the line, or cast Shield just before a shot reaches you: the first 0.65
seconds (gold shield rim) reflect it for +15 score and +12 mana. The rest of the
2-second shield blocks it. Orb hits cost 12 HP; contact costs 20 HP. After taking a
hit, a 0.65-second grace period prevents simultaneous attacks from draining all health.

Use these sequences within 3 seconds. On-screen prompts show the remaining window:

- Telekinesis -> Fireball (point -> open palm; keys 4 -> 1): Meteor Burst,
  a shot with 110-pixel splash and 3 boss damage.
- Shield -> Lightning (fist -> peace; keys 2 -> 3): Storm Counter,
  clears hostile shots, chains through up to 5 spirits, and deals 5 boss damage.

Both spells retain their normal mana costs and cooldowns. Failed casts do not arm combos.
Each sequence completes once; holding palm continues ordinary fireballs afterward.

The Grove Guardian first arrives at 45 seconds. It moves horizontally and fires
3-shot fans, expanding to 5 shots at half health. Its starting health is 12;
ordinary fireballs deal 1, lightning 3, reflected shots 2, and ultimate 8 damage.
Defeating it gives +150 score, +20 health and +35 mana, and clears hostile shots.
The next guardian arrives 60 seconds later with 4 additional health.
Ordinary defeats give +8 mana; downward attacks dodged off-screen give +3 mana.
Mana also regenerates at 10 per second.

Quick boss practice (boss after 5 seconds of play):

```bash
./.venv/bin/python gesture_game_adventure.py --boss-practice
```

See `TEST_NEW_GAME.md` for the play-through checklist. Automated checks in
`test_game_combat.py` cover counters, boss patterns/damage/rewards, combos,
pause/restart, UI rendering, and a 120-second simulation. `test_game_movement.py`
covers steering and transparent sprites. Run them with the project's `.venv/bin/python`;
for a headless run, set `SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy`.
