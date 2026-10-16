# Live gesture recognition

Close the data collector first, then double-click `Live Gesture Recognition.command`.
Alternatively, open Terminal and run:

```bash
cd "$HOME/Desktop/Hand Gesture Game "
./.venv/bin/python inference.py
```

All dependencies are already installed in the project's `.venv`. `requirements.txt`
records the tested environment. The local detector and trained weights are required:
`models/hand_landmarker.task`, `models/gesture_model.pth`, `models/gesture_model.json`.
The script runs offline and does not change your collected data or train a model.

## Poses

| Label | Pose | Spell |
|---|---|---|
| 0 | One hand, all five fingers extended | Fireball |
| 1 | One hand, closed fist | Shield |
| 2 | One hand, index and middle extended, others folded | Lightning |
| 3 | One hand, only index extended, others folded | Telekinesis |
| 4 | Both hands open, all fingers extended | Ultimate |

Hold each pose steady for at least five frames. The window shows the accepted gesture and
its mean supporting confidence, the raw current prediction, hand count, webcam FPS and
combined detection/classification latency. Latency excludes capture, drawing and display.
Three matching votes are required within the full five-frame window, and the current frame
must support that winner above 80% confidence. Threshold failures suppress the accepted output
immediately. Missing hands, invalid landmarks and hand-count changes clear smoothing history.

Click QUIT, press Q/Esc, or close the webcam window to exit.

## Game integration

```python
from inference import GestureRecognizer

with GestureRecognizer() as recognizer:
    # camera.read() supplies an unmirrored BGR frame.
    prediction = recognizer.process_frame(camera_bgr_frame)
    if prediction.action is not None:
        cast_spell(prediction.action)
```

`prediction.gesture_id` is the stable label (or None), `confidence` its supporting mean,
and `action` a one-shot spell string (or None). Use action for casting; gesture_id is suitable
for continuous pose feedback. Holding a pose emits once. To cast it again, remove your hands,
show a trained idle pose, or switch to another stable gesture. Events have a 0.6-second cooldown.
The mirrored frame and MediaPipe result are included for the game's webcam display.
`reset()` clears gesture history and rearms emission while preserving the cooldown.

Inference reuses `collect_data.extract_features` and `train_model.GestureClassifier` to keep
Left/Right slots, zero padding, normalization and network layout identical. Metadata is checked
before loading weights. MediaPipe uses increasing video timestamps; PyTorch runs in evaluation
and inference modes on CPU. References:
[MediaPipe guide](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker/python),
[PyTorch model loading](https://docs.pytorch.org/tutorials/beginner/saving_loading_models).

## Options and troubleshooting

- `--camera 1`: try another camera index.
- `--threshold 0.9`: demand stronger model confidence.
- `--window 7`: longer smoothing window (must be a positive odd number).
- `--cooldown 1.0`: minimum seconds between emitted spells.
- `--width 640 --height 480 --threads 2`: default CPU settings.
- `--model PATH --detector PATH`: alternate local models; matching JSON is required.

If the camera fails, close other camera apps and enable Terminal camera access in macOS
Privacy & Security settings. Use `.venv/bin/python` to avoid the earlier Anaconda conflict.
If predictions fluctuate, show your whole hand, improve lighting and move more slowly.
Do not mirror a frame before passing it to process_frame; that method mirrors it once itself.

Your five-class model has no idle class. No hands results in no spell, but unrelated visible
hand poses may still receive high confidence. Collect idle label 5 and retrain to improve this.
Test recognition on fresh poses, both left/right hands and new lighting before game integration.
No promise of 30 FPS is made: measure the displayed webcam FPS on your machine.
