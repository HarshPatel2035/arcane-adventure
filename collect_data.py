"""Collect labeled hand landmarks for the Gesture-Controlled Spellcasting Game.

Setup: python -m pip install opencv-contrib-python mediapipe numpy
First run: python collect_data.py --download-model
Later runs: python collect_data.py

Keys 0-4 select a gesture and PAUSE recording; Space toggles recording;
S saves one sample; Q/Esc quits. Clickable controls also appear in the window.
--label 2 --record starts collection without keyboard input.
--include-idle enables label 5.
--guided cycles through unfinished labels with a countdown before each pose.
--reset-data clears existing samples explicitly; ordinary starts preserve data.
Feature contract: mirrored camera input, normalized image landmarks, left
hand slot followed by right hand slot (MediaPipe handedness). Missing slots
are zeros. Training and inference must use this same preprocessing.
"""

from __future__ import annotations

import argparse
import csv
from collections import deque
import math
from pathlib import Path
import sys
import tempfile
import time
import urllib.request

import numpy as np


ROOT = Path(__file__).resolve().parent
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
LABELS = {
    0: "Open Palm / Fireball",
    1: "Closed Fist / Shield",
    2: "Peace Sign / Lightning",
    3: "Pointing / Telekinesis",
    4: "Both Hands Open / Ultimate",
    5: "Idle / No Spell",
}
POSE_INSTRUCTIONS = {
    0: "ONE hand: extend all five fingers, palm facing the camera.",
    1: "ONE hand: curl all fingers and your thumb into a closed fist.",
    2: "ONE hand: extend index + middle fingers; fold thumb, ring and pinky.",
    3: "ONE hand: extend ONLY your index finger; fold all other fingers.",
    4: "BOTH hands: extend all five fingers on each hand, palms to camera.",
    5: "Relax your hands or show other poses that should not cast a spell.",
}
HEADER = ["label"] + [f"f{i}" for i in range(126)]
CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14),
    (14, 15), (15, 16), (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
)


class GuidedSession:
    """Advance only after valid saved samples meet each label's target."""

    def __init__(self, counts, labels, target, countdown, now):
        self.counts = counts
        self.labels = [value for value in labels if counts[value] < target]
        self.target = target
        self.countdown = countdown
        self.index = 0
        self.deadline = now + countdown
        self.paused = False
        self.remaining = 0.0

    @property
    def done(self):
        return self.index >= len(self.labels)

    @property
    def label(self):
        return None if self.done else self.labels[self.index]

    def tick(self, now):
        if not self.done and self.counts[self.label] >= self.target:
            self.index += 1
            self.deadline = now + self.countdown
            self.remaining = self.countdown

    def ready(self, now):
        return not self.done and not self.paused and now >= self.deadline

    def toggle_pause(self, now):
        if self.done:
            return
        if self.paused:
            self.deadline = now + self.remaining
        else:
            self.remaining = max(0.0, self.deadline - now)
        self.paused = not self.paused

    def status(self, now):
        if self.done:
            return "COMPLETE! All labels reached the target. Click QUIT."
        pose = f"LABEL {self.label}: {LABELS[self.label]}"
        if self.paused:
            return f"GUIDED PAUSED | Next pose: {pose}"
        if now < self.deadline:
            seconds = math.ceil(self.deadline - now)
            return f"GET READY: {pose} | Starts in {seconds}s"
        return f"HOLD: {pose} | {self.counts[self.label]}/{self.target} saved"


def normalize_landmarks(landmarks) -> np.ndarray:
    """Return 63 wrist-relative coordinates divided by wrist-to-MCP length.

    Coordinates remain in MediaPipe normalized image units, including z.
    A collapsed or nonfinite hand is rejected, never saved as missing data.
    """
    points = np.array([[p.x, p.y, p.z] for p in landmarks], dtype=np.float32)
    if points.shape != (21, 3) or not np.isfinite(points).all():
        raise ValueError("Invalid hand landmarks")
    relative = points - points[0]
    scale = float(np.linalg.norm(relative[9]))
    if scale < 1e-6:
        raise ValueError("Hand is too small or degenerate")
    return (relative / scale).reshape(63)


def extract_features(result) -> np.ndarray:
    """Use fixed Left/Right slots, independent of detector output order."""
    features = np.zeros(126, dtype=np.float32)
    occupied = set()
    if len(result.hand_landmarks) != len(result.handedness):
        raise ValueError("Missing handedness information")
    for landmarks, categories in zip(result.hand_landmarks, result.handedness):
        if not categories or categories[0].category_name not in ("Left", "Right"):
            raise ValueError("Unknown handedness")
        slot = 0 if categories[0].category_name == "Left" else 1
        if slot in occupied:
            raise ValueError("Ambiguous handedness; reposition your hands")
        occupied.add(slot)
        features[slot * 63:(slot + 1) * 63] = normalize_landmarks(landmarks)
    return features


def validate_csv(path: Path) -> dict[int, int]:
    """Check an existing dataset before appending and recover its counts."""
    counts = dict.fromkeys(LABELS, 0)
    if not path.exists() or path.stat().st_size == 0:
        return counts
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        if next(reader, None) != HEADER:
            raise ValueError(f"{path}: expected label, f0, ..., f125 header")
        for line, row in enumerate(reader, 2):
            try:
                label = int(row[0])
                values = np.asarray(row[1:], dtype=np.float32)
                if len(row) != 127 or label not in LABELS:
                    raise ValueError("wrong column count or label")
                if not np.isfinite(values).all():
                    raise ValueError("nonfinite features")
            except (ValueError, IndexError) as exc:
                raise ValueError(f"{path}: invalid row at line {line}") from exc
            counts[label] += 1
    # Avoid merging the next sample into a valid final row without a newline.
    with path.open("rb") as handle:
        handle.seek(-1, 2)
        needs_newline = handle.read(1) not in (b"\n", b"\r")
    if needs_newline:
        with path.open("a", encoding="utf-8") as handle:
            handle.write("\n")
    return counts


def prepare_model(path: Path, download: bool) -> None:
    if path.is_file() and path.stat().st_size > 0:
        return
    if not download:
        raise FileNotFoundError(
            f"Missing detector model: {path}. Run with --download-model once."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        print("Downloading MediaPipe hand detector (one-time setup)...")
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            with urllib.request.urlopen(MODEL_URL, timeout=60) as response:
                while chunk := response.read(1024 * 1024):
                    handle.write(chunk)
        if temporary.stat().st_size == 0:
            raise RuntimeError("Downloaded model is empty")
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def draw_preview(cv2, frame, result, lines) -> None:
    height, width = frame.shape[:2]
    for landmarks in result.hand_landmarks:
        points = [(int(p.x * width), int(p.y * height)) for p in landmarks]
        for start, end in CONNECTIONS:
            cv2.line(frame, points[start], points[end], (60, 220, 100), 2)
        for point in points:
            cv2.circle(frame, point, 4, (0, 200, 255), -1)
    panel_height = min(height, 34 * len(lines) + 12)
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (width, panel_height), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
    for index, line in enumerate(lines):
        scale = 0.8 if index == 0 else 0.6
        text_width = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX,
                                     scale, 2)[0][0]
        scale *= min(1.0, (width - 24) / max(1, text_width))
        color = (0, 230, 255) if index == 0 else (255, 255, 255)
        cv2.putText(frame, line, (10, 34 * (index + 1)),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, color,
                    2 if index == 0 else 1, cv2.LINE_AA)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0, help="Webcam index")
    parser.add_argument("--output", type=Path, default=ROOT / "data/gestures.csv")
    parser.add_argument("--model", type=Path,
                        default=ROOT / "models/hand_landmarker.task")
    parser.add_argument("--download-model", action="store_true")
    parser.add_argument("--reset-data", action="store_true",
                        help="Erase existing CSV samples before starting collection")
    parser.add_argument("--include-idle", action="store_true")
    parser.add_argument("--label", type=int, choices=range(6),
                        help="Select a label at startup")
    parser.add_argument("--record", action="store_true",
                        help="Start recording immediately (requires --label)")
    parser.add_argument("--guided", action="store_true",
                        help="Automatically collect unfinished labels in order")
    parser.add_argument("--countdown", type=float, default=5,
                        help="Preparation seconds before each guided pose (default: 5)")
    parser.add_argument("--sample-rate", type=float, default=10,
                        help="Maximum saved samples/second (default: 10)")
    parser.add_argument("--target", type=int, default=200,
                        help="Sample goal per label; guided mode stops each label at this total")
    args = parser.parse_args(argv)
    if not math.isfinite(args.sample_rate) or not 0 < args.sample_rate <= 30:
        parser.error("--sample-rate must be greater than 0 and at most 30")
    if args.target < 1 or args.camera < 0:
        parser.error("--target must be positive; --camera must be nonnegative")
    if not math.isfinite(args.countdown) or args.countdown < 1:
        parser.error("--countdown must be at least 1 second")
    if args.guided and (args.label is not None or args.record):
        parser.error("--guided cannot be combined with --label or --record")
    if args.label == 5 and not args.include_idle:
        parser.error("label 5 requires --include-idle")
    if args.record and args.label is None:
        parser.error("--record requires --label")
    return args


def run(args) -> None:
    try:
        import cv2
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
    except ImportError as exc:
        raise RuntimeError(
            "Install dependencies: python -m pip install opencv-contrib-python "
            "mediapipe numpy (use desktop OpenCV, not headless)."
        ) from exc

    if args.reset_data:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow(HEADER)
        print(f"Cleared existing samples from {args.output}", flush=True)
    counts = validate_csv(args.output)
    active_labels = range(6 if args.include_idle else 5)
    if args.guided and all(counts[i] >= args.target for i in active_labels):
        print("All labels already meet the target. Use a larger --target for more samples.")
        return
    prepare_model(args.model, args.download_model)
    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(args.model)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6,
    )
    cap = cv2.VideoCapture(args.camera)
    window = "Spellcasting - Gesture Data Collection"
    try:
        if not cap.isOpened():
            raise RuntimeError("Cannot open webcam. Check permissions or --camera.")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        new_file = not args.output.exists() or args.output.stat().st_size == 0
        with vision.HandLandmarker.create_from_options(options) as detector:
            with args.output.open("a", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                if new_file:
                    writer.writerow(HEADER)
                    handle.flush()
                cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
                print("0-4: select label; Space: record/pause; S: one sample; Q: quit")
                print(f"Appending samples to {args.output}")
                label, recording = args.label, args.record
                guided = (
                    GuidedSession(counts, active_labels, args.target,
                                  args.countdown, time.monotonic())
                    if args.guided else None
                )
                if guided:
                    print("Guided session: follow the pose shown on screen. "
                          "Countdown frames are not saved.", flush=True)
                mouse_actions = deque()
                button_regions = []

                def on_mouse(event, x, y, flags, userdata):
                    if event == cv2.EVENT_LBUTTONDOWN:
                        for x0, y0, x1, y1, action in button_regions:
                            if x0 <= x < x1 and y0 <= y < y1:
                                mouse_actions.append(action)
                                break

                cv2.setMouseCallback(window, on_mouse)
                last_saved, timestamp = float("-inf"), -1
                previous_frame = time.monotonic()
                fps = 0.0
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        raise RuntimeError("Webcam stopped returning frames.")
                    frame = cv2.flip(frame, 1)
                    rgb = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    timestamp = max(timestamp + 1, time.monotonic_ns() // 1_000_000)
                    result = detector.detect_for_video(
                        mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), timestamp
                    )
                    now = time.monotonic()
                    elapsed = now - previous_frame
                    fps = 0.9 * fps + 0.1 / max(elapsed, 1e-6)
                    previous_frame = now
                    if guided:
                        guided.tick(now)
                        label, recording = guided.label, guided.ready(now)
                    hand_count = len(result.hand_landmarks)
                    features, reason = None, "Select a label"
                    if label is not None:
                        required = 2 if label == 4 else 1
                        if label != 5 and hand_count != required:
                            reason = f"Need exactly {required} hand(s); found {hand_count}"
                        else:
                            try:
                                features = extract_features(result)
                                reason = "Ready (labels are manual; check your pose)"
                            except ValueError as exc:
                                reason = str(exc)
                    active_labels = range(6 if args.include_idle else 5)
                    lines = [
                        f"Label: {LABELS.get(label, 'None')} | "
                        f"{'RECORDING' if recording else 'PAUSED'} | FPS: {fps:.1f}",
                        "0 Palm | 1 Fist | 2 Peace | 3 Point | 4 Both open"
                        + (" | 5 Idle" if args.include_idle else ""),
                        "Space: record/pause | S: single sample | Q/Esc: quit",
                        "Samples: " + "  ".join(
                            f"{i}: {counts[i]}/{args.target}" for i in active_labels
                        ),
                        reason,
                    ]
                    if guided:
                        lines = [
                            guided.status(now),
                            (POSE_INSTRUCTIONS[guided.label] if not guided.done
                             else "All five gestures collected. Your data is saved."),
                            "Labels follow the prompt; the pose is not classified yet.",
                            "Samples: " + "  ".join(
                                f"{i}: {counts[i]}/{args.target}" for i in active_labels
                            ),
                            "Complete! Click QUIT." if guided.done else reason,
                        ]
                    draw_preview(cv2, frame, result, lines)
                    # Mouse controls avoid relying on keyboard focus on macOS.
                    height, width = frame.shape[:2]
                    button_regions.clear()
                    labels = [] if guided else list(active_labels)
                    label_width = width // max(1, len(labels))
                    for index, value in enumerate(labels):
                        x0 = index * label_width
                        x1 = (index + 1) * label_width - 4
                        y0, y1 = height - 90, height - 50
                        color = (70, 130, 70) if value == label else (65, 65, 65)
                        cv2.rectangle(frame, (x0, y0), (x1, y1), color, -1)
                        caption = ["0 Palm", "1 Fist", "2 Peace", "3 Point",
                                   "4 Both Open", "5 Idle"][value]
                        cv2.putText(frame, caption, (x0 + 8, y0 + 26),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    (255, 255, 255), 1, cv2.LINE_AA)
                        button_regions.append((x0, y0, x1, y1, ord("0") + value))
                    actions = [("PAUSE" if recording else "RECORD", ord(" ")),
                               ("SAVE ONE", ord("s")), ("QUIT", ord("q"))]
                    if guided:
                        actions = [
                            ("RESUME" if guided.paused else "PAUSE", ord(" ")),
                            ("QUIT", ord("q")),
                        ]
                        if guided.done:
                            actions = [("QUIT", ord("q"))]
                    for index, (caption, action) in enumerate(actions):
                        x0 = index * width // len(actions)
                        x1 = (index + 1) * width // len(actions) - 4
                        y0, y1 = height - 45, height - 5
                        color = (40, 130, 40) if index == 0 else (65, 65, 65)
                        cv2.rectangle(frame, (x0, y0), (x1, y1), color, -1)
                        cv2.putText(frame, caption, (x0 + 10, y0 + 27),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                                    (255, 255, 255), 2, cv2.LINE_AA)
                        button_regions.append((x0, y0, x1, y1, action))
                    cv2.imshow(window, frame)
                    key = cv2.waitKey(1) & 0xFF
                    if mouse_actions:
                        key = mouse_actions.popleft()
                    if key in (27, ord("q"), ord("Q")):
                        break
                    if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                        break
                    if guided:
                        if key == ord(" "):
                            guided.toggle_pause(now)
                            continue
                        # Guided sessions ignore manual labeling and single saves.
                        key = 255
                    if ord("0") <= key <= ord("5"):
                        chosen = key - ord("0")
                        if chosen != 5 or args.include_idle:
                            label, recording = chosen, False
                            print(f"Selected {LABELS[label]}; click RECORD.", flush=True)
                        continue  # Do not save the frame captured before selection.
                    if key == ord(" ") and label is not None:
                        recording = not recording
                        print("Recording started." if recording else "Recording paused.",
                              flush=True)
                        continue
                    save_single = key in (ord("s"), ord("S"))
                    if features is not None and (recording or save_single):
                        if save_single or now - last_saved >= 1 / args.sample_rate:
                            writer.writerow([label] + features.tolist())
                            handle.flush()
                            counts[label] += 1
                            if counts[label] % 10 == 0:
                                print(f"Label {label}: {counts[label]} saved samples",
                                      flush=True)
                            last_saved = now
                            if counts[label] == args.target:
                                print(f"Label {label} reached {args.target} samples.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Saved totals: " + ", ".join(f"{i}={n}" for i, n in counts.items()))


def main() -> int:
    args = parse_args()
    try:
        run(args)
    except KeyboardInterrupt:
        print("\nCollection stopped.")
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
