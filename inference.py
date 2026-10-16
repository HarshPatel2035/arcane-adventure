"""Live gesture recognition for the Gesture-Controlled Spellcasting Game.

Run: ./.venv/bin/python inference.py
Uses the existing local detector, trained weights and matching JSON metadata.
No download, training or data collection occurs. Q/Esc or the QUIT button exits.

Game integration:
    with GestureRecognizer() as recognizer:
        prediction = recognizer.process_frame(camera_bgr_frame)
        if prediction.action is not None:
            cast_spell(prediction.action)

process_frame accepts an UNMIRRORED OpenCV BGR frame and returns a prediction
with a mirrored display frame, stable gesture_id, confidence and optional
one-shot action. Holding a gesture does not repeatedly emit the same action;
remove your hands, show idle, or switch to another stable gesture to rearm.
"""
from __future__ import annotations

import argparse
from collections import Counter, deque
from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys
import time

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import torch

from collect_data import LABELS, draw_preview, extract_features
from train_model import FEATURE_CONTRACT, GestureClassifier

ROOT = Path(__file__).resolve().parent
SPELLS = {0: 'Fireball', 1: 'Shield', 2: 'Lightning', 3: 'Telekinesis', 4: 'Ultimate'}


class GestureSmoother:
    """Require a strict majority in a full window and a matching current vote."""

    def __init__(self, window=5, threshold=0.8):
        if window < 1 or window % 2 == 0:
            raise ValueError('Smoothing window must be a positive odd number.')
        if not math.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError('Confidence threshold must be in [0, 1].')
        self.window, self.threshold = window, threshold
        self.history = deque(maxlen=window)

    def reset(self):
        self.history.clear()

    def update(self, label, confidence):
        valid = label is not None and math.isfinite(confidence) and confidence > self.threshold
        vote = label if valid else None
        self.history.append((vote, confidence if valid else 0.0))
        if len(self.history) < self.window or vote is None:
            return None, 0.0
        winner, count = Counter(item[0] for item in self.history).most_common(1)[0]
        if winner is None or count <= self.window // 2 or winner != vote:
            return None, 0.0
        scores = [score for candidate, score in self.history if candidate == winner]
        return winner, sum(scores) / len(scores)


@dataclass
class GesturePrediction:
    gesture_id: int | None = None
    gesture_name: str = 'No gesture'
    confidence: float = 0.0
    raw_gesture_id: int | None = None
    raw_confidence: float = 0.0
    action: str | None = None
    status: str = 'No hands detected'
    hand_count: int = 0
    latency_ms: float = 0.0
    display_frame: np.ndarray | None = None
    landmarks: object | None = None


class GestureRecognizer:
    """CPU classifier + hand detection; reusable without an OpenCV window."""

    def __init__(self, model_path=ROOT / 'models/gesture_model.pth',
                 detector_path=ROOT / 'models/hand_landmarker.task',
                 threshold=0.8, window=5, cooldown=0.6, threads=2):
        if not math.isfinite(cooldown) or cooldown < 0 or threads < 1:
            raise ValueError('Cooldown must be nonnegative and threads positive.')
        self.smoother = GestureSmoother(window, threshold)
        self.cooldown = cooldown
        self.last_emitted_label = None
        self.last_action_time = float('-inf')
        self.previous_hand_count = None
        self.timestamp = -1
        self.detector = None
        torch.set_num_threads(threads)
        model_path, detector_path = Path(model_path), Path(detector_path)
        metadata_path = model_path.with_suffix('.json')
        for path in (model_path, metadata_path, detector_path):
            if not path.is_file():
                raise FileNotFoundError(f'Missing {path}. Run training or collector setup first.')
        metadata = json.loads(metadata_path.read_text())
        self.class_ids = metadata.get('class_ids', [])
        if (not self.class_ids or any(type(i) is not int or i not in LABELS for i in self.class_ids)
                or len(set(self.class_ids)) != len(self.class_ids)):
            raise ValueError('Invalid class IDs in model metadata.')
        expected = {'input': 126, 'hidden': [128, 64], 'dropout': 0.3,
                    'num_classes': len(self.class_ids)}
        if metadata.get('architecture') != expected:
            raise ValueError('Model architecture does not match this project. Retrain the model.')
        if metadata.get('feature_contract') != FEATURE_CONTRACT:
            raise ValueError('Model preprocessing does not match the collector. Retrain the model.')
        self.model = GestureClassifier(len(self.class_ids))
        self.model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
        self.model.eval()
        options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(detector_path)),
            running_mode=vision.RunningMode.VIDEO, num_hands=2,
            min_hand_detection_confidence=0.6, min_hand_presence_confidence=0.6,
            min_tracking_confidence=0.6,
        )
        self.detector = vision.HandLandmarker.create_from_options(options)

    def close(self):
        if self.detector is not None:
            self.detector.close()
            self.detector = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def reset(self):
        self.smoother.reset()
        self.last_emitted_label = None
        self.previous_hand_count = None

    def classify_landmarks(self, result, now=None):
        """Classify a MediaPipe result from MIRRORED input; useful for testing."""
        now = time.monotonic() if now is None else now
        count = len(result.hand_landmarks)
        prediction = GesturePrediction(hand_count=count, landmarks=result)
        if count != self.previous_hand_count:
            self.smoother.reset()
            self.previous_hand_count = count
        if count == 0:
            self.smoother.reset()
            self.last_emitted_label = None
            return prediction
        try:
            features = extract_features(result)
        except ValueError as exc:
            self.smoother.reset()
            prediction.status = str(exc)
            return prediction
        with torch.inference_mode():
            probabilities = torch.softmax(self.model(torch.from_numpy(features).unsqueeze(0)), dim=1)
            probability, output_index = probabilities[0].max(dim=0)
        label, confidence = self.class_ids[output_index.item()], probability.item()
        prediction.raw_gesture_id, prediction.raw_confidence = label, confidence
        if not math.isfinite(confidence):
            self.smoother.reset()
            prediction.status = 'Invalid model confidence'
            return prediction
        # Keep hand-count rules identical to the collector's labels.
        if label != 5 and count != (2 if label == 4 else 1):
            self.smoother.reset()
            prediction.status = 'Pose does not match the required number of hands'
            return prediction
        stable, score = self.smoother.update(label, confidence)
        if stable is None:
            prediction.status = ('Low confidence: adjust your pose'
                                 if confidence <= self.smoother.threshold
                                 else 'Stabilizing: hold your pose')
            return prediction
        prediction.gesture_id, prediction.confidence = stable, score
        prediction.gesture_name = LABELS[stable].split(' / ')[0]
        if stable == 5:
            self.last_emitted_label = None
            prediction.status = 'Idle: no spell'
        elif stable != self.last_emitted_label and now - self.last_action_time >= self.cooldown:
            prediction.action = SPELLS[stable]
            self.last_emitted_label, self.last_action_time = stable, now
            prediction.status = f'Recognized: {prediction.action}'
        else:
            prediction.status = 'Holding pose; remove hands or change gesture to cast again'
        return prediction

    def process_frame(self, frame):
        if self.detector is None:
            raise RuntimeError('Recognizer is closed.')
        if (not isinstance(frame, np.ndarray) or frame.dtype != np.uint8
                or frame.ndim != 3 or frame.shape[2] != 3
                or not frame.shape[0] or not frame.shape[1]):
            raise ValueError('Expected a nonempty uint8 OpenCV BGR frame.')
        started = time.perf_counter()
        mirrored = cv2.flip(frame, 1)
        rgb = np.ascontiguousarray(cv2.cvtColor(mirrored, cv2.COLOR_BGR2RGB))
        self.timestamp = max(self.timestamp + 1, time.monotonic_ns() // 1_000_000)
        result = self.detector.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), self.timestamp
        )
        prediction = self.classify_landmarks(result)
        prediction.display_frame = mirrored
        prediction.latency_ms = (time.perf_counter() - started) * 1000
        return prediction


def render_prediction(prediction, fps, threshold, window):
    if prediction.display_frame is None:
        raise ValueError('Prediction has no display frame; use process_frame first.')
    frame = prediction.display_frame.copy()
    stable = (f'{prediction.gesture_name} | Confidence: {prediction.confidence:.1%}'
              if prediction.gesture_id is not None else 'No accepted gesture')
    raw = ('None' if prediction.raw_gesture_id is None else
           LABELS[prediction.raw_gesture_id].split(' / ')[0])
    lines = [stable, prediction.status,
             f'Raw: {raw} ({prediction.raw_confidence:.1%}) | Hands: {prediction.hand_count}',
             f'FPS: {fps:.1f} | Detection + classification: {prediction.latency_ms:.1f} ms',
             f'Threshold: >{threshold:.0%} | Smoothing: {window} frames | Q/Esc: quit']
    draw_preview(cv2, frame, prediction.landmarks, lines)
    height, width = frame.shape[:2]
    cv2.rectangle(frame, (width - 140, height - 48), (width - 8, height - 8), (50, 50, 160), -1)
    cv2.putText(frame, 'QUIT', (width - 115, height - 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return frame


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--model', type=Path, default=ROOT / 'models/gesture_model.pth')
    parser.add_argument('--detector', type=Path, default=ROOT / 'models/hand_landmarker.task')
    parser.add_argument('--threshold', type=float, default=0.8)
    parser.add_argument('--window', type=int, default=5)
    parser.add_argument('--cooldown', type=float, default=0.6)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=480)
    args = parser.parse_args(argv)
    if args.camera < 0 or min(args.threads, args.width, args.height) < 1:
        parser.error('Camera must be nonnegative; threads and dimensions positive.')
    if not math.isfinite(args.cooldown) or args.cooldown < 0:
        parser.error('Cooldown must be finite and nonnegative.')
    try:
        GestureSmoother(args.window, args.threshold)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def run(args):
    title = 'Spellcasting - Live Gesture Recognition'
    cap = None
    try:
        with GestureRecognizer(args.model, args.detector, args.threshold,
                               args.window, args.cooldown, args.threads) as recognizer:
            cap = cv2.VideoCapture(args.camera)
            if not cap.isOpened():
                raise RuntimeError('Cannot open webcam. Close the collector and check camera permissions or --camera 1.')
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
            cap.set(cv2.CAP_PROP_FPS, 30)
            cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)
            quit_requested = False
            frame_size = [args.width, args.height]

            def on_mouse(event, x, y, flags, userdata):
                nonlocal quit_requested
                if (event == cv2.EVENT_LBUTTONDOWN
                        and frame_size[0] - 140 <= x < frame_size[0] - 8
                        and frame_size[1] - 48 <= y < frame_size[1] - 8):
                    quit_requested = True

            cv2.setMouseCallback(title, on_mouse)
            print('Live recognition ready. Show one hand for labels 0-3, both for label 4.', flush=True)
            if 5 not in recognizer.class_ids:
                print('No idle class trained: unrelated hand poses may still match a spell.', flush=True)
            previous, fps = time.monotonic(), 0.0
            while not quit_requested:
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError('Webcam stopped returning frames.')
                prediction = recognizer.process_frame(frame)
                now = time.monotonic()
                measured = 1 / max(now - previous, 1e-6)
                fps = measured if fps == 0 else 0.9 * fps + 0.1 * measured
                previous = now
                frame_size[:] = [frame.shape[1], frame.shape[0]]
                cv2.imshow(title, render_prediction(prediction, fps, args.threshold, args.window))
                if prediction.action is not None:
                    print(f'{prediction.gesture_name}: {prediction.action} ({prediction.confidence:.1%})', flush=True)
                if cv2.waitKey(1) & 0xFF in (27, ord('q'), ord('Q')):
                    break
                if cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                    break
    finally:
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()


def main():
    args = parse_args()
    try:
        run(args)
    except KeyboardInterrupt:
        print('\nRecognition stopped.')
    except Exception as exc:
        print(f'Recognition failed: {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
