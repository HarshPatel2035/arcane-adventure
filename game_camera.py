"""Private camera helper for gesture_game.py; run by the game automatically.

Uses JSON lines over pipes so OpenCV and Pygame never share a process on macOS.
Only the latest camera image is used by the game. stdout is the data channel;
diagnostics stay on stderr. --synthetic/--max-frames support camera-free checks.
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import sys
import threading
import time


def palm_position(hands):
    """Average palm centers in the mirrored preview; two hands steer together."""
    centers = [
        sum(hand[i].x for i in (0, 5, 9, 13, 17)) / 5
        for hand in hands
        if len(hand) >= 18
    ]
    centers = [x for x in centers if math.isfinite(x)]
    return max(0.0, min(1.0, sum(centers) / len(centers))) if centers else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--motion", action="store_true")
    parser.add_argument("--max-frames", type=int, default=0)
    args = parser.parse_args()
    lock = threading.Lock()
    stopped = threading.Event()
    control = {"generation": 0, "reset": False}

    def commands():
        for line in sys.stdin:
            try:
                message = json.loads(line)
            except ValueError:
                continue
            if message.get("command") == "stop":
                stopped.set()
                break
            if message.get("command") == "reset":
                with lock:
                    control["generation"] = int(message["generation"])
                    control["reset"] = True
        stopped.set()

    def emit(packet):
        print(json.dumps(packet, separators=(",", ":")), flush=True)

    threading.Thread(target=commands, daemon=True).start()
    cap = None
    try:
        import cv2
        import numpy as np
        from inference import GestureRecognizer
        from collect_data import CONNECTIONS
        from motion_controls import MotionRecognizer

        motion = MotionRecognizer()

        with GestureRecognizer() as recognizer:
            if not args.synthetic:
                cap = cv2.VideoCapture(args.camera)
                if not cap.isOpened():
                    raise RuntimeError(
                        "Cannot open camera. Close other camera apps or try --camera 1."
                    )
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_FPS, 30)
            frames = 0
            while not stopped.is_set():
                started = time.monotonic()
                with lock:
                    generation, reset = control["generation"], control["reset"]
                    control["reset"] = False
                if reset:
                    recognizer.reset()
                    motion.reset()
                if args.synthetic:
                    ok, frame = True, np.zeros((480, 640, 3), dtype=np.uint8)
                else:
                    ok, frame = cap.read()
                if not ok:
                    raise RuntimeError(
                        "Camera stopped returning frames; keyboard controls remain available."
                    )
                prediction = recognizer.process_frame(frame)
                hands = prediction.landmarks.hand_landmarks
                hand_x = palm_position(hands)
                hand_y = (
                    sum(sum(hand[i].y for i in (0, 5, 9, 13, 17)) / 5 for hand in hands)
                    / len(hands)
                    if hands
                    else None
                )
                motion_action = (
                    motion.process(
                        hand_x,
                        hand_y,
                        prediction.gesture_id,
                        prediction.confidence,
                        prediction.hand_count,
                        time.monotonic(),
                    )
                    if args.motion
                    else None
                )
                display = prediction.display_frame.copy()
                h, w = display.shape[:2]
                for landmarks in prediction.landmarks.hand_landmarks:
                    points = [(int(p.x * w), int(p.y * h)) for p in landmarks]
                    for a, b in CONNECTIONS:
                        cv2.line(display, points[a], points[b], (80, 235, 160), 2)
                    for point in points:
                        cv2.circle(display, point, 3, (0, 210, 255), -1)
                display = cv2.resize(display, (362, 272))
                ok, encoded = cv2.imencode(
                    ".jpg", display, [cv2.IMWRITE_JPEG_QUALITY, 82]
                )
                if not ok:
                    raise RuntimeError("Could not encode webcam preview.")
                emit(
                    {
                        "generation": generation,
                        "observed_at": time.monotonic(),
                        "gesture_id": prediction.gesture_id,
                        "gesture_name": prediction.gesture_name,
                        "confidence": prediction.confidence,
                        "action": prediction.action,
                        "status": prediction.status,
                        "hand_count": prediction.hand_count,
                        "hand_x": hand_x,
                        "hand_y": hand_y,
                        "motion": motion_action,
                        "latency_ms": prediction.latency_ms,
                        "image": base64.b64encode(encoded.tobytes()).decode("ascii"),
                    }
                )
                frames += 1
                if args.max_frames and frames >= args.max_frames:
                    break
                stopped.wait(max(0, 1 / 30 - (time.monotonic() - started)))
    except BrokenPipeError:
        pass  # The game closed while the helper was finishing a frame.
    except Exception as exc:
        try:
            emit({"error": str(exc)})
        except BrokenPipeError:
            pass
        return 1
    finally:
        if cap is not None:
            cap.release()
    return 0


if __name__ == "__main__":
    sys.exit(main())
