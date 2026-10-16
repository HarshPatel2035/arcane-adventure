"""Conservative, optional motion recognizer using normalized mirrored palm centers.
No learned weights, camera libraries or Pygame are needed here.
"""

from collections import deque
import math


class MotionRecognizer:
    def __init__(self):
        self.reset()

    def reset(self):
        self.path = deque()
        self.pose = None
        self.anchor = None
        self.hold_since = None
        self.charged = False
        self.last = dict.fromkeys(("sweep", "circle", "charge"), float("-inf"))
        self.previous_time = None

    def process(self, x, y, pose, confidence, hand_count, now):
        if self.previous_time is not None and (
            now <= self.previous_time or now - self.previous_time > 0.3
        ):
            self.path.clear()
            self.hold_since = None
            self.anchor = None
            self.charged = False
        self.previous_time = now
        if (
            hand_count != 1
            or x is None
            or y is None
            or not all(math.isfinite(v) for v in (x, y, confidence, now))
            or confidence < 0.8
            or pose not in (0, 1, 2)
        ):
            self.path.clear()
            self.pose = None
            self.hold_since = None
            self.anchor = None
            self.charged = False
            return None
        if pose != self.pose:
            self.path.clear()
            self.pose = pose
            self.anchor = (x, y)
            self.hold_since = now
            self.charged = False
        if self.anchor is None:
            self.anchor = (x, y)
            self.hold_since = now
        self.path.append((now, x, y))
        while self.path and now - self.path[0][0] > 1.8:
            self.path.popleft()
        if pose == 1:
            if math.hypot(x - self.anchor[0], y - self.anchor[1]) > 0.06:
                self.anchor = (x, y)
                self.hold_since = now
            if (
                not self.charged
                and now - self.hold_since >= 1.3
                and now - self.last["charge"] >= 3
            ):
                self.charged = True
                self.last["charge"] = now
                return "charge"
        if pose == 0 and len(self.path) >= 5 and now - self.last["sweep"] >= 3:
            recent = [p for p in self.path if now - p[0] <= 1.0]
            duration = now - recent[0][0]
            dx = x - recent[0][1]
            length = sum(abs(b[1] - a[1]) for a, b in zip(recent, recent[1:]))
            if (
                0.22 <= duration <= 1
                and abs(dx) >= 0.34
                and max(p[2] for p in recent) - min(p[2] for p in recent) < 0.12
                and length
                and abs(dx) / length > 0.85
            ):
                self.last["sweep"] = now
                self.path.clear()
                return "sweep"
        if (
            pose == 2
            and len(self.path) >= 14
            and now - self.path[0][0] >= 0.6
            and now - self.last["circle"] >= 3
        ):
            points = list(self.path)
            xs = [p[1] for p in points]
            ys = [p[2] for p in points]
            if max(xs) - min(xs) < 0.14 or max(ys) - min(ys) < 0.14:
                return None
            if math.hypot(xs[0] - x, ys[0] - y) > 0.07:
                return None
            cx = sum(xs) / len(xs)
            cy = sum(ys) / len(ys)
            angles = [math.atan2(py - cy, px - cx) for px, py in zip(xs, ys)]
            turns = [
                (b - a + math.pi) % math.tau - math.pi
                for a, b in zip(angles, angles[1:])
            ]
            radii = [math.hypot(px - cx, py - cy) for px, py in zip(xs, ys)]
            mean = sum(radii) / len(radii)
            variation = (
                (sum((r - mean) ** 2 for r in radii) / len(radii)) ** 0.5 / mean
                if mean
                else 1
            )
            if (
                abs(sum(turns)) > 5.2
                and abs(sum(turns)) / max(0.001, sum(abs(t) for t in turns)) > 0.8
                and variation < 0.45
            ):
                self.last["circle"] = now
                self.path.clear()
                return "circle"
        return None
