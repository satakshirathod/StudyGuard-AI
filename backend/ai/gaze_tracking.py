"""Real-time gaze / attention tracking (Phase 3).

Pipeline for each processed frame, given the primary face's 478 MediaPipe
landmarks:

1. **Iris signal** — for each eye, how far the iris sits from the centre of
   the eye region (relative to eye width / height). Positive horizontal
   deviation = iris toward the *image right* = the student looking toward
   **their** left (MediaPipe eye/iris landmarks use the subject's
   perspective, so the subject's left eye appears on the image right).
2. **Head orientation** — ``cv2.solvePnP`` head pose estimation using six
   stable face landmarks and the Astrakhantsev-style 3D face model. Positive
   yaw = nose turned toward image right (student's left); positive pitch =
   nose turned down (and is negated internally so the vertical signal reads
   positive when the student looks up). Signs are pinned by the unit-test
   projection checks.
3. **Calibration** — optional in-memory baseline of the iris offsets so a
   student's natural "resting" gaze counts as centre.
4. **Classification** — weighted combination of the iris and head signals
   with a dominance check between horizontal and vertical deviation.
5. **Temporal smoothing** — a debounced majority-vote ``TemporalStateTracker``
   prevents flicker; a missing face lasting ``away_duration`` frames raises
   AWAY.

Attention score (0–100) and confidence (0–1) accompany every reading; when
there is no usable facial data the reading degrades gracefully to UNKNOWN /
AWAY and the numeric scores become ``None`` (never fake values).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from ai.face_detection import FaceData
from ai.temporal import TemporalStateTracker
from config import Settings
from utils.logger import get_logger

_log = get_logger("studygard.ai.gaze")

# Category labels — public API used by the status endpoint and the frontend.
DIRECTION_AT_SCREEN = "LOOKING_AT_SCREEN"
DIRECTION_LEFT = "LOOKING_LEFT"
DIRECTION_RIGHT = "LOOKING_RIGHT"
DIRECTION_UP = "LOOKING_UP"
DIRECTION_DOWN = "LOOKING_DOWN"
DIRECTION_AWAY = "AWAY"
DIRECTION_UNKNOWN = "UNKNOWN"

# Head pose model points (object-space, millimetres) mapped to face-puppet
# landmark indices. Positive Z approaches the camera; X is positive toward the
# person's right (image left); Y is positive downward — Astrakhantsev pattern.
HEAD_MODEL_INDICES: Tuple[int, int, int, int, int, int] = (1, 152, 33, 263, 61, 291)
HEAD_MODEL_POINTS: Tuple[Tuple[float, float, float], ...] = (
    (0.0, 0.0, 0.0),          # 1    nose bridge
    (0.0, -63.6, -12.5),      # 152  chin
    (-43.3, 32.7, -26.0),     # 33   left eye outer (subject's left, image right)
    (43.3, 32.7, -26.0),      # 263  right eye outer (subject's right, image left)
    (-28.9, -28.9, -24.1),    # 61   left mouth corner
    (28.9, -28.9, -24.1),     # 291  right mouth corner
)


@dataclass
class GazeReading:
    """Raw, per-frame gaze measurements before calibration/smoothing."""

    valid: bool = False
    iris_offset_x: Optional[float] = None  # [-1,1] + = student looking to their left
    iris_offset_y: Optional[float] = None  # [-1,1] + = student looking up
    iris_quality: float = 0.0              # fraction of the two-eye signal usable (0, 0.5, 1)
    iris_ok: bool = False
    head_yaw_deg: Optional[float] = None   # + = nose toward image right (student's left)
    head_pitch_deg: Optional[float] = None  # + = nose pointed down (raw solvePnP value)
    head_roll_deg: Optional[float] = None
    head_ok: bool = False
    error: Optional[str] = None


@dataclass
class GazeResult:
    """Final processed gaze output (JSON-serialisable via :meth:`to_dict`)."""

    direction: str = DIRECTION_UNKNOWN
    attention_score: Optional[int] = None  # 0..100, None when nothing usable
    confidence: Optional[float] = None     # 0..1
    head_yaw_deg: Optional[float] = None
    head_pitch_deg: Optional[float] = None
    head_roll_deg: Optional[float] = None
    iris_offset_x: Optional[float] = None
    iris_offset_y: Optional[float] = None
    deviation_x: Optional[float] = None  # combined signed signal, + = left
    deviation_y: Optional[float] = None  # combined signed signal, + = up
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "direction": self.direction,
            "attention_score": self.attention_score,
            "confidence": self.confidence,
            "head_yaw_deg": None if self.head_yaw_deg is None else round(self.head_yaw_deg, 1),
            "head_pitch_deg": None if self.head_pitch_deg is None else round(self.head_pitch_deg, 1),
            "head_roll_deg": None if self.head_roll_deg is None else round(self.head_roll_deg, 1),
            "iris_offset_x": None if self.iris_offset_x is None else round(self.iris_offset_x, 3),
            "iris_offset_y": None if self.iris_offset_y is None else round(self.iris_offset_y, 3),
            "deviation_x": None if self.deviation_x is None else round(self.deviation_x, 3),
            "deviation_y": None if self.deviation_y is None else round(self.deviation_y, 3),
            "error": self.error,
        }


def _clamp(value: float, low: float = -1.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


class GazeEstimator:
    """Stateless computation of gaze measurements from one face's landmarks."""

    def __init__(self, focal_length: Optional[float], frame_w: int, frame_h: int) -> None:
        if frame_w <= 0 or frame_h <= 0:
            raise ValueError("frame_w/frame_h must be positive")
        self._frame_w = int(frame_w)
        self._frame_h = int(frame_h)
        self._focal = float(focal_length) if focal_length and focal_length > 0 else float(frame_w)
        self._camera_matrix = np.array(
            [[self._focal, 0, frame_w / 2.0], [0, self._focal, frame_h / 2.0], [0, 0, 1]],
            dtype=np.float64,
        )
        self._dist_coeffs: np.ndarray = np.zeros((5, 1), dtype=np.float64)
        self._model_points = np.array(HEAD_MODEL_POINTS, dtype=np.float64)

    # ── iris signal ─────────────────────────────────────────────────────────

    @staticmethod
    def _eye_metrics(
        eye_pts: Sequence[Tuple[float, float]], iris: Optional[Tuple[float, float]]
    ) -> Tuple[Optional[float], Optional[float], float]:
        """Return (x_offset_norm, y_offset_norm, usable) for one eye.

        Offsets are the signed distance of the iris from the eye-region centre,
        expressed as a [−0.5, 0.5] fraction of the region's width / height.
        Horizontal is *positive toward image right*, vertical positive toward
        the bottom of the image. ``usable`` is 1.0 when computed, else 0.0.
        """
        if not eye_pts or len(eye_pts) < 3 or iris is None:
            return None, None, 0.0
        xs = [p[0] for p in eye_pts]
        ys = [p[1] for p in eye_pts]
        w = max(xs) - min(xs)
        h = max(ys) - min(ys)
        if w < 1e-6 or h < 1e-6:
            return None, None, 0.0
        cx, cy = (max(xs) + min(xs)) / 2.0, (max(ys) + min(ys)) / 2.0
        x_off = (iris[0] - cx) / w          # + → image right
        y_off = (iris[1] - cy) / h          # + → image bottom
        return x_off, y_off, 1.0

    def compute_reading(self, face: FaceData) -> GazeReading:
        """Compute a :class:`GazeReading` from one face, or an invalid reading."""
        reading = GazeReading()
        try:
            iris_ok = face.left_iris is not None and face.right_iris is not None

            x_offs: List[float] = []
            y_offs: List[float] = []
            for eye_pts, iris in (
                (face.left_eye, face.left_iris),
                (face.right_eye, face.right_iris),
            ):
                xo, yo, usable = self._eye_metrics(eye_pts, iris)
                if usable and xo is not None and yo is not None:
                    x_offs.append(xo)
                    y_offs.append(yo)
            reading.iris_quality = len(x_offs) / 2.0

            if x_offs and y_offs:
                x_off = float(np.mean(x_offs))  # image-right positive, both eyes agree
                y_off = float(np.mean(y_offs))  # image-bottom positive
                sx = _clamp(x_off * 2.0)        # → [−1, 1];   + = student's LEFT
                sy = _clamp(-y_off * 2.0)       # → [−1, 1];   + = student looks UP
                reading.iris_offset_x = sx
                reading.iris_offset_y = sy
                reading.iris_ok = True

            self._compute_head_pose(face, reading)

            if not reading.iris_ok and not reading.head_ok:
                reading.error = "no usable iris or head landmarks"
            elif reading.iris_ok:
                reading.valid = True
            else:
                reading.valid = True  # head-only readings still usable
        except Exception as exc:  # defensive: never crash the video loop
            _log.warning("gaze reading failed: %s", exc)
            reading.error = str(exc)
        return reading

    def _compute_head_pose(self, face: FaceData, reading: GazeReading) -> None:
        """Fill the head yaw / pitch / roll of ``reading`` via solvePnP."""
        if len(face.landmarks) <= max(HEAD_MODEL_INDICES):
            reading.error = "landmarks too sparse for head pose"
            return
        try:
            obj_pts = self._model_points.astype(np.float64)
            img_pts = np.array(
                [
                    (
                        face.landmarks[i][0] * self._frame_w,
                        face.landmarks[i][1] * self._frame_h,
                    )
                    for i in HEAD_MODEL_INDICES
                ],
                dtype=np.float64,
            ).reshape(-1, 1, 2)

            ok, rvec, _tvec = cv2.solvePnP(
                obj_pts, img_pts, self._camera_matrix, self._dist_coeffs
            )
            if not ok:
                reading.error = "solvePnP failed"
                return
            values = rvec.ravel()  # robust to (3,1), (3,) or (3,1,1) output shapes
            rx, ry, rz = (float(v) * 180.0 / math.pi for v in values)
            reading.head_yaw_deg = ry      # rotation about camera Y; + → student's left
            reading.head_pitch_deg = rx    # rotation about camera X; + → nose down
            reading.head_roll_deg = rz
            reading.head_ok = True
        except cv2.error as exc:
            reading.error = f"head pose error: {exc}"


class GazeCalibrator:
    """Collects a small baseline of iris offsets used as the 'resting gaze'."""

    def __init__(self) -> None:
        self._required = 30
        self._accept_range = 0.30
        self._collected: List[Tuple[float, float]] = []
        self._attempts = 0
        self._attempt_cap = 0
        self._collecting = False
        self._baseline_x: Optional[float] = None
        self._baseline_y: Optional[float] = None
        self._last_error: Optional[str] = None

    def start(self, required: int, accept_range: float) -> None:
        """Begin a new calibration run."""
        if required < 1:
            raise ValueError("required samples must be >= 1")
        self._required = int(required)
        self._accept_range = float(accept_range)
        self._collected = []
        self._attempts = 0
        self._attempt_cap = max(self._required * 4, 40)
        self._collecting = True
        self._last_error = None
        _log.info("gaze calibration started (need %d samples)", self._required)

    def cancel(self) -> None:
        self._collecting = False
        self._last_error = None
        _log.info("gaze calibration cancelled")

    @property
    def collecting(self) -> bool:
        return self._collecting

    @property
    def ready(self) -> bool:
        return self._baseline_x is not None

    @property
    def baseline(self) -> Tuple[Optional[float], Optional[float]]:
        """The resting-gaze baseline as (x, y) offsets, or (None, None)."""
        return self._baseline_x, self._baseline_y

    def add(self, reading: GazeReading) -> None:
        """Feed one raw reading; outliers (blinks/saccades) are ignored.

        Calibration finishes automatically once the required number of usable
        samples has been collected, or the attempt cap is reached (in which
        case it fails with an explanatory message).
        """
        if not self._collecting or not reading.iris_ok:
            return
        if reading.iris_offset_x is None or reading.iris_offset_y is None:
            return
        if (
            abs(reading.iris_offset_x) <= self._accept_range
            and abs(reading.iris_offset_y) <= self._accept_range
        ):
            self._collected.append((reading.iris_offset_x, reading.iris_offset_y))
        self._attempts += 1
        if len(self._collected) >= self._required or self._attempts >= self._attempt_cap:
            self.finish()

    def finish(self) -> bool:
        """Finalise the baseline; returns True when calibration succeeded."""
        if not self._collecting:
            return self.ready
        self._collecting = False
        if len(self._collected) < max(1, self._required):
            self._last_error = (
                f"only {len(self._collected)}/{self._required} acceptable samples "
                "— look straight at the camera"
            )
            _log.warning("gaze calibration failed: %s", self._last_error)
            return False
        self._baseline_x = float(np.mean([s[0] for s in self._collected]))
        self._baseline_y = float(np.mean([s[1] for s in self._collected]))
        _log.info(
            "gaze calibration complete (baseline x=%.3f y=%.3f)",
            self._baseline_x,
            self._baseline_y,
        )
        return True

    def status(self) -> dict:
        """Snapshot of the calibration state for the API."""
        progress = len(self._collected) if self._collecting else 0
        if self._collecting:
            state = "COLLECTING"
        elif self.ready:
            state = "READY"
        elif self._last_error:
            state = "FAILED"
        else:
            state = "REQUIRED"
        return {
            "status": state,
            "progress": progress,
            "required": self._required,
            "acceptable_range": self._accept_range,
            "baseline_x": self._baseline_x,
            "baseline_y": self._baseline_y,
            "error": self._last_error,
        }


class GazeTracker:
    """Stateful gaze analysis: estimator + calibrator + temporal smoothing."""

    def __init__(self, settings: Settings, frame_w: int, frame_h: int) -> None:
        self._settings = settings
        self._estimator = GazeEstimator(settings.gaze_focal_length, frame_w, frame_h)
        self._calibrator = GazeCalibrator()
        if settings.gaze_iris_weight + settings.gaze_head_weight <= 0:
            raise ValueError("gaze_iris_weight + gaze_head_weight must be > 0")
        self._smoother = TemporalStateTracker(
            window=settings.gaze_smoothing_window,
            change_threshold=settings.gaze_change_threshold,
            away_duration=settings.gaze_away_duration,
            away_label=DIRECTION_AWAY,
            unknown_label=DIRECTION_UNKNOWN,
        )

    # ── calibration API ────────────────────────────────────────────────────

    def start_calibration(self) -> dict:
        self._calibrator.start(
            self._settings.gaze_calibration_samples,
            self._settings.gaze_calibration_accept_range,
        )
        return self._calibrator.status()

    def cancel_calibration(self) -> dict:
        self._calibrator.cancel()
        return self._calibrator.status()

    def calibration_status(self) -> dict:
        return self._calibrator.status()

    # ── per-frame update ───────────────────────────────────────────────────

    def update(self, face: Optional[FaceData]) -> GazeResult:
        """Process one frame's primary face (or None when no face)."""
        if face is None:
            direction = self._smoother.update(None)
            return GazeResult(
                direction=direction,
                attention_score=None,
                confidence=None,
            )

        reading = self._estimator.compute_reading(face)
        if not reading.valid:
            direction = self._smoother.update(None)
            return GazeResult(direction=direction, attention_score=None, confidence=None)

        # Feed the calibrator (only while calibration is running; read the
        # *raw* offsets so the baseline itself is unbiased by an old baseline).
        if self._calibrator.collecting:
            self._calibrator.add(reading)

        # Apply the resting-gaze baseline.
        if reading.iris_ok and self._calibrator.ready:
            base_x, base_y = self._calibrator.baseline
            reading.iris_offset_x = _clamp((reading.iris_offset_x or 0.0) - (base_x or 0.0))
            reading.iris_offset_y = _clamp((reading.iris_offset_y or 0.0) - (base_y or 0.0))

        result = self._classify(reading)
        stable = self._smoother.update(result["direction"])
        result["direction"] = stable
        dev_x = result["deviation_x"]
        dev_y = result["deviation_y"]

        mag = math.hypot(dev_x, dev_y) if dev_x is not None and dev_y is not None else 0.0
        saturation = self._settings.gaze_attention_saturation
        attention = None
        if reading.iris_ok or reading.head_ok:
            attention = int(round(100.0 * (1.0 - min(1.0, mag / max(saturation, 1e-6)))))

        confidence = 0.0
        if reading.iris_ok:
            confidence += 0.6 * reading.iris_quality
        if reading.head_ok:
            confidence += 0.4
        if confidence <= 0.0:
            confidence = None
            attention = None

        return GazeResult(
            direction=stable,
            attention_score=attention,
            confidence=confidence,
            head_yaw_deg=reading.head_yaw_deg,
            head_pitch_deg=reading.head_pitch_deg,
            head_roll_deg=reading.head_roll_deg,
            iris_offset_x=reading.iris_offset_x,
            iris_offset_y=reading.iris_offset_y,
            deviation_x=dev_x,
            deviation_y=dev_y,
            error=reading.error,
        )

    def _classify(self, reading: GazeReading) -> dict:
        """Weight iris + head signals and pick a direction label."""
        s = self._settings
        iris_x = reading.iris_offset_x
        iris_y = reading.iris_offset_y
        yaw = reading.head_yaw_deg
        pitch = reading.head_pitch_deg
        have_iris = reading.iris_ok and iris_x is not None and iris_y is not None
        have_head = reading.head_ok and yaw is not None and pitch is not None

        w_iris = s.gaze_iris_weight
        w_head = s.gaze_head_weight

        h_part = 0.0
        v_part = 0.0
        denom = 0.0
        if have_iris:
            h_part += w_iris * iris_x
            v_part += w_iris * iris_y
            denom += w_iris
        if have_head:
            # solvePnP pitch is + when the nose points down; flip so a
            # positive contribution means "looking up".
            h_part += w_head * _clamp(yaw / s.gaze_yaw_threshold)
            v_part += w_head * _clamp(-pitch / s.gaze_pitch_threshold)
            denom += w_head

        if denom <= 0:
            return {"direction": DIRECTION_UNKNOWN, "deviation_x": None, "deviation_y": None}

        dev_x = _clamp(h_part / denom)
        dev_y = _clamp(v_part / denom)
        threshold = s.gaze_direction_threshold

        direction = DIRECTION_AT_SCREEN
        if abs(dev_x) >= threshold and abs(dev_x) >= abs(dev_y):
            direction = DIRECTION_LEFT if dev_x > 0 else DIRECTION_RIGHT
        elif abs(dev_y) >= threshold:
            direction = DIRECTION_UP if dev_y > 0 else DIRECTION_DOWN

        return {"direction": direction, "deviation_x": dev_x, "deviation_y": dev_y}


# ── overlay drawing (video feed) ──────────────────────────────────────────────

_DIR_COLORS: dict = {
    DIRECTION_AT_SCREEN: (70, 190, 90),   # green
    DIRECTION_LEFT: (0, 155, 255),        # orange
    DIRECTION_RIGHT: (0, 155, 255),
    DIRECTION_UP: (0, 155, 255),
    DIRECTION_DOWN: (0, 155, 255),
    DIRECTION_AWAY: (70, 70, 210),        # red
    DIRECTION_UNKNOWN: (170, 170, 170),   # grey
}
_DIRECTIONS_XY: dict = {
    DIRECTION_LEFT: (1.0, 0.0),
    DIRECTION_RIGHT: (-1.0, 0.0),
    DIRECTION_UP: (0.0, -1.0),
    DIRECTION_DOWN: (0.0, 1.0),
}


def draw_gaze_overlay(
    frame: np.ndarray,
    face: FaceData,
    result: GazeResult,
    show_crosshair: bool = True,
) -> np.ndarray:
    """Annotate an OpenCV BGR ``frame`` with the gaze result.

    Draws a label chip (direction + attention) above the primary face bbox and
    an optional crosshair showing the current deviation vector.
    """
    x_min, y_min, x_max, y_max = face.bbox_pixels
    color = _DIR_COLORS.get(result.direction, _DIR_COLORS[DIRECTION_UNKNOWN])
    label_parts = [result.direction.lower().replace("_", " ")]
    if result.attention_score is not None:
        label_parts.append(f"{result.attention_score}%")
    label = " ".join(label_parts)

    _draw_chip(frame, (x_min, max(0, y_min - 26)), label, color)

    if show_crosshair and result.deviation_x is not None and result.deviation_y is not None:
        cx = int((x_min + x_max) / 2)
        cy = int((y_min + y_max) / 2)
        span = max(10, int((x_max - x_min) / 4))
        px = int(_clamp(cx + result.deviation_x * span * 1.4, x_min, x_max))
        py = int(_clamp(cy - result.deviation_y * span * 1.4, y_min, y_max))
        cv2.circle(frame, (cx, cy), span, color, 1)
        cv2.line(frame, (cx, cy), (px, py), color, 2)
    return frame


def _draw_chip(frame: np.ndarray, origin: Tuple[int, int], text: str, color: Tuple[int, int, int]) -> None:
    """Draw a filled rounded-style label chip at ``origin`` (top-left)."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thickness = 1
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    pad_x, pad_y = 6, 4
    x, y = origin
    cv2.rectangle(
        frame,
        (x, y),
        (x + tw + 2 * pad_x, y + th + 2 * pad_y + baseline),
        color,
        -1,
    )
    cv2.putText(
        frame,
        text,
        (x + pad_x, y + th + pad_y),
        font,
        scale,
        (0, 0, 0),
        thickness,
        cv2.LINE_AA,
    )