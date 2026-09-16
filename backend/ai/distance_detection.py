"""Real-time screen-distance estimation via relative face width (Phase 5).

Uses the existing face pipeline — no second camera, no extra model. Each
processed frame the *primary* face's 478-normalised landmarks are converted to
pixel space using the camera frame dimensions, then a stable facial-width
measure is computed:

* **Interocular width** — the pixel distance between the outer eye corners
  (landmarks ``33`` and ``263``). It is a classic, roughly pose-stable proxy
  for "how wide the face appears", which scales inversely with distance.
* **Bounding-box width** — used only as a fallback when those landmark indices
  are unavailable. Boxes clipped at the frame edge (face partially out of
  frame) are rejected as unreliable.

Because the camera is not calibrated, the output is **relative**, not
centimetres:

``ratio = current_facial_width / reference_facial_width``

* ``ratio ≈ 1`` → face appears the same size as the reference distance (NORMAL)
* ``ratio ≥ SG_DISTANCE_TOO_CLOSE_RATIO`` → face is much larger → TOO_CLOSE
* ``ratio ≤ SG_DISTANCE_RECOVER_RATIO`` → clearly back at a safe distance

The reference width comes from either ``SG_DISTANCE_REFERENCE_WIDTH`` (px, set
during your own calibration) or an **auto-calibration** that captures the
average of the first stable readings after start-up — sit at your normal
working distance when it does. Without a reference no measurement exists, so
the result is honestly ``UNKNOWN`` (never fabricated).

Temporal logic mirrors the drowsiness module: a hysteresis band prevents
flip-flop, a streak counter means TOO_CLOSE only appears after
``SG_DISTANCE_TOO_CLOSE_FRAMES`` consecutive close frames, and the shared
``TemporalStateTracker`` majority-votes the final state. A noisy single frame
therefore never raises an alert.

State labels: ``NORMAL`` / ``TOO_CLOSE`` / ``UNKNOWN``.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Optional

from ai.face_detection import FaceData
from ai.temporal import TemporalStateTracker
from config import Settings
from utils.logger import get_logger

_log = get_logger("studygard.ai.distance")

# Public state labels — used by the status endpoint and the frontend.
STATE_NORMAL = "NORMAL"
STATE_TOO_CLOSE = "TOO_CLOSE"
STATE_UNKNOWN = "UNKNOWN"

# Outer-eye-corner landmark indices (478-point MediaPipe face mesh).
LEFT_OUTER_CORNER = 33
RIGHT_OUTER_CORNER = 263

CAL_CONFIRMED = "CONFIRMED"  # reference width set at construction (user-configured)
CAL_REQUIRED = "REQUIRED"    # no reference yet — auto-calibration will capture
CAL_COLLECTING = "COLLECTING"
CAL_READY = "READY"          # reference captured
CAL_DISABLED = "DISABLED"


@dataclass
class DistanceReading:
    """Raw, per-frame facial-width measurement before any temporal logic."""

    valid: bool = False
    face_width_px: Optional[float] = None  # pixel width of the facial-width measure
    method: Optional[str] = None  # "interocular" | "bbox"
    error: Optional[str] = None


@dataclass
class DistanceResult:
    """Final processed screen-distance output (JSON-serialisable)."""

    state: str = STATE_UNKNOWN  # NORMAL | TOO_CLOSE | UNKNOWN
    distance_estimate: Optional[float] = None  # relative ratio vs reference (>1 = closer)
    face_width: Optional[float] = None  # px
    too_close: Optional[bool] = None
    confidence: float = 0.0  # 0..1 signal quality (not a physical measurement)
    reference_width: Optional[float] = None  # px at the reference distance
    calibration: Optional[str] = None  # calibration sub-state
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "state": self.state,
            "distance_estimate": None if self.distance_estimate is None
            else round(self.distance_estimate, 3),
            "face_width": None if self.face_width is None else round(self.face_width, 1),
            "too_close": self.too_close,
            "confidence": round(self.confidence, 3),
            "reference_width": None if self.reference_width is None else round(self.reference_width, 1),
            "calibration": self.calibration,
            "error": self.error,
        }


def _finite(p):
    return p is not None and math.isfinite(p[0]) and math.isfinite(p[1])


class DistanceEstimator:
    """Stateless computation of the per-frame facial-width reading."""

    def __init__(self, frame_w: int, frame_h: int, min_face_width: float) -> None:
        self._frame_w = frame_w
        self._frame_h = frame_h
        self._min_face_width = min_face_width

    def compute_reading(self, face: Optional[FaceData]) -> DistanceReading:
        """Build a :class:`DistanceReading` from a face, or an invalid reading."""
        if not face:
            return DistanceReading(error="no face data")

        try:
            width = self._interocular_width(face)
            method = "interocular"
            if width is None:
                width = self._bbox_width(face)
                method = "bbox"
        except Exception as exc:  # defensive: never crash the video loop
            _log.warning("distance reading failed: %s", exc)
            return DistanceReading(error=str(exc))

        if width is None:
            return DistanceReading(error="no usable facial width")

        if width < self._min_face_width:
            return DistanceReading(
                error=f"face too small (width {width:.1f}px < {self._min_face_width:.0f}px)"
            )

        reading = DistanceReading(valid=True, face_width_px=width, method=method)
        return reading

    def _interocular_width(self, face: FaceData) -> Optional[float]:
        """Pixel distance between outer eye corners 33 and 263."""
        lms = face.landmarks
        if not lms or len(lms) <= RIGHT_OUTER_CORNER:
            return None
        left = lms[LEFT_OUTER_CORNER]
        right = lms[RIGHT_OUTER_CORNER]
        if not (_finite(left) and _finite(right)):
            return None
        dx = (right[0] - left[0]) * self._frame_w
        dy = (right[1] - left[1]) * self._frame_h
        return math.hypot(dx, dy)

    def _bbox_width(self, face: FaceData) -> Optional[float]:
        """Bounding-box width, rejected when the face is clipped at a frame edge."""
        bbox_norm = face.bbox_normalized
        if any(v <= 0.0 or v >= 1.0 for v in bbox_norm) or face.area <= 0:
            return None  # box touches the frame edge → face partially out of frame
        x_min, _, x_max, _ = face.bbox_pixels
        return float(x_max - x_min)


class DistanceDetector:
    """Stateful screen-distance analysis: estimator + hysteresis + temporal logic.

    Owns the reference-width calibration (user-configured or auto-captured)
    and the too-close streak counter; emissions are smoothed with the shared
    :class:`TemporalStateTracker` so one noisy frame never flips the state.
    """

    def __init__(self, settings: Settings, frame_w: int, frame_h: int) -> None:
        self._settings = settings
        self._frame_w = frame_w
        self._frame_h = frame_h
        self._estimator = DistanceEstimator(frame_w, frame_h, settings.distance_min_face_width)

        # Reference width — configured reference wins immediately; otherwise
        # auto-calibration captures the first stable readings after start-up.
        self._reference_width = (
            float(settings.distance_reference_width)
            if settings.distance_reference_width and settings.distance_reference_width > 0
            else None
        )
        self._previous_reference: Optional[float] = None  # restored on calibrate-cancel
        self._reference_samples: list[float] = []
        self._collecting = self._reference_width is None

        self._too_close_streak = 0
        self._held_raw = STATE_NORMAL  # hysteresis: state inside the dead band holds
        self._smoother = TemporalStateTracker(
            window=settings.distance_smoothing_window,
            change_threshold=settings.distance_change_threshold,
            away_duration=settings.distance_away_duration,
            away_label=STATE_UNKNOWN,
            unknown_label=STATE_UNKNOWN,
        )

    # ── calibration ───────────────────────────────────────────────────────

    def start_calibration(self) -> dict:
        """Re-capture the reference width from incoming frames."""
        self._previous_reference = self._reference_width
        self._reference_width = None
        self._collecting = True
        self._reference_samples = []
        self._too_close_streak = 0
        self._held_raw = STATE_NORMAL
        self._smoother.reset()
        return self.calibration_status()

    def cancel_calibration(self) -> dict:
        """Abort an in-progress calibration capture, restoring the old reference."""
        self._collecting = False
        self._reference_samples = []
        if self._reference_width is None and self._previous_reference is not None:
            self._reference_width = self._previous_reference
        self._previous_reference = None
        return self.calibration_status()

    def calibration_status(self) -> dict:
        """Report the calibration sub-state for the status endpoint."""
        mode = self._settings.distance_mode
        if not self._settings.distance_enabled:
            return {"status": CAL_DISABLED, "mode": mode, "progress": 0, "required": 0}
        if self._reference_width is not None and self._collecting is False:
            status = CAL_CONFIRMED if self._is_user_configured() else CAL_READY
            return {
                "status": status,
                "mode": mode,
                "reference_width": self._reference_width,
                "progress": 0,
                "required": 0,
            }
        if self._collecting:
            required = self._settings.distance_reference_samples
            return {
                "status": CAL_COLLECTING if self._reference_samples else CAL_REQUIRED,
                "mode": mode,
                "reference_width": None,
                "progress": len(self._reference_samples),
                "required": required,
            }
        return {"status": CAL_REQUIRED, "mode": mode, "progress": 0, "required": 0}

    def _is_user_configured(self) -> bool:
        return bool(self._settings.distance_reference_width and self._settings.distance_reference_width > 0)

    def _collect_sample(self, width: float) -> None:
        """Accumulate a stable facial width toward the auto-reference."""
        if not self._collecting or self._reference_width is not None:
            return
        self._reference_samples.append(width)
        if len(self._reference_samples) >= self._settings.distance_reference_samples:
            self._reference_width = sum(self._reference_samples) / len(self._reference_samples)
            self._collecting = False
            self._reference_samples = []
            _log.info("Distance reference captured: %.1f px", self._reference_width)

    @property
    def reference_width(self) -> Optional[float]:
        return self._reference_width

    # ── per-frame update ──────────────────────────────────────────────────

    def update(self, face: Optional[FaceData], now: Optional[float] = None) -> DistanceResult:
        """Process one frame's primary face (or None when no face)."""
        if not self._settings.distance_enabled:
            self._too_close_streak = 0
            self._held_raw = STATE_TOO_CLOSE  # neutral for a disabled module
            return DistanceResult(
                state=STATE_UNKNOWN, confidence=0.0,
                calibration=CAL_DISABLED, error="distance detection disabled by configuration",
            )
        return self._update(face, now)

    def _update(self, face: Optional[FaceData], now_: Optional[float]) -> DistanceResult:
        reading = self._estimator.compute_reading(face)
        if not reading.valid or reading.face_width_px is None:
            self._too_close_streak = 0
            stable = self._smoother.update(None)
            return DistanceResult(
                state=stable,
                confidence=0.0,
                reference_width=self._reference_width,
                calibration=self.calibration_status()["status"],
                error=reading.error,
            )

        width = reading.face_width_px

        # Auto-calibration happens at whatever distance the student sits at
        # start-up (documented as the reference/NORMAL distance).
        if self._reference_width is None:
            self._collect_sample(width)
            if self._reference_width is None:
                stable = self._smoother.update(None)
                return DistanceResult(
                    state=stable,
                    face_width=width,
                    confidence=0.0,
                    reference_width=None,
                    calibration=self.calibration_status()["status"],
                    error="calibrating reference width",
                )

        reference = self._reference_width
        if reference <= 0:
            return DistanceResult(
                state=STATE_UNKNOWN, face_width=width, confidence=0.0,
                reference_width=reference,
                calibration=self.calibration_status()["status"],
                error="invalid reference width",
            )

        ratio = width / reference
        too_close_ratio = self._settings.distance_too_close_ratio
        recover_ratio = self._settings.distance_recover_ratio

        # Hysteresis band: only strong signals move the state; borderline
        # frames keep whatever we last committed to (no flip-flop).
        if ratio >= too_close_ratio:
            raw = STATE_TOO_CLOSE
        elif ratio <= recover_ratio:
            raw = STATE_NORMAL
        else:
            raw = self._held_raw

        # Persistence: TOO_CLOSE needs N consecutive close frames.
        if raw == STATE_NORMAL:
            self._too_close_streak = 0
        else:
            self._too_close_streak += 1
            if self._too_close_streak < self._settings.distance_too_close_frames:
                raw = STATE_NORMAL  # brief approach noise — still normal

        self._held_raw = raw
        stable = self._smoother.update(raw)

        confidence = self._confidence(width, reading.method)
        return DistanceResult(
            state=stable,
            distance_estimate=ratio,
            face_width=width,
            too_close=(stable == STATE_TOO_CLOSE),
            confidence=confidence,
            reference_width=reference,
            calibration=self.calibration_status()["status"],
            error=reading.error,
        )

    @staticmethod
    def _confidence(width: float, method: Optional[str]) -> float:
        """Signal-quality confidence: measurement source × width margin.

        Interocular width is trusted above the bounding-box fallback, and very
        small faces (near the unreliable minimum) are discounted. This is a
        *measurement-quality* estimate, not a physical-distance guarantee.
        """
        base = 1.0 if method == "interocular" else 0.6 if method == "bbox" else 0.0
        # width-margin factor relative to an arbitrary 1× reference: 2× the
        # minimum width already yields full confidence; smaller → less certain.
        certainty = max(0.0, min(1.0, width / 100.0))
        return round(base * certainty, 3)

    def reset(self) -> None:
        """Clear streak + temporal history (e.g. between sessions)."""
        self._too_close_streak = 0
        self._held_raw = STATE_NORMAL
        self._smoother.reset()


# ── overlay drawing (video feed, dev-only) ──────────────────────────────────

_STATE_COLORS: dict = {
    STATE_NORMAL: (70, 190, 90),     # green
    STATE_TOO_CLOSE: (70, 110, 210),  # red
    STATE_UNKNOWN: (170, 170, 170),  # grey
}


def draw_distance_overlay(frame, face: "FaceData", result: DistanceResult):
    """Annotate an OpenCV BGR frame with the distance result chip."""
    import cv2

    x_min, y_min, x_max, y_max = face.bbox_pixels
    color = _STATE_COLORS.get(result.state, _STATE_COLORS[STATE_UNKNOWN])
    label = result.state.lower().replace("_", " ")
    if result.distance_estimate is not None:
        label += f" x{result.distance_estimate:.2f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thickness = 0.55, 1
    (tw, th), baseline = cv2.getTextSize(label, font, scale, thickness)
    pad_x, pad_y = 6, 4
    x, y = max(0, x_min + 4), max(0, y_min + 24)
    cv2.rectangle(
        frame,
        (x, y),
        (x + tw + 2 * pad_x, y + th + 2 * pad_y + baseline),
        color,
        -1,
    )
    cv2.putText(
        frame,
        label,
        (x + pad_x, y + th + pad_y),
        font,
        scale,
        (0, 0, 0),
        thickness,
        cv2.LINE_AA,
    )
    return frame