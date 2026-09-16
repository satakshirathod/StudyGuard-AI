"""Real-time drowsiness detection via Eye Aspect Ratio (EAR) (Phase 4).

Pipeline for each processed frame, given the primary face's 478 MediaPipe
landmarks (already extracted as ``FaceData.left_eye`` / ``FaceData.right_eye``):

1. **EAR per eye** — the classic eye-closure metric
   ``EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)`` for the six contour points
   of each eye. Values run ≈0.30–0.35 for a normally-open eye and drop toward
   ~0.10–0.20 for a closed eye, so the configured ``ear_threshold`` separates
   "open" from "closed" *per frame*.
2. **Blink vs drowsy** — a single closed frame (or a short blink) is normal.
   Drowsiness is only declared after the eye stays below threshold for
   ``ear_consecutive_frames`` consecutive processed frames **and** for at
   least ``drowsiness_duration_threshold`` seconds of sustained closure.
3. **Temporal smoothing** — a debounced majority-vote ``TemporalStateTracker``
   (same utility as the gaze module) prevents flicker; a long missing/no-face
   run decays the stable state to UNKNOWN.

States are ``NORMAL`` / ``DROWSY`` / ``UNKNOWN``. When there is no usable eye
data the result degrades honestly to UNKNOWN with numeric measures ``None``
(never fabricated values). Missing/zero-width/invalid eye landmarks for *one*
eye degrade gracefully to a single-eye reading; for *both* eyes → UNKNOWN.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

from ai.face_detection import FaceData
from ai.temporal import TemporalStateTracker
from config import Settings
from utils.logger import get_logger

_log = get_logger("studygard.ai.drowsiness")

# Public state labels — used by the status endpoint and the frontend.
STATE_NORMAL = "NORMAL"
STATE_DROWSY = "DROWSY"
STATE_UNKNOWN = "UNKNOWN"


@dataclass
class DrowsinessReading:
    """Raw, per-frame eye measurements before any temporal logic."""

    valid: bool = False
    left_ear: Optional[float] = None   # eye aspect ratio, left eye
    right_ear: Optional[float] = None  # eye aspect ratio, right eye
    avg_ear: Optional[float] = None    # mean of usable eyes (single-eye fallback)
    eyes_closed: Optional[bool] = None  # avg EAR strictly below threshold
    error: Optional[str] = None


@dataclass
class DrowsinessResult:
    """Final processed drowsiness output (JSON-serialisable)."""

    state: str = STATE_UNKNOWN  # NORMAL | DROWSY | UNKNOWN
    ear: Optional[float] = None  # current average EAR (debug metric)
    left_ear: Optional[float] = None
    right_ear: Optional[float] = None
    eyes_closed: Optional[bool] = None  # per-frame closure, before smoothing
    confidence: float = 0.0  # 0..1 signal quality (not medical certainty)
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "state": self.state,
            "ear": None if self.ear is None else round(self.ear, 3),
            "left_ear": None if self.left_ear is None else round(self.left_ear, 3),
            "right_ear": None if self.right_ear is None else round(self.right_ear, 3),
            "eyes_closed": self.eyes_closed,
            "confidence": round(self.confidence, 3),
            "error": self.error,
        }


def _euclidean(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def compute_ear(points: Sequence[Tuple[float, float]]) -> Optional[float]:
    """Compute the Eye Aspect Ratio for one eye's six contour points.

    Points must be ordered as the standard MediaPipe eye contour:
    ``[p1, p2, p3, p4, p5, p6]`` where ``p1``/``p4`` are the eye corners
    (outer/inner) and ``p2, p3, p5, p6`` the upper/lower lids.

    Returns ``None`` when the contour is unusable (missing points, non-finite
    coords, or zero/invalid corner separation) so the caller can degrade
    honestly to UNKNOWN / single-eye fallback.
    """
    if points is None or len(points) < 6:
        return None

    p1, p2, p3, p4, p5, p6 = points[:6]
    try:
        if not all(
            math.isfinite(p[0]) and math.isfinite(p[1])
            for p in (p1, p2, p3, p4, p5, p6)
        ):
            return None
        horizontal = _euclidean(p1, p4)
        if horizontal <= 1e-9:
            return None  # degenerate: zero eye width
        vertical = _euclidean(p2, p6) + _euclidean(p3, p5)
        return vertical / (2.0 * horizontal)
    except (TypeError, ValueError):
        return None


class DrowsinessEstimator:
    """Stateless computation of the per-frame EAR reading from one face."""

    def compute_reading(self, face: Optional[FaceData]) -> DrowsinessReading:
        """Build a :class:`DrowsinessReading` from a face, or an invalid reading."""
        reading = DrowsinessReading()
        if not face:
            reading.error = "no face data"
            return reading

        try:
            left_ear = compute_ear(face.left_eye) if face.left_eye else None
            right_ear = compute_ear(face.right_eye) if face.right_eye else None
        except Exception as exc:  # defensive: never crash the video loop
            _log.warning("ear reading failed: %s", exc)
            reading.error = str(exc)
            return reading

        reading.left_ear = left_ear
        reading.right_ear = right_ear

        usable = [ear for ear in (left_ear, right_ear) if ear is not None]
        if not usable:
            reading.error = "no usable eye landmarks"
            return reading

        reading.avg_ear = sum(usable) / len(usable)
        reading.valid = True
        return reading


class DrowsinessDetector:
    """Stateful drowsiness analysis: estimator + blink-safe temporal logic.

    The detector owns the per-frame closed-streak counter; the states it emits
    are then smoothed with the shared :class:`TemporalStateTracker` so a single
    stray frame never flips the reported state.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._estimator = DrowsinessEstimator()
        self._closed_streak = 0
        self._closure_started: Optional[float] = None
        self._smoother = TemporalStateTracker(
            window=settings.drowsiness_smoothing_window,
            change_threshold=settings.drowsiness_change_threshold,
            away_duration=settings.ear_consecutive_frames,
            away_label=STATE_UNKNOWN,
            unknown_label=STATE_UNKNOWN,
        )

    # ── per-frame update ──────────────────────────────────────────────────

    def update(
        self, face: Optional[FaceData], now: Optional[float] = None
    ) -> DrowsinessResult:
        """Process one frame's primary face (or None when no face).

        ``now`` is a monotonic clock (seconds) used for the sustained-closure
        duration; it defaults to the real clock and is injectable for tests.
        """
        if not self._settings.drowsiness_enabled:
            self._closed_streak = 0
            self._closure_started = None
            return DrowsinessResult(state=STATE_UNKNOWN, confidence=0.0,
                                    error="drowsiness disabled by configuration")
        return self._update(face, now)

    def _update(self, face: Optional[FaceData], now_: Optional[float]) -> DrowsinessResult:
        if face is None:
            self._closed_streak = 0
            self._closure_started = None
            stable = self._smoother.update(None)
            return DrowsinessResult(
                state=stable, confidence=0.0, error="no face data"
            )

        reading = self._estimator.compute_reading(face)
        if not reading.valid or reading.avg_ear is None:
            self._closed_streak = 0
            self._closure_started = None
            stable = self._smoother.update(None)
            return DrowsinessResult(
                state=stable,
                left_ear=reading.left_ear,
                right_ear=reading.right_ear,
                confidence=0.0,
                error=reading.error,
            )

        avg_ear = reading.avg_ear
        threshold = self._settings.ear_threshold
        eyes_closed = avg_ear < threshold
        reading.eyes_closed = eyes_closed
        now = now_ if now_ is not None else time.monotonic()

        if eyes_closed:
            self._closed_streak += 1
            if self._closure_started is None:
                self._closure_started = now
            sustained_seconds = now - self._closure_started
        else:
            self._closed_streak = 0
            self._closure_started = None
            sustained_seconds = 0.0

        raw_state = STATE_NORMAL
        if (
            eyes_closed
            and self._closed_streak >= self._settings.ear_consecutive_frames
            and sustained_seconds >= self._settings.drowsiness_duration_threshold
        ):
            raw_state = STATE_DROWSY

        stable = self._smoother.update(raw_state)
        confidence = self._confidence(avg_ear, threshold, reading)

        return DrowsinessResult(
            state=stable,
            ear=avg_ear,
            left_ear=reading.left_ear,
            right_ear=reading.right_ear,
            eyes_closed=eyes_closed,
            confidence=confidence,
            error=reading.error,
        )

    @staticmethod
    def _confidence(
        avg_ear: float, threshold: float, reading: DrowsinessReading
    ) -> float:
        """Signal-quality confidence: eye coverage × threshold separation.

        Uses how many eyes contributed (single-eye fallback halves the signal)
        and how far the EAR sits from the threshold — a reading right on the
        threshold is genuinely ambiguous and gets a low score. This is a
        *measurement-quality* estimate, not a medical/driver-confidence value.
        """
        left_ok = reading.left_ear is not None
        right_ok = reading.right_ear is not None
        coverage = (left_ok + right_ok) / 2.0
        separation = abs(avg_ear - threshold) / max(threshold, 1e-9)
        certainty = max(0.0, min(1.0, separation / 0.5))  # full at half-threshold away
        return round(coverage * certainty, 3)

    def reset(self) -> None:
        """Clear per-frame state and temporal history (e.g. between sessions)."""
        self._closed_streak = 0
        self._closure_started = None
        self._smoother.reset()


# ── overlay drawing (video feed, dev-only) ──────────────────────────────────

_STATE_COLORS: dict = {
    STATE_NORMAL: (70, 190, 90),   # green
    STATE_DROWSY: (70, 70, 210),   # red
    STATE_UNKNOWN: (170, 170, 170),  # grey
}


def draw_drowsiness_overlay(
    frame, face: "FaceData", result: DrowsinessResult
):
    """Annotate an OpenCV BGR frame with the drowsiness result chip."""
    import cv2

    x_min, y_min, x_max, y_max = face.bbox_pixels
    color = _STATE_COLORS.get(result.state, _STATE_COLORS[STATE_UNKNOWN])
    label = result.state.lower().replace("_", " ")
    if result.ear is not None:
        label += f" {result.ear:.2f}"
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