"""Real-time sitting-posture estimation via face-landmark head-pitch signals (Phase 6).

Uses the existing 478-point MediaPipe face mesh — no second camera, no extra
model.  Two complementary signals derived from the *primary* face's normalised
landmarks are computed each processed frame and combined into a single 0–100
posture score:

1. **Head-pitch (nose–eye–chin ratio)** — when the nose tip sits farther
   *below* the eye centre relative to the chin the head is pitched forward
   (slouching); when the nose is roughly centred the head is upright.

2. **Face vertical offset** — a face positioned unusually low in the camera
   frame is likely leaning forward; one centred high is upright.

The combined score is smoothed with an exponential moving average, then
classified into states that match the existing session-service aggregation
logic: **GOOD** (≥ ``posture_good_threshold``), **MODERATE** (≥
``posture_moderate_threshold``), **POOR** (below), or **UNKNOWN** when no
reliable reading exists.

States are further smoothed with the shared ``TemporalStateTracker`` so a
single noisy frame cannot flip the card.  No per-frame database writes are
performed — posture state and score persist only via the periodic behaviour-log
snapshot, matching every other module.

Limitations (honest, must be documented)
----------------------------------------
This is an *approximate head-pitch heuristic* — not a full body-posture
analysis.  A student with good head position but a slouched torso will appear
GOOD, and a forward-leaning student with a tilted screen may appear POOR.  Camera
position, resolution and framing all affect the thresholds; re-calibration is
done via tunable env vars (``SG_POSTURE_GOOD_THRESHOLD``,
``SG_POSTURE_MODERATE_THRESHOLD``).  No accuracy claim is made without a formal
evaluation (Phase 15).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Optional

from ai.face_detection import FaceData, NOSE_TIP, FACE_OVAL
from ai.temporal import TemporalStateTracker
from config import Settings
from utils.logger import get_logger

_log = get_logger("studygard.ai.posture")

# Public state labels — used by the status endpoint and the frontend.
STATE_GOOD = "GOOD"
STATE_MODERATE = "MODERATE"
STATE_POOR = "POOR"
STATE_UNKNOWN = "UNKNOWN"

# Landmark indices reused from face_detection.py constants.
_LEFT_EYE_OUTER = 33
_RIGHT_EYE_OUTER = 263
_CHIN = 152       # bottom of FACE_OVAL


@dataclass
class PostureReading:
    """Raw, per-frame head-pitch measurements before any temporal logic."""

    valid: bool = False
    head_pitch_score: Optional[float] = None   # 0..100, 100 = ideal head pitch
    face_offset_score: Optional[float] = None   # 0..100, 100 = ideal position
    raw_score: Optional[float] = None           # combined 0..100
    error: Optional[str] = None


@dataclass
class PostureResult:
    """Final processed posture output (JSON-serialisable)."""

    state: str = STATE_UNKNOWN  # GOOD | MODERATE | POOR | UNKNOWN
    posture_score: Optional[float] = None  # 0..100 (smoothed)
    head_pitch_score: Optional[float] = None
    face_offset_score: Optional[float] = None
    confidence: float = 0.0  # 0..1 signal quality
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "state": self.state,
            "posture_score": (
                None if self.posture_score is None else round(self.posture_score, 1)
            ),
            "head_pitch_score": (
                None if self.head_pitch_score is None else round(self.head_pitch_score, 1)
            ),
            "face_offset_score": (
                None if self.face_offset_score is None else round(self.face_offset_score, 1)
            ),
            "confidence": round(self.confidence, 3),
            "error": self.error,
        }


# ── per-frame reading ───────────────────────────────────────────────────────

def _clamp(v: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, v))


def _finite(p) -> bool:
    return (
        p is not None
        and len(p) >= 2
        and math.isfinite(p[0])
        and math.isfinite(p[1])
    )


def compute_posture_score(face: FaceData, frame_h: int) -> PostureReading:
    """Compute a 0–100 posture score from face landmarks.

    Two complementary signals are blended:
    * **Head pitch** (nose–eye–chin ratio) — detects forward/downward head tilt.
    * **Face vertical offset** — detects a face shifted low in the frame.

    Returns ``None`` scores when critical landmarks are missing/invalid.
    """
    lms = face.landmarks
    if not lms or len(lms) <= max(NOSE_TIP, _CHIN, _LEFT_EYE_OUTER, _RIGHT_EYE_OUTER):
        return PostureReading(error="insufficient landmarks")

    nose = lms[NOSE_TIP]
    chin = lms[_CHIN]
    left_eye = lms[_LEFT_EYE_OUTER]
    right_eye = lms[_RIGHT_EYE_OUTER]

    if not all(_finite(p) for p in (nose, chin, left_eye, right_eye)):
        return PostureReading(error="non-finite landmark coordinates")

    # ── signal 1: head pitch (nose–eye–chin ratio) ────────────────────
    eye_y = (left_eye[1] + right_eye[1]) / 2.0
    nose_y = nose[1]
    chin_y = chin[1]
    eye_to_chin = abs(chin_y - eye_y)
    if eye_to_chin < 1e-9:
        head_pitch_score = 50.0  # degenerate face geometry → mid-range
    else:
        ratio = (nose_y - eye_y) / eye_to_chin
        # Upright head: nose is roughly 0.40–0.55 of the way down from eyes to chin.
        # Slouching head: nose drops to ~0.55–0.70+.  The ratio is mapped to 0–100.
        # 0.45 → 100 (very upright), 0.65 → 0 (very slouched).
        head_pitch_score = _clamp(_linear_map(ratio, 0.45, 0.65, 100.0, 0.0))

    # ── signal 2: face vertical offset ────────────────────────────────
    face_cy = (face.bbox_normalized[1] + face.bbox_normalized[3]) / 2.0
    # Upright student: face centre sits roughly at 0.35–0.45 of the frame
    # (eyebrows near the top third).  A low face (0.55–0.70) suggests leaning
    # forward.  Map 0.35 → 100, 0.60 → 0.
    face_offset_score = _clamp(_linear_map(face_cy, 0.35, 0.60, 100.0, 0.0))

    # ── blend ─────────────────────────────────────────────────────────
    raw = 0.65 * head_pitch_score + 0.35 * face_offset_score
    return PostureReading(
        valid=True,
        head_pitch_score=round(head_pitch_score, 1),
        face_offset_score=round(face_offset_score, 1),
        raw_score=round(raw, 1),
    )


def _linear_map(x: float, x0: float, x1: float, y0: float, y1: float) -> float:
    """Linearly map x from [x0, x1] to [y0, y1], clamped outside the range."""
    if abs(x1 - x0) < 1e-12:
        return (y0 + y1) / 2.0
    return y0 + (x - x0) * (y1 - y0) / (x1 - x0)


# ── stateful detector ───────────────────────────────────────────────────────

class PostureDetector:
    """Stateful posture analysis: estimator + EMA smoothing + temporal logic.

    The detector owns:
    * An exponential moving average (EMA) of the raw posture score so a
      single jittery frame cannot jump the displayed value.
    * A streak counter for the *unknown* state (missing face frames) so the
      card only flips to UNKNOWN after ``posture_away_frames`` consecutive
      missing readings — matching every other module.

    States are fed into the shared ``TemporalStateTracker`` majority-vote
    so a brief slouch does not immediately raise POOR.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._ema_score: Optional[float] = None
        self._missing_streak = 0
        self._smoother = TemporalStateTracker(
            window=settings.posture_smoothing_window,
            change_threshold=settings.posture_change_threshold,
            away_duration=settings.posture_away_frames,
            away_label=STATE_UNKNOWN,
            unknown_label=STATE_UNKNOWN,
        )

    # ── per-frame update ──────────────────────────────────────────────────

    def update(
        self, face: Optional[FaceData], now: Optional[float] = None,
    ) -> PostureResult:
        """Process one frame's primary face (or ``None`` when no face)."""
        if not self._settings.posture_enabled:
            self._ema_score = None
            self._missing_streak = 0
            return PostureResult(
                state=STATE_UNKNOWN, confidence=0.0,
                error="posture detection disabled by configuration",
            )
        return self._update(face)

    def _update(self, face: Optional[FaceData]) -> PostureResult:
        if face is None:
            self._missing_streak += 1
            stable = self._smoother.update(None)
            return PostureResult(
                state=stable,
                posture_score=self._ema_score,
                confidence=0.0,
                error="no face data",
            )

        reading = compute_posture_score(face, frame_h=0)  # frame_h unused here
        if not reading.valid or reading.raw_score is None:
            self._missing_streak += 1
            stable = self._smoother.update(None)
            return PostureResult(
                state=stable,
                posture_score=self._ema_score,
                confidence=0.0,
                error=reading.error,
            )

        self._missing_streak = 0
        raw = reading.raw_score

        # EMA smoothing
        alpha = self._settings.posture_ema_alpha
        if self._ema_score is None:
            self._ema_score = raw
        else:
            self._ema_score = alpha * raw + (1.0 - alpha) * self._ema_score

        # Classify
        state = self._classify(self._ema_score)
        stable = self._smoother.update(state)
        confidence = self._confidence(reading)

        return PostureResult(
            state=stable,
            posture_score=self._ema_score,
            head_pitch_score=reading.head_pitch_score,
            face_offset_score=reading.face_offset_score,
            confidence=confidence,
            error=reading.error,
        )

    def _classify(self, score: float) -> str:
        if score >= self._settings.posture_good_threshold:
            return STATE_GOOD
        if score >= self._settings.posture_moderate_threshold:
            return STATE_MODERATE
        return STATE_POOR

    @staticmethod
    def _confidence(reading: PostureReading) -> float:
        """Signal quality based on measurement completeness."""
        scores = [s for s in (reading.head_pitch_score, reading.face_offset_score)
                  if s is not None]
        if not scores:
            return 0.0
        return round(len(scores) / 2.0, 3)

    def reset(self) -> None:
        """Clear EMA + streak + temporal history (e.g. between sessions)."""
        self._ema_score = None
        self._missing_streak = 0
        self._smoother.reset()


# ── overlay drawing (video feed, dev-only) ──────────────────────────────────

_STATE_COLORS: dict = {
    STATE_GOOD: (70, 190, 90),       # green
    STATE_MODERATE: (200, 170, 50),   # amber
    STATE_POOR: (70, 70, 210),       # red
    STATE_UNKNOWN: (170, 170, 170),  # grey
}


def draw_posture_overlay(frame, face: FaceData, result: PostureResult):
    """Annotate an OpenCV BGR frame with the posture result chip."""
    import cv2

    x_min, y_min, x_max, y_max = face.bbox_pixels
    color = _STATE_COLORS.get(result.state, _STATE_COLORS[STATE_UNKNOWN])
    label = result.state.lower().replace("_", " ")
    if result.posture_score is not None:
        label += f" {result.posture_score:.0f}"
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
