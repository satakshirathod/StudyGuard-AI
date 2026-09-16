"""Tests for the drowsiness detection module (Phase 4).

Uses fully synthetic, geometrically-consistent eyes: each ``_eye_region`` is
built so its Eye Aspect Ratio is *exactly* the requested ``ear`` value (EAR =
``h/w`` by construction), so thresholds and blink logic are deterministic.
Camera-dependent accuracy against real faces is out of scope here, but the
EAR maths, single-eye fallback, blink-vs-drowsy logic, temporal smoothing and
honest UNKNOWN handling are all covered.
"""

from __future__ import annotations

import math
import sys
from dataclasses import replace
from pathlib import Path

import pytest

_BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from ai.drowsiness_detection import (  # noqa: E402
    STATE_NORMAL,
    STATE_DROWSY,
    STATE_UNKNOWN,
    DrowsinessDetector,
    DrowsinessEstimator,
    DrowsinessResult,
    DrowsinessReading,
    compute_ear,
)
from ai.face_detection import FaceData  # noqa: E402
from config import get_settings  # noqa: E402

_s = get_settings()

OPEN_EAR = 0.375   # clearly above the default threshold of 0.25
CLOSED_EAR = 0.025  # clearly below it


def make_settings(**overrides):
    return replace(_s, **overrides)


# ── synthetic geometry ───────────────────────────────────────────────────────

def _eye_region(cx: float, cy: float, ear: float, half_w: float = 0.08):
    """Six-point eye contour whose EAR is exactly ``ear`` (EAR = h/w)."""
    w = half_w
    h = ear * w
    return [
        (cx - w, cy),           # p1 outer corner
        (cx, cy - h),           # p2 upper lid
        (cx + 0.5 * w, cy - h),  # p3 upper lid
        (cx + w, cy),           # p4 inner corner
        (cx + 0.5 * w, cy + h),  # p5 lower lid
        (cx, cy + h),           # p6 lower lid
    ]


def make_face(left_ear=None, right_ear=None) -> FaceData:
    """Face with controlled per-eye EAR (None → that eye has no landmarks)."""
    left = _eye_region(0.55, 0.50, left_ear) if left_ear is not None else []
    right = _eye_region(0.45, 0.50, right_ear) if right_ear is not None else []
    return FaceData(
        bbox_pixels=(400, 160, 880, 560),
        bbox_normalized=(0.31, 0.22, 0.69, 0.78),
        area=100.0,
        confidence=0.98,
        landmarks=[(0.5, 0.5, 0.0)] * 478,
        left_eye=left,
        right_eye=right,
        left_iris=None,
        right_iris=None,
        nose_tip=(0.5, 0.5),
        mouth=[],
        face_outline=[],
    )


def make_invalid_face() -> FaceData:
    """Face with corrupt eye coordinates in BOTH eyes (no valid fallback)."""
    face = make_face(None, None)
    bad = (float("nan"), 0.5)
    face.left_eye = [bad, bad, bad, bad, bad, bad]
    face.right_eye = [bad, bad, bad, bad, bad, bad]
    return face


def make_zerowidth_face() -> FaceData:
    """Face whose left eye has coincident corners (zero width)."""
    face = make_face(None, CLOSED_EAR)
    face.left_eye = [(0.5, 0.5)] * 6
    return face


# ── EAR maths ───────────────────────────────────────────────────────────────

class TestEarMath:
    def test_ear_matches_constructed_geometry(self):
        for ear in (0.375, 0.30, 0.25, 0.15, 0.05):
            region = _eye_region(0.5, 0.5, ear)
            assert compute_ear(region) == pytest.approx(ear, abs=1e-9)

    def test_ear_open_eye_above_threshold(self):
        assert compute_ear(_eye_region(0.5, 0.5, OPEN_EAR)) > _s.ear_threshold

    def test_ear_closed_eye_below_threshold(self):
        assert compute_ear(_eye_region(0.5, 0.5, CLOSED_EAR)) < _s.ear_threshold

    def test_ear_requires_six_points(self):
        region = _eye_region(0.5, 0.5, OPEN_EAR)
        assert compute_ear(region[:5]) is None
        assert compute_ear([]) is None
        assert compute_ear(None) is None

    def test_ear_zero_width_is_invalid(self):
        assert compute_ear([(0.5, 0.5)] * 6) is None  # p1 == p4 → no horizontal span

    def test_ear_nonfinite_coords_are_invalid(self):
        assert compute_ear([(float("nan"), 0.5)] * 6) is None
        assert compute_ear([(float("inf"), 0.5)] * 6) is None

    def test_ear_resists_garbage(self):
        assert compute_ear([(None, 1), (1, 1), (1, 1), (2, 1), (1, 2), (1, 2)]) is None


# ── estimator ───────────────────────────────────────────────────────────────

class TestEstimator:
    def _est(self):
        return DrowsinessEstimator()

    def test_open_eyes_valid_average(self):
        reading = self._est().compute_reading(make_face(OPEN_EAR, OPEN_EAR))
        assert reading.valid
        assert reading.avg_ear == pytest.approx(OPEN_EAR, abs=1e-9)
        assert reading.left_ear == pytest.approx(OPEN_EAR, abs=1e-9)
        assert reading.right_ear == pytest.approx(OPEN_EAR, abs=1e-9)
        assert reading.error is None

    def test_left_eye_only_fallback(self):
        reading = self._est().compute_reading(make_face(OPEN_EAR, None))
        assert reading.valid
        assert reading.left_ear is not None
        assert reading.right_ear is None
        assert reading.avg_ear == pytest.approx(OPEN_EAR, abs=1e-9)

    def test_right_eye_only_fallback(self):
        reading = self._est().compute_reading(make_face(None, OPEN_EAR))
        assert reading.valid
        assert reading.left_ear is None
        assert reading.right_ear is not None
        assert reading.avg_ear == pytest.approx(OPEN_EAR, abs=1e-9)

    def test_both_eyes_average(self):
        reading = self._est().compute_reading(make_face(0.30, 0.10))
        assert reading.valid
        assert reading.avg_ear == pytest.approx(0.20, abs=1e-9)

    def test_missing_both_eyes_invalid(self):
        reading = self._est().compute_reading(make_face(None, None))
        assert not reading.valid
        assert reading.avg_ear is None
        assert reading.error is not None

    def test_missing_face_invalid(self):
        reading = self._est().compute_reading(None)
        assert not reading.valid

    def test_invalid_coords_invalid(self):
        reading = self._est().compute_reading(make_invalid_face())
        assert not reading.valid
        assert reading.avg_ear is None

    def test_zero_width_degrades_to_usable_eye(self):
        reading = self._est().compute_reading(make_zerowidth_face())
        assert reading.valid
        assert reading.left_ear is None
        assert reading.right_ear == pytest.approx(CLOSED_EAR, abs=1e-9)
        assert reading.avg_ear == pytest.approx(CLOSED_EAR, abs=1e-9)


# ── detector (temporal + blink) ─────────────────────────────────────────────

class TestDetector:
    def _detector(self, **overrides):
        return DrowsinessDetector(make_settings(**overrides))

    def test_no_face_is_unknown_no_fake_values(self):
        det = self._detector()
        r = det.update(None, now=0.0)
        assert r.state == STATE_UNKNOWN
        assert r.ear is None and r.left_ear is None and r.right_ear is None
        assert r.eyes_closed is None
        assert r.confidence == 0.0

    def test_open_eye_stays_normal(self):
        det = self._detector()
        face = make_face(OPEN_EAR, OPEN_EAR)
        # The temporal smoother starts UNKNOWN and needs a few frames to confirm NORMAL.
        states = []
        for i in range(15):
            r = det.update(face, now=float(i) * 0.1)
            states.append(r.state)
            assert r.eyes_closed is False
        # After warmup, every state must be NORMAL.
        assert all(s == STATE_NORMAL for s in states[3:]), states

    def test_closed_eye_requires_persistence(self):
        det = self._detector(ear_consecutive_frames=12, drowsiness_duration_threshold=0.8)
        face = make_face(CLOSED_EAR, CLOSED_EAR)
        # Short closure (few frames, under the frame threshold) is a blink.
        for i in range(6):
            r = det.update(face, now=float(i) * 0.1)
            # During the temporal warmup and blink window, state must NOT be DROWSY.
            assert r.state in (STATE_UNKNOWN, STATE_NORMAL)
        # Continue closure past the frame + duration thresholds → DROWSY.
        drowsy_seen = False
        for i in range(6, 40):
            r = det.update(face, now=float(i) * 0.1)
            if r.state == STATE_DROWSY:
                drowsy_seen = True
                break
        assert drowsy_seen
        assert r.eyes_closed is True

    def test_blink_never_reports_drowsy(self):
        det = self._detector(ear_consecutive_frames=12, drowsiness_duration_threshold=0.8)
        # A quick blink (a handful of closed frames) then reopen → never DROWSY.
        for i in range(6):
            r = det.update(make_face(CLOSED_EAR, CLOSED_EAR), now=float(i) * 0.05)
            assert r.state in (STATE_UNKNOWN, STATE_NORMAL)
        for i in range(6, 30):
            r = det.update(make_face(OPEN_EAR, OPEN_EAR), now=float(i) * 0.05)
            if i < 10:
                # Still within the temporal warmup or just after the blink.
                assert r.state in (STATE_UNKNOWN, STATE_NORMAL)
            else:
                assert r.state == STATE_NORMAL

    def test_temporal_smoothing_resists_single_frame_flip(self):
        det = self._detector(ear_consecutive_frames=3, drowsiness_duration_threshold=0.2)
        closed = make_face(CLOSED_EAR, CLOSED_EAR)
        open_face = make_face(OPEN_EAR, OPEN_EAR)
        for i in range(6):
            det.update(closed, now=float(i) * 0.1)
        assert det.update(closed, now=0.6).state == STATE_DROWSY
        # A single open frame must not instantly un-drowsify (majority vote).
        assert det.update(open_face, now=0.7).state == STATE_DROWSY

    def test_sustained_open_clears_drowsy(self):
        det = self._detector(ear_consecutive_frames=3, drowsiness_duration_threshold=0.2)
        closed = make_face(CLOSED_EAR, CLOSED_EAR)
        open_face = make_face(OPEN_EAR, OPEN_EAR)
        for i in range(6):
            det.update(closed, now=float(i) * 0.1)
        assert det.update(closed, now=0.6).state == STATE_DROWSY
        # Reopen; the smoother needs a few frames of NORMAL to confirm the switch.
        normal_seen = False
        for i in range(7, 20):
            r = det.update(open_face, now=float(i) * 0.1)
            if r.state == STATE_NORMAL:
                normal_seen = True
                break
        assert normal_seen

    def test_no_face_after_drowsy_decays_to_unknown(self):
        det = self._detector(ear_consecutive_frames=2, drowsiness_duration_threshold=0.1)
        for i in range(4):
            det.update(make_face(CLOSED_EAR, CLOSED_EAR), now=float(i) * 0.1)
        assert det.update(make_face(CLOSED_EAR, CLOSED_EAR), now=0.4).state == STATE_DROWSY
        for i in range(5, 15):
            r = det.update(None, now=float(i) * 0.1)
            if r.state == STATE_UNKNOWN:
                break
        assert r.state == STATE_UNKNOWN

    def test_disabled_returns_unknown(self):
        det = self._detector(drowsiness_enabled=False)
        r = det.update(make_face(OPEN_EAR, OPEN_EAR), now=1.0)
        assert r.state == STATE_UNKNOWN
        assert r.confidence == 0.0
        assert r.error is not None

    def test_confidence_reflects_signal_quality(self):
        det = self._detector()
        both = det.update(make_face(OPEN_EAR, OPEN_EAR), now=0.0)
        assert both.confidence > 0.9  # both eyes, far from threshold
        single = det.update(make_face(OPEN_EAR, None), now=0.0)
        assert single.confidence == pytest.approx(0.5, abs=0.05)  # half coverage


# ── result serialisation ────────────────────────────────────────────────────

class TestResultSerialisation:
    def test_to_dict_is_honest_for_unknown(self):
        d = DrowsinessResult().to_dict()
        assert d["state"] == STATE_UNKNOWN
        assert d["ear"] is None
        assert d["eyes_closed"] is None
        assert d["confidence"] == 0.0

    def test_to_dict_round_trips_measured_values(self):
        r = DrowsinessResult(
            state=STATE_DROWSY,
            ear=0.12345,
            left_ear=0.12345,
            right_ear=0.12345,
            eyes_closed=True,
            confidence=1.0,
        ).to_dict()
        assert r["state"] == STATE_DROWSY
        assert r["ear"] == 0.123
        assert r["eyes_closed"] is True