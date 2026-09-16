"""Tests for the screen-distance estimation module (Phase 5).

Uses fully synthetic faces whose interocular width (pixel distance between
landmark 33 and 263) is controlled exactly, so the relative-ratio thresholds,
hysteresis band, streak persistence and temporal smoothing are deterministic.
Physical-accuracy against real cameras is out of scope (and explicitly not
claimed by the module), but the maths, fallbacks, calibration and honest
UNKNOWN handling are all covered.
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

from ai.distance_detection import (  # noqa: E402
    STATE_NORMAL,
    STATE_TOO_CLOSE,
    STATE_UNKNOWN,
    DistanceDetector,
    DistanceEstimator,
    DistanceResult,
    CAL_CONFIRMED,
    CAL_REQUIRED,
    CAL_COLLECTING,
    CAL_READY,
)
from ai.face_detection import FaceData  # noqa: E402
from config import get_settings  # noqa: E402

_s = get_settings()

FRAME_W = 1000
FRAME_H = 700
REF = 40.0  # reference interocular width (px)


def make_settings(**overrides):
    return replace(_s, **overrides)


# ── synthetic geometry ───────────────────────────────────────────────────────

def make_face(width_px: float, frame_w: int = FRAME_W, frame_h: int = FRAME_H,
              invalid_corners: bool = False, clipped: bool = False,
              no_landmarks: bool = False) -> FaceData:
    """Face whose outer-eye-corner landmark distance is exactly ``width_px``."""
    cx, cy = 0.5, 0.5
    half = (width_px / frame_w) / 2.0
    lms = [(0.5, 0.5, 0.0)] * 478
    if not no_landmarks:
        if invalid_corners:
            lms[33] = (float("nan"), cy, 0.0)
            lms[263] = (float("nan"), cy, 0.0)
        else:
            lms[33] = (cx - half, cy, 0.0)
            lms[263] = (cx + half, cy, 0.0)

    if clipped:
        bbox_norm = (0.0, 0.1, 0.9, 0.9)
        bbox_px = (int(0.0 * frame_w), int(0.1 * frame_h),
                   int(0.9 * frame_w), int(0.9 * frame_h))
        area = (bbox_px[2] - bbox_px[0]) * (bbox_px[3] - bbox_px[1])
    else:
        bbox_norm = (0.1, 0.1, 0.9, 0.9)
        bbox_px = (int(0.1 * frame_w), int(0.1 * frame_h),
                   int(0.9 * frame_w), int(0.9 * frame_h))
        area = (bbox_px[2] - bbox_px[0]) * (bbox_px[3] - bbox_px[1])

    return FaceData(
        bbox_pixels=bbox_px,
        bbox_normalized=bbox_norm,
        area=float(area),
        confidence=0.98,
        landmarks=lms,
        left_eye=[],
        right_eye=[],
        left_iris=None,
        right_iris=None,
        nose_tip=(0.5, 0.5),
        mouth=[],
        face_outline=[],
    )


def _warmup(det, face, n: int = 8, dt: float = 0.1):
    """Feed normal-distance frames so the smoother reaches NORMAL."""
    for i in range(n):
        det.update(face, now=float(i) * dt)


# ── estimator ───────────────────────────────────────────────────────────────

class TestEstimator:
    def _est(self, **kw):
        return DistanceEstimator(FRAME_W, FRAME_H, _s.distance_min_face_width)

    def test_interocular_width_is_exact(self):
        reading = self._est().compute_reading(make_face(60.0))
        assert reading.valid
        assert reading.face_width_px == pytest.approx(60.0, abs=1e-6)
        assert reading.method == "interocular"

    def test_no_face_invalid(self):
        reading = self._est().compute_reading(None)
        assert not reading.valid
        assert reading.face_width_px is None

    def test_invalid_corners_fall_back_to_bbox(self):
        reading = self._est().compute_reading(make_face(60.0, invalid_corners=True))
        assert reading.valid
        assert reading.method == "bbox"
        assert reading.face_width_px == pytest.approx(0.8 * FRAME_W, abs=1)

    def test_clipped_bbox_invalid(self):
        reading = self._est().compute_reading(make_face(60.0, no_landmarks=True, clipped=True))
        assert not reading.valid

    def test_small_face_below_min_invalid(self):
        reading = self._est().compute_reading(make_face(20.0))
        assert not reading.valid
        assert "too small" in (reading.error or "")

    def test_zero_area_ignored(self):
        face = make_face(60.0, invalid_corners=True)
        face.area = 0.0
        reading = self._est().compute_reading(face)
        assert not reading.valid


# ── detector: calibration ────────────────────────────────────────────────────

class TestCalibration:
    def _detector(self, **kw):
        return DistanceDetector(make_settings(**kw), FRAME_W, FRAME_H)

    def test_configured_reference_is_confirmed(self):
        det = self._detector(distance_reference_width=REF)
        assert det.reference_width == REF
        assert det.calibration_status()["status"] == CAL_CONFIRMED

    def test_auto_calibration_captures_reference(self):
        det = self._detector(distance_reference_width=0.0, distance_reference_samples=3)
        assert det.calibration_status()["status"] == CAL_REQUIRED
        face = make_face(50.0)
        for i in range(2):
            det.update(face, now=float(i) * 0.1)
            assert det.reference_width is None
            assert det.calibration_status()["status"] in (CAL_REQUIRED, CAL_COLLECTING)
        det.update(face, now=2.0)  # third stable sample locks the reference
        assert det.reference_width == pytest.approx(50.0, abs=1e-6)
        assert det.calibration_status()["status"] == CAL_READY

    def test_cancel_calibration_keeps_no_reference(self):
        det = self._detector(distance_reference_width=0.0)
        det.start_calibration()
        det.cancel_calibration()
        assert det.reference_width is None

    def test_recalibrate_replaces_reference(self):
        det = self._detector(distance_reference_width=REF)
        det.start_calibration()
        for i in range(_s.distance_reference_samples):
            det.update(make_face(80.0), now=float(i) * 0.1)
        assert det.reference_width == pytest.approx(80.0, abs=1e-6)


# ── detector: classification ─────────────────────────────────────────────────

class TestDetector:
    def _detector(self, **kw):
        default = dict(
            distance_reference_width=REF,
            distance_too_close_frames=4,
            distance_smoothing_window=5,
        )
        default.update(kw)
        return DistanceDetector(make_settings(**default), FRAME_W, FRAME_H)

    def test_normal_distance(self):
        det = self._detector()
        _warmup(det, make_face(REF))
        r = det.update(make_face(REF), now=10.0)
        assert r.state == STATE_NORMAL
        assert r.distance_estimate == pytest.approx(1.0, abs=1e-3)
        assert r.too_close is False
        assert r.face_width == pytest.approx(REF, abs=1e-6)
        assert r.confidence > 0

    def test_too_close_persists(self):
        det = self._detector()
        _warmup(det, make_face(REF))
        close = make_face(2.0 * REF)  # ratio 2.0 ≫ 1.45
        state = STATE_NORMAL
        for i in range(10):
            r = det.update(close, now=20.0 + i * 0.1)
            state = r.state
            if state == STATE_TOO_CLOSE:
                break
        assert state == STATE_TOO_CLOSE
        assert r.too_close is True
        assert r.distance_estimate == pytest.approx(2.0, abs=1e-3)

    def test_no_face_is_unknown_nothing_fabricated(self):
        det = self._detector()
        _warmup(det, make_face(REF))
        r = det.update(None, now=30.0)
        assert r.state in (STATE_UNKNOWN, STATE_NORMAL)  # smoother holds briefly
        # After enough consecutive missing frames it must decay to UNKNOWN.
        for i in range(_s.distance_away_duration + 1):
            r = det.update(None, now=31.0 + i * 0.1)
        assert r.state == STATE_UNKNOWN
        assert r.distance_estimate is None
        assert r.face_width is None
        assert r.too_close is None
        assert r.confidence == 0.0

    def test_invalid_landmarks_falls_back_bbox(self):
        det = self._detector()
        # bbox width is 800px, ratio 800/40 = 20 → too close (bbox source trusted less)
        face = make_face(REF, invalid_corners=True)
        _warmup(det, face)
        for i in range(10):
            r = det.update(face, now=40.0 + i * 0.1)
            if r.state == STATE_TOO_CLOSE:
                break
        assert r.face_width == pytest.approx(0.8 * FRAME_W, abs=1)
        assert r.state == STATE_TOO_CLOSE

    def test_small_face_is_unknown(self):
        det = self._detector()
        _warmup(det, make_face(REF))
        r = det.update(make_face(20.0), now=50.0)  # below min width
        for _ in range(_s.distance_away_duration + 1):
            r = det.update(make_face(20.0), now=52.0)
        assert r.state == STATE_UNKNOWN
        assert r.face_width is None

    def test_disabled_returns_unknown(self):
        det = self._detector(distance_enabled=False)
        r = det.update(make_face(REF), now=1.0)
        assert r.state == STATE_UNKNOWN
        assert r.confidence == 0.0
        assert r.error is not None


# ── temporal smoothing + hysteresis ──────────────────────────────────────────

class TestTemporal:
    def _detector(self, **kw):
        default = dict(
            distance_reference_width=REF,
            distance_too_close_frames=4,
            distance_smoothing_window=5,
        )
        default.update(kw)
        return DistanceDetector(make_settings(**default), FRAME_W, FRAME_H)

    def test_single_noise_frame_does_not_flip(self):
        det = self._detector()
        _warmup(det, make_face(REF))
        # One very-close frame must not unseat a stable NORMAL (majority vote).
        r = det.update(make_face(4.0 * REF), now=10.0)
        assert r.state == STATE_NORMAL

    def test_brief_approach_stays_normal(self):
        det = self._detector()
        _warmup(det, make_face(REF))
        # A few close frames below the streak threshold stay NORMAL.
        for i in range(3):
            r = det.update(make_face(2.0 * REF), now=20.0 + i * 0.1)
            assert r.state == STATE_NORMAL

    def test_borderline_hysteresis_holds_state(self):
        det = self._detector()
        _warmup(det, make_face(REF))
        # ratios strictly between recover (1.25) and too_close (1.45) hold NORMAL.
        for i in range(8):
            r = det.update(make_face(REF * 1.35), now=30.0 + i * 0.1)
            assert r.state == STATE_NORMAL

    def test_return_from_too_close_to_normal(self):
        det = self._detector()
        _warmup(det, make_face(REF))
        for i in range(10):
            r = det.update(make_face(2.0 * REF), now=40.0 + i * 0.1)
            if r.state == STATE_TOO_CLOSE:
                break
        assert r.state == STATE_TOO_CLOSE
        normal_seen = False
        for i in range(10, 30):
            r = det.update(make_face(REF), now=50.0 + i * 0.1)
            if r.state == STATE_NORMAL:
                normal_seen = True
                break
        assert normal_seen
        assert r.too_close is False


# ── result serialisation ─────────────────────────────────────────────────────

class TestResultSerialisation:
    def test_to_dict_is_honest_for_unknown(self):
        d = DistanceResult().to_dict()
        assert d["state"] == STATE_UNKNOWN
        assert d["distance_estimate"] is None
        assert d["face_width"] is None
        assert d["too_close"] is None
        assert d["confidence"] == 0.0

    def test_to_dict_round_trips_measured_values(self):
        r = DistanceResult(
            state=STATE_TOO_CLOSE,
            distance_estimate=1.45678,
            face_width=123.456,
            too_close=True,
            confidence=1.0,
            reference_width=80.0,
        ).to_dict()
        assert r["state"] == STATE_TOO_CLOSE
        assert r["distance_estimate"] == 1.457
        assert r["face_width"] == 123.5
        assert r["too_close"] is True