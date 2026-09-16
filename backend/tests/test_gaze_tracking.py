"""Tests for the gaze / attention tracking module (Phase 3).

Uses synthetic, geometrically-consistent faces:

* ``_make_eye_face`` builds eyes and irises at controlled normalised offsets so
  the iris signal is fully deterministic.
* ``_make_head_face`` projects the Astrakhantsev model through the same camera
  parameters used by :class:`GazeEstimator`, so ``solvePnP`` recovers a known
  rotation — this is also the empirical anchor for the yaw/pitch *sign*
  conventions used elsewhere in the module.

Detection accuracy against real faces is not covered here (requires a camera)
but the maths, classifications, calibration lifecycle and smoothing are.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest

_BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from ai.face_detection import FaceData  # noqa: E402
from ai.gaze_tracking import (  # noqa: E402
    HEAD_MODEL_INDICES,
    HEAD_MODEL_POINTS,
    DIRECTION_AT_SCREEN,
    DIRECTION_LEFT,
    DIRECTION_RIGHT,
    DIRECTION_UP,
    DIRECTION_DOWN,
    DIRECTION_AWAY,
    DIRECTION_UNKNOWN,
    GazeEstimator,
    GazeTracker,
    GazeReading,
    draw_gaze_overlay,
)
from config import get_settings  # noqa: E402

FRAME_W, FRAME_H = 1280, 720
_CAM = np.array([[FRAME_W, 0, FRAME_W / 2], [0, FRAME_W, FRAME_H / 2], [0, 0, 1]], np.float64)
_TV = np.array([[0.0], [0.0], [300.0]], np.float64)

_LEFT_EYE_CENTER = (0.55, 0.50)   # subject's left eye sits on the image right
_RIGHT_EYE_CENTER = (0.45, 0.50)  # subject's right eye sits on the image left
_EYE_W, _EYE_H = 0.08, 0.07


def make_settings(**overrides):
    return replace(get_settings(), **overrides)


def _blank_landmarks(n=478):
    return [(0.5, 0.5, 0.0)] * n


def _eye_region(center):
    cx, cy = center
    half_w, half_h = _EYE_W / 2, _EYE_H / 2
    return [
        (cx - half_w, cy),
        (cx - half_w * 0.4, cy - half_h),
        (cx + half_w * 0.4, cy - half_h),
        (cx + half_w, cy),
        (cx + half_w * 0.4, cy + half_h),
        (cx - half_w * 0.4, cy + half_h),
    ]


def make_eye_face(iris_dx=0.0, iris_dy=0.0, w_head: float = 0.4, w_iris: float = 0.6):
    """Face with controlled iris displacements (as fraction of eye size)."""
    lcx, lcy = _LEFT_EYE_CENTER
    rcx, rcy = _RIGHT_EYE_CENTER
    left_iris = (lcx + iris_dx * _EYE_W, lcy + iris_dy * _EYE_H)
    right_iris = (rcx + iris_dx * _EYE_W, rcy + iris_dy * _EYE_H)
    lm = _blank_landmarks()
    return FaceData(
        bbox_pixels=(400, 160, 880, 560),
        bbox_normalized=(0.31, 0.22, 0.69, 0.78),
        area=100.0,
        confidence=0.98,
        landmarks=lm,
        left_eye=_eye_region(_LEFT_EYE_CENTER),
        right_eye=_eye_region(_RIGHT_EYE_CENTER),
        left_iris=left_iris,
        right_iris=right_iris,
        nose_tip=(0.5, 0.5),
        mouth=[],
        face_outline=[],
    )


def make_head_face(yaw_deg=0.0, pitch_deg=0.0, **eye_kwargs):
    """Project the head model through the estimator's camera at a known pose."""
    face = make_eye_face(**eye_kwargs)
    rvec = np.array([[np.deg2rad(pitch_deg)], [np.deg2rad(yaw_deg)], [0.0]], np.float64)
    img, _ = cv2.projectPoints(np.array(HEAD_MODEL_POINTS, np.float64), rvec, _TV, _CAM, np.zeros(5))
    pts = img.reshape(-1, 2)
    lm = face.landmarks
    for idx, (u, v) in zip(HEAD_MODEL_INDICES, pts):
        lm[idx] = (u / FRAME_W, v / FRAME_H, 0.0)
    return face


# ── estimator: iris ──────────────────────────────────────────────────────────

class TestIrisSignal:
    def test_iris_shift_to_image_right_reads_left(self):
        est = GazeEstimator(None, FRAME_W, FRAME_H)
        r = est.compute_reading(make_eye_face(iris_dx=0.25))
        assert r.iris_ok
        assert r.iris_offset_x is not None and r.iris_offset_x > 0  # student's left
        assert abs(r.iris_offset_y or 0) < 0.01

    def test_iris_shift_to_image_left_reads_right(self):
        est = GazeEstimator(None, FRAME_W, FRAME_H)
        r = est.compute_reading(make_eye_face(iris_dx=-0.25))
        assert r.iris_offset_x is not None and r.iris_offset_x < 0

    def test_iris_down_reads_down(self):
        est = GazeEstimator(None, FRAME_W, FRAME_H)
        r = est.compute_reading(make_eye_face(iris_dy=0.25))
        assert r.iris_offset_y is not None and r.iris_offset_y < 0  # up is positive

    def test_iris_up_reads_up(self):
        est = GazeEstimator(None, FRAME_W, FRAME_H)
        r = est.compute_reading(make_eye_face(iris_dy=-0.25))
        assert r.iris_offset_y is not None and r.iris_offset_y > 0

    def test_missing_irises_means_no_iris_signal(self):
        face = make_eye_face(iris_dx=0.25)
        face.left_iris = None
        face.right_iris = None
        est = GazeEstimator(None, FRAME_W, FRAME_H)
        r = est.compute_reading(face)
        assert not r.iris_ok


# ── estimator: head pose (signs are the empirical anchor) ────────────────────

class TestHeadPose:
    def test_yaw_positive_is_student_left(self):
        est = GazeEstimator(None, FRAME_W, FRAME_H)
        r = est.compute_reading(make_head_face(yaw_deg=15))
        assert r.head_ok
        assert r.head_yaw_deg is not None and r.head_yaw_deg > 0
        assert abs(r.head_yaw_deg - 15) < 4

    def test_yaw_negative_is_student_right(self):
        est = GazeEstimator(None, FRAME_W, FRAME_H)
        r = est.compute_reading(make_head_face(yaw_deg=-15))
        assert r.head_yaw_deg is not None and r.head_yaw_deg < 0

    def test_pitch_positive_is_nose_down(self):
        est = GazeEstimator(None, FRAME_W, FRAME_H)
        r = est.compute_reading(make_head_face(pitch_deg=10))
        assert r.head_pitch_deg is not None and r.head_pitch_deg > 0

    def test_pitch_negative_is_nose_up(self):
        est = GazeEstimator(None, FRAME_W, FRAME_H)
        r = est.compute_reading(make_head_face(pitch_deg=-10))
        assert r.head_pitch_deg is not None and r.head_pitch_deg < 0


# ── tracker: classification + smoothing ──────────────────────────────────────

class TestTrackerClassification:
    def _tracker(self, **over):
        return GazeTracker(make_settings(**over), FRAME_W, FRAME_H)

    def test_smoothed_left(self):
        t = self._tracker()
        face = make_head_face(yaw_deg=15, iris_dx=0.25)
        result = None
        for _ in range(5):
            result = t.update(face)
        assert result.direction == DIRECTION_LEFT
        assert result.attention_score is not None and 0 <= result.attention_score <= 100
        assert result.confidence is not None and result.confidence > 0

    def test_smoothed_right(self):
        t = self._tracker()
        face = make_head_face(yaw_deg=-15, iris_dx=-0.25)
        for _ in range(5):
            t.update(face)
        assert t.update(face).direction == DIRECTION_RIGHT

    def test_smoothed_up(self):
        t = self._tracker()
        face = make_head_face(pitch_deg=-10, iris_dy=-0.25)
        for _ in range(5):
            t.update(face)
        assert t.update(face).direction == DIRECTION_UP

    def test_smoothed_down(self):
        t = self._tracker()
        face = make_head_face(pitch_deg=10, iris_dy=0.25)
        for _ in range(5):
            t.update(face)
        assert t.update(face).direction == DIRECTION_DOWN

    def test_centered_is_at_screen_with_high_attention(self):
        t = self._tracker()
        face = make_head_face(yaw_deg=1.0, pitch_deg=-1.0, iris_dx=0.02, iris_dy=-0.02)
        for _ in range(5):
            t.update(face)
        r = t.update(face)
        assert r.direction == DIRECTION_AT_SCREEN
        assert r.attention_score is not None and r.attention_score >= 85

    def test_no_face_is_unknown_then_away(self):
        t = self._tracker(gaze_away_duration=3)
        assert t.update(None).direction == DIRECTION_UNKNOWN
        t.update(None)
        assert t.update(None).direction == DIRECTION_AWAY

    def test_invalid_face_fields_never_crash(self):
        t = self._tracker()
        face = make_eye_face()
        face.landmarks = []  # head pose impossible
        face.left_eye = []
        face.right_eye = []
        face.left_iris = None
        face.right_iris = None
        r = t.update(face)
        assert r.direction in (DIRECTION_UNKNOWN, DIRECTION_AWAY)
        assert r.attention_score is None

    def test_gaze_result_dict_shape(self):
        t = self._tracker()
        face = make_head_face(yaw_deg=15, iris_dx=0.25)
        for _ in range(5):
            t.update(face)
        d = t.update(face).to_dict()
        for key in ("direction", "attention_score", "confidence", "head_yaw_deg",
                    "head_pitch_deg", "head_roll_deg", "iris_offset_x",
                    "iris_offset_y", "deviation_x", "deviation_y", "error"):
            assert key in d
        assert d["direction"] == DIRECTION_LEFT


# ── calibration ──────────────────────────────────────────────────────────────

class TestCalibration:
    def _tracker(self, **over):
        defaults = dict(gaze_calibration_samples=4, gaze_calibration_accept_range=0.30)
        defaults.update(over)
        return GazeTracker(make_settings(**defaults), FRAME_W, FRAME_H)

    def test_full_lifecycle(self):
        t = self._tracker()
        assert t.calibration_status()["status"] == "REQUIRED"
        t.start_calibration()
        assert t.calibration_status()["status"] == "COLLECTING"
        face = make_eye_face(iris_dx=0.10)
        for _ in range(6):
            t.update(face)
        status = t.calibration_status()
        assert status["status"] == "READY"
        assert status["baseline_x"] is not None
        assert abs(status["baseline_x"] - 0.20) < 0.05  # iris_dx 0.10 → offset 0.20

    def test_cancel_returns_to_required(self):
        t = self._tracker()
        t.start_calibration()
        t.cancel_calibration()
        assert t.calibration_status()["status"] == "REQUIRED"

    def test_calibration_shifts_centered_gaze(self):
        t = self._tracker()
        t.start_calibration()
        for _ in range(6):
            t.update(make_eye_face(iris_dx=0.10))  # resting bias of offset 0.20 (within range)
        assert t.calibration_status()["status"] == "READY"
        # After calibration the previously-"left" face is now roughly centred.
        r = None
        for _ in range(5):
            r = t.update(make_eye_face(iris_dx=0.10))
        assert r.iris_offset_x is not None and abs(r.iris_offset_x) < 0.15

    def test_fails_when_samples_too_skewed(self):
        t = self._tracker(gaze_calibration_accept_range=0.05)
        t.start_calibration()
        t.update(make_eye_face(iris_dx=0.6))  # will be rejected every time
        for _ in range(120):
            t.update(make_eye_face(iris_dx=0.6))
        status = t.calibration_status()
        assert status["status"] == "FAILED"
        assert status.get("error")


# ── overlay drawing ──────────────────────────────────────────────────────────

def test_overlay_draws_without_error():
    from ai.gaze_tracking import GazeResult

    frame = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
    face = make_eye_face()
    result = GazeResult(direction=DIRECTION_LEFT, attention_score=77, confidence=0.8,
                        deviation_x=0.5, deviation_y=0.0)
    out = draw_gaze_overlay(frame, face, result)
    assert out.shape == frame.shape
    assert out.dtype == frame.dtype

    # Away result draws the red chip and crosshair safely too
    away = GazeResult(direction=DIRECTION_AWAY)
    out = draw_gaze_overlay(frame, face, away, show_crosshair=True)
    assert out.shape == frame.shape


def test_reading_defaults_are_safe():
    r = GazeReading()
    assert r.valid is False
    assert r.iris_ok is False
    assert r.head_ok is False