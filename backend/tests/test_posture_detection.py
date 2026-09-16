"""Tests for the sitting-posture estimation module (Phase 6).

Uses fully synthetic faces whose head-pitch ratio (nose–eye–chin) and face
vertical offset are controlled exactly, so the posture score maths, the
GOOD/MODERATE/POOR classification and the temporal smoothing are deterministic.
Physical accuracy against real cameras is out of scope (and explicitly not
claimed by the module — it is a head-pitch heuristic, not full body analysis).
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import pytest

_BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from ai.posture_detection import (  # noqa: E402
    STATE_GOOD,
    STATE_MODERATE,
    STATE_POOR,
    STATE_UNKNOWN,
    PostureDetector,
    PostureReading,
    compute_posture_score,
)
from ai.face_detection import FaceData  # noqa: E402
from config import get_settings  # noqa: E402

_s = get_settings()

FRAME_W = 1000
FRAME_H = 700


def make_settings(**overrides):
    return replace(_s, **overrides)


# ── synthetic geometry ───────────────────────────────────────────────────────

def make_face(
    ratio: float = 0.50,  # (nose_y - eye_y) / (chin_y - eye_y)
    face_cy: float = 0.40,
    frame_w: int = FRAME_W,
    frame_h: int = FRAME_H,
    no_landmarks: bool = False,
    nan_landmarks: bool = False,
    degenerate: bool = False,
) -> FaceData:
    """Face whose head-pitch ratio and frame position are controlled exactly."""
    eye_y = 0.40
    eye_to_chin = 0.30
    chin_y = eye_y + eye_to_chin
    nose_y = eye_y + ratio * eye_to_chin

    if no_landmarks:
        lms: list = []
    else:
        lms = [(0.5, 0.5, 0.0)] * 478
        if nan_landmarks:
            lms[4] = (float("nan"), nose_y, 0.0)
            lms[33] = (0.35, eye_y, 0.0)
            lms[263] = (0.65, eye_y, 0.0)
            lms[152] = (0.5, chin_y, 0.0)
        elif degenerate:
            lms[4] = (0.5, eye_y, 0.0)          # nose at same Y as eyes
            lms[33] = (0.35, eye_y, 0.0)
            lms[263] = (0.65, eye_y, 0.0)
            lms[152] = (0.5, eye_y, 0.0)         # chin at same Y as eyes
        else:
            lms[4] = (0.5, nose_y, 0.0)
            lms[33] = (0.35, eye_y, 0.0)
            lms[263] = (0.65, eye_y, 0.0)
            lms[152] = (0.5, chin_y, 0.0)

    half_h = 0.15
    bbox_norm = (0.2, face_cy - half_h, 0.8, face_cy + half_h)
    bbox_px = (int(0.2 * frame_w), int(bbox_norm[1] * frame_h),
               int(0.8 * frame_w), int(bbox_norm[3] * frame_h))
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
        nose_tip=(0.5, 0.40 + ratio * 0.30),
        mouth=[],
        face_outline=[],
    )


def _warmup(det: PostureDetector, face: FaceData, n: int = 8, state: str = STATE_GOOD):
    """Feed upright frames so the smoother reaches GOOD (or the given state)."""
    for _ in range(n):
        det.update(face)


# ── score maths ──────────────────────────────────────────────────────────────

class TestEstimator:
    def test_upright_head_scores_high(self):
        reading = compute_posture_score(make_face(ratio=0.50), FRAME_H)
        assert reading.valid
        # ratio 0.50 → head pitch 75; face centre 0.40 → offset 80
        assert reading.head_pitch_score == pytest.approx(75.0, abs=0.1)
        assert reading.face_offset_score == pytest.approx(80.0, abs=0.1)
        assert reading.raw_score == pytest.approx(76.75, abs=0.1)

    def test_slouched_head_scores_low(self):
        reading = compute_posture_score(make_face(ratio=0.65, face_cy=0.55), FRAME_H)
        assert reading.valid
        assert reading.head_pitch_score == pytest.approx(0.0, abs=0.1)
        assert reading.face_offset_score == pytest.approx(20.0, abs=0.1)
        assert reading.raw_score == pytest.approx(7.0, abs=0.1)

    def test_ratio_mapping_is_clamped(self):
        very_upright = compute_posture_score(make_face(ratio=0.20), FRAME_H)
        very_slouched = compute_posture_score(make_face(ratio=0.95), FRAME_H)
        assert very_upright.head_pitch_score == 100.0
        assert very_slouched.head_pitch_score == 0.0

    def test_no_landmarks_invalid(self):
        reading = compute_posture_score(make_face(no_landmarks=True), FRAME_H)
        assert not reading.valid
        assert reading.error  # honest, not fabricated

    def test_sparse_landmarks_invalid(self):
        face = make_face()
        face.landmarks = [(0.5, 0.5, 0.0)] * 5
        reading = compute_posture_score(face, FRAME_H)
        assert not reading.valid

    def test_nan_landmarks_invalid(self):
        reading = compute_posture_score(make_face(nan_landmarks=True), FRAME_H)
        assert not reading.valid
        assert reading.error

    def test_degenerate_geometry_mid_scale_not_crash(self):
        reading = compute_posture_score(make_face(degenerate=True), FRAME_H)
        assert reading.valid  # no crash; falls back to mid-range head pitch


# ── stateful detector ────────────────────────────────────────────────────────

class TestDetector:
    def test_warmup_reaches_good(self):
        det = PostureDetector(make_settings())
        _warmup(det, make_face(ratio=0.50, face_cy=0.40))
        result = det.update(make_face(ratio=0.50, face_cy=0.40))
        assert result.state == STATE_GOOD
        assert result.posture_score is not None
        assert result.confidence == pytest.approx(1.0, abs=1e-3)

    def test_poor_posture_after_warmup(self):
        det = PostureDetector(make_settings())
        _warmup(det, make_face(ratio=0.50, face_cy=0.40))
        poor = make_face(ratio=0.65, face_cy=0.55)
        for _ in range(5):  # 3 of 5 required for the majority vote
            det.update(poor)
        assert det.update(poor).state == STATE_POOR

    def test_moderate_posture(self):
        det = PostureDetector(make_settings())
        _warmup(det, make_face(ratio=0.50, face_cy=0.40))
        moderate = make_face(ratio=0.55, face_cy=0.45)
        for _ in range(5):
            det.update(moderate)
        assert det.update(moderate).state == STATE_MODERATE

    def test_no_face_becomes_unknown(self):
        det = PostureDetector(make_settings())
        _warmup(det, make_face(ratio=0.50, face_cy=0.40))
        for _ in range(12):  # >= 3 missing frames flips the majority to UNKNOWN
            result = det.update(None)
        assert result.state == STATE_UNKNOWN
        assert result.posture_score is not None  # EMA retained, state honest

    def test_disabled_returns_unknown_with_reason(self):
        det = PostureDetector(make_settings(posture_enabled=False))
        result = det.update(make_face(ratio=0.50, face_cy=0.40))
        assert result.state == STATE_UNKNOWN
        assert result.confidence == 0.0
        assert "disabled" in (result.error or "")

    def test_single_noise_frame_does_not_flip(self):
        det = PostureDetector(make_settings())
        _warmup(det, make_face(ratio=0.50, face_cy=0.40))
        # One POOR frame inside a GOOD window must not switch the stable state.
        det.update(make_face(ratio=0.65, face_cy=0.55))
        good = det.update(make_face(ratio=0.50, face_cy=0.40))
        assert good.state == STATE_GOOD

    def test_reset_clears_history(self):
        det = PostureDetector(make_settings())
        _warmup(det, make_face(ratio=0.50, face_cy=0.40))
        det.reset()
        result = det.update(make_face(ratio=0.50, face_cy=0.40))
        # Fill the window from scratch so the majority reaches GOOD again.
        for _ in range(5):
            result = det.update(make_face(ratio=0.50, face_cy=0.40))
        assert result.state == STATE_GOOD

    def test_result_round_trips(self):
        det = PostureDetector(make_settings())
        _warmup(det, make_face(ratio=0.50, face_cy=0.40))
        result = det.update(make_face(ratio=0.50, face_cy=0.40))
        d = result.to_dict()
        assert set(d) >= {"state", "posture_score", "head_pitch_score",
                          "face_offset_score", "confidence", "error"}
        assert d["state"] == STATE_GOOD