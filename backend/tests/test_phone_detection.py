"""Tests for the real-time phone detection module (Phase 7).

Uses a FakeObjectDetector injected into PhoneDetector so all tests are
deterministic, fast, and require no real model inference.  Confirms:
  * PHONE_DETECTED / NO_PHONE / UNKNOWN states match spec.
  * Confidence threshold filters real positives.
  * A single noisy frame never raises a false phone alert.
  * Recovery when the phone leaves the frame is correctly handled.
  * Inference errors and disabled config both degrade to UNKNOWN honestly.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

_BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from ai.phone_detection import (  # noqa: E402
    STATE_DETECTED,
    STATE_CLEAR,
    STATE_UNKNOWN,
    ObjectDetection,
    PhoneDetector,
    PhoneResult,
)
from config import get_settings  # noqa: E402

_s = get_settings()

FRAME_W = 1280
FRAME_H = 720
_DUMMY_FRAME = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)


def make_settings(**overrides):
    return replace(_s, **overrides)


def _phone_det(score: float = 0.95, label: str = "cell phone") -> ObjectDetection:
    return ObjectDetection(box=(100, 200, 300, 400), label=label, score=score)


# ── Fake object detector (dependency injection) ─────────────────────────────

class FakeObjectDetector:
    def __init__(
        self,
        detections: list[ObjectDetection] | None = None,
        ready: bool = True,
        available: bool = True,
        error: str | None = None,
        raise_on_detect: bool = False,
    ):
        self._detections = detections or []
        self._ready = ready
        self._available = available
        self._error = error
        self._raise_on_detect = raise_on_detect

    @property
    def available(self):
        return self._available

    @property
    def ready(self):
        return self._ready

    @property
    def last_error(self):
        return self._error

    def initialize(self):
        return self._ready

    def detect(self, frame_bgr):
        if self._raise_on_detect:
            raise RuntimeError("inference exploded")
        return self._detections

    def close(self):
        pass


# ── detection / state logic ─────────────────────────────────────────────────

class TestPhoneDetectionLogic:
    def test_detected_with_confident_box(self):
        fake = FakeObjectDetector(detections=[_phone_det(0.95)])
        det = PhoneDetector(make_settings(), object_detector=fake)
        # Warm up: window=5, change_threshold=0.6 → need 3/5 agreeing to flip.
        results = []
        for _ in range(5):
            r = det.update(_DUMMY_FRAME)
            results.append(r.state)
        # Frames 1-2: only 1-2 DETECTED slots filled → UNKNOWN.
        assert results[0] == STATE_UNKNOWN
        assert results[1] == STATE_UNKNOWN
        # Frame 3+: 3+ DETECTED slots → stable flips to PHONE_DETECTED.
        assert results[2] == STATE_DETECTED
        assert results[3] == STATE_DETECTED
        assert r.detected is True
        assert r.confidence == 0.95
        assert len(r.boxes) == 1

    def test_no_phone_when_below_threshold(self):
        fake = FakeObjectDetector(detections=[_phone_det(0.2)])
        det = PhoneDetector(make_settings(), object_detector=fake)
        # TemporalStateTracker needs several agreeing frames to flip from UNKNOWN.
        for _ in range(6):
            r = det.update(_DUMMY_FRAME)
        assert r.state == STATE_CLEAR
        assert r.detected is False
        assert r.confidence == 0.0
        assert r.boxes == []

    def test_no_phone_when_label_not_matching(self):
        fake = FakeObjectDetector(detections=[
            ObjectDetection(box=(0, 0, 10, 10), label="person", score=0.99),
        ])
        det = PhoneDetector(make_settings(), object_detector=fake)
        for _ in range(6):
            r = det.update(_DUMMY_FRAME)
        assert r.state == STATE_CLEAR
        assert r.detected is False

    def test_unknown_when_detector_not_ready(self):
        fake = FakeObjectDetector(detections=[_phone_det(0.95)], ready=False)
        det = PhoneDetector(make_settings(), object_detector=fake)
        r = det.update(_DUMMY_FRAME)
        assert r.state == STATE_UNKNOWN
        assert r.detected is None
        assert "model not loaded" in (r.error or "").lower()

    def test_unknown_when_disabled(self):
        fake = FakeObjectDetector(detections=[_phone_det(0.95)])
        det = PhoneDetector(make_settings(phone_enabled=False), object_detector=fake)
        r = det.update(_DUMMY_FRAME)
        assert r.state == STATE_UNKNOWN
        assert r.detected is None
        assert "disabled" in (r.error or "").lower()

    def test_unknown_when_frame_is_none(self):
        fake = FakeObjectDetector(detections=[_phone_det(0.95)])
        det = PhoneDetector(make_settings(), object_detector=fake)
        r = det.update(None)
        assert r.state == STATE_UNKNOWN
        assert "no video frame" in (r.error or "").lower()

    def test_single_false_positive_suppressed(self):
        fake_clear = FakeObjectDetector(detections=[])
        det = PhoneDetector(make_settings(), object_detector=fake_clear)
        for _ in range(6):
            det.update(_DUMMY_FRAME)
        assert det._smoother.stable_state == STATE_CLEAR
        # One noisy phone frame
        det._object_detector = FakeObjectDetector(detections=[_phone_det(0.95)])
        det.update(_DUMMY_FRAME)
        assert det._smoother.stable_state == STATE_CLEAR
        # Back to no phone
        det._object_detector = fake_clear
        for _ in range(5):
            det.update(_DUMMY_FRAME)
        assert det._smoother.stable_state == STATE_CLEAR

    def test_persistent_detection_flips_to_detected(self):
        fake = FakeObjectDetector(detections=[_phone_det(0.9)])
        det = PhoneDetector(make_settings(), object_detector=fake)
        results = [det.update(_DUMMY_FRAME).state for _ in range(8)]
        assert results[-1] == STATE_DETECTED

    def test_recovery_after_phone_removed(self):
        fake_phone = FakeObjectDetector(detections=[_phone_det(0.9)])
        det = PhoneDetector(make_settings(), object_detector=fake_phone)
        for _ in range(7):
            det.update(_DUMMY_FRAME)
        assert det.update(_DUMMY_FRAME).state == STATE_DETECTED
        det._object_detector = FakeObjectDetector(detections=[])
        for _ in range(8):
            r = det.update(_DUMMY_FRAME)
        assert r.state == STATE_CLEAR

    def test_inference_error_returns_unknown(self):
        fake = FakeObjectDetector(raise_on_detect=True)
        det = PhoneDetector(make_settings(), object_detector=fake)
        r = det.update(_DUMMY_FRAME)
        assert r.state == STATE_UNKNOWN
        assert r.detected is None
        assert "failed" in (r.error or "").lower()


# ── confidence + box serialization ──────────────────────────────────────────

class TestConfidenceAndSerialization:
    def test_confidence_is_max_phone_score(self):
        fake = FakeObjectDetector(detections=[_phone_det(0.7), _phone_det(0.92)])
        det = PhoneDetector(make_settings(), object_detector=fake)
        r = det.update(_DUMMY_FRAME)
        assert r.confidence == 0.92

    def test_to_dict_keys(self):
        fake = FakeObjectDetector(detections=[_phone_det(0.8)])
        det = PhoneDetector(make_settings(), object_detector=fake)
        r = det.update(_DUMMY_FRAME).to_dict()
        assert set(r.keys()) == {"state", "detected", "confidence", "boxes", "error"}

    def test_box_dict_fields(self):
        fake = FakeObjectDetector(detections=[_phone_det(0.88)])
        det = PhoneDetector(make_settings(), object_detector=fake)
        r = det.update(_DUMMY_FRAME)
        assert len(r.boxes) == 1
        box = r.boxes[0]
        for k in ("x_min", "y_min", "x_max", "y_max", "label", "score"):
            assert k in box
        assert box["score"] == 0.88

    def test_boxes_empty_when_not_detected(self):
        fake = FakeObjectDetector(detections=[])
        det = PhoneDetector(make_settings(), object_detector=fake)
        r = det.update(_DUMMY_FRAME)
        assert r.boxes == []
        assert r.detected is False


# ── availability / model existence ──────────────────────────────────────────

class TestAvailability:
    def test_available_false_when_model_missing(self, tmp_path):
        fake = FakeObjectDetector(available=False)
        det = PhoneDetector(make_settings(), object_detector=fake)
        assert det.available is False

    def test_available_true_when_model_present(self):
        fake = FakeObjectDetector(available=True)
        det = PhoneDetector(make_settings(), object_detector=fake)
        assert det.available is True

    def test_reset_clears_state(self):
        fake = FakeObjectDetector(detections=[_phone_det(0.9)])
        det = PhoneDetector(make_settings(), object_detector=fake)
        for _ in range(6):
            det.update(_DUMMY_FRAME)
        det.reset()
        assert det._smoother.stable_state == STATE_UNKNOWN
        assert det._last_boxes == []
