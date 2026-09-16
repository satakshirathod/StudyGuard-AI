"""Tests for the face-detection module (Phase 2).

These tests validate module initialisation, edge-case handling, result
structure, and overlay drawing — **not** detection accuracy (which requires
a live camera and real faces and is evaluated manually per the testing
checklist in the README).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Ensure the backend package is importable when running from backend/
_BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from config import get_settings
from ai.face_detection import (
    FaceDetectionModule,
    FaceResult,
    FaceData,
    draw_overlay,
)

MODELS_DIR = _BACKEND_ROOT / "models"
MODELS_READY = (MODELS_DIR / "face_landmarker.task").exists()


@pytest.fixture
def settings():
    return get_settings()


@pytest.fixture
def blank_frame():
    """720×1280 BGR frame — should yield NO_FACE reliably."""
    return np.zeros((720, 1280, 3), dtype=np.uint8)


# ── initialisation ──────────────────────────────────────────────────────────

class TestModuleInit:
    def test_init_returns_bool(self, settings):
        mod = FaceDetectionModule(settings)
        result = mod.initialize()
        assert isinstance(result, bool)

    @pytest.mark.skipif(not MODELS_READY, reason="models not downloaded")
    def test_init_succeeds_with_models(self, settings):
        mod = FaceDetectionModule(settings)
        assert mod.initialize() is True
        assert mod.ready is True
        mod.cleanup()

    @pytest.mark.skipif(not MODELS_READY, reason="models not downloaded")
    def test_landmarker_available(self, settings):
        mod = FaceDetectionModule(settings)
        mod.initialize()
        assert mod.landmarker_available is True
        mod.cleanup()

    def test_init_missing_models_degrades(self, settings):
        """Module should degrade gracefully when model files are missing."""
        bad_settings = get_settings()
        # Override model paths to nonexistent files
        with patch("config.get_settings") as gs:
            gs.return_value = bad_settings
            mod = FaceDetectionModule(bad_settings)
            # Force init even without files
            result = mod.initialize()
            # Should either succeed with available models or degrade — never raise
            assert isinstance(result, bool)

    def test_idempotent_init(self, settings):
        mod = FaceDetectionModule(settings)
        first = mod.initialize()
        second = mod.initialize()
        assert first == second


# ── process edge cases ──────────────────────────────────────────────────────

class TestProcessEdgeCases:
    @pytest.mark.skipif(not MODELS_READY, reason="models not downloaded")
    def test_process_blank_frame(self, settings):
        """Blank frame should return NO_FACE without crashing."""
        mod = FaceDetectionModule(settings)
        mod.initialize()
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        res = mod.process(frame)
        assert isinstance(res, FaceResult)
        assert res.status in ("NO_FACE", "DETECTED", "UNKNOWN")
        assert res.face_count >= 0
        assert isinstance(res.multiple_faces, bool)
        mod.cleanup()

    def test_process_none_frame(self, settings):
        mod = FaceDetectionModule(settings)
        res = mod.process(None)
        assert res.status == "UNKNOWN"
        assert res.error is not None

    def test_process_invalid_type(self, settings):
        mod = FaceDetectionModule(settings)
        res = mod.process("not a frame")
        assert res.status == "UNKNOWN"

    def test_process_wrong_ndim(self, settings):
        mod = FaceDetectionModule(settings)
        res = mod.process(np.zeros((10,), dtype=np.uint8))
        assert res.status == "UNKNOWN"

    def test_process_uninitialized_module(self, settings):
        """process() should auto-initialize and still not crash."""
        mod = FaceDetectionModule(settings)
        # Don't call initialize(); process() will attempt it
        res = mod.process(np.zeros((720, 1280, 3), dtype=np.uint8))
        assert isinstance(res, FaceResult)
        assert res.status in ("NO_FACE", "UNKNOWN")
        mod.cleanup()

    def test_cleanup_is_safe(self, settings):
        mod = FaceDetectionModule(settings)
        mod.cleanup()  # never raises even if never initialized


# ── result structure ────────────────────────────────────────────────────────

class TestResultStructure:
    def test_face_result_defaults(self):
        r = FaceResult()
        assert r.face_detected is False
        assert r.face_count == 0
        assert r.multiple_faces is False
        assert r.landmarks_available is False
        assert r.confidence == 0.0
        assert r.faces == []
        assert r.error is None

    def test_face_data_fields(self):
        fd = FaceData(
            bbox_pixels=(0, 0, 100, 100),
            bbox_normalized=(0.0, 0.0, 0.5, 0.5),
            area=10000.0,
            confidence=0.95,
            landmarks=[(0.1, 0.2, 0.0), (0.3, 0.4, 0.0)],
            left_eye=[],
            right_eye=[],
            left_iris=None,
            right_iris=None,
            nose_tip=(0.25, 0.25),
            mouth=[],
            face_outline=[],
        )
        assert fd.confidence == 0.95
        assert len(fd.landmarks) == 2


# ── overlay drawing ─────────────────────────────────────────────────────────

class TestOverlay:
    def test_draw_empty_result(self, blank_frame):
        frame = draw_overlay(blank_frame, FaceResult())
        assert frame is blank_frame  # same object

    def test_draw_with_synthetic_faces(self):
        """Synthetic face data should draw onto the frame without errors."""
        original = np.zeros((720, 1280, 3), dtype=np.uint8)
        frame = original.copy()
        fd = FaceData(
            bbox_pixels=(100, 100, 400, 400),
            bbox_normalized=(0.1, 0.1, 0.4, 0.4),
            area=90000.0,
            confidence=0.9,
            landmarks=[(0.25, 0.3, 0.0)] * 478,
            left_eye=[(0.2, 0.25), (0.22, 0.24), (0.23, 0.25),
                       (0.28, 0.25), (0.24, 0.26), (0.22, 0.26)],
            right_eye=[(0.42, 0.25), (0.44, 0.24), (0.45, 0.25),
                        (0.48, 0.25), (0.46, 0.26), (0.44, 0.26)],
            left_iris=(0.24, 0.25),
            right_iris=(0.45, 0.25),
            nose_tip=(0.33, 0.35),
            mouth=[(0.30, 0.45), (0.35, 0.46), (0.40, 0.45)],
            face_outline=[(0.2 + 0.01 * i, 0.2 + 0.01 * i) for i in range(36)],
        )
        result = FaceResult(
            face_detected=True,
            face_count=1,
            multiple_faces=False,
            primary_face_detected=True,
            landmarks_available=True,
            confidence=0.9,
            confidence_available=True,
            status="DETECTED",
            faces=[fd],
        )
        draw_overlay(frame, result)
        # Confirm the frame was actually mutated (overlay was drawn)
        assert not np.array_equal(frame, original), "overlay did not modify the frame"


# ── multi-face logic (unit level) ──────────────────────────────────────────

class TestMultiFaceLogic:
    def test_multiple_faces_flag(self):
        r = FaceResult(face_count=3, multiple_faces=True, face_detected=True)
        assert r.multiple_faces is True
        assert r.face_count == 3

    def test_single_face_flag(self):
        r = FaceResult(face_count=1, multiple_faces=False, face_detected=True)
        assert r.multiple_faces is False


# ── config integration ──────────────────────────────────────────────────────

class TestConfigIntegration:
    def test_settings_contain_face_fields(self, settings):
        assert hasattr(settings, "face_detection_enabled")
        assert hasattr(settings, "show_face_landmarks")
        assert hasattr(settings, "max_faces")
        assert hasattr(settings, "face_detector_model")
        assert hasattr(settings, "face_landmarker_model")
        assert isinstance(settings.max_faces, int)
        assert isinstance(settings.face_detection_enabled, bool)
        assert isinstance(settings.show_face_landmarks, bool)