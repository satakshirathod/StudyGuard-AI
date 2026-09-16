"""Minimal API-level smoke tests for the Phase 7 phone integration.

Verifies the FastAPI wiring (root phase/modules + `/api/video/status` phone
block) through an in-process TestClient.  Environment for a temp SQLite database
is set *before* importing ``main`` so settings are resolved correctly.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

_BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

_fd, _db_path = tempfile.mkstemp(suffix=".db")
os.close(_fd)
os.environ["SG_DATABASE_URL"] = f"sqlite:///{_db_path}"

from fastapi.testclient import TestClient  # noqa: E402

import main  # noqa: E402

client = TestClient(main.app)


def test_root_reports_phone_phase():
    with client:
        resp = client.get("/")
        assert resp.status_code == 200
        body = resp.json()
        assert body["phase"] == "phone-detection"
        assert "phone" in body["modules"]
        assert body["version"].startswith("0.8")


def test_health_ok():
    with client:
        resp = client.get("/api/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


def test_status_includes_phone_block():
    with client:
        resp = client.get("/api/video/status")
        assert resp.status_code == 200
        body = resp.json()
        phone = body.get("phone")
        assert phone is not None
        # Shape matches the frontend contract.
        for key in ("state", "detected", "confidence", "boxes",
                    "error", "phone_enabled", "model_available", "detector_ready"):
            assert key in phone
        assert phone["state"] in ("PHONE_DETECTED", "NO_PHONE", "UNKNOWN")
        assert isinstance(phone["phone_enabled"], bool)
        assert isinstance(phone["model_available"], bool)


def test_status_posture_block_still_present():
    with client:
        resp = client.get("/api/video/status")
        assert resp.status_code == 200
        posture = resp.json().get("posture")
        assert posture is not None
        assert posture["state"] in ("GOOD", "MODERATE", "POOR", "UNKNOWN")


def test_root_modules_list_complete():
    with client:
        body = client.get("/").json()
        expected = {"face", "gaze", "drowsiness", "distance", "posture", "phone"}
        assert set(body["modules"]) == expected
