"""StudyGuard AI — FastAPI backend entrypoint.

Phase 1: webcam capture + live streaming.
Phase 2: face detection + MediaPipe face-landmark analysis.
Phase 3: real-time gaze / attention tracking (iris + head orientation).
Phase 4: real-time drowsiness detection (eye aspect ratio + temporal logic).
Phase 5: real-time screen-distance estimation (relative face width).
Phase 6: real-time sitting-posture analysis (head-pitch signals from face landmarks).
Phase 7: real-time phone / mobile-device presence detection (on-device object detector).
Phase (application): full app foundation — student profiles, study sessions,
SQLite persistence, behavior engine + focus scoring, real-time alerts,
dashboard/analytics/report APIs.
AI modules are wired into the capture layer without changing the webcam itself.
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import Literal

import cv2
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

import runtime
from api import (
    alerts_router,
    analytics_router,
    dashboard_router,
    sessions_router,
    students_router,
)
from config import get_settings
from database.db import init_db
from utils.logger import setup_logger

__version__ = "0.8.0"

settings = get_settings()
setup_logger()

log = logging.getLogger("studygard.main")

JPEG_QUALITY = 80


# ── Webcam & AI module instances ────────────────────────────────────────────

def create_webcam():
    """Build the shared webcam capture instance (importable for tests)."""
    from ai.webcam import WebcamCapture
    return WebcamCapture(
        camera_index=settings.camera_index,
        width=settings.frame_width,
        height=settings.frame_height,
    )


def create_face_module():
    """Build the face-detection module instance."""
    from ai.face_detection import FaceDetectionModule
    return FaceDetectionModule(settings)


def create_gaze_tracker():
    """Build the real-time gaze/attention tracker."""
    from ai.gaze_tracking import GazeTracker
    return GazeTracker(
        settings,
        frame_w=settings.frame_width,
        frame_h=settings.frame_height,
    )


def create_drowsiness_detector():
    """Build the real-time drowsiness (EAR) detector."""
    from ai.drowsiness_detection import DrowsinessDetector
    return DrowsinessDetector(settings)


def create_distance_detector():
    """Build the real-time screen-distance (relative face width) detector."""
    from ai.distance_detection import DistanceDetector
    return DistanceDetector(
        settings,
        frame_w=settings.frame_width,
        frame_h=settings.frame_height,
    )


def create_posture_detector():
    """Build the real-time sitting-posture (head-pitch) detector."""
    from ai.posture_detection import PostureDetector
    return PostureDetector(settings)


def create_phone_detector():
    """Build the real-time phone / mobile-device presence detector."""
    from ai.phone_detection import PhoneDetector
    return PhoneDetector(settings)


webcam = create_webcam()
face_module = create_face_module()
gaze_tracker = create_gaze_tracker()
drowsiness_detector = create_drowsiness_detector()
distance_detector = create_distance_detector()
posture_detector = create_posture_detector()
phone_detector = create_phone_detector()

# Latest frame-level AI state (written in the MJPEG generator, read by status endpoint)
_latest_face_result = None
_latest_gaze_result = None
_latest_drowsiness_result = None
_latest_distance_result = None
_latest_posture_result = None
_latest_phone_result = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Open camera and AI models on startup; release on shutdown."""
    log.info("StudyGuard AI backend starting (v%s)", __version__)

    # Database (application foundation)
    try:
        init_db(settings.database_url)
        log.info("Database ready")
    except Exception as exc:  # pragma: no cover - environment specific
        log.error("Database initialisation failed: %s", exc)
        raise

    # Camera
    log.info(
        "Starting camera capture (index=%s, target=%sx%s)",
        settings.camera_index, settings.frame_width, settings.frame_height,
    )
    webcam.open()
    if not webcam.available:
        log.warning("Camera unavailable at startup — video feed will be offline")

    # Face detection (Phase 2)
    if settings.face_detection_enabled:
        log.info("Initialising face detection module…")
        face_module.initialize()
    else:
        log.info("Face detection disabled by configuration")

    # Phone detection (Phase 7)
    if settings.phone_enabled:
        log.info("Initialising phone detection module…")
        phone_detector.initialize()
    else:
        log.info("Phone detection disabled by configuration")

    yield

    # Shutdown
    face_module.cleanup()
    phone_detector.cleanup()
    webcam.release()
    log.info("StudyGuard AI backend stopped")


app = FastAPI(
    title="StudyGuard AI",
    description="Real-Time Student Learning Behavior and Attention Analysis System",
    version=__version__,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Application API routers (profiles, sessions, dashboard, analytics, alerts)
app.include_router(students_router)
app.include_router(sessions_router)
app.include_router(dashboard_router)
app.include_router(analytics_router)
app.include_router(alerts_router)


# ── Video helpers ───────────────────────────────────────────────────────────

def _frame_to_jpeg(frame: np.ndarray) -> bytes:
    """Encode a BGR frame to JPEG bytes."""
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        raise RuntimeError("Failed to encode frame as JPEG")
    return buf.tobytes()


# ── Live behavior persistence (application foundation) ───────────────────────

def _record_behavior_snapshot(snapshot) -> None:
    """Persist a periodic behavior row + raised alerts for the active session."""
    from database.db import get_session

    db = get_session()
    try:
        session = runtime.session_service.get_active_session(db)
        if session is None:
            return
        runtime.session_service.record_behavior_log(db, session.id, snapshot)
        fired = runtime.alert_engine.evaluate(snapshot)
        if fired:
            runtime.session_service.add_alerts(db, session.id, fired)
    finally:
        db.close()


def _mjpeg_iter():
    """Infinite MJPEG generator with optional face/gaze/drowsiness/distance/posture/phone overlays + behavior logging."""
    from ai.face_detection import draw_overlay
    from ai.gaze_tracking import draw_gaze_overlay
    from ai.drowsiness_detection import draw_drowsiness_overlay
    from ai.distance_detection import draw_distance_overlay
    from ai.posture_detection import draw_posture_overlay
    from ai.phone_detection import draw_phone_overlay

    min_interval = 1.0 / max(settings.max_fps, 1)
    frame_count = 0
    face_result = None  # persists across non-processed frames
    gaze_result = None
    drowsiness_result = None
    distance_result = None
    posture_result = None
    phone_result = None

    while True:
        ok, frame = webcam.read()
        if not ok:
            log.warning("Camera read failed — stopping feed")
            break

        frame_count += 1

        # Run AI on every Nth frame
        if settings.face_detection_enabled and face_module.ready:
            if frame_count % settings.process_every_n_frames == 0:
                face_result = face_module.process(frame)
                primary = face_result.faces[0] if face_result.faces else None
                gaze_result = gaze_tracker.update(primary)
                drowsiness_result = drowsiness_detector.update(primary)
                distance_result = distance_detector.update(primary)
                posture_result = posture_detector.update(primary)
                phone_result = phone_detector.update(frame)

                # Behavior engine → live status + (periodically) database
                snapshot = runtime.behavior_engine.evaluate(
                    face_result,
                    gaze_result,
                    drowsiness_result,
                    distance_result,
                    posture_result,
                    phone_result,
                )
                runtime.latest_behavior = snapshot
                now = time.monotonic()
                if runtime.next_behavior_log_monotonic is None:
                    runtime.next_behavior_log_monotonic = (
                        now + settings.behavior_log_interval_seconds
                    )
                if now >= runtime.next_behavior_log_monotonic:
                    _record_behavior_snapshot(snapshot)
                    runtime.next_behavior_log_monotonic = (
                        now + settings.behavior_log_interval_seconds
                    )

            # Draw overlays on every frame using the last available results
            if settings.show_face_landmarks and face_result is not None:
                frame = draw_overlay(frame, face_result)
            if settings.show_gaze_overlay and gaze_result is not None:
                primary = face_result.faces[0] if face_result and face_result.faces else None
                if primary is not None:
                    frame = draw_gaze_overlay(frame, primary, gaze_result)
            if settings.show_drowsiness_overlay and drowsiness_result is not None:
                primary = face_result.faces[0] if face_result and face_result.faces else None
                if primary is not None:
                    frame = draw_drowsiness_overlay(frame, primary, drowsiness_result)
            if settings.show_distance_overlay and distance_result is not None:
                primary = face_result.faces[0] if face_result and face_result.faces else None
                if primary is not None:
                    frame = draw_distance_overlay(frame, primary, distance_result)
            if settings.show_posture_overlay and posture_result is not None:
                primary = face_result.faces[0] if face_result and face_result.faces else None
                if primary is not None:
                    frame = draw_posture_overlay(frame, primary, posture_result)
            if settings.show_phone_overlay and phone_result is not None:
                frame = draw_phone_overlay(frame, phone_result)

        # Expose latest result to the status endpoint (module-level state)
        global _latest_face_result, _latest_gaze_result, _latest_drowsiness_result, _latest_distance_result, _latest_posture_result, _latest_phone_result
        _latest_face_result = face_result
        _latest_gaze_result = gaze_result
        _latest_drowsiness_result = drowsiness_result
        _latest_distance_result = distance_result
        _latest_posture_result = posture_result
        _latest_phone_result = phone_result

        try:
            jpeg = _frame_to_jpeg(frame)
        except RuntimeError as exc:
            log.error("Frame encode failed: %s", exc)
            break

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
        )
        time.sleep(min_interval)


# ── Endpoints ───────────────────────────────────────────────────────────────

@app.get("/api/health")
def health() -> JSONResponse:
    """Liveness check for the backend."""
    return JSONResponse({
        "status": "ok",
        "service": "StudyGuard AI",
        "version": __version__,
        "camera_available": webcam.available,
    })


@app.get("/api/video/status")
def video_status() -> JSONResponse:
    """Camera, face-detection, gaze, drowsiness, distance, and posture status."""
    from ai.gaze_tracking import GazeResult

    w, h = webcam.resolution
    fr = _latest_face_result

    face_info = {
        "face_detection_enabled": settings.face_detection_enabled,
        "face_detected": fr.face_detected if fr else False,
        "face_count": fr.face_count if fr else 0,
        "multiple_faces": fr.multiple_faces if fr else False,
        "primary_face_detected": fr.primary_face_detected if fr else False,
        "landmarks_available": fr.landmarks_available if fr else False,
        "confidence": round(fr.confidence, 3) if fr else 0.0,
        "confidence_available": (fr.confidence_available if fr else face_module.detector_available),
        "status": fr.status if fr else ("UNKNOWN" if not face_module.ready else "NO_FACE"),
        "error": fr.error if fr else (None if face_module.ready else "face module not loaded"),
    }

    gz = _latest_gaze_result if _latest_gaze_result is not None else GazeResult()
    gaze_info = gz.to_dict()
    gaze_info["gaze_tracking_enabled"] = settings.face_detection_enabled
    gaze_info["calibration"] = gaze_tracker.calibration_status()

    from ai.drowsiness_detection import DrowsinessResult

    dr = (
        _latest_drowsiness_result
        if _latest_drowsiness_result is not None
        else DrowsinessResult()
    )
    drowsiness_info = dr.to_dict()
    drowsiness_info["drowsiness_enabled"] = settings.drowsiness_enabled

    from ai.distance_detection import DistanceResult

    ds = (
        _latest_distance_result
        if _latest_distance_result is not None
        else DistanceResult()
    )
    distance_info = ds.to_dict()
    distance_info["distance_enabled"] = settings.distance_enabled
    distance_info["calibration_details"] = distance_detector.calibration_status()

    from ai.posture_detection import PostureResult

    ps = (
        _latest_posture_result
        if _latest_posture_result is not None
        else PostureResult()
    )
    posture_info = ps.to_dict()
    posture_info["posture_enabled"] = settings.posture_enabled

    from ai.phone_detection import PhoneResult

    ph = _latest_phone_result if _latest_phone_result is not None else PhoneResult()
    phone_info = ph.to_dict()
    phone_info["phone_enabled"] = settings.phone_enabled
    phone_info["model_available"] = phone_detector.available
    phone_info["detector_ready"] = phone_detector.ready

    return JSONResponse({
        "camera_available": webcam.available,
        "camera_index": settings.camera_index,
        "frame_width": w,
        "frame_height": h,
        "fps": round(webcam.fps, 1),
        "max_fps": settings.max_fps,
        "last_error": webcam.last_error,
        "face": face_info,
        "gaze": gaze_info,
        "drowsiness": drowsiness_info,
        "distance": distance_info,
        "posture": posture_info,
        "phone": phone_info,
        "behavior": (
            runtime.latest_behavior.to_dict() if runtime.latest_behavior is not None else None
        ),
    })


@app.get("/api/video/feed")
def video_feed() -> StreamingResponse:
    """Live MJPEG webcam stream with optional face overlay."""
    if not webcam.available:
        reason = webcam.last_error or "camera unavailable"
        log.warning("Video feed requested but camera unavailable: %s", reason)
        return JSONResponse(status_code=503, content={"detail": reason})

    return StreamingResponse(
        _mjpeg_iter(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-store, no-cache, must-revalidate"},
    )


@app.get("/")
def root() -> JSONResponse:
    """API root with pointer to docs."""
    return JSONResponse({
        "service": "StudyGuard AI",
        "version": __version__,
        "docs": "/docs",
        "phase": "phone-detection",
        "modules": ["face", "gaze", "drowsiness", "distance", "posture", "phone"],
    })


# ── Gaze calibration ─────────────────────────────────────────────────────────

class CalibrateRequest(BaseModel):
    """Body for the gaze calibration endpoint."""

    action: Literal["start", "cancel"]


@app.post("/api/ai/gaze/calibrate")
def gaze_calibrate(req: CalibrateRequest) -> JSONResponse:
    """Start or cancel the gaze calibration run."""
    if not settings.face_detection_enabled:
        return JSONResponse(
            status_code=400,
            content={"detail": "gaze tracking requires face detection to be enabled"},
        )
    if req.action == "start":
        status = gaze_tracker.start_calibration()
        log.info("Calibration requested from client: start")
    else:
        status = gaze_tracker.cancel_calibration()
        log.info("Calibration requested from client: cancel")
    return JSONResponse(status)


# ── Distance calibration ─────────────────────────────────────────────────────

@app.post("/api/ai/distance/calibrate")
def distance_calibrate(req: CalibrateRequest) -> JSONResponse:
    """Start or cancel the screen-distance reference capture."""
    if not settings.face_detection_enabled:
        return JSONResponse(
            status_code=400,
            content={"detail": "distance estimation requires face detection to be enabled"},
        )
    if req.action == "start":
        status = distance_detector.start_calibration()
        log.info("Distance calibration requested from client: start")
    else:
        status = distance_detector.cancel_calibration()
        log.info("Distance calibration requested from client: cancel")
    return JSONResponse(status)