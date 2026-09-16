"""Central configuration for StudyGuard AI.

All tunable parameters live here (or in ``.env``) so magic numbers are not
scattered through the codebase. Values are read from environment variables
prefixed with ``SG_``; the ``.env`` file is loaded automatically.

Later phases add their AI thresholds (EAR, gaze, distance, posture, phone)
to this same module.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

_BACKEND_DIR = Path(__file__).resolve().parent


def _env_int(name: str, default: int) -> int:
    """Read an integer environment variable with a fallback default."""
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    """Read a boolean environment variable (true/false/1/0)."""
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return value.strip().lower() in ("true", "1", "yes")


def _env_float(name: str, default: float) -> float:
    """Read a float environment variable with a fallback default."""
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    """Application settings resolved from environment variables."""

    # Server
    host: str
    port: int

    # Webcam capture
    camera_index: int
    frame_width: int
    frame_height: int
    process_every_n_frames: int
    max_fps: int

    # Face detection (Phase 2)
    face_detection_enabled: bool
    show_face_landmarks: bool
    max_faces: int
    face_detector_model: str
    face_landmarker_model: str

    # Gaze / attention tracking (Phase 3)
    show_gaze_overlay: bool
    gaze_smoothing_window: int
    gaze_change_threshold: float
    gaze_away_duration: int  # consecutive frames of no-face before AWAY
    gaze_calibration_samples: int
    gaze_calibration_accept_range: float  # max |iris offset| accepted during calibration
    gaze_iris_x_threshold: float  # iris shift (fraction of eye width) considered off-centre
    gaze_iris_y_threshold: float
    gaze_yaw_threshold: float  # degrees — head turn considered off-centre
    gaze_pitch_threshold: float  # degrees — head tilt considered off-centre
    gaze_iris_weight: float  # weight of iris signal (combined with head weight = 1)
    gaze_head_weight: float
    gaze_direction_threshold: float  # fraction of full-scale needed to switch direction
    gaze_attention_saturation: float  # deviation magnitude that yields attention 0
    gaze_focal_length: float | None  # camera focal in px; None → frame width

    # Drowsiness detection (Phase 4)
    drowsiness_enabled: bool
    show_drowsiness_overlay: bool  # draw EAR/state on the video feed
    ear_threshold: float  # avg EAR below this → eyes considered closed this frame
    ear_consecutive_frames: int  # consecutive closed frames before DROWSY (blink-safe)
    drowsiness_duration_threshold: float  # seconds of sustained closure before DROWSY
    drowsiness_smoothing_window: int  # smoothing of the final NORMAL/DROWSY/UNKNOWN state
    drowsiness_change_threshold: float  # fraction of window that must agree to switch state

    # Screen-distance estimation (Phase 5)
    distance_enabled: bool
    show_distance_overlay: bool  # draw distance/state chip on the video feed
    distance_mode: str  # "relative" — no physical calibration available yet
    distance_reference_width: float  # px at the reference distance (0 → auto-calibrate)
    distance_reference_samples: int  # frames averaged during auto-calibration
    distance_too_close_ratio: float  # width/reference above this → TOO_CLOSE
    distance_recover_ratio: float  # width/reference below this → NORMAL (hysteresis)
    distance_min_face_width: float  # px below this the face is unreliable → UNKNOWN
    distance_too_close_frames: int  # consecutive close frames before TOO_CLOSE (debounce)
    distance_smoothing_window: int  # smoothing of the final NORMAL/TOO_CLOSE/UNKNOWN state
    distance_change_threshold: float  # fraction of window that must agree to switch state
    distance_away_duration: int  # consecutive missing-face frames before UNKNOWN

    # Posture analysis (Phase 6)
    posture_enabled: bool
    show_posture_overlay: bool
    posture_good_threshold: float    # score ≥ this → GOOD
    posture_moderate_threshold: float  # score ≥ this → MODERATE, below → POOR
    posture_smoothing_window: int    # smoothing of the final GOOD/MODERATE/POOR/UNKNOWN state
    posture_change_threshold: float  # fraction of window that must agree to switch state
    posture_ema_alpha: float         # EMA smoothing factor (0..1, higher = more responsive)
    posture_away_frames: int         # consecutive missing-face frames before UNKNOWN

    # Phone detection (Phase 7)
    phone_enabled: bool
    show_phone_overlay: bool
    phone_detector_model: str          # EfficientDet-Lite0 (COCO) object detector
    phone_confidence_threshold: float  # per-detection score ≥ this → counts as a phone
    phone_label_filter: str            # substring matched against the COCO category (default "phone")
    phone_smoothing_window: int        # smoothing of PHONE_DETECTED/NO_PHONE/UNKNOWN state
    phone_change_threshold: float      # fraction of window that must agree to switch state
    phone_away_frames: int             # consecutive unusable frames before UNKNOWN

    # Database (Phase: application foundation)
    database_url: str  # SQLAlchemy connection URL (sqlite by default)

    # Behavior engine + focus score (Phase: application foundation)
    behavior_log_interval_seconds: float  # persist a behavior row at most this often
    focus_weight_gaze: float
    focus_weight_presence: float
    focus_weight_drowsiness: float
    focus_weight_posture: float
    focus_weight_distance: float
    focus_weight_phone: float
    focus_level_high: int  # focus ≥ this → "high"
    focus_level_medium: int  # focus ≥ this → "medium", below → "low"

    # Real-time alerts (Phase: application foundation)
    alert_away_seconds: float  # student absent from camera before an alert
    alert_gaze_off_seconds: float  # gaze off-screen before an alert
    alert_drowsiness_seconds: float  # extra persistence beyond DROWSY state before alert (0 = fire immediately)
    alert_too_close_seconds: float  # extra persistence beyond TOO_CLOSE state before alert
    alert_poor_posture_seconds: float  # extra persistence beyond POOR state before alert
    alert_phone_seconds: float  # extra persistence beyond PHONE_DETECTED before alert
    alert_cooldown_seconds: float  # minimum between repeated alerts of a type


def get_settings() -> Settings:
    """Load settings from ``.env`` / process environment."""
    load_dotenv()
    models_dir = _BACKEND_DIR / "models"
    return Settings(
        host=os.getenv("SG_HOST", "0.0.0.0"),
        port=_env_int("SG_PORT", 8000),
        camera_index=_env_int("SG_CAMERA_INDEX", 0),
        frame_width=_env_int("SG_FRAME_WIDTH", 1280),
        frame_height=_env_int("SG_FRAME_HEIGHT", 720),
        process_every_n_frames=_env_int("SG_PROCESS_EVERY_N_FRAMES", 2),
        max_fps=_env_int("SG_MAX_FPS", 30),
        # Face detection (Phase 2)
        face_detection_enabled=_env_bool("SG_FACE_DETECTION_ENABLED", True),
        show_face_landmarks=_env_bool("SG_SHOW_FACE_LANDMARKS", False),
        max_faces=_env_int("SG_MAX_FACES", 2),
        face_detector_model=os.getenv(
            "SG_FACE_DETECTOR_MODEL", str(models_dir / "face_detector.tflite")
        ),
        face_landmarker_model=os.getenv(
            "SG_FACE_LANDMARKER_MODEL", str(models_dir / "face_landmarker.task")
        ),
        # Gaze / attention tracking (Phase 3)
        show_gaze_overlay=_env_bool("SG_SHOW_GAZE_OVERLAY", False),
        gaze_smoothing_window=_env_int("SG_GAZE_SMOOTHING_WINDOW", 5),
        gaze_change_threshold=_env_float("SG_GAZE_CHANGE_THRESHOLD", 0.6),
        gaze_away_duration=_env_int("SG_GAZE_AWAY_DURATION", 10),
        gaze_calibration_samples=_env_int("SG_GAZE_CALIBRATION_SAMPLES", 30),
        gaze_calibration_accept_range=_env_float("SG_GAZE_CALIBRATION_ACCEPT_RANGE", 0.30),
        gaze_iris_x_threshold=_env_float("SG_GAZE_IRIS_X_THRESHOLD", 0.10),
        gaze_iris_y_threshold=_env_float("SG_GAZE_IRIS_Y_THRESHOLD", 0.10),
        gaze_yaw_threshold=_env_float("SG_GAZE_YAW_THRESHOLD", 12.0),
        gaze_pitch_threshold=_env_float("SG_GAZE_PITCH_THRESHOLD", 10.0),
        gaze_iris_weight=_env_float("SG_GAZE_IRIS_WEIGHT", 0.6),
        gaze_head_weight=_env_float("SG_GAZE_HEAD_WEIGHT", 0.4),
        gaze_direction_threshold=_env_float("SG_GAZE_DIRECTION_THRESHOLD", 0.40),
        gaze_attention_saturation=_env_float("SG_GAZE_ATTENTION_SATURATION", 1.5),
        gaze_focal_length=(
            float(os.getenv("SG_GAZE_FOCAL_LENGTH")) if os.getenv("SG_GAZE_FOCAL_LENGTH") else None
        ),
        # Drowsiness detection (Phase 4)
        drowsiness_enabled=_env_bool("SG_DROWSINESS_ENABLED", True),
        show_drowsiness_overlay=_env_bool("SG_SHOW_DROWSINESS_OVERLAY", False),
        ear_threshold=_env_float("SG_EAR_THRESHOLD", 0.25),
        ear_consecutive_frames=_env_int("SG_EAR_CONSECUTIVE_FRAMES", 12),
        drowsiness_duration_threshold=_env_float("SG_DROWSINESS_DURATION_THRESHOLD", 0.8),
        drowsiness_smoothing_window=_env_int("SG_DROWSINESS_SMOOTHING_WINDOW", 5),
        drowsiness_change_threshold=_env_float("SG_DROWSINESS_CHANGE_THRESHOLD", 0.6),
        # Screen-distance estimation (Phase 5)
        distance_enabled=_env_bool("SG_DISTANCE_ENABLED", True),
        show_distance_overlay=_env_bool("SG_SHOW_DISTANCE_OVERLAY", False),
        distance_mode=os.getenv("SG_DISTANCE_MODE", "relative"),
        distance_reference_width=_env_float("SG_DISTANCE_REFERENCE_WIDTH", 0.0),
        distance_reference_samples=_env_int("SG_DISTANCE_REFERENCE_SAMPLES", 15),
        distance_too_close_ratio=_env_float("SG_DISTANCE_TOO_CLOSE_RATIO", 1.45),
        distance_recover_ratio=_env_float("SG_DISTANCE_RECOVER_RATIO", 1.25),
        distance_min_face_width=_env_float("SG_DISTANCE_MIN_FACE_WIDTH", 40.0),
        distance_too_close_frames=_env_int("SG_DISTANCE_TOO_CLOSE_FRAMES", 10),
        distance_smoothing_window=_env_int("SG_DISTANCE_SMOOTHING_WINDOW", 5),
        distance_change_threshold=_env_float("SG_DISTANCE_CHANGE_THRESHOLD", 0.6),
        distance_away_duration=_env_int("SG_DISTANCE_AWAY_DURATION", 10),
        # Posture analysis (Phase 6)
        posture_enabled=_env_bool("SG_POSTURE_ENABLED", True),
        show_posture_overlay=_env_bool("SG_SHOW_POSTURE_OVERLAY", False),
        posture_good_threshold=_env_float("SG_POSTURE_GOOD_THRESHOLD", 70.0),
        posture_moderate_threshold=_env_float("SG_POSTURE_MODERATE_THRESHOLD", 40.0),
        posture_smoothing_window=_env_int("SG_POSTURE_SMOOTHING_WINDOW", 5),
        posture_change_threshold=_env_float("SG_POSTURE_CHANGE_THRESHOLD", 0.6),
        posture_ema_alpha=_env_float("SG_POSTURE_EMA_ALPHA", 0.4),
        posture_away_frames=_env_int("SG_POSTURE_AWAY_FRAMES", 10),
        # Phone detection (Phase 7)
        phone_enabled=_env_bool("SG_PHONE_ENABLED", True),
        show_phone_overlay=_env_bool("SG_SHOW_PHONE_OVERLAY", False),
        phone_detector_model=os.getenv(
            "SG_PHONE_DETECTOR_MODEL", str(models_dir / "efficientdet_lite0.tflite")
        ),
        phone_confidence_threshold=_env_float("SG_PHONE_CONFIDENCE_THRESHOLD", 0.5),
        phone_label_filter=os.getenv("SG_PHONE_LABEL_FILTER", "phone"),
        phone_smoothing_window=_env_int("SG_PHONE_SMOOTHING_WINDOW", 5),
        phone_change_threshold=_env_float("SG_PHONE_CHANGE_THRESHOLD", 0.6),
        phone_away_frames=_env_int("SG_PHONE_AWAY_FRAMES", 10),
        # Database (Phase: application foundation)
        database_url=os.getenv(
            "SG_DATABASE_URL", f"sqlite:///{_BACKEND_DIR / 'studygard.db'}"
        ),
        # Behavior engine + focus score
        behavior_log_interval_seconds=_env_float("SG_BEHAVIOR_LOG_INTERVAL_SECONDS", 2.0),
        focus_weight_gaze=_env_float("SG_FOCUS_WEIGHT_GAZE", 0.35),
        focus_weight_presence=_env_float("SG_FOCUS_WEIGHT_PRESENCE", 0.30),
        focus_weight_drowsiness=_env_float("SG_FOCUS_WEIGHT_DROWSINESS", 0.15),
        focus_weight_posture=_env_float("SG_FOCUS_WEIGHT_POSTURE", 0.10),
        focus_weight_distance=_env_float("SG_FOCUS_WEIGHT_DISTANCE", 0.05),
        focus_weight_phone=_env_float("SG_FOCUS_WEIGHT_PHONE", 0.05),
        focus_level_high=_env_int("SG_FOCUS_LEVEL_HIGH", 80),
        focus_level_medium=_env_int("SG_FOCUS_LEVEL_MEDIUM", 55),
        # Real-time alerts
        alert_away_seconds=_env_float("SG_ALERT_AWAY_SECONDS", 5.0),
        alert_gaze_off_seconds=_env_float("SG_ALERT_GAZE_OFF_SECONDS", 8.0),
        alert_drowsiness_seconds=_env_float("SG_ALERT_DROWSINESS_SECONDS", 0.0),
        alert_too_close_seconds=_env_float("SG_ALERT_TOO_CLOSE_SECONDS", 5.0),
        alert_poor_posture_seconds=_env_float("SG_ALERT_POOR_POSTURE_SECONDS", 10.0),
        alert_phone_seconds=_env_float("SG_ALERT_PHONE_SECONDS", 5.0),
        alert_cooldown_seconds=_env_float("SG_ALERT_COOLDOWN_SECONDS", 60.0),
    )
