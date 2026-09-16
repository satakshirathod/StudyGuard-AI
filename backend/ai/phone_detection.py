"""
Real-time phone / mobile-device presence detection (Phase 7).

Detects whether a mobile phone is present in the student's visible area using
MediaPipe ObjectDetector (EfficientDet-Lite0, COCO-trained).

Output states:
    PHONE_DETECTED
    NO_PHONE
    UNKNOWN
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import time

import numpy as np

from ai.temporal import TemporalStateTracker
from config import Settings
from utils.logger import get_logger


_log = get_logger("studygard.ai.phone")


# ---------------------------------------------------------------------------
# Public state labels
# ---------------------------------------------------------------------------

STATE_DETECTED = "PHONE_DETECTED"
STATE_CLEAR = "NO_PHONE"
STATE_UNKNOWN = "UNKNOWN"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ObjectDetection:
    """One raw object-detector box in pixels."""

    box: tuple[int, int, int, int]
    label: str
    score: float


@dataclass
class PhoneDetection:
    """Raw, per-frame phone presence before temporal logic."""

    valid: bool = False
    raw_state: Optional[str] = None
    phone_boxes: list[dict] = field(default_factory=list)
    max_confidence: float = 0.0
    error: Optional[str] = None


@dataclass
class PhoneResult:
    """Final processed phone output."""

    state: str = STATE_UNKNOWN
    detected: Optional[bool] = None
    confidence: float = 0.0
    boxes: list[dict] = field(default_factory=list)
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "state": self.state,
            "detected": self.detected,
            "confidence": round(self.confidence, 3),
            "boxes": self.boxes,
            "error": self.error,
        }


# ---------------------------------------------------------------------------
# MediaPipe Object Detector
# ---------------------------------------------------------------------------

class PhoneObjectDetector:
    """
    Wrapper around MediaPipe Tasks ObjectDetector.

    Uses EfficientDet-Lite0 in VIDEO mode.

    Important:
    MediaPipe requires timestamps passed to detect_for_video()
    to be strictly increasing.

    We therefore keep our own last timestamp and guarantee:

        current_timestamp > previous_timestamp
    """

    def __init__(self, model_path: str) -> None:
        self._model_path = Path(model_path)

        self._detector = None
        self._initialized = False
        self._ready = False
        self._last_error: Optional[str] = None

        # IMPORTANT:
        # MediaPipe VIDEO mode requires strictly increasing timestamps.
        self._last_timestamp_ms = -1

    @property
    def available(self) -> bool:
        """True when the model file exists."""
        return self._model_path.exists()

    @property
    def ready(self) -> bool:
        """True when the detector has been initialized successfully."""
        return self._ready

    @property
    def last_error(self) -> Optional[str]:
        return self._last_error

    def initialize(self) -> bool:
        """Load the MediaPipe object detection model."""

        if self._initialized:
            return self._ready

        self._initialized = True

        if not self._model_path.exists():
            self._last_error = (
                f"phone detection model not found: {self._model_path}"
            )

            _log.warning(
                "Phone detection model missing — module will report UNKNOWN"
            )

            return False

        try:
            import mediapipe as mp

            base_options = mp.tasks.BaseOptions(
                model_asset_path=str(self._model_path)
            )

            options = mp.tasks.vision.ObjectDetectorOptions(
                base_options=base_options,
                running_mode=mp.tasks.vision.RunningMode.VIDEO,

                # Allow low-confidence detections to reach our own
                # configured semantic threshold.
                score_threshold=0.1,
            )

            self._detector = (
                mp.tasks.vision.ObjectDetector.create_from_options(options)
            )

            self._ready = True
            self._last_error = None

            # Reset timestamp whenever a new detector is initialized.
            self._last_timestamp_ms = -1

            _log.info(
                "Phone detection model loaded: %s",
                self._model_path.name,
            )

            return True

        except Exception as exc:
            self._last_error = (
                f"phone detection model failed to load: {exc}"
            )

            _log.error(
                "Phone detection init failed: %s",
                exc,
            )

            self._ready = False

            return False

    def close(self) -> None:
        """Release MediaPipe detector resources."""

        if self._detector is not None:
            try:
                self._detector.close()
            except Exception as exc:
                _log.warning(
                    "Error while closing phone detector: %s",
                    exc,
                )

        self._detector = None
        self._ready = False

        # Reset timestamp so a future detector session starts cleanly.
        self._last_timestamp_ms = -1

    def _next_timestamp_ms(self) -> int:
        """
        Generate a strictly increasing timestamp for MediaPipe VIDEO mode.

        time.monotonic_ns() gives a monotonic clock.

        However, converting it to milliseconds can still produce the same
        millisecond value for multiple frames.

        Therefore we explicitly enforce:

            timestamp > previous_timestamp
        """

        timestamp_ms = time.monotonic_ns() // 1_000_000

        if timestamp_ms <= self._last_timestamp_ms:
            timestamp_ms = self._last_timestamp_ms + 1

        self._last_timestamp_ms = timestamp_ms

        return timestamp_ms

    def detect(
        self,
        frame_bgr: np.ndarray,
    ) -> list[ObjectDetection]:
        """
        Run object detection on one BGR OpenCV frame.

        Raises an exception if inference fails.
        The caller handles the exception and returns UNKNOWN.
        """

        if not self._initialized:
            self.initialize()

        if not self._ready or self._detector is None:
            return []

        if frame_bgr is None:
            raise ValueError("frame is None")

        if not isinstance(frame_bgr, np.ndarray):
            raise TypeError("frame must be a numpy ndarray")

        if frame_bgr.size == 0:
            raise ValueError("frame is empty")

        if frame_bgr.ndim != 3:
            raise ValueError(
                f"expected BGR image with 3 dimensions, got shape "
                f"{frame_bgr.shape}"
            )

        frame_h, frame_w = frame_bgr.shape[:2]

        if frame_h <= 0 or frame_w <= 0:
            raise ValueError(
                f"invalid frame dimensions: {frame_w}x{frame_h}"
            )

        import cv2
        import mediapipe as mp

        # ---------------------------------------------------------------
        # OpenCV BGR -> RGB
        # ---------------------------------------------------------------

        rgb = cv2.cvtColor(
            frame_bgr,
            cv2.COLOR_BGR2RGB,
        )

        # ---------------------------------------------------------------
        # MediaPipe image
        # ---------------------------------------------------------------

        mp_img = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb,
        )

        # ---------------------------------------------------------------
        # IMPORTANT FIX:
        # Strictly increasing timestamp
        # ---------------------------------------------------------------

        timestamp_ms = self._next_timestamp_ms()

        # ---------------------------------------------------------------
        # Run detector
        # ---------------------------------------------------------------

        result = self._detector.detect_for_video(
            mp_img,
            timestamp_ms,
        )

        # ---------------------------------------------------------------
        # Convert detections
        # ---------------------------------------------------------------

        out: list[ObjectDetection] = []

        for det in (result.detections or []):

            bb = det.bounding_box

            x_min = int(bb.origin_x)
            y_min = int(bb.origin_y)
            x_max = int(bb.origin_x + bb.width)
            y_max = int(bb.origin_y + bb.height)

            # Keep coordinates inside actual frame.
            x_min = max(0, min(x_min, frame_w))
            y_min = max(0, min(y_min, frame_h))
            x_max = max(0, min(x_max, frame_w))
            y_max = max(0, min(y_max, frame_h))

            box = (
                x_min,
                y_min,
                x_max,
                y_max,
            )

            for category in (det.categories or []):

                label = str(
                    category.category_name or ""
                )

                score = float(
                    category.score or 0.0
                )

                out.append(
                    ObjectDetection(
                        box=box,
                        label=label,
                        score=score,
                    )
                )

        return out


# ---------------------------------------------------------------------------
# Phone Detector
# ---------------------------------------------------------------------------

class PhoneDetector:
    """
    Stateful phone presence detector.

    Pipeline:

        Webcam Frame
             ↓
        MediaPipe Object Detector
             ↓
        Phone label filtering
             ↓
        Confidence threshold
             ↓
        Temporal smoothing
             ↓
        PHONE_DETECTED / NO_PHONE / UNKNOWN
    """

    def __init__(
        self,
        settings: Settings,
        object_detector: Optional[PhoneObjectDetector] = None,
    ) -> None:

        self._settings = settings

        self._object_detector = (
            object_detector
            or PhoneObjectDetector(
                settings.phone_detector_model
            )
        )

        self._filter = (
            settings.phone_label_filter or ""
        ).lower()

        self._missing_streak = 0

        self._last_boxes: list[dict] = []

        self._last_confidence = 0.0

        self._smoother = TemporalStateTracker(
            window=settings.phone_smoothing_window,
            change_threshold=settings.phone_change_threshold,
            away_duration=settings.phone_away_frames,
            away_label=STATE_UNKNOWN,
            unknown_label=STATE_UNKNOWN,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def available(self) -> bool:
        """True when the model file exists."""
        return self._object_detector.available

    @property
    def ready(self) -> bool:
        """True when detector is initialized."""
        return self._object_detector.ready

    @property
    def last_error(self) -> Optional[str]:
        return self._object_detector.last_error

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self) -> bool:
        return self._object_detector.initialize()

    def cleanup(self) -> None:
        self._object_detector.close()

    # ------------------------------------------------------------------
    # Per-frame update
    # ------------------------------------------------------------------

    def update(
        self,
        frame: Optional[np.ndarray],
    ) -> PhoneResult:
        """
        Process one webcam frame.

        If the frame/model is unavailable, return UNKNOWN instead of
        fabricating a phone detection.
        """

        # ---------------------------------------------------------------
        # Detection disabled
        # ---------------------------------------------------------------

        if not self._settings.phone_enabled:

            self._missing_streak = 0
            self._last_boxes = []
            self._last_confidence = 0.0

            return PhoneResult(
                state=STATE_UNKNOWN,
                detected=None,
                confidence=0.0,
                boxes=[],
                error="phone detection disabled by configuration",
            )

        # ---------------------------------------------------------------
        # No frame
        # ---------------------------------------------------------------

        if frame is None:

            self._missing_streak += 1

            stable = self._smoother.update(None)

            return PhoneResult(
                state=stable,
                detected=(
                    stable == STATE_DETECTED
                    if stable != STATE_UNKNOWN
                    else None
                ),
                confidence=self._last_confidence,
                boxes=list(self._last_boxes),
                error="no video frame",
            )

        # ---------------------------------------------------------------
        # Detector not ready
        # ---------------------------------------------------------------

        if not self._object_detector.ready:

            self._missing_streak += 1

            stable = self._smoother.update(None)

            reason = (
                self._object_detector.last_error
                or "phone detection model not loaded"
            )

            return PhoneResult(
                state=stable,
                detected=(
                    stable == STATE_DETECTED
                    if stable != STATE_UNKNOWN
                    else None
                ),
                confidence=0.0,
                boxes=[],
                error=reason,
            )

        # ---------------------------------------------------------------
        # Run object detection
        # ---------------------------------------------------------------

        try:

            detections = self._object_detector.detect(
                frame
            )

        except Exception as exc:

            _log.warning(
                "Phone detection failed this frame: %s",
                exc,
            )

            self._missing_streak += 1

            stable = self._smoother.update(None)

            return PhoneResult(
                state=stable,
                detected=(
                    stable == STATE_DETECTED
                    if stable != STATE_UNKNOWN
                    else None
                ),
                confidence=self._last_confidence,
                boxes=list(self._last_boxes),
                error=f"phone detection failed: {exc}",
            )

        # ---------------------------------------------------------------
        # Valid frame
        # ---------------------------------------------------------------

        self._missing_streak = 0

        # ---------------------------------------------------------------
        # Filter phone labels
        #
        # COCO normally uses "cell phone".
        # If settings filter is "phone", it matches "cell phone".
        # ---------------------------------------------------------------

        phones = [
            detection
            for detection in detections
            if self._matches_label(detection.label)
        ]

        # ---------------------------------------------------------------
        # Confidence threshold
        # ---------------------------------------------------------------

        confident = [
            detection
            for detection in phones
            if detection.score
            >= self._settings.phone_confidence_threshold
        ]

        # ---------------------------------------------------------------
        # Phone found
        # ---------------------------------------------------------------

        if confident:

            self._last_confidence = max(
                detection.score
                for detection in confident
            )

            self._last_boxes = [
                self._box_to_dict(detection)
                for detection in confident
            ]

            raw_state = STATE_DETECTED

        # ---------------------------------------------------------------
        # No phone
        # ---------------------------------------------------------------

        else:

            self._last_confidence = 0.0
            self._last_boxes = []

            raw_state = STATE_CLEAR

        # ---------------------------------------------------------------
        # Temporal smoothing
        # ---------------------------------------------------------------

        stable = self._smoother.update(
            raw_state
        )

        # ---------------------------------------------------------------
        # Final result
        # ---------------------------------------------------------------

        return PhoneResult(
            state=stable,
            detected=(
                stable == STATE_DETECTED
                if stable != STATE_UNKNOWN
                else None
            ),
            confidence=round(
                self._last_confidence,
                3,
            ),
            boxes=list(self._last_boxes),
            error=None,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _matches_label(
        self,
        label: str,
    ) -> bool:
        """
        Match configured phone label.

        Example:
            filter = "phone"
            label  = "cell phone"

        Result:
            True
        """

        if not self._filter:
            return False

        return self._filter in label.lower()

    @staticmethod
    def _box_to_dict(
        detection: ObjectDetection,
    ) -> dict:

        x_min, y_min, x_max, y_max = detection.box

        return {
            "x_min": x_min,
            "y_min": y_min,
            "x_max": x_max,
            "y_max": y_max,
            "label": detection.label,
            "score": round(
                detection.score,
                3,
            ),
        }

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """
        Clear detection state and temporal history.

        Call this between study sessions.
        """

        self._missing_streak = 0
        self._last_boxes = []
        self._last_confidence = 0.0

        self._smoother.reset()


# ---------------------------------------------------------------------------
# Overlay drawing
# ---------------------------------------------------------------------------

def draw_phone_overlay(
    frame: np.ndarray,
    result: PhoneResult,
) -> np.ndarray:
    """
    Annotate an OpenCV BGR frame with detected phone boxes
    and a state chip.
    """

    import cv2

    # ------------------------------------------------------------------
    # Draw phone bounding boxes
    # ------------------------------------------------------------------

    for box in result.boxes:

        x_min = int(box["x_min"])
        y_min = int(box["y_min"])
        x_max = int(box["x_max"])
        y_max = int(box["y_max"])

        color = (70, 90, 220)

        cv2.rectangle(
            frame,
            (x_min, y_min),
            (x_max, y_max),
            color,
            2,
        )

        label = (
            f"{box['label']} "
            f"{box['score']:.2f}"
        )

        font = cv2.FONT_HERSHEY_SIMPLEX

        scale = 0.5
        thickness = 1

        (
            (tw, th),
            baseline,
        ) = cv2.getTextSize(
            label,
            font,
            scale,
            thickness,
        )

        cv2.rectangle(
            frame,
            (
                x_min,
                max(
                    0,
                    y_min - th - 8 - baseline,
                ),
            ),
            (
                x_min + tw + 8,
                y_min,
            ),
            color,
            -1,
        )

        cv2.putText(
            frame,
            label,
            (
                x_min + 4,
                y_min - 4,
            ),
            font,
            scale,
            (0, 0, 0),
            thickness,
            cv2.LINE_AA,
        )

    # ------------------------------------------------------------------
    # State chip color
    # ------------------------------------------------------------------

    chip_color = {
        STATE_DETECTED: (70, 70, 210),
        STATE_CLEAR: (70, 190, 90),
        STATE_UNKNOWN: (170, 170, 170),
    }.get(
        result.state,
        (170, 170, 170),
    )

    chip_text = (
        result.state
        .lower()
        .replace("_", " ")
    )

    font = cv2.FONT_HERSHEY_SIMPLEX

    scale = 0.55
    thickness = 1

    (
        (tw, th),
        baseline,
    ) = cv2.getTextSize(
        chip_text,
        font,
        scale,
        thickness,
    )

    pad_x = 6
    pad_y = 4

    x = 12
    y = int(frame.shape[0]) - 24

    cv2.rectangle(
        frame,
        (x, y),
        (
            x + tw + 2 * pad_x,
            y + th + 2 * pad_y + baseline,
        ),
        chip_color,
        -1,
    )

    cv2.putText(
        frame,
        chip_text,
        (
            x + pad_x,
            y + th + pad_y,
        ),
        font,
        scale,
        (0, 0, 0),
        thickness,
        cv2.LINE_AA,
    )

    return frame