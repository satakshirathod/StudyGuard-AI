"""Face detection and facial landmark extraction (Phase 2).

Uses two complementary MediaPipe 0.10.x tasks:

* FaceDetector — BlazeFace short-range model producing real detection
  confidence scores and bounding boxes.
* FaceLandmarker — 478-point face mesh producing normalised (x, y, z)
  landmark coordinates per face.

Both tasks run in VIDEO mode.

Important:
MediaPipe VIDEO mode requires timestamps to be strictly increasing.
This module therefore maintains separate monotonic timestamps for the
FaceDetector and FaceLandmarker.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from config import Settings
from utils.logger import get_logger


log = get_logger("studygard.ai.face_detection")


# ─────────────────────────────────────────────────────────────────────────────
# MediaPipe landmark indices
# ─────────────────────────────────────────────────────────────────────────────

LEFT_EYE_CONTOUR: List[int] = [33, 160, 158, 133, 153, 144]

RIGHT_EYE_CONTOUR: List[int] = [362, 385, 387, 263, 373, 380]

LEFT_IRIS: int = 468
RIGHT_IRIS: int = 473

NOSE_TIP: int = 4

MOUTH_OUTER: List[int] = [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375,
    291, 409, 270, 269, 267, 0, 37, 39, 40, 185,
]

FACE_OVAL: List[int] = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323,
    361, 288, 397, 365, 379, 378, 400, 377, 152, 148,
    176, 149, 150, 136, 172, 58, 132, 93, 234, 127,
    162, 21, 54, 103, 67, 109,
]


# ─────────────────────────────────────────────────────────────────────────────
# Data containers
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FaceData:
    """Landmarks and metadata for a single detected face."""

    bbox_pixels: Tuple[int, int, int, int]
    bbox_normalized: Tuple[float, float, float, float]
    area: float

    confidence: float

    # Full landmark points: (x, y, z)
    landmarks: List[Tuple[float, float, float]]

    # Region subsets
    left_eye: List[Tuple[float, float]]
    right_eye: List[Tuple[float, float]]

    left_iris: Optional[Tuple[float, float]]
    right_iris: Optional[Tuple[float, float]]

    nose_tip: Optional[Tuple[float, float]]

    mouth: List[Tuple[float, float]]
    face_outline: List[Tuple[float, float]]


@dataclass
class FaceResult:
    """Aggregated face-detection result for a single frame."""

    face_detected: bool = False
    face_count: int = 0
    multiple_faces: bool = False

    primary_face_detected: bool = False
    landmarks_available: bool = False

    confidence: float = 0.0
    confidence_available: bool = False

    status: str = "UNKNOWN"

    faces: List[FaceData] = field(default_factory=list)

    error: Optional[str] = None


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _euclidean(
    a: Tuple[float, float],
    b: Tuple[float, float],
) -> float:
    """Calculate Euclidean distance between two 2D points."""

    return math.hypot(
        a[0] - b[0],
        a[1] - b[1],
    )


def _bbox_from_landmarks(
    landmarks: list,
    frame_w: int,
    frame_h: int,
) -> Tuple[
    Tuple[int, int, int, int],
    Tuple[float, float, float, float],
    float,
]:
    """Create a bounding box from MediaPipe face landmarks."""

    xs = [lm.x for lm in landmarks]
    ys = [lm.y for lm in landmarks]

    if not xs or not ys:
        return (
            (0, 0, 0, 0),
            (0.0, 0.0, 0.0, 0.0),
            0.0,
        )

    x_min_n = min(xs)
    x_max_n = max(xs)

    y_min_n = min(ys)
    y_max_n = max(ys)

    # Small padding around face
    pad_x = (x_max_n - x_min_n) * 0.05
    pad_y = (y_max_n - y_min_n) * 0.05

    x_min_n = max(0.0, x_min_n - pad_x)
    x_max_n = min(1.0, x_max_n + pad_x)

    y_min_n = max(0.0, y_min_n - pad_y)
    y_max_n = min(1.0, y_max_n + pad_y)

    x_min_px = int(x_min_n * frame_w)
    y_min_px = int(y_min_n * frame_h)

    x_max_px = int(x_max_n * frame_w)
    y_max_px = int(y_max_n * frame_h)

    area = float(
        max(0, x_max_px - x_min_px)
        * max(0, y_max_px - y_min_px)
    )

    return (
        (x_min_px, y_min_px, x_max_px, y_max_px),
        (x_min_n, y_min_n, x_max_n, y_max_n),
        area,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main module
# ─────────────────────────────────────────────────────────────────────────────

class FaceDetectionModule:
    """Initialise and run MediaPipe FaceDetector + FaceLandmarker."""

    def __init__(self, settings: Settings) -> None:

        self._settings = settings

        self._detector = None
        self._landmarker = None

        self._detector_ok = False
        self._landmarker_ok = False

        self._initialized = False

        # IMPORTANT:
        # MediaPipe VIDEO mode requires timestamps to strictly increase.
        self._last_detector_timestamp_ms = 0
        self._last_landmarker_timestamp_ms = 0

    # ─────────────────────────────────────────────────────────────────────
    # Timestamp helper
    # ─────────────────────────────────────────────────────────────────────

    @staticmethod
    def _next_timestamp_ms(last_timestamp: int) -> int:
        """
        Generate a strictly increasing timestamp.

        time.monotonic_ns() is used instead of wall-clock time because
        monotonic clocks never move backwards.
        """

        current = time.monotonic_ns() // 1_000_000

        if current <= last_timestamp:
            current = last_timestamp + 1

        return current

    # ─────────────────────────────────────────────────────────────────────
    # Lifecycle
    # ─────────────────────────────────────────────────────────────────────

    def initialize(self) -> bool:
        """Load MediaPipe models."""

        if self._initialized:
            return self._detector_ok or self._landmarker_ok

        try:
            from mediapipe.tasks.python import BaseOptions
            from mediapipe.tasks.python import vision as mp_vision

            # ─────────────────────────────────────────────────────────────
            # Face Detector
            # ─────────────────────────────────────────────────────────────

            det_path = self._settings.face_detector_model

            if Path(det_path).exists():

                det_opts = mp_vision.FaceDetectorOptions(
                    base_options=BaseOptions(
                        model_asset_path=det_path
                    ),
                    running_mode=mp_vision.RunningMode.VIDEO,
                )

                self._detector = (
                    mp_vision.FaceDetector
                    .create_from_options(det_opts)
                )

                self._detector_ok = True

                log.info(
                    "Face detector initialised (%s)",
                    det_path,
                )

            else:

                log.warning(
                    "Face detector model not found: %s",
                    det_path,
                )

            # ─────────────────────────────────────────────────────────────
            # Face Landmarker
            # ─────────────────────────────────────────────────────────────

            lm_path = self._settings.face_landmarker_model

            if Path(lm_path).exists():

                lm_opts = mp_vision.FaceLandmarkerOptions(
                    base_options=BaseOptions(
                        model_asset_path=lm_path
                    ),
                    running_mode=mp_vision.RunningMode.VIDEO,
                    output_face_blendshapes=False,
                    num_faces=self._settings.max_faces,
                )

                self._landmarker = (
                    mp_vision.FaceLandmarker
                    .create_from_options(lm_opts)
                )

                self._landmarker_ok = True

                log.info(
                    "Face landmarker initialised (%s)",
                    lm_path,
                )

            else:

                log.warning(
                    "Face landmarker model not found: %s",
                    lm_path,
                )

        except Exception as exc:

            log.error(
                "Face detection initialisation failed: %s",
                exc,
            )

            self._detector_ok = False
            self._landmarker_ok = False

        self._initialized = True

        ok = self._detector_ok or self._landmarker_ok

        if ok:

            log.info(
                "Face detection module ready "
                "(detector=%s, landmarker=%s)",
                self._detector_ok,
                self._landmarker_ok,
            )

        else:

            log.warning(
                "Face detection module degraded: "
                "no models available"
            )

        return ok

    # ─────────────────────────────────────────────────────────────────────
    # Cleanup
    # ─────────────────────────────────────────────────────────────────────

    def cleanup(self) -> None:
        """Release MediaPipe model resources."""

        if self._detector is not None:

            try:
                self._detector.close()
            except Exception:
                pass

            self._detector = None

        if self._landmarker is not None:

            try:
                self._landmarker.close()
            except Exception:
                pass

            self._landmarker = None

        self._detector_ok = False
        self._landmarker_ok = False

        self._last_detector_timestamp_ms = 0
        self._last_landmarker_timestamp_ms = 0

        log.info(
            "Face detection module cleaned up"
        )

    # ─────────────────────────────────────────────────────────────────────
    # Properties
    # ─────────────────────────────────────────────────────────────────────

    @property
    def ready(self) -> bool:
        return (
            self._initialized
            and (
                self._detector_ok
                or self._landmarker_ok
            )
        )

    @property
    def detector_available(self) -> bool:
        return self._detector_ok

    @property
    def landmarker_available(self) -> bool:
        return self._landmarker_ok

    # ─────────────────────────────────────────────────────────────────────
    # Frame processing
    # ─────────────────────────────────────────────────────────────────────

    def process(
        self,
        frame_bgr: np.ndarray,
    ) -> FaceResult:
        """Analyse a BGR frame and return structured face data."""

        if not self._initialized:
            self.initialize()

        if (
            frame_bgr is None
            or not isinstance(frame_bgr, np.ndarray)
            or frame_bgr.ndim < 2
        ):
            return FaceResult(
                status="UNKNOWN",
                error="invalid frame",
            )

        if not self.ready:
            return FaceResult(
                status="UNKNOWN",
                error="models not loaded",
            )

        frame_h, frame_w = frame_bgr.shape[:2]

        # ─────────────────────────────────────────────────────────────────
        # Convert OpenCV BGR -> MediaPipe RGB
        # ─────────────────────────────────────────────────────────────────

        try:

            import mediapipe as mp

            rgb = cv2.cvtColor(
                frame_bgr,
                cv2.COLOR_BGR2RGB,
            )

            mp_img = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb,
            )

        except Exception as exc:

            log.warning(
                "Frame conversion failed: %s",
                exc,
            )

            return FaceResult(
                status="UNKNOWN",
                error=str(exc),
            )

        # ─────────────────────────────────────────────────────────────────
        # Run models
        # ─────────────────────────────────────────────────────────────────

        try:

            detections, lm_faces = self._run_models(
                mp_img,
                frame_w,
                frame_h,
            )

        except Exception as exc:

            log.error(
                "Model inference failed: %s",
                exc,
            )

            return FaceResult(
                status="UNKNOWN",
                error=str(exc),
            )

        return self._build_result(
            detections,
            lm_faces,
            frame_w,
            frame_h,
        )

    # ─────────────────────────────────────────────────────────────────────
    # MediaPipe inference
    # ─────────────────────────────────────────────────────────────────────

    def _run_models(
        self,
        mp_img,
        frame_w: int,
        frame_h: int,
    ):
        """
        Run detector and landmarker.

        IMPORTANT:
        Each MediaPipe VIDEO task receives its own strictly increasing
        timestamp.
        """

        detections = []
        lm_faces = []

        # ─────────────────────────────────────────────────────────────────
        # Face Detector
        # ─────────────────────────────────────────────────────────────────

        if (
            self._detector_ok
            and self._detector is not None
        ):

            timestamp_ms = self._next_timestamp_ms(
                self._last_detector_timestamp_ms
            )

            self._last_detector_timestamp_ms = timestamp_ms

            det_res = self._detector.detect_for_video(
                mp_img,
                timestamp_ms,
            )

            detections = (
                det_res.detections
                if det_res.detections
                else []
            )

        # ─────────────────────────────────────────────────────────────────
        # Face Landmarker
        # ─────────────────────────────────────────────────────────────────

        if (
            self._landmarker_ok
            and self._landmarker is not None
        ):

            timestamp_ms = self._next_timestamp_ms(
                self._last_landmarker_timestamp_ms
            )

            self._last_landmarker_timestamp_ms = timestamp_ms

            lm_res = self._landmarker.detect_for_video(
                mp_img,
                timestamp_ms,
            )

            lm_faces = (
                lm_res.face_landmarks
                if lm_res.face_landmarks
                else []
            )

        return detections, lm_faces

    # ─────────────────────────────────────────────────────────────────────
    # Result construction
    # ─────────────────────────────────────────────────────────────────────

    def _build_result(
        self,
        detections,
        lm_faces,
        frame_w: int,
        frame_h: int,
    ) -> FaceResult:
        """Match detections to landmarks and build FaceResult."""

        n_det = len(detections)
        n_lm = len(lm_faces)

        # No face
        if n_lm == 0 and n_det == 0:

            return FaceResult(
                face_detected=False,
                face_count=0,
                multiple_faces=False,
                primary_face_detected=False,
                landmarks_available=False,
                confidence=0.0,
                confidence_available=self._detector_ok,
                status="NO_FACE",
            )

        # ─────────────────────────────────────────────────────────────────
        # Detection centres
        # ─────────────────────────────────────────────────────────────────

        det_centers = []

        for det in detections:

            bb = det.bounding_box

            cx = (
                bb.origin_x
                + bb.width / 2
            )

            cy = (
                bb.origin_y
                + bb.height / 2
            )

            det_centers.append(
                (cx, cy)
            )

        used_det: set = set()

        faces: List[FaceData] = []

        # ─────────────────────────────────────────────────────────────────
        # Build landmark faces
        # ─────────────────────────────────────────────────────────────────

        for face_lms in lm_faces[
            : self._settings.max_faces
        ]:

            if not face_lms:
                continue

            (
                bbox_px,
                bbox_norm,
                area,
            ) = _bbox_from_landmarks(
                face_lms,
                frame_w,
                frame_h,
            )

            cx = (
                (bbox_norm[0] + bbox_norm[2])
                / 2
                * frame_w
            )

            cy = (
                (bbox_norm[1] + bbox_norm[3])
                / 2
                * frame_h
            )

            # Find closest detector result
            best_det_idx = -1
            best_dist = float("inf")

            for i, (
                dcx,
                dcy,
            ) in enumerate(det_centers):

                if i in used_det:
                    continue

                distance = _euclidean(
                    (cx, cy),
                    (dcx, dcy),
                )

                if distance < best_dist:

                    best_dist = distance
                    best_det_idx = i

            # Detector confidence
            conf = 0.0

            if best_det_idx >= 0:

                used_det.add(
                    best_det_idx
                )

                cats = detections[
                    best_det_idx
                ].categories

                if cats:
                    conf = float(
                        cats[0].score
                    )

            # ─────────────────────────────────────────────────────────────
            # Landmark helpers
            # ─────────────────────────────────────────────────────────────

            def _lm_xy(
                idx: int,
            ) -> Tuple[float, float]:

                lm = face_lms[idx]

                return (
                    float(lm.x),
                    float(lm.y),
                )

            def _lm_xyz(
                idx: int,
            ) -> Tuple[
                float,
                float,
                float,
            ]:

                lm = face_lms[idx]

                return (
                    float(lm.x),
                    float(lm.y),
                    float(lm.z),
                )

            # Full landmarks
            all_lms = [
                _lm_xyz(i)
                for i in range(
                    len(face_lms)
                )
            ]

            # ─────────────────────────────────────────────────────────────
            # FaceData
            # ─────────────────────────────────────────────────────────────

            faces.append(
                FaceData(
                    bbox_pixels=bbox_px,

                    bbox_normalized=bbox_norm,

                    area=area,

                    confidence=conf,

                    landmarks=all_lms,

                    left_eye=[
                        _lm_xy(i)
                        for i in LEFT_EYE_CONTOUR
                        if i < len(face_lms)
                    ],

                    right_eye=[
                        _lm_xy(i)
                        for i in RIGHT_EYE_CONTOUR
                        if i < len(face_lms)
                    ],

                    left_iris=(
                        _lm_xy(LEFT_IRIS)
                        if LEFT_IRIS
                        < len(face_lms)
                        else None
                    ),

                    right_iris=(
                        _lm_xy(RIGHT_IRIS)
                        if RIGHT_IRIS
                        < len(face_lms)
                        else None
                    ),

                    nose_tip=(
                        _lm_xy(NOSE_TIP)
                        if NOSE_TIP
                        < len(face_lms)
                        else None
                    ),

                    mouth=[
                        _lm_xy(i)
                        for i in MOUTH_OUTER
                        if i < len(face_lms)
                    ],

                    face_outline=[
                        _lm_xy(i)
                        for i in FACE_OVAL
                        if i < len(face_lms)
                    ],
                )
            )

        # ─────────────────────────────────────────────────────────────────
        # Add detector faces without landmarks
        # ─────────────────────────────────────────────────────────────────

        for i, det in enumerate(detections):

            if i in used_det:
                continue

            bb = det.bounding_box

            conf = (
                float(
                    det.categories[0].score
                )
                if det.categories
                else 0.0
            )

            x1 = bb.origin_x
            y1 = bb.origin_y

            x2 = (
                bb.origin_x
                + bb.width
            )

            y2 = (
                bb.origin_y
                + bb.height
            )

            faces.append(
                FaceData(
                    bbox_pixels=(
                        x1,
                        y1,
                        x2,
                        y2,
                    ),

                    bbox_normalized=(
                        x1 / frame_w,
                        y1 / frame_h,
                        x2 / frame_w,
                        y2 / frame_h,
                    ),

                    area=float(
                        bb.width
                        * bb.height
                    ),

                    confidence=conf,

                    landmarks=[],

                    left_eye=[],
                    right_eye=[],

                    left_iris=None,
                    right_iris=None,

                    nose_tip=None,

                    mouth=[],
                    face_outline=[],
                )
            )

        # ─────────────────────────────────────────────────────────────────
        # Largest face = primary face
        # ─────────────────────────────────────────────────────────────────

        faces.sort(
            key=lambda f: f.area,
            reverse=True,
        )

        primary = (
            faces[0]
            if faces
            else None
        )

        lm_faces_only = [
            f
            for f in faces
            if len(f.landmarks) > 0
        ]

        total_faces = max(
            n_det,
            n_lm,
        )

        return FaceResult(
            face_detected=True,

            face_count=total_faces,

            multiple_faces=(
                total_faces > 1
            ),

            primary_face_detected=(
                primary is not None
                and len(primary.landmarks) > 0
            ),

            landmarks_available=(
                len(lm_faces_only) > 0
            ),

            confidence=(
                primary.confidence
                if primary
                else 0.0
            ),

            confidence_available=(
                self._detector_ok
            ),

            status="DETECTED",

            faces=faces[
                : self._settings.max_faces
            ],
        )


# ─────────────────────────────────────────────────────────────────────────────
# Overlay
# ─────────────────────────────────────────────────────────────────────────────

def draw_overlay(
    frame: np.ndarray,
    result: FaceResult,
) -> np.ndarray:
    """Draw face bounding boxes and landmarks on the frame."""

    if (
        not result.face_detected
        or not result.faces
    ):
        return frame

    h, w = frame.shape[:2]

    for i, face in enumerate(
        result.faces
    ):

        is_primary = i == 0

        colour = (
            (0, 200, 255)
            if is_primary
            else (120, 120, 120)
        )

        thick = (
            2
            if is_primary
            else 1
        )

        # ─────────────────────────────────────────────────────────────────
        # Bounding box
        # ─────────────────────────────────────────────────────────────────

        x1, y1, x2, y2 = (
            face.bbox_pixels
        )

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            colour,
            thick,
        )

        # ─────────────────────────────────────────────────────────────────
        # Label
        # ─────────────────────────────────────────────────────────────────

        label = (
            "Primary"
            if is_primary
            else f"Face {i + 1}"
        )

        cv2.putText(
            frame,
            label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            colour,
            1,
            cv2.LINE_AA,
        )

        # ─────────────────────────────────────────────────────────────────
        # Landmarks
        # ─────────────────────────────────────────────────────────────────

        if not face.landmarks:
            continue

        # Left eye
        for lx, ly in face.left_eye:

            px = int(lx * w)
            py = int(ly * h)

            cv2.circle(
                frame,
                (px, py),
                2,
                (0, 255, 0),
                -1,
            )

        # Right eye
        for rx, ry in face.right_eye:

            px = int(rx * w)
            py = int(ry * h)

            cv2.circle(
                frame,
                (px, py),
                2,
                (0, 255, 0),
                -1,
            )

        # ─────────────────────────────────────────────────────────────────
        # Iris
        # ─────────────────────────────────────────────────────────────────

        if face.left_iris:

            px = int(
                face.left_iris[0] * w
            )

            py = int(
                face.left_iris[1] * h
            )

            cv2.circle(
                frame,
                (px, py),
                4,
                (0, 200, 255),
                1,
            )

        if face.right_iris:

            px = int(
                face.right_iris[0] * w
            )

            py = int(
                face.right_iris[1] * h
            )

            cv2.circle(
                frame,
                (px, py),
                4,
                (0, 200, 255),
                1,
            )

        # ─────────────────────────────────────────────────────────────────
        # Face outline
        # ─────────────────────────────────────────────────────────────────

        if face.face_outline:

            pts = np.array(
                [
                    (
                        int(p[0] * w),
                        int(p[1] * h),
                    )
                    for p in face.face_outline
                ],
                dtype=np.int32,
            )

            cv2.polylines(
                frame,
                [pts],
                True,
                (200, 200, 200),
                1,
                cv2.LINE_AA,
            )

        # ─────────────────────────────────────────────────────────────────
        # Nose tip
        # ─────────────────────────────────────────────────────────────────

        if face.nose_tip:

            px = int(
                face.nose_tip[0] * w
            )

            py = int(
                face.nose_tip[1] * h
            )

            cv2.circle(
                frame,
                (px, py),
                3,
                (255, 200, 0),
                -1,
            )

    return frame