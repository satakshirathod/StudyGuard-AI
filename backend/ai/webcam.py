"""Webcam capture module (Phase 1).

A thin, reusable wrapper around OpenCV's ``VideoCapture`` that:

* opens the camera once and reuses the instance (performance),
* exposes graceful degradation when the camera is unavailable or permission
  is denied (returns ``available=False`` instead of crashing),
* tracks a rolling FPS estimate,
* supports context-manager use and explicit ``release``.
"""

from __future__ import annotations

import time
from collections import deque

import cv2
import numpy as np

from utils.logger import get_logger

log = get_logger("studygard.ai.webcam")

_FPS_WINDOW = 30  # number of recent frames used for the rolling average


class WebcamCapture:
    """Wrapper around OpenCV ``VideoCapture`` with FPS tracking."""

    def __init__(self, camera_index: int = 0, width: int = 1280, height: int = 720) -> None:
        """Prepare the wrapper.

        Note: the camera is not opened here. Call :meth:`open`.
        """
        self._camera_index = camera_index
        self._width = width
        self._height = height
        self._capture: cv2.VideoCapture | None = None
        self._times: deque[float] = deque(maxlen=_FPS_WINDOW)
        self._last_error: str | None = None

    # ------------------------------------------------------------------ lifecycle

    def open(self) -> bool:
        """Open the camera and apply the requested frame size.

        Returns:
            ``True`` if the camera opened and is readable.
        """
        if self._capture is not None:
            return True

        try:
            self._capture = cv2.VideoCapture(self._camera_index)
            if not self._capture.isOpened():
                self._capture = None
                self._last_error = (
                    f"Camera index {self._camera_index} could not be opened. "
                    "Check it exists and that camera permission is granted."
                )
                log.error("Camera unavailable: %s", self._last_error)
                return False

            self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
            self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
            log.info(
                "Camera opened (index=%s, requested %sx%s)",
                self._camera_index, self._width, self._height,
            )
            return True
        except cv2.error as exc:  # pragma: no cover - environment dependent
            self._capture = None
            self._last_error = f"OpenCV failed to open the camera: {exc}"
            log.error("Camera error: %s", self._last_error)
            return False

    def release(self) -> None:
        """Release the camera if it is open (safe to call multiple times)."""
        if self._capture is not None:
            self._capture.release()
            self._capture = None
            log.info("Camera released")
        self._times.clear()

    # ------------------------------------------------------------------ accessors

    @property
    def available(self) -> bool:
        """Whether the camera is currently open and readable."""
        return self._capture is not None and (
            not hasattr(self._capture, "isOpened") or self._capture.isOpened()
        )

    @property
    def last_error(self) -> str | None:
        """Description of the last camera failure, if any."""
        return self._last_error

    @property
    def fps(self) -> float:
        """Recent rolling frames-per-second estimate (0.0 if not streaming)."""
        if len(self._times) < 2:
            return 0.0
        span = self._times[-1] - self._times[0]
        if span <= 0:
            return 0.0
        return (len(self._times) - 1) / span

    @property
    def resolution(self) -> tuple[int, int]:
        """Actual frame resolution, or the requested size if unknown."""
        if self._capture is not None:
            w = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if w > 0 and h > 0:
                return w, h
        return self._width, self._height

    # ------------------------------------------------------------------ capture

    def read(self) -> tuple[bool, np.ndarray | None]:
        """Grab a single frame.

        Returns:
            A tuple ``(ok, frame)``. ``ok`` is ``False`` and ``frame`` is
            ``None`` if the camera is closed or produced no frame.
        """
        if self._capture is None:
            return False, None
        ok, frame = self._capture.read()
        if not ok or frame is None:
            return False, None
        self._times.append(time.monotonic())
        return True, frame

    # ------------------------------------------------------------------ context manager

    def __enter__(self) -> "WebcamCapture":
        self.open()
        return self

    def __exit__(self, *exc) -> None:
        self.release()