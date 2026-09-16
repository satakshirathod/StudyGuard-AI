"""Real-time behavior engine + focus scoring (application foundation).

Aggregates the independent AI modules (face, gaze, drowsiness, posture,
distance, phone) into a single per-frame :class:`BehaviorSnapshot`, computes a
configurable 0–100 focus score, and decides when to raise *persistent* alerts.

Design rules
============
* Modules are independent. If a module is not implemented yet its section
  returns ``{"available": False, "state": "UNKNOWN"}`` — nothing is fabricated.
* The focus score is a weighted average over **available** modules only.
  Unavailable modules drop out of the score (and their weight), so the score
  is always a real measure of what the pipeline currently knows.
* Alerts require temporal persistence (seconds, not a single frame) and are
  rate-limited by a per-type cooldown.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

from ai.drowsiness_detection import STATE_NORMAL, DrowsinessResult
from ai.distance_detection import STATE_NORMAL as _DIST_NORMAL, DistanceResult
from ai.posture_detection import PostureResult
from ai.phone_detection import PhoneResult
from ai.gaze_tracking import (
    DIRECTION_AT_SCREEN,
    DIRECTION_AWAY,
    DIRECTION_UNKNOWN,
    GazeResult,
)
from ai.face_detection import FaceResult
from config import Settings
from utils.logger import get_logger

_log = get_logger("studygard.ai.behavior")

# Shared severity labels
SEVERITY_INFO = "info"
SEVERITY_WARNING = "warning"
SEVERITY_CRITICAL = "critical"

# Focus levels
FOCUS_HIGH = "high"
FOCUS_MEDIUM = "medium"
FOCUS_LOW = "low"

# Alert types (stable identifiers used by the alerts table + frontend)
ALERT_AWAY = "STUDENT_AWAY"
ALERT_GAZE_OFF = "LOOKING_AWAY"
ALERT_DROWSINESS = "DROWSINESS"
ALERT_PHONE = "PHONE"
ALERT_TOO_CLOSE = "TOO_CLOSE"
ALERT_POSTURE = "POOR_POSTURE"

# All alert modules are live since Phase 7 — previously pending modules get a
# real reading on every processed frame, never a fabricated zero/false value.
_PENDING_ALERTS: dict = {}


@dataclass
class BehaviorSnapshot:
    """One aggregated snapshot of the whole behavior pipeline."""

    # Section per module — matches the frontend status panel one-for-one.
    presence: dict = field(default_factory=dict)        # state: PRESENT|ABSENT|UNKNOWN
    gaze: dict = field(default_factory=dict)            # state, attention, confidence...
    drowsiness: dict = field(default_factory=dict)      # state, available
    posture: dict = field(default_factory=dict)
    distance: dict = field(default_factory=dict)
    phone: dict = field(default_factory=dict)
    distraction: dict = field(default_factory=dict)     # state: DISTRACTED|FOCUSED|UNKNOWN
    focus_score: Optional[int] = None                   # 0..100
    focus_level: Optional[str] = None                   # high | medium | low
    sampled_at: Optional[float] = None                  # unix seconds
    modules: dict = field(default_factory=dict)         # module -> available

    def to_dict(self) -> dict:
        return {
            "presence": self.presence,
            "gaze": self.gaze,
            "drowsiness": self.drowsiness,
            "posture": self.posture,
            "distance": self.distance,
            "phone": self.phone,
            "distraction": self.distraction,
            "focus_score": self.focus_score,
            "focus_level": self.focus_level,
            "sampled_at": self.sampled_at,
            "modules": self.modules,
        }


def _undefined_module() -> dict:
    return {"available": False, "state": "UNKNOWN"}


class BehaviorEngine:
    """Evaluate the current AI signals into a :class:`BehaviorSnapshot`."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._weights = {
            "gaze": settings.focus_weight_gaze,
            "presence": settings.focus_weight_presence,
            "drowsiness": settings.focus_weight_drowsiness,
            "posture": settings.focus_weight_posture,
            "distance": settings.focus_weight_distance,
            "phone": settings.focus_weight_phone,
        }
        if sum(self._weights.values()) <= 0:
            raise ValueError("sum of focus weights must be > 0")

    # ── evaluation ─────────────────────────────────────────────────────────

    def evaluate(
        self,
        face_result: Optional[FaceResult],
        gaze_result: Optional[GazeResult],
        drowsiness_result: Optional[DrowsinessResult] = None,
        distance_result: Optional[DistanceResult] = None,
        posture_result: Optional[PostureResult] = None,
        phone_result: Optional[PhoneResult] = None,
        sampled_at: Optional[float] = None,
    ) -> BehaviorSnapshot:
        snapshot = BehaviorSnapshot()
        snapshot.sampled_at = sampled_at if sampled_at is not None else time.time()
        snapshot.modules = {
            "face": face_result is not None,
            "gaze": gaze_result is not None,
            "drowsiness": drowsiness_result is not None,
            "posture": posture_result is not None,
            "distance": distance_result is not None,
            "phone": phone_result is not None,
        }

        # Presence (real)
        if face_result is not None:
            if face_result.face_detected:
                snapshot.presence = {
                    "state": "PRESENT",
                    "available": True,
                    "face_count": face_result.face_count,
                    "confidence": round(face_result.confidence, 3),
                }
            else:
                snapshot.presence = {"state": "ABSENT", "available": True, "face_count": 0}
        else:
            snapshot.presence = _undefined_module()

        # Gaze (real)
        if gaze_result is not None:
            snapshot.gaze = {
                "state": gaze_result.direction,
                "available": True,
                "attention": gaze_result.attention_score,
                "confidence": gaze_result.confidence,
                "head_yaw_deg": gaze_result.head_yaw_deg,
                "head_pitch_deg": gaze_result.head_pitch_deg,
            }
        else:
            snapshot.gaze = _undefined_module()

        # Drowsiness (real, Phase 4)
        if drowsiness_result is not None:
            snapshot.drowsiness = {
                "state": drowsiness_result.state,
                "available": True,
                "ear": drowsiness_result.ear,
                "left_ear": drowsiness_result.left_ear,
                "right_ear": drowsiness_result.right_ear,
                "eyes_closed": drowsiness_result.eyes_closed,
                "confidence": drowsiness_result.confidence,
            }
        else:
            snapshot.drowsiness = _undefined_module()
        # Posture (real, Phase 6)
        if posture_result is not None:
            snapshot.posture = {
                "state": posture_result.state,
                "available": True,
                "posture_score": posture_result.posture_score,
                "head_pitch_score": posture_result.head_pitch_score,
                "face_offset_score": posture_result.face_offset_score,
                "confidence": posture_result.confidence,
            }
        else:
            snapshot.posture = _undefined_module()
        # Phone detection (real, Phase 7)
        if phone_result is not None:
            snapshot.phone = {
                "state": phone_result.state,
                "available": True,
                "detected": phone_result.detected,
                "confidence": phone_result.confidence,
                "boxes": phone_result.boxes,
            }
        else:
            snapshot.phone = _undefined_module()

        # Screen distance (real, Phase 5)
        if distance_result is not None:
            snapshot.distance = {
                "state": distance_result.state,
                "distance_estimate": distance_result.distance_estimate,
                "face_width": distance_result.face_width,
                "too_close": distance_result.too_close,
                "confidence": distance_result.confidence,
                "available": True,
            }
        else:
            snapshot.distance = _undefined_module()

        # Distraction (derived from real signals: gaze + presence)
        state = "UNKNOWN"
        if gaze_result is not None and face_result is not None:
            if snapshot.presence.get("state") == "ABSENT" or gaze_result.direction == DIRECTION_AWAY:
                state = "DISTRACTED"
            elif gaze_result.direction == DIRECTION_AT_SCREEN:
                state = "FOCUSED"
        elif gaze_result is not None and gaze_result.direction in (DIRECTION_AT_SCREEN,):
            state = "FOCUSED"
        elif gaze_result is not None:
            state = "DISTRACTED" if gaze_result.direction != DIRECTION_UNKNOWN else "UNKNOWN"
        snapshot.distraction = {"state": state, "available": True}

        # Focus score from available modules only.
        snapshot.focus_score, snapshot.focus_level = self.focus_score(snapshot)
        return snapshot

    # ── focus score ────────────────────────────────────────────────────────

    def focus_score(self, snapshot: BehaviorSnapshot) -> tuple[Optional[int], Optional[str]]:
        """Compute the weighted 0–100 focus score over *available* modules."""
        scores: dict[str, Optional[int]] = {}

        if snapshot.gaze.get("available"):
            scores["gaze"] = snapshot.gaze.get("attention")
        if snapshot.presence.get("available"):
            scores["presence"] = 100 if snapshot.presence.get("state") == "PRESENT" else 0
        if snapshot.drowsiness.get("available") and snapshot.drowsiness.get("state") in (
            "NORMAL",
            "DROWSY",
        ):
            scores["drowsiness"] = 100 if snapshot.drowsiness.get("state") == STATE_NORMAL else 0
        if snapshot.posture.get("available") and snapshot.posture.get("state") in (
            "GOOD",
            "MODERATE",
            "POOR",
        ):
            scores["posture"] = _state_to_score(snapshot.posture.get("state"))
        if snapshot.distance.get("available") and snapshot.distance.get("state") in (
            _DIST_NORMAL,
            "TOO_CLOSE",
        ):
            scores["distance"] = 100 if snapshot.distance.get("state") == _DIST_NORMAL else 0
        if snapshot.phone.get("available") and snapshot.phone.get("detected") is not None:
            scores["phone"] = 0 if snapshot.phone.get("detected") else 100

        if not scores:
            return None, None

        total_weight = sum(self._weights[name] for name in scores)
        numerator = 0.0
        for name, value in scores.items():
            if value is not None:
                numerator += self._weights[name] * value
        if total_weight <= 0:
            return None, None
        score = int(round(numerator / total_weight))
        return score, self._focus_level(score)

    def _focus_level(self, score: int) -> str:
        if score >= self._settings.focus_level_high:
            return FOCUS_HIGH
        if score >= self._settings.focus_level_medium:
            return FOCUS_MEDIUM
        return FOCUS_LOW

    # ── weights (for API introspection) ────────────────────────────────────

    def weights(self) -> dict:
        return dict(self._weights)


def _state_to_score(state: str) -> int:
    return {"GOOD": 100, "MODERATE": 50, "POOR": 0}.get(state, 100)


class AlertEngine:
    """Persistence + cooldown for real-time alerts (per session).

    A monotonic ``clock`` callable is injectable so tests control time.
    """

    def __init__(self, settings: Settings, clock=time.monotonic) -> None:
        self._settings = settings
        self._clock = clock
        self._started_at: dict[str, float] = {}      # current condition onset
        self._last_alerted: dict[str, float] = {}    # last time an alert fired

    def reset_current_session(self) -> None:
        """Clear onset/last-alert tracking between sessions."""
        self._started_at.clear()
        self._last_alerted.clear()

    def evaluate(self, snapshot: BehaviorSnapshot) -> list[dict]:
        """Return alerts that should be persisted *now* (usually none)."""
        now = self._clock()
        fired: list[dict] = []

        # (type, active_now, required_seconds, severity, message)
        active = snapshot.presence.get("state")
        gaze_state = snapshot.gaze.get("state")
        drowsiness_state = snapshot.drowsiness.get("state")

        conditions: list[tuple[str, bool, float, str, str]] = [
            (
                ALERT_AWAY,
                active == "ABSENT",
                self._settings.alert_away_seconds,
                SEVERITY_WARNING,
                "You appear to be away from the camera.",
            ),
            (
                ALERT_GAZE_OFF,
                gaze_state in (DIRECTION_AWAY,)
                or (gaze_state not in (DIRECTION_AT_SCREEN, DIRECTION_UNKNOWN, DIRECTION_AWAY)),
                self._settings.alert_gaze_off_seconds,
                SEVERITY_WARNING,
                "Your attention seems to have drifted away from the screen.",
            ),
            (
                ALERT_DROWSINESS,
                drowsiness_state == "DROWSY",
                self._settings.alert_drowsiness_seconds,
                SEVERITY_CRITICAL,
                "Persistent eye closure detected — you may be getting drowsy.",
            ),
            (
                ALERT_TOO_CLOSE,
                snapshot.distance.get("state") == "TOO_CLOSE"
                and snapshot.distance.get("available"),
                self._settings.alert_too_close_seconds,
                SEVERITY_WARNING,
                "You are sitting too close to the screen. Move back to a comfortable distance.",
            ),
            (
                ALERT_POSTURE,
                snapshot.posture.get("state") == "POOR"
                and snapshot.posture.get("available"),
                self._settings.alert_poor_posture_seconds,
                SEVERITY_WARNING,
                "Your sitting posture looks slouched — try sitting upright for better focus.",
            ),
            (
                ALERT_PHONE,
                snapshot.phone.get("available")
                and snapshot.phone.get("detected") is True,
                self._settings.alert_phone_seconds,
                SEVERITY_WARNING,
                "A phone appears to be in view — put it down and stay focused.",
            ),
        ]

        for alert_type, is_active, threshold, severity, message in conditions:
            if is_active:
                self._started_at.setdefault(alert_type, now)
                elapsed = now - self._started_at[alert_type]
                if elapsed >= threshold:
                    last = self._last_alerted.get(alert_type, 0.0)
                    if now - last >= self._settings.alert_cooldown_seconds:
                        self._last_alerted[alert_type] = now
                        fired.append(
                            {
                                "alert_type": alert_type,
                                "severity": severity,
                                "message": message,
                            }
                        )
            else:
                self._started_at.pop(alert_type, None)

        return fired


def pending_alert_types() -> dict:
    """Expose the list of alert types gated on not-yet-built modules."""
    return {k: v for k, v in _PENDING_ALERTS.items()}