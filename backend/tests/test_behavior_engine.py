"""Behavior engine + alert engine tests (application foundation)."""

from __future__ import annotations

import pytest

from ai.behavior_engine import (
    BehaviorEngine,
    BehaviorSnapshot,
    AlertEngine,
    FOCUS_HIGH,
    FOCUS_MEDIUM,
    FOCUS_LOW,
    ALERT_DROWSINESS,
    ALERT_TOO_CLOSE,
    ALERT_POSTURE,
    ALERT_PHONE,
)
from ai.drowsiness_detection import (
    STATE_NORMAL,
    STATE_DROWSY,
    STATE_UNKNOWN,
    DrowsinessResult,
)
from ai.distance_detection import (
    STATE_NORMAL as DIST_NORMAL,
    STATE_TOO_CLOSE,
    STATE_UNKNOWN as DIST_UNKNOWN,
    DistanceResult,
)
from ai.posture_detection import (
    STATE_GOOD,
    STATE_MODERATE,
    STATE_POOR,
    STATE_UNKNOWN as POSTURE_UNKNOWN,
    PostureResult,
)
from ai.phone_detection import (
    STATE_DETECTED as PHONE_DETECTED,
    STATE_UNKNOWN as PHONE_UNKNOWN,
    PhoneResult,
)
from ai.face_detection import FaceResult
from ai.gaze_tracking import DIRECTION_AT_SCREEN, DIRECTION_LEFT, DIRECTION_AWAY
from config import get_settings

_s = get_settings()


def _face(present: bool = True) -> FaceResult:
    result = FaceResult()
    result.face_detected = present
    result.face_count = 1 if present else 0
    result.confidence = 0.95 if present else 0.0
    result.confidence_available = True
    return result


def _gaze(state: str, attention: int | None = None, confidence: float = 1.0):
    from ai.gaze_tracking import GazeResult

    return GazeResult(direction=state, attention_score=attention, confidence=confidence)


class TestBehaviorEngine:
    def test_undefined_modules_are_honest(self):
        engine = BehaviorEngine(_s)
        snap = engine.evaluate(None, None)
        assert snap.presence["state"] == "UNKNOWN"
        assert snap.presence["available"] is False
        assert snap.drowsiness["available"] is False
        assert snap.posture["available"] is False
        assert snap.distance["available"] is False
        assert snap.phone["available"] is False
        assert snap.focus_score is None
        assert snap.focus_level is None
        # no fabricated numbers anywhere
        assert snap.gaze.get("attention") is None
        assert snap.distance.get("distance_cm") is None
        assert snap.phone.get("detected") is None

    def test_present_at_screen_scores_high(self):
        engine = BehaviorEngine(_s)
        snap = engine.evaluate(_face(True), _gaze(DIRECTION_AT_SCREEN, attention=100))
        assert snap.presence["state"] == "PRESENT"
        assert snap.gaze["state"] == DIRECTION_AT_SCREEN
        assert snap.distraction["state"] == "FOCUSED"
        assert snap.focus_score == 100
        assert snap.focus_level == FOCUS_HIGH

    def test_absent_away_scores_low(self):
        engine = BehaviorEngine(_s)
        snap = engine.evaluate(_face(False), _gaze(DIRECTION_AWAY, attention=0))
        assert snap.presence["state"] == "ABSENT"
        assert snap.distraction["state"] == "DISTRACTED"
        assert snap.focus_score == 0
        assert snap.focus_level == FOCUS_LOW

    def test_mixed_signal_uses_available_weights(self):
        engine = BehaviorEngine(_s)
        snap = engine.evaluate(_face(True), _gaze(DIRECTION_LEFT, attention=50))
        # presence=100, gaze=50 → weighted over {gaze,presence} only
        expected = round(
            (engine.weights()["gaze"] * 50 + engine.weights()["presence"] * 100)
            / (engine.weights()["gaze"] + engine.weights()["presence"])
        )
        assert snap.focus_score == expected

    def test_focus_level_boundaries(self):
        engine = BehaviorEngine(_s)
        assert engine._focus_level(_s.focus_level_high) == FOCUS_HIGH
        assert engine._focus_level(_s.focus_level_medium) == FOCUS_MEDIUM
        assert engine._focus_level(_s.focus_level_medium - 1) == FOCUS_LOW

    def test_drowsiness_available_fills_snapshot(self):
        engine = BehaviorEngine(_s)
        snap = engine.evaluate(
            _face(True),
            _gaze(DIRECTION_AT_SCREEN, attention=100),
            drowsiness_result=DrowsinessResult(
                state=STATE_NORMAL, ear=0.34, confidence=1.0
            ),
        )
        assert snap.drowsiness["available"] is True
        assert snap.drowsiness["state"] == STATE_NORMAL
        assert snap.drowsiness["ear"] == 0.34
        assert snap.modules["drowsiness"] is True

    def test_drowsy_lowers_focus_score(self):
        engine = BehaviorEngine(_s)
        normal = engine.evaluate(
            _face(True),
            _gaze(DIRECTION_AT_SCREEN, attention=100),
            drowsiness_result=DrowsinessResult(state=STATE_NORMAL, ear=0.34),
        )
        drowsy = engine.evaluate(
            _face(True),
            _gaze(DIRECTION_AT_SCREEN, attention=100),
            drowsiness_result=DrowsinessResult(state=STATE_DROWSY, ear=0.10),
        )
        # With drowsiness weighted, a drowsy student scores below a normal one.
        assert drowsy.focus_score < normal.focus_score

    def test_drowsiness_unknown_does_not_fake_zero(self):
        engine = BehaviorEngine(_s)
        snap = engine.evaluate(
            _face(True),
            _gaze(DIRECTION_AT_SCREEN, attention=100),
            drowsiness_result=DrowsinessResult(state=STATE_UNKNOWN),
        )
        # UNKNOWN drowsiness must not drag the score down as if it were drowsy.
        expected = engine.weights()["gaze"] * 100 + engine.weights()["presence"] * 100
        expected /= engine.weights()["gaze"] + engine.weights()["presence"]
        assert snap.focus_score == round(expected)

    def test_distance_available_fills_snapshot(self):
        engine = BehaviorEngine(_s)
        snap = engine.evaluate(
            _face(True),
            _gaze(DIRECTION_AT_SCREEN, attention=100),
            drowsiness_result=DrowsinessResult(state=STATE_NORMAL, ear=0.34),
            distance_result=DistanceResult(
                state=DIST_NORMAL, distance_estimate=1.0,
                face_width=80.0, too_close=False, confidence=1.0,
            ),
        )
        assert snap.distance["available"] is True
        assert snap.distance["state"] == DIST_NORMAL
        assert snap.distance["distance_estimate"] == 1.0
        assert snap.distance["too_close"] is False
        assert snap.modules["distance"] is True

    def test_too_close_lowers_focus_score(self):
        engine = BehaviorEngine(_s)
        normal = engine.evaluate(
            _face(True),
            _gaze(DIRECTION_AT_SCREEN, attention=100),
            drowsiness_result=DrowsinessResult(state=STATE_NORMAL, ear=0.34),
            distance_result=DistanceResult(state=DIST_NORMAL, too_close=False),
        )
        close = engine.evaluate(
            _face(True),
            _gaze(DIRECTION_AT_SCREEN, attention=100),
            drowsiness_result=DrowsinessResult(state=STATE_NORMAL, ear=0.34),
            distance_result=DistanceResult(
                state=STATE_TOO_CLOSE, distance_estimate=2.0, too_close=True,
            ),
        )
        assert close.focus_score < normal.focus_score
        assert normal.focus_score in (100,)
        assert close.focus_score > 0  # distance is weighted, not the whole score

    def test_distance_unknown_does_not_fake_zero(self):
        engine = BehaviorEngine(_s)
        snap = engine.evaluate(
            _face(True),
            _gaze(DIRECTION_AT_SCREEN, attention=100),
            drowsiness_result=DrowsinessResult(state=STATE_NORMAL, ear=0.34),
            distance_result=DistanceResult(state=DIST_UNKNOWN),
        )
        # UNKNOWN distance is excluded from the weighted average, not scored 0.
        expected = engine.weights()["gaze"] * 100 + engine.weights()["presence"] * 100
        expected += engine.weights()["drowsiness"] * 100
        expected /= (
            engine.weights()["gaze"] + engine.weights()["presence"]
            + engine.weights()["drowsiness"]
        )
        assert snap.focus_score == round(expected)

    def test_posture_available_fills_snapshot(self):
        engine = BehaviorEngine(_s)
        snap = engine.evaluate(
            _face(True),
            _gaze(DIRECTION_AT_SCREEN, attention=100),
            posture_result=PostureResult(
                state=STATE_GOOD, posture_score=76.0,
                head_pitch_score=75.0, face_offset_score=80.0, confidence=1.0,
            ),
        )
        assert snap.posture["available"] is True
        assert snap.posture["state"] == STATE_GOOD
        assert snap.posture["posture_score"] == 76.0
        assert snap.modules["posture"] is True

    def test_poor_posture_lowers_focus_score(self):
        engine = BehaviorEngine(_s)
        good = engine.evaluate(
            _face(True),
            _gaze(DIRECTION_AT_SCREEN, attention=100),
            posture_result=PostureResult(state=STATE_GOOD, posture_score=100),
        )
        poor = engine.evaluate(
            _face(True),
            _gaze(DIRECTION_AT_SCREEN, attention=100),
            posture_result=PostureResult(state=STATE_POOR, posture_score=0),
        )
        assert poor.focus_score < good.focus_score

    def test_posture_unknown_excluded_from_score(self):
        engine = BehaviorEngine(_s)
        no_posture = engine.evaluate(
            _face(True),
            _gaze(DIRECTION_AT_SCREEN, attention=100),
        )
        with_posture = engine.evaluate(
            _face(True),
            _gaze(DIRECTION_AT_SCREEN, attention=100),
            posture_result=PostureResult(state=POSTURE_UNKNOWN),
        )
        # UNKNOWN posture must not change the focus score.
        assert no_posture.focus_score == with_posture.focus_score


class FakeClock:
    def __init__(self, start: float = 1000.0) -> None:
        self.t = start

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


class TestAlertEngine:
    def _make_session(self, clock: FakeClock | None = None):
        return AlertEngine(_s, clock=(clock if clock is not None else FakeClock()))

    def test_no_alert_for_focused_student(self):
        engine = AlertEngine(_s)
        snap = BehaviorSnapshot()
        snap.presence = {"state": "PRESENT", "available": True}
        snap.gaze = {"state": DIRECTION_AT_SCREEN, "available": True}
        assert engine.evaluate(snap) == []

    def test_away_requires_persistence(self):
        clock = FakeClock()
        engine = AlertEngine(_s, clock=clock)
        snap = BehaviorSnapshot()
        snap.presence = {"state": "ABSENT", "available": True}
        snap.gaze = {"state": DIRECTION_AWAY, "available": True, "attention": 0}

        # Advance in small steps; short absences (below threshold) must not fire.
        for _ in range(20):
            clock.advance(0.1)
            assert engine.evaluate(snap) == []

        # Push past the threshold → exactly one alert fires.
        clock.advance(max(_s.alert_away_seconds + 1, 2.0))
        fired = engine.evaluate(snap)
        assert any(a["alert_type"] == "STUDENT_AWAY" for a in fired)

    def test_gaze_off_alert_tracks_sustained_off_screen(self):
        clock = FakeClock()
        engine = AlertEngine(_s, clock=clock)
        snap = BehaviorSnapshot()
        snap.presence = {"state": "PRESENT", "available": True}
        snap.gaze = {"state": "LOOKING_LEFT", "available": True, "attention": 30}
        # 5s of stepping stays below the 8s threshold → no alert.
        for _ in range(10):
            clock.advance(0.5)
            assert engine.evaluate(snap) == []
        clock.advance(_s.alert_gaze_off_seconds + 1)
        fired = engine.evaluate(snap)
        assert any(a["alert_type"] == "LOOKING_AWAY" for a in fired)

    def test_cooldown_limits_repeats(self):
        clock = FakeClock()
        engine = AlertEngine(_s, clock=clock)
        snap = BehaviorSnapshot()
        snap.presence = {"state": "ABSENT", "available": True}
        snap.gaze = {"state": DIRECTION_AWAY, "available": True, "attention": 0}

        # Drive past threshold and keep the condition present.
        clock.advance(_s.alert_away_seconds + 1)
        for _ in range(10):
            clock.advance(1.0)
            clock.advance(0)  # no-op keeps type-consistent
            assert sum(
                1 for a in engine.evaluate(snap) if a["alert_type"] == "STUDENT_AWAY"
            ) <= 1

    def test_reset_between_sessions(self):
        clock = FakeClock()
        engine = AlertEngine(_s, clock=clock)
        snap = BehaviorSnapshot()
        snap.presence = {"state": "ABSENT", "available": True}
        snap.gaze = {"state": DIRECTION_AWAY, "available": True, "attention": 0}

        clock.advance(_s.alert_away_seconds + 1)
        engine.evaluate(snap)  # establishes onset
        clock.advance(_s.alert_away_seconds + 1)
        assert engine.evaluate(snap)  # now fires
        engine.reset_current_session()
        # After reset, build-up must start over (below threshold → no alert).
        for _ in range(20):
            clock.advance(0.1)
            assert engine.evaluate(snap) == []

    def test_normal_drowsiness_no_alert(self):
        engine = AlertEngine(_s)
        snap = BehaviorSnapshot()
        snap.presence = {"state": "PRESENT", "available": True}
        snap.gaze = {"state": DIRECTION_AT_SCREEN, "available": True}
        snap.drowsiness = {
            "state": STATE_NORMAL, "available": True, "ear": 0.34, "confidence": 1.0,
        }
        assert engine.evaluate(snap) == []

    def test_drowsy_state_fires_alert(self):
        clock = FakeClock()
        engine = AlertEngine(_s, clock=clock)
        snap = BehaviorSnapshot()
        snap.presence = {"state": "PRESENT", "available": True}
        snap.gaze = {"state": DIRECTION_AT_SCREEN, "available": True}
        snap.drowsiness = {
            "state": STATE_DROWSY, "available": True, "ear": 0.10, "confidence": 1.0,
        }
        clock.advance(max(_s.alert_drowsiness_seconds + 0.1, 0.1))
        fired = engine.evaluate(snap)
        assert any(a["alert_type"] == ALERT_DROWSINESS for a in fired)

    def test_drowsy_alert_respects_cooldown(self):
        clock = FakeClock()
        engine = AlertEngine(_s, clock=clock)
        snap = BehaviorSnapshot()
        snap.presence = {"state": "PRESENT", "available": True}
        snap.gaze = {"state": DIRECTION_AT_SCREEN, "available": True}
        snap.drowsiness = {
            "state": STATE_DROWSY, "available": True, "ear": 0.10, "confidence": 1.0,
        }
        clock.advance(max(_s.alert_drowsiness_seconds + 0.1, 0.1))
        assert sum(
            1 for a in engine.evaluate(snap) if a["alert_type"] == ALERT_DROWSINESS
        ) == 1
        # The condition stays active but stays inside the cooldown window.
        step = _s.alert_cooldown_seconds / 6.0
        for _ in range(5):
            clock.advance(step)  # 5 × step = 5/6 of the cooldown
            assert sum(
                1 for a in engine.evaluate(snap) if a["alert_type"] == ALERT_DROWSINESS
            ) == 0

    def test_drowsy_alert_stops_after_recovery(self):
        clock = FakeClock()
        engine = AlertEngine(_s, clock=clock)
        drowsy = dict(state=STATE_DROWSY, available=True, ear=0.10, confidence=1.0)
        normal = dict(state=STATE_NORMAL, available=True, ear=0.34, confidence=1.0)
        snap = BehaviorSnapshot()
        snap.presence = {"state": "PRESENT", "available": True}
        snap.gaze = {"state": DIRECTION_AT_SCREEN, "available": True}

        snap.drowsiness = drowsy
        clock.advance(max(_s.alert_drowsiness_seconds + 0.1, 0.1))
        assert any(a["alert_type"] == ALERT_DROWSINESS for a in engine.evaluate(snap))

        snap.drowsiness = normal
        for _ in range(5):
            assert engine.evaluate(snap) == []

    def test_normal_distance_no_alert(self):
        engine = AlertEngine(_s)
        snap = BehaviorSnapshot()
        snap.presence = {"state": "PRESENT", "available": True}
        snap.gaze = {"state": DIRECTION_AT_SCREEN, "available": True}
        snap.distance = {"state": DIST_NORMAL, "available": True, "too_close": False}
        assert engine.evaluate(snap) == []

    def test_too_close_fires_alert_after_persistence(self):
        clock = FakeClock()
        engine = AlertEngine(_s, clock=clock)
        snap = BehaviorSnapshot()
        snap.presence = {"state": "PRESENT", "available": True}
        snap.gaze = {"state": DIRECTION_AT_SCREEN, "available": True}
        snap.distance = {"state": STATE_TOO_CLOSE, "available": True, "too_close": True}

        # First evaluate sets the onset at the current clock time.
        assert engine.evaluate(snap) == []  # onset established, elapsed=0
        # Short duration below the threshold → no alert.
        for _ in range(20):
            clock.advance(0.1)
            assert engine.evaluate(snap) == []

        clock.advance(_s.alert_too_close_seconds + 1)
        fired = engine.evaluate(snap)
        assert any(a["alert_type"] == ALERT_TOO_CLOSE for a in fired)

    def test_too_close_alert_respects_cooldown(self):
        clock = FakeClock()
        engine = AlertEngine(_s, clock=clock)
        snap = BehaviorSnapshot()
        snap.presence = {"state": "PRESENT", "available": True}
        snap.gaze = {"state": DIRECTION_AT_SCREEN, "available": True}
        snap.distance = {"state": STATE_TOO_CLOSE, "available": True, "too_close": True}

        # Establish the onset, then advance past the threshold to fire.
        engine.evaluate(snap)
        clock.advance(_s.alert_too_close_seconds + 1)
        assert sum(
            1 for a in engine.evaluate(snap) if a["alert_type"] == ALERT_TOO_CLOSE
        ) == 1
        step = _s.alert_cooldown_seconds / 6.0
        for _ in range(5):
            clock.advance(step)
            assert sum(
                1 for a in engine.evaluate(snap) if a["alert_type"] == ALERT_TOO_CLOSE
            ) == 0

    def test_too_close_alert_stops_after_recovery(self):
        clock = FakeClock()
        engine = AlertEngine(_s, clock=clock)
        snap = BehaviorSnapshot()
        snap.presence = {"state": "PRESENT", "available": True}
        snap.gaze = {"state": DIRECTION_AT_SCREEN, "available": True}
        snap.distance = {"state": STATE_TOO_CLOSE, "available": True, "too_close": True}

        engine.evaluate(snap)  # establish onset
        clock.advance(_s.alert_too_close_seconds + 1)
        assert any(a["alert_type"] == ALERT_TOO_CLOSE for a in engine.evaluate(snap))

        snap.distance = {"state": DIST_NORMAL, "available": True, "too_close": False}
        for _ in range(5):
            assert engine.evaluate(snap) == []

    def test_normal_posture_no_alert(self):
        engine = AlertEngine(_s)
        snap = BehaviorSnapshot()
        snap.presence = {"state": "PRESENT", "available": True}
        snap.gaze = {"state": DIRECTION_AT_SCREEN, "available": True}
        snap.posture = {"state": STATE_GOOD, "available": True, "posture_score": 80}
        assert engine.evaluate(snap) == []

    def test_poor_posture_fires_alert_after_persistence(self):
        clock = FakeClock()
        engine = AlertEngine(_s, clock=clock)
        snap = BehaviorSnapshot()
        snap.presence = {"state": "PRESENT", "available": True}
        snap.gaze = {"state": DIRECTION_AT_SCREEN, "available": True}
        snap.posture = {"state": STATE_POOR, "available": True, "posture_score": 10}

        # First call establishes onset — no alert yet.
        assert engine.evaluate(snap) == []
        # Short durations stay below threshold.
        for _ in range(20):
            clock.advance(0.1)
            assert engine.evaluate(snap) == []
        # Past the threshold → exactly one alert fires.
        clock.advance(_s.alert_poor_posture_seconds + 1)
        fired = engine.evaluate(snap)
        assert any(a["alert_type"] == ALERT_POSTURE for a in fired)

    def test_poor_posture_alert_respects_cooldown(self):
        clock = FakeClock()
        engine = AlertEngine(_s, clock=clock)
        snap = BehaviorSnapshot()
        snap.presence = {"state": "PRESENT", "available": True}
        snap.gaze = {"state": DIRECTION_AT_SCREEN, "available": True}
        snap.posture = {"state": STATE_POOR, "available": True, "posture_score": 10}

        engine.evaluate(snap)  # establish onset
        clock.advance(_s.alert_poor_posture_seconds + 1)
        assert sum(
            1 for a in engine.evaluate(snap) if a["alert_type"] == ALERT_POSTURE
        ) == 1
        step = _s.alert_cooldown_seconds / 6.0
        for _ in range(5):
            clock.advance(step)
            assert sum(
                1 for a in engine.evaluate(snap) if a["alert_type"] == ALERT_POSTURE
            ) == 0

    def test_poor_posture_alert_stops_after_recovery(self):
        clock = FakeClock()
        engine = AlertEngine(_s, clock=clock)
        snap = BehaviorSnapshot()
        snap.presence = {"state": "PRESENT", "available": True}
        snap.gaze = {"state": DIRECTION_AT_SCREEN, "available": True}
        snap.posture = {"state": STATE_POOR, "available": True, "posture_score": 10}

        engine.evaluate(snap)  # establish onset
        clock.advance(_s.alert_poor_posture_seconds + 1)
        assert any(a["alert_type"] == ALERT_POSTURE for a in engine.evaluate(snap))

        snap.posture = {"state": STATE_GOOD, "available": True, "posture_score": 80}
        for _ in range(5):
            assert engine.evaluate(snap) == []


# ── Phone detection (Phase 7) ──────────────────────────────────────────────

class TestPhoneBehaviorEngine:
    def test_phone_available_fills_snapshot(self):
        engine = BehaviorEngine(_s)
        snap = engine.evaluate(
            _face(True),
            _gaze(DIRECTION_AT_SCREEN, attention=100),
            phone_result=PhoneResult(
                state=PHONE_DETECTED, detected=True,
                confidence=0.88, boxes=[{"x_min": 1, "y_min": 2, "x_max": 3, "y_max": 4,
                                         "label": "cell phone", "score": 0.88}],
            ),
        )
        assert snap.phone["available"] is True
        assert snap.phone["detected"] is True
        assert snap.phone["confidence"] == 0.88
        assert snap.modules["phone"] is True

    def test_phone_unknown_excluded_from_focus_score(self):
        engine = BehaviorEngine(_s)
        no_phone = engine.evaluate(
            _face(True),
            _gaze(DIRECTION_AT_SCREEN, attention=100),
        )
        with_phone_unknown = engine.evaluate(
            _face(True),
            _gaze(DIRECTION_AT_SCREEN, attention=100),
            phone_result=PhoneResult(state=PHONE_UNKNOWN, detected=None),
        )
        # UNKNOWN phone (detected=None) must not change the focus score.
        assert no_phone.focus_score == with_phone_unknown.focus_score

    def test_phone_detected_lowers_focus_score(self):
        engine = BehaviorEngine(_s)
        no_phone = engine.evaluate(
            _face(True),
            _gaze(DIRECTION_AT_SCREEN, attention=100),
        )
        phone_seen = engine.evaluate(
            _face(True),
            _gaze(DIRECTION_AT_SCREEN, attention=100),
            phone_result=PhoneResult(state=PHONE_DETECTED, detected=True, confidence=0.9),
        )
        assert no_phone.focus_score is not None
        assert phone_seen.focus_score is not None
        assert phone_seen.focus_score < no_phone.focus_score


class TestPhoneAlerts:
    def _make_snap(self, detected: bool) -> BehaviorSnapshot:
        snap = BehaviorSnapshot()
        snap.presence = {"state": "PRESENT", "available": True}
        snap.gaze = {"state": DIRECTION_AT_SCREEN, "available": True}
        snap.phone = {
            "state": PHONE_DETECTED if detected else "NO_PHONE",
            "available": True,
            "detected": detected,
            "confidence": 0.9 if detected else 0.0,
        }
        return snap

    def test_no_phone_no_alert(self):
        engine = AlertEngine(_s)
        assert engine.evaluate(self._make_snap(detected=False)) == []

    def test_phone_alert_fires_after_persistence(self):
        clock = FakeClock()
        engine = AlertEngine(_s, clock=clock)
        snap = self._make_snap(detected=True)

        assert engine.evaluate(snap) == []
        for _ in range(20):
            clock.advance(0.1)
            assert engine.evaluate(snap) == []
        clock.advance(_s.alert_phone_seconds + 1)
        fired = engine.evaluate(snap)
        assert any(a["alert_type"] == ALERT_PHONE for a in fired)

    def test_phone_alert_respects_cooldown(self):
        clock = FakeClock()
        engine = AlertEngine(_s, clock=clock)
        snap = self._make_snap(detected=True)

        engine.evaluate(snap)  # establish onset
        clock.advance(_s.alert_phone_seconds + 1)
        assert sum(
            1 for a in engine.evaluate(snap) if a["alert_type"] == ALERT_PHONE
        ) == 1
        step = _s.alert_cooldown_seconds / 6.0
        for _ in range(5):
            clock.advance(step)
            assert sum(
                1 for a in engine.evaluate(snap) if a["alert_type"] == ALERT_PHONE
            ) == 0

    def test_phone_alert_stops_after_recovery(self):
        clock = FakeClock()
        engine = AlertEngine(_s, clock=clock)
        snap = self._make_snap(detected=True)

        engine.evaluate(snap)
        clock.advance(_s.alert_phone_seconds + 1)
        assert any(a["alert_type"] == ALERT_PHONE for a in engine.evaluate(snap))

        snap = self._make_snap(detected=False)
        for _ in range(5):
            assert engine.evaluate(snap) == []