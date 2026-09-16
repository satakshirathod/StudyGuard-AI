"""Session service + dashboard + analytics tests (application foundation)."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from ai.behavior_engine import BehaviorSnapshot
from config import get_settings
from database.db import create_all_from_url
from database.models import BehaviorLog, Student
from services.analytics_service import AnalyticsService
from services.dashboard_service import DashboardService
from services.session_service import SessionError, SessionService
from services.student_service import (
    StudentError,
    create_student,
    get_active_student,
    set_active_student,
)

_s = get_settings()


@pytest.fixture()
def db():
    """In-memory database session (fresh per test)."""
    session_factory = create_all_from_url("sqlite:///:memory:")
    session = session_factory()
    yield session
    session.close()


def _snapshot(
    present=True,
    gaze="AT_SCREEN",
    attention=95,
    focus_level="high",
    focus_score=95,
) -> BehaviorSnapshot:
    snap = BehaviorSnapshot()
    snap.presence = {"state": "PRESENT" if present else "ABSENT", "available": True}
    snap.gaze = {"state": gaze, "available": True, "attention": attention, "confidence": 0.9}
    snap.drowsiness = {"state": None, "available": False}
    snap.posture = {"state": None, "available": False}
    snap.distance = {"distance_cm": None, "available": False}
    snap.phone = {"detected": None, "available": False}
    snap.distraction = {"state": "FOCUSED" if present else "DISTRACTED", "available": True}
    snap.focus_score = focus_score
    snap.focus_level = focus_level
    return snap


class TestStudentService:
    def test_create_and_validate(self, db):
        student = create_student(db, "Ada", "ada@example.com")
        assert student.id is not None
        assert get_active_student(db) is None  # not automatically active

    def test_validation(self, db):
        with pytest.raises(StudentError):
            create_student(db, "")
        with pytest.raises(StudentError):
            create_student(db, "Ada", "not-an-email")

    def test_email_unique(self, db):
        create_student(db, "Ada", "ada@example.com")
        with pytest.raises(StudentError):
            create_student(db, "Another", "ada@example.com")

    def test_set_active_exclusivity(self, db):
        a = create_student(db, "A", "a@example.com")
        b = create_student(db, "B", "b@example.com")
        set_active_student(db, a.id)
        set_active_student(db, b.id)
        assert get_active_student(db).id == b.id


class TestSessionLifecycle:
    def test_start_and_active(self, db):
        student = create_student(db, "Ada")
        svc = SessionService(_s)
        session, created = svc.start_session(db, student.id)
        assert created is True
        assert svc.get_active_session(db).id == session.id
        assert svc.get_session(db, session.id).status == "active"

    def test_only_one_live_session(self, db):
        student = create_student(db, "Ada")
        svc = SessionService(_s)
        first, _ = svc.start_session(db, student.id)
        again, created = svc.start_session(db, student.id)
        assert created is False
        assert again.id == first.id

    def test_conflict_across_students(self, db):
        a = create_student(db, "A")
        b = create_student(db, "B")
        svc = SessionService(_s)
        svc.start_session(db, a.id)
        with pytest.raises(SessionError):
            svc.start_session(db, b.id)

    def test_end_computes_aggregates(self, db):
        student = create_student(db, "Ada")
        svc = SessionService(_s)
        session, _ = svc.start_session(db, student.id)
        session.started_at = datetime.utcnow() - timedelta(minutes=5)
        svc.record_behavior_log(db, session.id, _snapshot(focus_level="high", focus_score=96))
        svc.record_behavior_log(db, session.id, _snapshot(focus_level="medium", focus_score=60))
        svc.record_behavior_log(db, session.id, _snapshot(focus_level="low", focus_score=20))
        ended = svc.end_session(db, session.id)
        assert ended.status == "ended"
        assert ended.avg_focus_score == pytest.approx((96 + 60 + 20) / 3, abs=0.1)
        assert ended.focused_seconds == int(1 * _s.behavior_log_interval_seconds)
        assert ended.distracted_seconds == int(1 * _s.behavior_log_interval_seconds)
        assert ended.drowsy_seconds is None  # module unavailable → honest None
        assert ended.phone_seconds is None
        assert ended.avg_distance_cm is None
        assert ended.duration_seconds == 300

    def test_end_without_logs(self, db):
        student = create_student(db, "Ada")
        svc = SessionService(_s)
        session, _ = svc.start_session(db, student.id)
        ended = svc.end_session(db, session.id)
        assert ended.avg_focus_score is None
        assert ended.focused_seconds == 0

    def test_alert_recording_bumps_count(self, db):
        student = create_student(db, "Ada")
        svc = SessionService(_s)
        session, _ = svc.start_session(db, student.id)
        svc.add_alerts(
            db, session.id,
            [{"alert_type": "STUDENT_AWAY", "severity": "warning", "message": "away"}],
        )
        ended = svc.end_session(db, session.id)
        assert ended.alert_count == 1

    def test_cannot_end_twice(self, db):
        student = create_student(db, "Ada")
        svc = SessionService(_s)
        session, _ = svc.start_session(db, student.id)
        svc.end_session(db, session.id)
        with pytest.raises(SessionError):
            svc.end_session(db, session.id)


class TestDashboard:
    def test_empty_overview(self, db):
        student = create_student(db, "Ada")
        d = DashboardService(_s.behavior_log_interval_seconds)
        overview = d.today_overview(db, student.id)
        assert overview["session_count"] == 0
        assert overview["study_seconds"] == 0
        assert overview["avg_focus_score"] is None
        assert overview["alert_count"] == 0

    def test_overview_after_session(self, db):
        student = create_student(db, "Ada")
        svc = SessionService(_s)
        session, _ = svc.start_session(db, student.id)
        session.started_at = datetime.utcnow() - timedelta(minutes=5)
        svc.record_behavior_log(db, session.id, _snapshot(focus_level="high", focus_score=90))
        svc.end_session(db, session.id)
        d = DashboardService(_s.behavior_log_interval_seconds)
        overview = d.today_overview(db, student.id)
        assert overview["session_count"] == 1
        assert overview["avg_focus_score"] == pytest.approx(90, abs=0.1)
        assert overview["study_seconds"] == 300

    def test_trend_shape(self, db):
        student = create_student(db, "Ada")
        d = DashboardService(_s.behavior_log_interval_seconds)
        trend = d.trend(db, student.id, days=7)
        assert len(trend) == 7
        assert trend[-1]["date"] == __import__("datetime").date.today().isoformat()

    def test_recent_sessions(self, db):
        student = create_student(db, "Ada")
        svc = SessionService(_s)
        s1, _ = svc.start_session(db, student.id)
        svc.end_session(db, s1.id)
        s2, _ = svc.start_session(db, student.id)
        svc.end_session(db, s2.id)
        recent = DashboardService().recent_sessions(db, student.id, limit=2)
        assert [r["id"] for r in recent] == [s2.id, s1.id]


class TestAnalytics:
    def test_session_stats(self, db):
        student = create_student(db, "Ada")
        svc = SessionService(_s)
        session, _ = svc.start_session(db, student.id)
        svc.record_behavior_log(db, session.id, _snapshot(gaze="LEFT", attention=70, focus_level="medium", focus_score=70))
        svc.end_session(db, session.id)
        stats = AnalyticsService(_s.behavior_log_interval_seconds).session_stats(db, session.id)
        assert stats["student_name"] == "Ada"
        assert stats["gaze_distribution"].get("LEFT") == 100.0
        assert stats["modules"]["drowsiness"] is False  # honest

    def test_missing_session(self, db):
        assert AnalyticsService().session_stats(db, 9999) is None