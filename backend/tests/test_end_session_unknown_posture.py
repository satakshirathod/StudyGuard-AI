"""Regression: ending a session with UNKNOWN posture logs must not raise KeyError."""

from config import get_settings
from database.db import create_all_from_url
from database.models import BehaviorLog, Student
from services.session_service import SessionService


def test_end_session_with_unknown_posture():
    Session = create_all_from_url("sqlite://")
    db = Session()
    student = Student(name="Test")
    db.add(student)
    db.commit()

    service = SessionService(get_settings())
    session, _ = service.start_session(db, student.id)
    for state in ("GOOD", "UNKNOWN", "POOR", "UNKNOWN"):
        db.add(
            BehaviorLog(
                session_id=session.id,
                presence="PRESENT",
                gaze_direction="CENTER",
                posture_state=state,
                focus_level="high",
                focus_score=80,
            )
        )
    db.commit()

    ended = service.end_session(db, session.id)
    assert ended.status == "ended"
    assert ended.posture_score == 50.0  # mean(GOOD=100, POOR=0); UNKNOWN ignored
    assert ended.duration_seconds is not None
