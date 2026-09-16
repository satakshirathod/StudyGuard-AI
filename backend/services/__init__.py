"""Business-logic services for StudyGuard AI.

Each service takes an explicit SQLAlchemy ``Session`` so it is trivially
testable against an in-memory database and reusable from API routers.
"""

from services.student_service import (
    create_student,
    get_student,
    list_students,
    set_active_student,
    get_active_student,
    clear_active_student,
)
from services.session_service import SessionService
from services.alert_service import AlertService
from services.dashboard_service import DashboardService
from services.analytics_service import AnalyticsService

__all__ = [
    "create_student",
    "get_student",
    "list_students",
    "set_active_student",
    "get_active_student",
    "clear_active_student",
    "SessionService",
    "AlertService",
    "DashboardService",
    "AnalyticsService",
]