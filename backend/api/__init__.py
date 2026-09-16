"""API routers for StudyGuard AI (students, sessions, dashboard, analytics, alerts)."""

from api.alerts import router as alerts_router
from api.analytics import router as analytics_router
from api.dashboard import router as dashboard_router
from api.sessions import router as sessions_router
from api.students import router as students_router

__all__ = [
    "students_router",
    "sessions_router",
    "dashboard_router",
    "analytics_router",
    "alerts_router",
]