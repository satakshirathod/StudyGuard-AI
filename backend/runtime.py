"""Application-wide singletons shared by main.py and the API routers.

Kept import-safe: routers import this module freely while main.py populates it
during startup.
"""

from __future__ import annotations

import threading
from typing import Optional

from ai.behavior_engine import AlertEngine, BehaviorEngine, BehaviorSnapshot
from config import get_settings
from services.alert_service import AlertService
from services.analytics_service import AnalyticsService
from services.dashboard_service import DashboardService
from services.session_service import SessionService

settings = get_settings()

# Services (constructed here so routers/tests can import them)
behavior_engine = BehaviorEngine(settings)
session_service = SessionService(settings)
alert_service = AlertService()
dashboard_service = DashboardService(settings.behavior_log_interval_seconds)
analytics_service = AnalyticsService(settings.behavior_log_interval_seconds)
alert_engine = AlertEngine(settings)

# Frontend scheduling guard (feed writer + no other camera pump touches this)
gate = threading.Lock()

# Latest live behavior snapshot (updated by the feed writer while a session runs)
latest_behavior: Optional[BehaviorSnapshot] = None

# Next timestamp at which a behavior row is persisted
next_behavior_log_monotonic: Optional[float] = None


def reset_alert_state_for_new_session() -> None:
    alert_engine.reset_current_session()