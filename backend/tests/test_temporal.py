"""Tests for the temporal state smoother (Phase 3)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from ai.temporal import TemporalStateTracker  # noqa: E402


def test_windows_must_be_positive():
    with pytest.raises(ValueError):
        TemporalStateTracker(window=0)
    with pytest.raises(ValueError):
        TemporalStateTracker(change_threshold=0.0)
    with pytest.raises(ValueError):
        TemporalStateTracker(change_threshold=1.5)
    with pytest.raises(ValueError):
        TemporalStateTracker(away_duration=0)


def test_smoothes_single_small_window():
    t = TemporalStateTracker(window=1, change_threshold=1.0)
    assert t.update("A") == "A"
    assert t.update("B") == "B"


def test_needs_majority_to_switch():
    t = TemporalStateTracker(window=5, change_threshold=0.6)
    # first stable state after majority build-up
    for _ in range(3):
        t.update("A")
    assert t.stable_state == "A"
    # a single "B" must not flip the stable state
    t.update("B")
    assert t.stable_state == "A"
    # majority of B then flips
    for _ in range(3):
        t.update("B")
    assert t.stable_state == "B"


def test_flicker_suppression():
    t = TemporalStateTracker(window=5, change_threshold=0.6)
    for _ in range(5):
        t.update("A")
    seq = ["A", "B", "A", "A", "A", "B", "A", "A"]
    for s in seq:
        t.update(s)
    assert t.stable_state == "A"


def test_unknown_frame_is_unknown():
    t = TemporalStateTracker(window=3, change_threshold=0.6)
    assert t.update(None) == "UNKNOWN"


def test_away_after_missing_face():
    t = TemporalStateTracker(window=3, away_duration=3, away_label="AWAY")
    t.update("A")
    t.update(None)
    assert t.stable_state == "UNKNOWN"
    t.update(None)
    assert t.stable_state == "UNKNOWN"  # streak just got started
    t.update(None)
    assert t.stable_state == "AWAY"


def test_returns_to_valid_after_away():
    t = TemporalStateTracker(window=3, change_threshold=0.6, away_duration=2)
    for _ in range(3):
        t.update(None)
    assert t.stable_state == "AWAY"
    for _ in range(4):
        t.update("A")
    assert t.stable_state == "A"


def test_reset_clears_state():
    t = TemporalStateTracker(window=3, away_duration=2)
    for _ in range(3):
        t.update(None)
    assert t.stable_state == "AWAY"
    t.reset()
    assert t.stable_state == "UNKNOWN"


def test_unknown_label_passthrough():
    t = TemporalStateTracker(window=3, change_threshold=0.6, unknown_label="NOTHING")
    assert t.update("NOTHING") == "NOTHING"