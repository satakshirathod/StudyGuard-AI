"""Reusable temporal state smoothing.

Wraps a simple yet tunable debounce/majority-vote scheme used by the gaze,
drowsiness and (later) posture modules so the per-frame signals do not
flicker between states on every frame.

The tracker keeps the ``window`` most recent raw states. The *stable* state
only changes when a different state holds at least ``change_threshold`` of
the window (a hysteresis/majority vote). While the raw signal is *unknown*
for ``away_duration`` consecutive frames, the AWAY state is raised so
downstream logic can react to a learner who has left the frame.

The class is deliberately statistic-free and pure-python so it is trivial to
unit-test without heavy dependencies.
"""

from __future__ import annotations

from collections import deque

from utils.logger import get_logger

_log = get_logger("studygard.ai.temporal")


class TemporalStateTracker:
    """Smooth a noisy per-frame state label into a stable one.

    Args:
        window: Size of the rolling window of raw observations.
        change_threshold: Fraction (0..1] of the window that must agree on a
            new state before the stable state actually changes.
        away_duration: Consecutive ``unknown`` raw frames before ``AWAY`` is
            returned as the stable state.
        away_label: Label used for the "away" state.
        unknown_label: Label used for "no reliable reading this frame".
    """

    def __init__(
        self,
        window: int = 5,
        change_threshold: float = 0.6,
        away_duration: int = 10,
        away_label: str = "AWAY",
        unknown_label: str = "UNKNOWN",
    ) -> None:
        if window < 1:
            raise ValueError("window must be >= 1")
        if not 0.0 < change_threshold <= 1.0:
            raise ValueError("change_threshold must be in (0, 1]")
        if away_duration < 1:
            raise ValueError("away_duration must be >= 1")
        self.window = int(window)
        self.change_threshold = float(change_threshold)
        self.away_duration = int(away_duration)
        self.away_label = away_label
        self.unknown_label = unknown_label
        self._history: deque[str] = deque(maxlen=self.window)
        self._stable_state: str = unknown_label
        self._unknown_streak = 0

    def reset(self) -> None:
        """Clear all history and return to the unknown state."""
        self._history.clear()
        self._stable_state = self.unknown_label
        self._unknown_streak = 0

    def _majority(self) -> str:
        """Return the most frequent label in the window (ties → first seen)."""
        counts: dict[str, int] = {}
        for label in self._history:
            counts[label] = counts.get(label, 0) + 1
        best = self.unknown_label
        best_count = -1
        for label in self._history:  # preserve insertion order → stable ties
            if counts[label] > best_count:
                best, best_count = label, counts[label]
        return best

    def update(self, raw_state: str | None) -> str:
        """Feed one raw observation and return the current stable state.

        Args:
            raw_state: Per-frame label, or ``None``/falsy for "couldn't tell".

        Returns:
            Stable state after applying the debounce rules.
        """
        if not raw_state or raw_state == self.unknown_label:
            self._unknown_streak += 1
            raw = self.unknown_label
        else:
            self._unknown_streak = 0
            raw = raw_state

        self._history.append(raw)

        # Away wins as soon as the face has been missing long enough.
        if (
            self._unknown_streak >= self.away_duration
            and not raw == self.away_label
        ):
            self._stable_state = self.away_label
            return self._stable_state

        candidate = self._majority()
        need = max(1, int(round(self.window * self.change_threshold)))
        if _vote(candidate, self._history) >= need:
            self._stable_state = candidate
        return self._stable_state

    @property
    def stable_state(self) -> str:
        """Current stable state without feeding a new observation."""
        return self._stable_state

    @property
    def is_away(self) -> bool:
        return self._stable_state == self.away_label


def _vote(label: str, history: deque[str]) -> int:
    """Count occurrences of ``label`` in ``history``."""
    return sum(1 for item in history if item == label)