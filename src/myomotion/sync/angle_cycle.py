"""One joint angle closing and opening again = one rep: the state machine shared by the curl and the RDL trackers.

    REST (open) --closes below open_exit--> closing phase --below closed_enter--> ISOMETRIC_HOLD
        --opens above closed_exit--> opening phase --above open_enter--> REST   (rep counted)

The phase NAMES depend on the exercise, the maths does not:
    biceps curl  (elbow angle): closing = CONCENTRIC (curl up),  opening = ECCENTRIC (lower)
    RDL          (hip angle):   closing = ECCENTRIC (lower),     opening = CONCENTRIC (stand up)

Separate enter / exit angles (hysteresis) and a short dwell stop frame noise from flickering the phase. The live
phase is causal (it lags a little); in each rep's record the start is moved back to the last moment the joint was at
its open turning point, and the start of the opening to the last moment at its closed turning point, so the phases
in merged.csv match the movement. A rep may only start after the joint has been open once (no counting from a bent
start), and one that stays in a phase longer than stall_s is dropped (e.g. a joint simply held bent).

Data structures (this file): a dict key -> start time for the dwell timers; (angle, time) tuples for the two turning
    points; a dict per rep record (with a list of phase dicts); lists of reps and phase changes.
Time complexity (this file): O(1) per angle sample (a constant number of comparisons and dict updates); O(R + T)
    memory for R reps and T phase changes.
"""

from __future__ import annotations   # > type hints are stored as text, not run: allows forward references

from dataclasses import dataclass
from typing import Any

REST, HOLD = "REST", "ISOMETRIC_HOLD"
STARTED, FINISHED, DROPPED = "started", "finished", "dropped"


# ---------------------------------------------------------------------------
# CycleConfig
# What:  the angles and times of one exercise's cycle (each tracker builds its own).
# Data structures: frozen dataclass.
# Time:  O(1) to build (fixed fields).
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CycleConfig:
    """Thresholds of one angle cycle (degrees, seconds)."""

    open_enter_deg: float              # back at the open position (REST) above this
    open_exit_deg: float               # a rep starts when the angle closes below this
    closed_enter_deg: float            # the closed turning zone (HOLD) starts below this
    closed_exit_deg: float             # the opening starts when the angle rises above this
    dwell_s: float = 0.08              # a zone change must last this long (frame noise)
    turn_tolerance_deg: float = 3.0    # "still at the turning point" = within this of the extreme
    stall_s: float = 6.0               # stuck in one phase of a rep this long -> not a rep


# ---------------------------------------------------------------------------
# AngleCycle
# What:  the state machine (see the file docstring). step() feeds one angle and says
#        whether a rep started, finished or was dropped; the trackers add their own
#        fields (cues, maxima) to `current` while a rep runs.
# Time:  O(1) per step.  Space: O(R) rep records + O(T) phase changes.
# Data structures: dict key -> start time for the dwell timers (O(1)); (angle, t) tuples
#        for the turning points; lists of dicts for reps and transitions.
# Files used: none.
# ---------------------------------------------------------------------------
class AngleCycle:
    """Rep counting on one joint angle that closes and opens again."""

    # Time:  O(1).  Space: O(1).
    # Data structures: empty lists (reps, transitions), a dict of dwell timers, (angle, t) tuples.
    def __init__(self, cfg: CycleConfig, closing_phase: str, opening_phase: str) -> None:
        self.cfg = cfg
        self.closing_phase, self.opening_phase = closing_phase, opening_phase
        self.phase = REST
        self.rep_count = 0
        self.reps: list[dict[str, Any]] = []
        self.transitions: list[dict[str, Any]] = []
        self.current: dict[str, Any] | None = None
        self.armed = False
        self._since: dict[str, float | None] = {}
        self._open_turn: tuple[float, float] | None = None      # (largest angle, last time near it)
        self._closed_turn: tuple[float, float] | None = None    # (smallest angle, last time near it)
        self._phase_since = 0.0

    # -----------------------------------------------------------------------
    # step(t, angle)
    # What:  advance the machine by one smoothed angle at time t.
    # Time / Space: O(1).
    # Data structures: a dict key -> start time (dwell timers, O(1) lookups); (angle, t) tuples for the turning
    #                  points; a dict for the running rep record.
    # -----------------------------------------------------------------------
    def step(self, t: float, angle: float) -> str | None:
        """One angle; returns STARTED, FINISHED, DROPPED or None."""
        cfg = self.cfg
        event = None
        if self.phase != REST and t - self._phase_since > cfg.stall_s:
            self.drop(t)
            event = DROPPED
        if self.phase == REST:
            if angle > cfg.open_enter_deg:
                self.armed = True
            self._open_turn = turning_point(self._open_turn, angle, t, cfg.turn_tolerance_deg, larger=True)
            started = self._sustained("close", self.armed and angle < cfg.open_exit_deg, t)
            if started is not None:
                start_t, open_angle = (self._open_turn[1], self._open_turn[0]) if self._open_turn else (started, angle)
                self.current = {"t_start_s": start_t, "open_angle_deg": open_angle, "turn_angle_deg": angle}
                self._enter(self.closing_phase, t)
                event = STARTED
        elif self.phase == self.closing_phase:
            reached = self._sustained("turn", angle < cfg.closed_enter_deg, t)
            if reached is not None:
                self.current["t_turn_s"] = reached
                self._closed_turn = (angle, t)
                self._enter(HOLD, t)
            elif self._sustained("back", angle > cfg.open_enter_deg, t) is not None:
                self.drop(t)                                     # > went back without reaching the turn
                event = DROPPED
        elif self.phase == HOLD:
            self._closed_turn = turning_point(self._closed_turn, angle, t, cfg.turn_tolerance_deg, larger=False)
            if self._sustained("open", angle > cfg.closed_exit_deg, t) is not None:
                self.current["t_return_s"] = self._closed_turn[1]
                self._enter(self.opening_phase, t)
        elif self.phase == self.opening_phase:
            ended = self._sustained("end", angle > cfg.open_enter_deg, t)
            if ended is not None:
                self._finish(ended)
                event = FINISHED
            elif self._sustained("turn", angle < cfg.closed_enter_deg, t) is not None:
                self._closed_turn = (angle, t)                   # > closed again before reaching the open end
                self._enter(HOLD, t)
        if self.current is not None:
            self.current["turn_angle_deg"] = min(self.current["turn_angle_deg"], angle)
        return event

    # Time:  O(1).
    # Data structures: resets the rep dict and the turning-point tuples.
    def drop(self, t: float) -> None:
        """Forget an unfinished rep (stalled, went back, or the joint was lost); re-arm on the next open position."""
        self.current, self._open_turn, self._closed_turn = None, None, None
        self.armed = False
        if self.phase != REST:
            self._enter(REST, t)

    # Time:  O(1).  Space: O(1).
    # Data structures: a dict key -> start time (O(1) lookup and update).
    def _sustained(self, key: str, condition: bool, t: float) -> float | None:
        """Start time of a condition that has lasted dwell_s, else None (the start time adds no dwell lag)."""
        if not condition:
            self._since[key] = None
            return None
        start = self._since.get(key)
        if start is None:
            self._since[key] = start = t
        return start if t - start >= self.cfg.dwell_s else None

    # Time:  O(1) amortised (one list append; clears a dict of at most 4 timers).
    # Data structures: the list of transition dicts; the dwell-timer dict.
    def _enter(self, phase: str, t: float) -> None:
        """Change phase and log it."""
        self.transitions.append({"t_s": t, "from": self.phase, "to": phase})
        self.phase = phase
        self._phase_since = t
        self._since.clear()

    # -----------------------------------------------------------------------
    # _finish(t_end)
    # What:  close the rep: its three phases with start and end times ("phases", read by
    #        src/myomotion/sync/align.py intervals_from_cv in any exercise's order).
    # Time / Space: O(1).
    # Data structures: the rep dict gains a list of three phase dicts and is appended to the reps list.
    # -----------------------------------------------------------------------
    def _finish(self, t_end: float) -> None:
        """Store the finished rep."""
        rep = self.current
        self.rep_count += 1
        rep.update({"rep_index": self.rep_count, "t_end_s": t_end, "phases": [
            {"phase": self.closing_phase, "start_s": rep["t_start_s"], "end_s": rep["t_turn_s"]},
            {"phase": HOLD, "start_s": rep["t_turn_s"], "end_s": rep["t_return_s"]},
            {"phase": self.opening_phase, "start_s": rep["t_return_s"], "end_s": t_end}]})
        self.reps.append(rep)
        self.current, self._open_turn, self._closed_turn = None, None, None
        self._enter(REST, t_end)


# ---------------------------------------------------------------------------
# turning_point(best, angle, t, tolerance, larger)
# What:  (extreme angle, LAST time the angle was within `tolerance` of it). So after a long
#        rest the rep starts when the joint last was open, not at the most open moment
#        seconds earlier (the same idea as Ahmad's lift-off time).
# Time / Space: O(1).  Data structures: tuple (angle, t).
# ---------------------------------------------------------------------------
def turning_point(best: tuple[float, float] | None, angle: float, t: float, tolerance: float,
                  larger: bool) -> tuple[float, float]:
    """Updated (extreme angle, last time near it)."""
    if best is None or (angle > best[0] if larger else angle < best[0]):
        return angle, t
    if abs(angle - best[0]) <= tolerance:
        return best[0], t
    return best
