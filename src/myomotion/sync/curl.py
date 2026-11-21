"""Biceps curl from MediaPipe landmarks: elbow angle, rep phases and two form cues (the FYP-1 synced demo).

Why the curl for the demo: the sEMG electrodes go on the biceps (easy to reach with a T-shirt sleeve), the body
hardly moves (seated or standing, side-on to the camera), and the elbow angle comes straight from three landmarks
MediaPipe tracks well (shoulder, elbow, wrist). Ahmad's biomechanics package does the deadlift; this file only
USES two of his helpers (biomechanics/geometry.py joint_angle, biomechanics/filters.py OneEuroFilter) and lives in
the sync package so his files stay untouched (it can move into biomechanics/ if he agrees).

One rep, concentric first (like the deadlift, and the recorder's UP / HOLD / DOWN cues):

    REST (arm down) --elbow closes below 120--> CONCENTRIC --below 80--> ISOMETRIC_HOLD
        --opens above 90--> ECCENTRIC --above 130--> REST  (rep counted)

So a rep needs at least ~50 degrees of real movement. Many people do not straighten the arm fully between curls
(first real test, 21 Nov, TESTCV/S06: bottoms of 123-155 deg with the biceps still working hard); those reps are
counted, with the cue "Lower your arm all the way" when the arm stayed bent below 150 deg.

The rep state machine is shared with the RDL (src/myomotion/sync/angle_cycle.py: hysteresis, dwell, turning points,
stall). Form cues: the upper arm swinging forward ("keep your elbow at your side") and the trunk swinging
("don't swing your body"), both against the person's own rest posture.

Data structures (this file): a dict joint -> 2-D NumPy point; the AngleCycle state machine; Ahmad's OneEuroFilter;
    CurlFrame dataclasses; dict rep records; a dict column -> text per CSV row.
Time complexity (this file): O(1) per frame for the arm choice, the elbow angle, the cues and the state machine;
    O(P) per frame to draw the overlay panel (P pixels).
"""

from __future__ import annotations   # > type hints are stored as text, not run: allows forward references

import math
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from myomotion.biomechanics.filters import OneEuroFilter
from myomotion.biomechanics.geometry import joint_angle
from myomotion.sync.angle_cycle import FINISHED, STARTED, AngleCycle, CycleConfig

CURL_EXERCISES = frozenset({"curl", "biceps", "bicep_curl", "biceps_curl"})
SCHEMA = "myomotion.curl_session/1"
REST, CONCENTRIC, HOLD, ECCENTRIC = "REST", "CONCENTRIC", "ISOMETRIC_HOLD", "ECCENTRIC"
PHASE_LABELS = {REST: "Rest", CONCENTRIC: "Curl up", HOLD: "Hold", ECCENTRIC: "Lower"}
# > MediaPipe landmark numbers of the arm and hip on each side
ARM_INDICES = {"left": {"shoulder": 11, "elbow": 13, "wrist": 15, "hip": 23},
               "right": {"shoulder": 12, "elbow": 14, "wrist": 16, "hip": 24}}
CUE_ELBOW = "Keep your elbow at your side"
CUE_SWING = "Don't swing your body"
CUE_RANGE = "Lower your arm all the way"
DISCLAIMERS = ["Angle thresholds are provisional (not validated on labelled curls).", "Not a medical device."]


# ---------------------------------------------------------------------------
# CurlConfig  (the CONFIG for this file)
# What:  every threshold of the curl tracker.
# Data structures: frozen dataclass.
# Time:  O(1) to build (fixed fields).
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CurlConfig:
    """Settings of the curl tracker."""

    min_visibility: float = 0.5        # shoulder, elbow and wrist must be at least this visible
    side_switch_margin: float = 0.2    # the other arm must be this much more visible to take over
    filter_min_cutoff_hz: float = 1.5  # One-Euro smoothing of the elbow angle (jitter vs lag)
    filter_beta: float = 0.02
    bottom_enter_deg: float = 130.0    # arm counted as down (REST) above this (S06: 145 missed half the reps)
    bottom_exit_deg: float = 120.0     # a curl starts when the elbow closes below this
    full_extension_deg: float = 150.0  # straightest angle before a curl below this -> "lower your arm all the way"
    top_enter_deg: float = 80.0        # the top (HOLD) starts below this
    top_exit_deg: float = 90.0         # lowering starts when the elbow opens above this
    dwell_s: float = 0.08              # a zone change must last this long (frame noise)
    turn_tolerance_deg: float = 3.0    # "still at the turning point" = within this of the straightest / most curled
    lost_after_s: float = 1.0          # arm not seen this long -> the unfinished rep is dropped
    stall_s: float = 6.0               # stuck in one phase of a rep this long -> not a rep (arm just held bent)
    elbow_drift_deg: float = 30.0      # upper arm this far forward of the trunk -> cue
    swing_deg: float = 10.0            # trunk lean this far from its rest value -> cue
    baseline_weight: float = 0.05      # how fast the rest trunk lean follows the person (per frame)


CURL_CONFIG = CurlConfig()


# ---------------------------------------------------------------------------
# CurlFrame
# What:  what the tracker says about one frame (for the overlay and cv_frames.csv).
# Data structures: frozen dataclass; arm points as a dict name -> (x, y) pixels.
# Time:  O(1) to build (references to the landmark array and a 4-entry dict).
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CurlFrame:
    """One frame of the curl tracker."""

    t_s: float
    tracked: bool                       # shoulder, elbow and wrist seen
    side: str | None
    phase: str                          # REST / CONCENTRIC / ISOMETRIC_HOLD / ECCENTRIC (our names)
    rep_count: int
    elbow_deg: float | None = None      # 180 = straight arm, ~40 = fully curled
    upper_arm_deg: float | None = None  # angle between the trunk and the upper arm (elbow drifting forward)
    trunk_swing_deg: float | None = None    # trunk lean minus the rest lean
    cues: tuple[str, ...] = ()
    landmarks: np.ndarray | None = field(default=None, repr=False)
    points: dict[str, tuple[float, float]] = field(default_factory=dict, repr=False)


# ---------------------------------------------------------------------------
# CurlAnalyzer
# What:  the tracker: pick the arm the camera sees best, smooth the elbow angle, feed it
#        to the shared rep state machine (src/myomotion/sync/angle_cycle.py; closing =
#        CONCENTRIC, opening = ECCENTRIC), and collect the form cues of each rep.
# Time:  O(1) per frame.  Space: O(R) rep records + O(T) phase changes.
# Data structures: dicts for the arm points (O(1) lookups), the AngleCycle, OneEuroFilter state.
# Files used: src/myomotion/sync/angle_cycle.py; src/myomotion/biomechanics/geometry.py,
#        filters.py (Ahmad's, unchanged).
# ---------------------------------------------------------------------------
class CurlAnalyzer:
    """Biceps curl reps and phases from MediaPipe landmarks."""

    # Time:  O(1).  Space: O(1).
    # Data structures: an AngleCycle, a OneEuroFilter, floats.
    def __init__(self, cfg: CurlConfig = CURL_CONFIG, side: str | None = None) -> None:
        self.cfg = cfg
        self.side = side
        self._side_fixed = side is not None
        self._filter = OneEuroFilter(cfg.filter_min_cutoff_hz, cfg.filter_beta)
        self.cycle = AngleCycle(CycleConfig(open_enter_deg=cfg.bottom_enter_deg, open_exit_deg=cfg.bottom_exit_deg,
                                            closed_enter_deg=cfg.top_enter_deg, closed_exit_deg=cfg.top_exit_deg,
                                            dwell_s=cfg.dwell_s, turn_tolerance_deg=cfg.turn_tolerance_deg,
                                            stall_s=cfg.stall_s),
                                closing_phase=CONCENTRIC, opening_phase=ECCENTRIC)
        self._rest_lean: float | None = None
        self._last_seen: float | None = None
        self.frames_total = 0
        self.frames_tracked = 0
        self.t_first: float | None = None
        self.t_last: float | None = None

    # Time:  O(1).
    # Data structures: a str.
    @property
    def phase(self) -> str:
        """Current phase (our names)."""
        return self.cycle.phase

    # Time:  O(1).
    # Data structures: an int.
    @property
    def rep_count(self) -> int:
        """Reps finished so far."""
        return self.cycle.rep_count

    # Time:  O(1) (returned by reference, not copied).
    # Data structures: the list of rep dicts.
    @property
    def reps(self) -> list[dict[str, Any]]:
        """One record per finished rep."""
        return self.cycle.reps

    # Time:  O(1) (returned by reference).
    # Data structures: the list of transition dicts.
    @property
    def transitions(self) -> list[dict[str, Any]]:
        """Every phase change {t_s, from, to}."""
        return self.cycle.transitions

    # -----------------------------------------------------------------------
    # process(t_s, lm)
    # What:  one frame: lm = MediaPipe image landmarks [33, 3] (x_px, y_px, visibility) or None.
    # Time / Space: O(1).
    # Data structures: a dict joint -> 2-D NumPy point for the arm (O(1) lookups); the running rep dict; a CurlFrame
    #                  dataclass out.
    # -----------------------------------------------------------------------
    def process(self, t_s: float, lm: np.ndarray | None) -> CurlFrame:
        """Analyse one frame."""
        self.frames_total += 1
        self.t_first = t_s if self.t_first is None else self.t_first
        self.t_last = t_s
        arm = self._arm(lm)
        if arm is None:
            if self._last_seen is not None and t_s - self._last_seen > self.cfg.lost_after_s and self.phase != REST:
                self.cycle.drop(t_s)                            # > lost the arm mid-rep: not a rep
            return CurlFrame(t_s, False, self.side, self.phase, self.rep_count, landmarks=lm)
        self.frames_tracked += 1
        self._last_seen = t_s
        raw = joint_angle(arm["shoulder"], arm["elbow"], arm["wrist"])
        elbow = self._filter(t_s, raw) if math.isfinite(raw) else raw
        upper_arm = joint_angle(arm["hip"], arm["shoulder"], arm["elbow"]) if "hip" in arm else None
        lean = _lean_from_vertical(arm["hip"], arm["shoulder"]) if "hip" in arm else None
        if math.isfinite(elbow):
            event = self.cycle.step(t_s, elbow)
            if event == STARTED:                                # > the curl's own fields on the new rep record
                straightest = self.cycle.current["open_angle_deg"]
                self.cycle.current.update({"max_upper_arm_deg": 0.0, "max_trunk_swing_deg": 0.0,
                                           "cues": [CUE_RANGE] if straightest < self.cfg.full_extension_deg else []})
            elif event == FINISHED:
                _add_curl_keys(self.cycle.reps[-1])
        if self.phase == REST and lean is not None:             # > the person's own upright posture
            w = self.cfg.baseline_weight
            self._rest_lean = lean if self._rest_lean is None else (1 - w) * self._rest_lean + w * lean
        swing = abs(lean - self._rest_lean) if lean is not None and self._rest_lean is not None else None
        cues = self._cues(upper_arm, swing)
        return CurlFrame(t_s, True, self.side, self.phase, self.rep_count, elbow_deg=elbow, upper_arm_deg=upper_arm,
                         trunk_swing_deg=swing, cues=cues, landmarks=lm,
                         points={name: (float(p[0]), float(p[1])) for name, p in arm.items()})

    # -----------------------------------------------------------------------
    # _arm(lm)
    # What:  the shoulder, elbow, wrist (and hip if seen) of the arm the camera sees best.
    #        The other arm takes over only when it is clearly more visible (no flicker).
    # Time / Space: O(1).
    # Data structures: a dict side -> visibility score; a dict joint -> NumPy point.
    # -----------------------------------------------------------------------
    def _arm(self, lm: np.ndarray | None) -> dict[str, np.ndarray] | None:
        """Arm points in pixels, or None if the arm is not visible enough."""
        if lm is None:
            return None
        score = {side: float(np.mean([lm[idx[j], 2] for j in ("shoulder", "elbow", "wrist")]))
                 for side, idx in ARM_INDICES.items()}
        best = max(score, key=score.get)
        if self.side is None or (not self._side_fixed and score[best] > score[self.side] + self.cfg.side_switch_margin):
            self.side = best
        idx = ARM_INDICES[self.side]
        if any(lm[idx[j], 2] < self.cfg.min_visibility for j in ("shoulder", "elbow", "wrist")):
            return None
        arm = {j: lm[idx[j], :2].astype(float) for j in ("shoulder", "elbow", "wrist")}
        if lm[idx["hip"], 2] >= self.cfg.min_visibility:
            arm["hip"] = lm[idx["hip"], :2].astype(float)
        return arm

    # -----------------------------------------------------------------------
    # _cues(upper_arm, swing)
    # What:  the form cues active now (only during a rep), kept in the rep's record too.
    # Time / Space: O(1).
    # Data structures: a list of cue strings (returned as a tuple); the running rep dict.
    # -----------------------------------------------------------------------
    def _cues(self, upper_arm: float | None, swing: float | None) -> tuple[str, ...]:
        """Active form cues."""
        rep = self.cycle.current
        if rep is None:
            return ()
        cues = [CUE_RANGE] if self.phase == CONCENTRIC and CUE_RANGE in rep["cues"] else []
        if upper_arm is not None:
            rep["max_upper_arm_deg"] = max(rep["max_upper_arm_deg"], upper_arm)
            if upper_arm > self.cfg.elbow_drift_deg:
                cues.append(CUE_ELBOW)
        if swing is not None:
            rep["max_trunk_swing_deg"] = max(rep["max_trunk_swing_deg"], swing)
            if swing > self.cfg.swing_deg:
                cues.append(CUE_SWING)
        for cue in cues:
            if cue not in rep["cues"]:
                rep["cues"].append(cue)
        return tuple(cues)

    # -----------------------------------------------------------------------
    # summary(meta)
    # What:  the session summary. Each rep has "phases" (start/end of every phase, read by
    #        src/myomotion/sync/align.py intervals_from_cv) and the deadlift-style keys
    #        (t_concentric_start_s, t_lockout_s, t_descent_start_s, t_end_s).
    # Time / Space: O(R + T).  Data structures: dict -> JSON.
    # -----------------------------------------------------------------------
    def summary(self, meta: dict[str, Any] | None = None) -> dict[str, Any]:
        """JSON-ready session summary."""
        return {"schema": SCHEMA, "meta": meta or {}, "side": self.side, "frames_total": self.frames_total,
                "frames_tracked": self.frames_tracked, "frames_back_monitored": 0,
                "duration_s": (self.t_last - self.t_first) if self.t_first is not None else 0.0,
                "rep_count": self.rep_count, "reps": self.reps, "phase_transitions": self.transitions,
                "config": asdict(self.cfg), "disclaimers": DISCLAIMERS}


# Time:  O(1).  Space: O(1).
# Data structures: a dict update on the finished rep record.
def _add_curl_keys(rep: dict[str, Any]) -> None:
    """The curl's names for a finished rep's times and angles (concentric first, like the deadlift)."""
    rep.update({"t_concentric_start_s": rep["t_start_s"], "t_lockout_s": rep["t_turn_s"],
                "t_descent_start_s": rep["t_return_s"], "min_elbow_deg": rep["turn_angle_deg"],
                "start_elbow_deg": rep["open_angle_deg"],
                "concentric_s": rep["t_turn_s"] - rep["t_start_s"], "hold_s": rep["t_return_s"] - rep["t_turn_s"],
                "eccentric_s": rep["t_end_s"] - rep["t_return_s"]})


# Time:  O(1).  Space: O(1).
# Data structures: two 2-D NumPy points in, a float out.
def _lean_from_vertical(hip: np.ndarray, shoulder: np.ndarray) -> float:
    """Unsigned angle of hip->shoulder from straight up, degrees (image y points down)."""
    dx, dy = float(shoulder[0] - hip[0]), float(hip[1] - shoulder[1])
    return math.degrees(math.atan2(abs(dx), dy))


# ---------------------------------------------------------------------------
# curl_row(frame, t_s, result, process_ms)
# What:  one CurlFrame as one cv_frames.csv row (blank cells where nothing was seen).
# Time / Space: O(1).  Data structures: dict column -> text.  Files used: none.
# ---------------------------------------------------------------------------
CURL_COLUMNS = ("frame", "t_s", "person", "tracked", "side", "phase", "phase_label", "rep_count", "elbow_deg",
                "upper_arm_deg", "trunk_swing_deg", "cues", "process_ms")


# Time:  O(K) for the K = 13 columns.  Space: O(K).
# Data structures: a dict column -> text; nested helper number().
def curl_row(frame: int, t_s: float, result: CurlFrame, process_ms: float) -> dict[str, str]:
    """One row of cv_frames.csv for the curl."""
    # Time:  O(1).
    # Data structures: a float (or None) in, a str out.
    def number(value: float | None) -> str:
        return "" if value is None or not math.isfinite(value) else f"{value:.2f}"

    return {"frame": str(frame), "t_s": f"{t_s:.6f}", "person": str(int(result.landmarks is not None)),
            "tracked": str(int(result.tracked)), "side": result.side or "", "phase": result.phase,
            "phase_label": PHASE_LABELS[result.phase], "rep_count": str(result.rep_count),
            "elbow_deg": number(result.elbow_deg), "upper_arm_deg": number(result.upper_arm_deg),
            "trunk_swing_deg": number(result.trunk_swing_deg), "cues": "|".join(result.cues),
            "process_ms": f"{process_ms:.1f}"}


# ---------------------------------------------------------------------------
# draw_curl(frame, result, analyzer)
# What:  the overlay: the arm (shoulder-elbow-wrist) and trunk line, the elbow angle at
#        the elbow, a panel with the phase (UI colours), reps and angles, and the active
#        cue as a banner. Drawn in place on the frame given (a copy, never the recording).
# Time:  O(panel pixels).  Space: O(panel pixels) for the translucent panel.
# Data structures: NumPy image.  Files used: none.
# ---------------------------------------------------------------------------
PHASE_COLOURS = {REST: (170, 170, 170), CONCENTRIC: (224, 79, 31), HOLD: (222, 210, 205), ECCENTRIC: (255, 191, 169)}
CUE_COLOUR = (36, 165, 245)            # > #F5A524 warning amber, BGR


# Time:  O(P) for the P pixels of the translucent panel (cv2.addWeighted), plus a fixed number of shapes.  Space:
#        O(P).
# Data structures: a NumPy image drawn in place; a dict name -> (x, y) pixel point.
def draw_curl(frame: np.ndarray, result: CurlFrame, analyzer: CurlAnalyzer) -> np.ndarray:
    """Curl overlay, drawn in place; returns the frame."""
    import cv2
    h, w = frame.shape[:2]
    k = max(0.6, h / 720.0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    p = {name: (int(x), int(y)) for name, (x, y) in result.points.items()}
    if "hip" in p:
        cv2.line(frame, p["hip"], p["shoulder"], (160, 160, 160), max(2, int(3 * k)), cv2.LINE_AA)
    if result.tracked:
        cv2.line(frame, p["shoulder"], p["elbow"], (255, 255, 255), max(3, int(5 * k)), cv2.LINE_AA)
        cv2.line(frame, p["elbow"], p["wrist"], PHASE_COLOURS[result.phase], max(3, int(5 * k)), cv2.LINE_AA)
        for name in ("shoulder", "elbow", "wrist"):
            cv2.circle(frame, p[name], max(4, int(7 * k)), (40, 40, 40), -1, cv2.LINE_AA)
            cv2.circle(frame, p[name], max(3, int(5 * k)), (255, 255, 255), -1, cv2.LINE_AA)
        if result.elbow_deg is not None and math.isfinite(result.elbow_deg):
            cv2.putText(frame, f"{result.elbow_deg:.0f} deg", (p["elbow"][0] + int(12 * k), p["elbow"][1]), font,
                        0.7 * k, (255, 255, 255), 2, cv2.LINE_AA)
    panel = frame.copy()
    cv2.rectangle(panel, (0, 0), (int(300 * k), int(150 * k)), (25, 25, 25), -1)
    frame[:] = cv2.addWeighted(panel, 0.6, frame, 0.4, 0)
    x, y, dy = int(12 * k), int(34 * k), int(28 * k)
    title = PHASE_LABELS[result.phase].upper() if result.tracked else "SHOW YOUR ARM"
    cv2.putText(frame, title, (x, y), font, 0.9 * k, PHASE_COLOURS[result.phase] if result.tracked else CUE_COLOUR,
                2, cv2.LINE_AA)
    cv2.putText(frame, f"Reps: {result.rep_count}", (x, y + dy), font, 0.7 * k, (255, 255, 255), 2, cv2.LINE_AA)
    rows = []
    if result.tracked:
        rows.append(f"Elbow {result.elbow_deg:5.0f} deg  ({result.side} arm)")
        if result.upper_arm_deg is not None:
            rows.append(f"Upper arm {result.upper_arm_deg:4.0f}  swing {result.trunk_swing_deg or 0:4.0f} deg")
    else:
        rows.append("side-on, shoulder to wrist in view")
    for index, text in enumerate(rows):
        cv2.putText(frame, text, (x, y + dy * (index + 2)), font, 0.5 * k, (210, 210, 210), 1, cv2.LINE_AA)
    if result.cues:
        text = result.cues[0]
        (tw, th), _ = cv2.getTextSize(text, font, 0.8 * k, 2)
        x0, y0 = (w - tw) // 2, h - int(24 * k)
        cv2.rectangle(frame, (x0 - 10, y0 - th - 10), (x0 + tw + 10, y0 + 10), (25, 25, 25), -1)
        cv2.putText(frame, text, (x0, y0), font, 0.8 * k, CUE_COLOUR, 2, cv2.LINE_AA)
    return frame
