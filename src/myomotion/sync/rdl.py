"""Dumbbell Romanian deadlift (RDL): reps and phases on top of Ahmad's deadlift analysis (the FYP-1 back demo).

Why the RDL: a hip hinge with a loaded back, so the lower-back muscle (erector spinae, where the electrodes go) works
hard while the spine must stay flat; filmed side-on, that is exactly what SpinePose and Ahmad's spine angles and back
cues are built for. Everything per frame (MediaPipe, SpinePose, joint and spine angles, back-rounding rules, the
overlay) is Ahmad's DeadliftAnalyzer, used unchanged. Only the rep counting differs: his counter expects a deadlift
from the floor (pull -> lockout -> lower) and, on a synthetic RDL that starts standing, counts 2 of 3 reps. Here an
RDL rep is the hip angle closing and opening again, with the shared state machine (src/myomotion/sync/angle_cycle.py):

    REST (standing, hip > 160) --below 150--> ECCENTRIC (lower) --below 130--> ISOMETRIC_HOLD (bottom)
        --above 140--> CONCENTRIC (stand up) --above 160--> REST   (rep counted)

The zones are the ones of his segmenter (set-up 130/140, lockout 160/150), on the same angle (hip_signed_deg).
Each rep also records the trunk lean, the spine angles and the back flags his rules raised during it.

Data structures (this file): Ahmad's FrameResult in; the AngleCycle state machine; a set of form-flag names per rep;
    RdlFrame dataclasses; dict rep records.
Time complexity (this file): O(1) per frame for the state machine plus O(F) for the F active form flags; drawing the
    HUD is O(P) for the patch's P pixels.
"""

from __future__ import annotations   # > type hints are stored as text, not run: allows forward references

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from myomotion.biomechanics import FrameResult
from myomotion.sync.angle_cycle import FINISHED, HOLD, REST, STARTED, AngleCycle, CycleConfig
from myomotion.sync.curl import PHASE_COLOURS

RDL_EXERCISES = frozenset({"rdl", "romanian_deadlift", "dumbbell_rdl"})
SCHEMA = "myomotion.rdl_session/1"
CONCENTRIC, ECCENTRIC = "CONCENTRIC", "ECCENTRIC"
PHASE_LABELS = {REST: "Standing", ECCENTRIC: "Lower", HOLD: "Hold", CONCENTRIC: "Stand up"}


# ---------------------------------------------------------------------------
# RdlConfig  (the CONFIG for this file)
# What:  the hip-angle zones and timing of an RDL rep.
# Data structures: frozen dataclass.
# Time:  O(1) to build (fixed fields).
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RdlConfig:
    """Settings of the RDL rep counter."""

    stand_enter_deg: float = 160.0     # standing (REST) above this; Ahmad's lockout zone
    stand_exit_deg: float = 150.0      # lowering starts below this
    bottom_enter_deg: float = 130.0    # the bottom (HOLD) below this; Ahmad's set-up zone
    bottom_exit_deg: float = 140.0     # standing up starts above this
    dwell_s: float = 0.08
    turn_tolerance_deg: float = 3.0
    stall_s: float = 8.0               # an RDL rep is slower than a curl
    lost_after_s: float = 1.0          # legs or trunk not seen this long -> the unfinished rep is dropped


RDL_CONFIG = RdlConfig()


# ---------------------------------------------------------------------------
# RdlFrame
# What:  one frame: Ahmad's full FrameResult (angles, spine, cues; drawn by his overlay)
#        plus the RDL phase and rep count.
# Data structures: frozen dataclass.
# Time:  O(1) to build (fixed fields).
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RdlFrame:
    """One frame of the RDL tracker."""

    t_s: float
    deadlift: FrameResult
    phase: str
    rep_count: int
    hip_deg: float | None


# ---------------------------------------------------------------------------
# RdlCounter
# What:  feed Ahmad's FrameResult of every frame; count RDL reps on his hip angle and
#        keep, for each rep, the deepest trunk lean, the largest spine angles and the
#        back flags that were active.
# Time:  O(1) per frame.  Space: O(R) rep records.
# Data structures: AngleCycle; per-rep dict; set of flag names (O(1) add).
# Files used: src/myomotion/sync/angle_cycle.py; Ahmad's FrameResult (biomechanics).
# ---------------------------------------------------------------------------
class RdlCounter:
    """RDL reps from Ahmad's per-frame deadlift analysis."""

    # Time:  O(1).  Space: O(1).
    # Data structures: an AngleCycle; a set of flag names.
    def __init__(self, cfg: RdlConfig = RDL_CONFIG) -> None:
        self.cfg = cfg
        self.cycle = AngleCycle(CycleConfig(open_enter_deg=cfg.stand_enter_deg, open_exit_deg=cfg.stand_exit_deg,
                                            closed_enter_deg=cfg.bottom_enter_deg,
                                            closed_exit_deg=cfg.bottom_exit_deg, dwell_s=cfg.dwell_s,
                                            turn_tolerance_deg=cfg.turn_tolerance_deg, stall_s=cfg.stall_s),
                                closing_phase=ECCENTRIC, opening_phase=CONCENTRIC)
        self._last_seen: float | None = None
        self._flags: set[str] = set()

    # Time:  O(1) per frame (one state-machine step) plus O(F) for the F active form flags.  Space: O(F).
    # Data structures: Ahmad's FrameResult in; the running rep dict; a set of flag names (O(1) add); an RdlFrame out.
    def update(self, t_s: float, result: FrameResult) -> RdlFrame:
        """One frame of Ahmad's analysis -> the RDL phase and count."""
        angles = result.angles
        hip = float(angles.hip_signed_deg) if angles is not None else None
        if hip is None or not math.isfinite(hip):
            if self._last_seen is not None and t_s - self._last_seen > self.cfg.lost_after_s:
                self.cycle.drop(t_s)                             # > lost the legs mid-rep: not a rep
            return RdlFrame(t_s, result, self.cycle.phase, self.cycle.rep_count, None)
        self._last_seen = t_s
        event = self.cycle.step(t_s, hip)
        if event == STARTED:
            self._flags = set()
            self.cycle.current.update({"max_trunk_incl_deg": 0.0, "max_thoracic_deg": None, "max_lumbar_deg": None,
                                       "spine_source": None, "back_flags": []})
        rep = self.cycle.current
        if rep is not None:
            self._track_back(rep, result)
        if event == FINISHED:
            self.cycle.reps[-1]["back_flags"] = sorted(self._flags)
        return RdlFrame(t_s, result, self.cycle.phase, self.cycle.rep_count, hip)

    # Time:  O(F) for the F active flags.  Space: O(F).
    # Data structures: the rep dict; a set of flag names, sorted into a list.
    def _track_back(self, rep: dict[str, Any], result: FrameResult) -> None:
        """Deepest trunk lean, largest spine angles and back flags during the rep."""
        angles = result.angles
        rep["max_trunk_incl_deg"] = max(rep["max_trunk_incl_deg"], float(angles.trunk_incl_deg))
        spine = angles.spine
        if spine is not None and spine.curvature_visible:
            rep["spine_source"] = spine.source
            for key, value in (("max_thoracic_deg", spine.thoracic_deg), ("max_lumbar_deg", spine.lumbar_deg)):
                rep[key] = float(value) if rep[key] is None else max(rep[key], float(value))
        for flag in result.active_flags:
            self._flags.add(flag.value)
            rep["back_flags"] = sorted(self._flags)

    # -----------------------------------------------------------------------
    # summary(deadlift_summary)
    # What:  Ahmad's session summary (spine, events, flags: all kept) with the RDL reps in
    #        "reps" (each with its "phases") and his own deadlift count kept for comparison.
    # Time / Space: O(R).  Data structures: dict -> JSON.
    # -----------------------------------------------------------------------
    def summary(self, deadlift_summary: dict[str, Any]) -> dict[str, Any]:
        """The RDL session summary."""
        out = dict(deadlift_summary)
        out.update({"schema": SCHEMA, "deadlift_rep_count": deadlift_summary.get("rep_count"),
                    "deadlift_reps": deadlift_summary.get("reps", []), "rep_count": self.cycle.rep_count,
                    "reps": self.cycle.reps, "rdl_phase_transitions": self.cycle.transitions,
                    "rdl_config": asdict(self.cfg)})
        return out


# ---------------------------------------------------------------------------
# draw_rdl_hud(frame, result)
# What:  after Ahmad's overlay is drawn, replace its first two lines (his deadlift phase
#        and rep count) with the RDL phase (UI colours) and the RDL rep count; his spine
#        angles, checklist and back cues stay as he draws them.
# Time:  O(patch pixels).  Space: O(1) extra.
# Data structures: NumPy image (drawn in place).  Files used: none.
# ---------------------------------------------------------------------------
def draw_rdl_hud(frame: np.ndarray, result: RdlFrame) -> np.ndarray:
    """RDL phase and reps over the top of Ahmad's panel."""
    import cv2
    k = max(0.5, frame.shape[0] / 720.0)                          # > the same scale as his overlay
    cv2.rectangle(frame, (0, 0), (int(400 * k), int(76 * k)), (25, 25, 25), -1)
    label = PHASE_LABELS[result.phase].upper() if result.hip_deg is not None else "SHOW YOUR WHOLE BODY"
    colour = PHASE_COLOURS.get(result.phase, (200, 200, 200)) if result.hip_deg is not None else (36, 165, 245)
    cv2.putText(frame, f"RDL  {label}", (int(14 * k), int(36 * k)), cv2.FONT_HERSHEY_SIMPLEX, 0.85 * k, colour, 2,
                cv2.LINE_AA)
    hip = f"   hip {result.hip_deg:.0f} deg" if result.hip_deg is not None else ""
    cv2.putText(frame, f"Reps: {result.rep_count}{hip}", (int(14 * k), int(66 * k)), cv2.FONT_HERSHEY_SIMPLEX,
                0.7 * k, (255, 255, 255), 2, cv2.LINE_AA)
    return frame
