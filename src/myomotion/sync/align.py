"""Lining up the ESP32 clock with the video clock, and putting the sEMG on every video frame.

Two clocks run during a recording: the ESP32's micros() (every sEMG sample and every LED_ON event) and the
phone's video clock (every frame). They differ by an offset (they started at different moments) and a small
drift (no two crystals tick at exactly the same rate; the old Uno ran 1446 ppm slow). With the same flashes
seen on both clocks, the straight line

    t_video = offset + slope * t_esp32          (slope = 1 + drift)

is fitted by least squares. Two flashes fix the line exactly. A third (the recorder's middle flash) is an
independent check: its distance from the line through the first and last flashes is the sync error that
test TC03 measures (SRS limit: 33 ms, one frame at 30 fps).

Then every frame gets the sEMG value at its own moment (straight-line interpolation between the two
samples around it) and a rep phase: from Ahmad's CV summary JSON when given (phases seen in the video), or
else from the recorder's own cues (phases the person was told to do).

Data structures (this file): ClockFit, SyncReport and PhaseInterval dataclasses; NumPy float64 arrays; Python lists
    for the two-pointer interpolation walk; lists of per-frame phase labels.
Time complexity (this file): O(F) to fit and check the clock line over F flashes; O(N + M) to interpolate N samples
    onto M frames (two pointers that only move forward); O(M + I) to label M frames with I phase intervals; O(N * I)
    for the per-phase statistics.
"""

from __future__ import annotations   # > type hints are stored as text, not run: allows forward references

from dataclasses import dataclass

import numpy as np

from myomotion.semg.mvc import PhaseStat
from myomotion.semg.quality import ACTIVE_PHASES, US_PER_S, Cue

REST_PHASE = "REST"                   # frames outside every rep phase
MS_PER_S = 1000.0
PPM = 1e6                             # parts per million


# ---------------------------------------------------------------------------
# SyncConfig  (the CONFIG for this file)
# What:  the TC03 limit and the drift sanity limit.
# Data structures: a frozen dataclass.
# Time:  O(1) to build (fixed fields).
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SyncConfig:
    """Limits for the sync check."""

    max_check_error_ms: float = 33.0  # TC03 / SRS: sync error <= 33 ms (one frame at 30 fps)
    max_drift_ppm: float = 1000.0     # ESP32 and phone crystals are within ~100 ppm; much more = wrong pairing


SYNC_CONFIG = SyncConfig()


# ---------------------------------------------------------------------------
# ClockFit
# What:  the fitted line between the two clocks, and how far each flash is from it.
# Data structures: a frozen dataclass; residuals as a tuple (fixed once fitted).
# Time:  O(1) to build; to_video_s / to_esp32_us are O(n) for n times.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ClockFit:
    """t_video_s = video_ref_s + slope * (t_esp32_us - esp32_ref_us) / 1e6.

    The line is anchored at the first flash, not at ESP32 time 0: the ESP32 has usually been running for
    minutes, and extending the line that far back would multiply any small slope error by those minutes.
    """

    slope: float                      # video seconds per ESP32 second (1 + drift)
    esp32_ref_us: int                 # anchor on the ESP32 clock (the first flash)
    video_ref_s: float                # the fitted video time at that anchor
    residuals_ms: tuple[float, ...]   # each flash's video time minus the line, in milliseconds

    # Time:  O(1).
    # Data structures: a float.
    @property
    def drift_ppm(self) -> float:
        """How much faster the video clock runs than the ESP32's (parts per million)."""
        return (self.slope - 1.0) * PPM

    # Time:  O(1).
    # Data structures: a float.
    @property
    def esp32_s_at_video_start(self) -> float:
        """What the ESP32 clock read when the video started (video time 0), in seconds."""
        return float(self.to_esp32_us(0.0)) / US_PER_S

    # -----------------------------------------------------------------------
    # to_video_s(t_us) / to_esp32_us(t_video_s)
    # What:  convert ESP32 microseconds to video seconds, and back. The ESP32 times are
    #        made relative to the anchor while still integers, so no precision is lost.
    # Time:  O(N) for N times.  Space: O(N).
    # Data structures: NumPy arrays (or one float).  Files used: none.
    # -----------------------------------------------------------------------
    def to_video_s(self, t_us: np.ndarray | float) -> np.ndarray | float:
        """ESP32 time (us) -> video time (s)."""
        relative_s = (np.asarray(t_us) - self.esp32_ref_us).astype(np.float64) / US_PER_S
        return self.video_ref_s + self.slope * relative_s

    # Time:  O(n) for n times.  Space: O(n).
    # Data structures: a NumPy float64 array (or one float).
    def to_esp32_us(self, t_video_s: np.ndarray | float) -> np.ndarray | float:
        """Video time (s) -> ESP32 time (us)."""
        relative_s = (np.asarray(t_video_s, dtype=np.float64) - self.video_ref_s) / self.slope
        return self.esp32_ref_us + relative_s * US_PER_S


# ---------------------------------------------------------------------------
# fit_clock(esp32_us, video_s)
# What:  least-squares straight line through the flash pairs (our own maths):
#            slope     = sum((x - mean_x) * (y - mean_y)) / sum((x - mean_x)^2)
#            intercept = mean_y - slope * mean_x
#        with x = ESP32 seconds since the first flash and y = video seconds. Measuring x
#        from the first flash keeps the numbers small and the intercept meaningful
#        (it is the video time of the first flash).
# Time:  O(F) for F flashes.  Space: O(F).
# Data structures: NumPy float64 arrays.  Files used: none.
# ---------------------------------------------------------------------------
def fit_clock(esp32_us: list[int], video_s: list[float]) -> ClockFit:
    """Fit t_video = video_ref + slope * (t_esp32 - first flash) through matching flashes."""
    if len(esp32_us) != len(video_s):
        raise ValueError(f"{len(esp32_us)} LED_ON events but {len(video_s)} flashes in the video")
    if len(esp32_us) < 2:
        raise ValueError("at least 2 flashes are needed to line up the clocks")
    anchor_us = int(esp32_us[0])
    x = (np.asarray(esp32_us, dtype=np.int64) - anchor_us).astype(np.float64) / US_PER_S
    y = np.asarray(video_s, dtype=np.float64)
    x_mean, y_mean = float(np.sum(x)) / len(x), float(np.sum(y)) / len(y)
    spread = float(np.sum((x - x_mean) ** 2))
    if spread == 0:
        raise ValueError("the LED_ON events all have the same time")
    slope = float(np.sum((x - x_mean) * (y - y_mean))) / spread
    intercept = y_mean - slope * x_mean
    residuals = (y - (intercept + slope * x)) * MS_PER_S
    return ClockFit(slope=slope, esp32_ref_us=anchor_us, video_ref_s=intercept,
                    residuals_ms=tuple(float(r) for r in residuals))


# ---------------------------------------------------------------------------
# SyncReport
# What:  the TC03 result. check_errors_ms holds, for every middle flash, its distance
#        from the line through the first and last flashes (an independent check: that
#        flash was not used to draw the line). With only 2 flashes there is nothing to
#        check against, so the verdict is NOT MEASURED.
# Data structures: frozen dataclass; tuples for the per-flash numbers.
# Time:  O(F) to build for F check flashes; verdict is O(F).
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SyncReport:
    """TC03 numbers and verdict."""

    flashes: int
    esp32_s_at_video_start: float     # what the ESP32 clock read at video time 0
    drift_ppm: float
    check_errors_ms: tuple[float, ...]
    max_check_error_ms: float
    drift_plausible: bool             # info: False suggests the flashes were paired wrongly

    # Time:  O(F) for the F check flashes.  Space: O(1).
    # Data structures: a tuple of floats; a str out.
    @property
    def verdict(self) -> str:
        """PASS, FAIL or NOT MEASURED (no middle flash)."""
        if not self.check_errors_ms:
            return "NOT MEASURED"
        worst = max(abs(error) for error in self.check_errors_ms)
        return "PASS" if worst <= self.max_check_error_ms and self.drift_plausible else "FAIL"


# ---------------------------------------------------------------------------
# check_sync(esp32_us, video_s, fit, frame_interval_s, cfg)
# What:  builds the SyncReport. The drift limit is widened by the drift the frame
#        timing alone can cause (one frame over the span of the recording), so a short
#        clip with honest flashes is not flagged.
# Time:  O(F).  Space: O(F).
# Data structures: lists, tuples.  Files used: none.
# ---------------------------------------------------------------------------
def check_sync(esp32_us: list[int], video_s: list[float], fit: ClockFit, frame_interval_s: float,
               cfg: SyncConfig = SYNC_CONFIG) -> SyncReport:
    """TC03: how well the middle flashes agree with the clock line."""
    errors: list[float] = []
    if len(esp32_us) >= 3:
        outer = fit_clock([esp32_us[0], esp32_us[-1]], [video_s[0], video_s[-1]])
        for t_us, t_video in zip(esp32_us[1:-1], video_s[1:-1]):
            errors.append((t_video - float(outer.to_video_s(t_us))) * MS_PER_S)
    span_s = (esp32_us[-1] - esp32_us[0]) / US_PER_S
    timing_ppm = frame_interval_s / span_s * PPM if span_s > 0 else float("inf")
    return SyncReport(flashes=len(esp32_us), esp32_s_at_video_start=fit.esp32_s_at_video_start,
                      drift_ppm=fit.drift_ppm,
                      check_errors_ms=tuple(errors), max_check_error_ms=cfg.max_check_error_ms,
                      drift_plausible=abs(fit.drift_ppm) <= cfg.max_drift_ppm + timing_ppm)


# ---------------------------------------------------------------------------
# interpolate_sorted(x_known, y_known, x_new)
# What:  straight-line interpolation of y at every x_new, NaN outside the known range.
#        Both x lists are sorted (sample times and frame times only go forward), so one
#        walk with two pointers does it: the sample pointer never moves backwards.
# Time:  O(N + F) for N samples and F frames (a search per frame would be O(F log N)).
# Space: O(F) for the result.
# Data structures: Python lists for the walk (indexing a list is much faster than
#        indexing a NumPy array one item at a time); NumPy array out.  Files used: none.
# ---------------------------------------------------------------------------
def interpolate_sorted(x_known: np.ndarray, y_known: np.ndarray, x_new: np.ndarray) -> np.ndarray:
    """y at each x_new by linear interpolation (x_known and x_new ascending)."""
    xs, ys, targets = list(map(float, x_known)), list(map(float, y_known)), list(map(float, x_new))
    out = np.full(len(targets), np.nan)
    if not xs:
        return out
    i = 0
    last = len(xs) - 1
    for j, x in enumerate(targets):
        if x < xs[0] or x > xs[last]:
            continue
        while i < last - 1 and xs[i + 1] < x:          # > move the sample pointer up to the frame
            i += 1
        if last == 0 or xs[i + 1] == xs[i]:
            out[j] = ys[i]
            continue
        weight = (x - xs[i]) / (xs[i + 1] - xs[i])
        out[j] = ys[i] + weight * (ys[i + 1] - ys[i])
    return out


# ---------------------------------------------------------------------------
# PhaseInterval
# What:  one phase of one rep on the video clock, named as in events.csv.
# Data structures: frozen dataclass.
# Time:  O(1) to build (fixed fields).
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PhaseInterval:
    """[start_s, end_s) of one rep phase, in video seconds."""

    rep: int
    phase: str                        # CONCENTRIC, ISOMETRIC_HOLD or ECCENTRIC
    start_s: float
    end_s: float


# > Ahmad's RepTiming fields (src/myomotion/biomechanics/rep_segmenter.py) -> our phase names.
# > Deadlift order: concentric pull -> lockout (an isometric hold) -> eccentric descent.
CV_REP_PHASES = (
    ("CONCENTRIC", "t_concentric_start_s", "t_lockout_s"),
    ("ISOMETRIC_HOLD", "t_lockout_s", "t_descent_start_s"),
    ("ECCENTRIC", "t_descent_start_s", "t_end_s"),
)


# ---------------------------------------------------------------------------
# intervals_from_cv(summary)
# What:  rep phases from a CV summary JSON, on the video clock. A rep with a "phases"
#        list (the curl and RDL trackers, src/myomotion/sync/angle_cycle.py) gives each
#        phase in its own order (the RDL lowers first); otherwise Ahmad's deadlift keys
#        are read (CV_REP_PHASES). Ahmad's detector is causal, so his times lag the real
#        movement by a few hundred ms (his note in RepTiming); that lag belongs to the CV
#        side, not to the sync.
# Time:  O(R) for R reps.  Space: O(R).
# Data structures: list of PhaseInterval.  Files used: none (the caller reads the JSON).
# ---------------------------------------------------------------------------
def intervals_from_cv(summary: dict) -> list[PhaseInterval]:
    """Rep phases from a CV session summary (curl, RDL or Ahmad's deadlift)."""
    intervals: list[PhaseInterval] = []
    for rep in summary.get("reps", []):
        if "phases" in rep:
            spans = [(part["phase"], part["start_s"], part["end_s"]) for part in rep["phases"]]
        else:
            spans = [(phase, rep.get(start_key), rep.get(end_key)) for phase, start_key, end_key in CV_REP_PHASES]
        for phase, start, end in spans:
            if start is not None and end is not None and end > start:
                intervals.append(PhaseInterval(rep=int(rep["rep_index"]), phase=phase,
                                               start_s=float(start), end_s=float(end)))
    return sorted(intervals, key=lambda interval: interval.start_s)


# ---------------------------------------------------------------------------
# intervals_from_cues(cues, last_t_us, fit)
# What:  rep phases from the recorder's cues (events.csv), moved onto the video clock.
#        A cue lasts until the next cue; a new rep starts when a working phase follows
#        a non-working one (same rule as src/myomotion/semg/mvc.py phase_summary).
# Time:  O(E) for E cues.  Space: O(E).
# Data structures: list of PhaseInterval.  Files used: none.
# ---------------------------------------------------------------------------
def intervals_from_cues(cues: list[Cue], last_t_us: int, fit: ClockFit) -> list[PhaseInterval]:
    """Rep phases from the recorder's cues, in video seconds."""
    intervals: list[PhaseInterval] = []
    rep = 0
    previous_active = False
    for index, cue in enumerate(cues):
        active = cue.event in ACTIVE_PHASES
        if active and not previous_active:
            rep += 1
        previous_active = active
        if not active:
            continue
        end_us = cues[index + 1].t_us if index + 1 < len(cues) else last_t_us
        intervals.append(PhaseInterval(rep=rep, phase=cue.event, start_s=float(fit.to_video_s(cue.t_us)),
                                       end_s=float(fit.to_video_s(end_us))))
    return intervals


# ---------------------------------------------------------------------------
# label_frames(frame_times_s, intervals)
# What:  the phase and rep of every frame (REST and rep 0 outside every phase).
#        Frames and intervals are both in time order, so one two-pointer walk does it.
# Time:  O(F + I).  Space: O(F).
# Data structures: two Python lists (phase names, rep numbers), one entry per frame.
# Files used: none.
# ---------------------------------------------------------------------------
def label_frames(frame_times_s: np.ndarray, intervals: list[PhaseInterval]) -> tuple[list[str], list[int]]:
    """(phase per frame, rep per frame)."""
    ordered = sorted(intervals, key=lambda interval: interval.start_s)
    phases = [REST_PHASE] * len(frame_times_s)
    reps = [0] * len(frame_times_s)
    k = 0
    for j, t_s in enumerate(map(float, frame_times_s)):
        while k < len(ordered) and ordered[k].end_s <= t_s:   # > skip phases that ended before this frame
            k += 1
        if k < len(ordered) and ordered[k].start_s <= t_s:
            phases[j] = ordered[k].phase
            reps[j] = ordered[k].rep
    return phases, reps


# ---------------------------------------------------------------------------
# phase_stats(sample_video_s, percent, intervals)
# What:  mean and peak %MVC of every phase of every rep, using all sEMG samples (500
#        per second, finer than the frames) that fall inside it.
# Time:  O(N * I) (one comparison of all samples per interval).  Space: O(N).
# Data structures: boolean NumPy masks; list of PhaseStat (from src/myomotion/semg/mvc.py).
# Files used: none.
# ---------------------------------------------------------------------------
def phase_stats(sample_video_s: np.ndarray, percent: np.ndarray, intervals: list[PhaseInterval]) -> list[PhaseStat]:
    """Per-phase %MVC summary on the video's phase times."""
    stats: list[PhaseStat] = []
    for interval in intervals:
        values = percent[(sample_video_s >= interval.start_s) & (sample_video_s < interval.end_s)]
        if len(values) == 0:
            continue
        stats.append(PhaseStat(rep=interval.rep, phase=interval.phase, seconds=interval.end_s - interval.start_s,
                               mean_percent=float(np.sum(values)) / len(values), peak_percent=float(np.max(values))))
    return stats
