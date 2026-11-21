"""MVC and %MVC for the raw-EMG ESP32 build (Step 4, test TC02).

MVC = Maximum Voluntary Contraction: the strongest signal the muscle makes when you push as
hard as you can. Every later number is a percentage of it, which makes recordings comparable
between people, days and electrode placements:

    %MVC(t) = 100 * (envelope(t) - rest) / (MVC - rest_mvc)

Protocol (record_session.py --mvc): relaxed baseline, then 3 maximal pushes of 5 s with 30 s
rest between them (shorter rests risk fatigue: later pushes come out weaker). Measurement:
  1. envelope = 20 Hz high-pass + mains notches + 100 ms RMS   (dsp.emg_envelope)
  2. rest     = mean envelope during the relaxed baseline (the noise floor)
  3. each push's peak = the highest 1-second mean of the envelope inside that push
  4. MVC      = the best push's peak (the protocol's "highest 1-second mean")
TC02 passes when all pushes were recorded, each is clearly a real contraction, the two best
agree (a big gap means one push was not maximal: redo the session), and no push reached the
ESP32's ceiling (a clipped push would make the MVC too small and every %MVC too big).
Subtracting the rest level makes a relaxed muscle read about 0% instead of the noise floor.

Data structures (this file): NumPy arrays for the envelope; lists of (start, end) push windows; MvcResult, PhaseStat
    and NormalisedSet dataclasses; a dict for mvc.json; a list of Paths sorted by time (find_latest_mvc).
Time complexity (this file): O(N * S) for the envelope (N samples, S = 6 filter sections); O(n) per push window
    (moving average); O(N * P) for the per-phase summary of P phase windows; O(F log F) to find the newest passed
    MVC among F folders.
"""

from __future__ import annotations   # > type hints are stored as text, not run: allows forward references

import json                           # > writes mvc.json, percent_mvc.json
import math                           # > checks for finite numbers
from dataclasses import asdict, dataclass, field
from pathlib import Path              # > file paths that work on any operating system

import numpy as np                    # > arrays for the envelope

from myomotion.semg.dsp import DSP_CONFIG, DspConfig, emg_envelope, moving_average
from myomotion.semg.quality import ACTIVE_PHASES, Cue, Recording, measured_rate_hz, rest_mask

MVC_EVENT = "MVC"                     # cue name record_session.py --mvc writes for each maximal push
PERCENT = 100.0                       # %MVC is a percentage


# ---------------------------------------------------------------------------
# MvcConfig  (the CONFIG for this file)
# What:  every setting of the MVC measurement in one place.
# Data structures: a frozen dataclass (named, typed, cannot change by accident).
# Values marked "provisional" are starting guesses to tune on real sessions.
# Time:  O(1) to build (fixed fields).
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class MvcConfig:
    """Settings for the MVC measurement and the TC02 check."""

    channel: str = "ch1"              # emg.csv column with the muscle
    dsp: DspConfig = DSP_CONFIG       # the same envelope as everywhere else (Step 2)
    peak_window_s: float = 1.0        # MVC = highest mean over 1 second (the protocol)
    expected_trials: int = 3          # pushes the protocol asks for
    min_peak_to_rest: float = 3.0     # provisional: a real maximal push is at least 3x the rest envelope
    max_best_two_gap: float = 0.10    # provisional: the two best pushes within 10% of each other
    ceiling_mv: float = 3100.0        # ESP32 ADC top (analogReadMilliVolts)
    clip_margin_mv: float = 50.0      # within 50 mV of the ceiling counts as clipped
    max_clipped_samples: int = 0      # no clipping allowed inside a push


MVC_CONFIG = MvcConfig()


# ---------------------------------------------------------------------------
# MvcResult
# What:  the outcome of one MVC session: the rest level, each push's peak, the
#        MVC itself and the three TC02 verdicts.
# Data structures: a dataclass; per-push numbers are lists (one entry per push,
#        in order); `passed` is a computed property.
# Time:  O(T) to build for the T pushes (per-push lists); passed is O(1).
# ---------------------------------------------------------------------------
@dataclass
class MvcResult:
    """MVC session result (envelope values in mV at the ESP32 pin)."""

    rate_hz: float                   # measured samples per second
    rest_envelope_mv: float          # mean envelope during the relaxed baseline
    trial_peaks_mv: list[float]      # each push's highest 1-second mean envelope
    trial_peak_start_us: list[int]   # where each push's best 1-second window starts (ESP32 clock)
    mvc_envelope_mv: float           # the best push's peak (rest included)
    mvc_above_rest_mv: float         # MVC minus rest: the 100% step used for %MVC
    best_two_gap: float              # (best - second best) / best; 0 = identical pushes
    clipped_samples: int             # raw samples near the ceiling inside the pushes
    trials_ok: bool                  # all expected pushes were found
    strength_ok: bool                # every push >= min_peak_to_rest x rest
    consistency_ok: bool             # best two within max_best_two_gap
    clipping_ok: bool                # no push reached the ceiling

    # Time:  O(1).
    # Data structures: four bools combined with `and`.
    @property
    def passed(self) -> bool:
        """TC02 passes when all four checks pass."""
        return self.trials_ok and self.strength_ok and self.consistency_ok and self.clipping_ok


# ---------------------------------------------------------------------------
# PhaseStat
# What:  %MVC summary of one phase of one rep: the project's "activation per rep
#        phase" in its simplest form.
# Data structures: a frozen dataclass (a finished measurement does not change).
# Time:  O(1) to build (fixed fields).
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PhaseStat:
    """Mean and peak %MVC of one phase of one rep."""

    rep: int
    phase: str                       # CONCENTRIC, ISOMETRIC_HOLD or ECCENTRIC
    seconds: float
    mean_percent: float
    peak_percent: float


# ---------------------------------------------------------------------------
# cue_windows(cues, event, last_t_us)
# What:  (start_us, end_us) for every cue called `event`: from that cue to the next
#        cue (or to the end of the recording for the last one).
# Time:  O(E) for E cues.  Space: O(W) windows.
# Data structures: list of (start, end) tuples (a window never changes once found).
# Files used: none.
# ---------------------------------------------------------------------------
def cue_windows(cues: list[Cue], event: str, last_t_us: int) -> list[tuple[int, int]]:
    """Time window of every cue named `event`."""
    windows = []
    for index, cue in enumerate(cues):
        if cue.event == event:
            end_us = cues[index + 1].t_us if index + 1 < len(cues) else last_t_us + 1
            windows.append((cue.t_us, end_us))
    return windows


# ---------------------------------------------------------------------------
# best_window_mean(values, window)
# What:  the highest mean over any `window` consecutive samples, and where that
#        window starts. Uses dsp.moving_average (running totals) and keeps only the
#        outputs whose window is full (index >= window - 1), so a short partial window
#        at the start cannot win.
# Time:  O(n): one moving average and one scan for the maximum.
# Space: O(n): the moving-average array.
# Data structures: NumPy arrays; returns a tuple (best mean, start index).
# Files used: src/myomotion/semg/dsp.py (moving_average).
# ---------------------------------------------------------------------------
def best_window_mean(values: np.ndarray, window: int) -> tuple[float, int]:
    """Highest mean over `window` consecutive samples, and the index where that window starts."""
    if len(values) < window:
        raise ValueError(f"need at least {window} samples, got {len(values)}")
    full_windows = moving_average(values, window)[window - 1:]   # > entry i covers values[i .. i+window-1]
    best_start = int(np.argmax(full_windows))             # > argmax = index of the largest mean
    return float(full_windows[best_start]), best_start


# ---------------------------------------------------------------------------
# rest_level(recording, envelope)
# What:  mean envelope during the relaxed baseline (countdown excluded): the noise
#        floor that %MVC subtracts.
# Time:  O(N).  Space: O(N) for the mask.
# Data structures: boolean NumPy mask from quality.rest_mask.
# Files used: src/myomotion/semg/quality.py (rest_mask).
# ---------------------------------------------------------------------------
def rest_level(recording: Recording, envelope: np.ndarray) -> float:
    """Mean envelope (mV) over the relaxed baseline."""
    rest = envelope[rest_mask(recording.t_us, recording.cues, recording.baseline_s)]
    if len(rest) == 0:
        raise ValueError("the baseline is empty: check events.csv and baseline_s")
    return float(np.sum(rest)) / len(rest)


# ---------------------------------------------------------------------------
# measure_mvc(recording, cfg)
# What:  envelope -> rest level -> each push's best 1-second mean -> MVC and the
#        TC02 checks (see the file docstring), plus a count of raw samples at the
#        ceiling inside the pushes.
# Time:  O(N * S) for the envelope filter (N samples, S = 6 filter sections) plus
#        O(N * T) for the T push windows (each compares all timestamps once).
# Space: O(N).
# Data structures: NumPy arrays; per-push results in lists, sorted copy for the
#        best two.
# Files used: src/myomotion/semg/dsp.py (emg_envelope), src/myomotion/semg/quality.py
#        (measured_rate_hz, rest_mask through rest_level).
# ---------------------------------------------------------------------------
def measure_mvc(recording: Recording, cfg: MvcConfig = MVC_CONFIG) -> MvcResult:
    """Measure the MVC from a record_session.py --mvc recording."""
    rate_hz = measured_rate_hz(recording.t_us)
    envelope = emg_envelope(recording.signal_mv, rate_hz, cfg.dsp)
    rest = rest_level(recording, envelope)
    window = max(1, round(cfg.peak_window_s * rate_hz))  # > 1 s * 500 Hz = 500 samples
    peaks: list[float] = []
    starts_us: list[int] = []
    clipped = 0
    for start_us, end_us in cue_windows(recording.cues, MVC_EVENT, int(recording.t_us[-1])):
        in_push = (recording.t_us >= start_us) & (recording.t_us < end_us)
        push_indices = np.flatnonzero(in_push)            # > positions of this push's samples
        # > clipping is checked on the RAW millivolts: the envelope hides flat tops
        clipped += int(np.sum(recording.signal_mv[push_indices] >= cfg.ceiling_mv - cfg.clip_margin_mv))
        if len(push_indices) < window:                    # > aborted or too short: cannot give a 1 s mean
            continue
        peak, offset = best_window_mean(envelope[push_indices], window)
        peaks.append(peak)
        starts_us.append(int(recording.t_us[push_indices[offset]]))
    if not peaks:
        raise ValueError("no complete MVC push found: was this recorded with record_session.py --mvc?")

    ranked = sorted(peaks, reverse=True)                  # > best first
    best = ranked[0]
    gap = (ranked[0] - ranked[1]) / ranked[0] if len(ranked) > 1 and ranked[0] > 0 else 0.0
    return MvcResult(
        rate_hz=rate_hz,
        rest_envelope_mv=rest,
        trial_peaks_mv=peaks,
        trial_peak_start_us=starts_us,
        mvc_envelope_mv=best,
        mvc_above_rest_mv=best - rest,
        best_two_gap=gap,
        clipped_samples=clipped,
        trials_ok=len(peaks) >= cfg.expected_trials,
        strength_ok=rest > 0 and min(peaks) >= cfg.min_peak_to_rest * rest,
        consistency_ok=len(peaks) > 1 and gap <= cfg.max_best_two_gap,
        clipping_ok=clipped <= cfg.max_clipped_samples,
    )


# ---------------------------------------------------------------------------
# percent_mvc(envelope, rest_mv, mvc_above_rest_mv)
# What:  %MVC(t) = 100 * (envelope(t) - rest) / (MVC - rest_mvc). Not clipped: a
#        value slightly below 0 just means "quieter than the average rest".
# Time:  O(n).  Space: O(n).
# Data structures: NumPy arrays (element-wise arithmetic).
# Files used: none.
# ---------------------------------------------------------------------------
def percent_mvc(envelope: np.ndarray, rest_mv: float, mvc_above_rest_mv: float) -> np.ndarray:
    """Envelope (mV) to %MVC, rest-subtracted."""
    if not (mvc_above_rest_mv > 0 and math.isfinite(mvc_above_rest_mv)):
        raise ValueError("the MVC must be above the rest level")
    return PERCENT * (envelope - rest_mv) / mvc_above_rest_mv


# ---------------------------------------------------------------------------
# phase_summary(t_us, percent, cues)
# What:  mean and peak %MVC for every active phase of every rep. A new rep starts at
#        each active cue that follows a non-active one (works for SQUEEZE/HOLD/RELEASE
#        and DOWN/HOLD/UP alike).
# Time:  O(N * P): each of the P phase windows compares all N timestamps once.
# Space: O(N) for one mask at a time, O(P) for the results.
# Data structures: list of PhaseStat; boolean masks to pick each phase's samples.
# Files used: none.
# ---------------------------------------------------------------------------
def phase_summary(t_us: np.ndarray, percent: np.ndarray, cues: list[Cue]) -> list[PhaseStat]:
    """Mean and peak %MVC per rep phase."""
    stats: list[PhaseStat] = []
    rep = 0
    previous_active = False
    for index, cue in enumerate(cues):
        is_active = cue.event in ACTIVE_PHASES
        if is_active and not previous_active:
            rep += 1
        previous_active = is_active
        if not is_active:
            continue
        end_us = cues[index + 1].t_us if index + 1 < len(cues) else int(t_us[-1]) + 1
        values = percent[(t_us >= cue.t_us) & (t_us < end_us)]
        if len(values) == 0:
            continue
        stats.append(PhaseStat(rep=rep, phase=cue.event, seconds=(end_us - cue.t_us) / 1e6,
                               mean_percent=float(np.sum(values)) / len(values),
                               peak_percent=float(np.max(values))))
    return stats


# ---------------------------------------------------------------------------
# NormalisedSet
# What:  one exercise set converted to %MVC: the time series (this is the measured
#        Y(t) the models will learn from) and the per-phase summary.
# Data structures: dataclass with NumPy arrays and a list of PhaseStat.
# Time:  O(1) to build (it holds references to the arrays, no copies).
# ---------------------------------------------------------------------------
@dataclass
class NormalisedSet:
    """An exercise set in %MVC."""

    t_us: np.ndarray
    envelope_mv: np.ndarray
    percent: np.ndarray
    rest_envelope_mv: float
    phases: list[PhaseStat] = field(default_factory=list)


# ---------------------------------------------------------------------------
# normalise_set(recording, mvc_above_rest_mv, cfg)
# What:  envelope of the set -> its own rest level (from its own baseline, so a
#        small change in noise between recordings does not shift 0%) -> %MVC ->
#        per-phase summary.
# Time:  O(N * S) for the envelope, O(N * P) for the summary.
# Space: O(N).
# Data structures: see NormalisedSet.
# Files used: src/myomotion/semg/dsp.py (emg_envelope).
# ---------------------------------------------------------------------------
def normalise_set(recording: Recording, mvc_above_rest_mv: float, cfg: MvcConfig = MVC_CONFIG) -> NormalisedSet:
    """Convert an exercise set to %MVC using an MVC measured in the same session."""
    rate_hz = measured_rate_hz(recording.t_us)
    envelope = emg_envelope(recording.signal_mv, rate_hz, cfg.dsp)
    rest = rest_level(recording, envelope)
    percent = percent_mvc(envelope, rest, mvc_above_rest_mv)
    return NormalisedSet(t_us=recording.t_us, envelope_mv=envelope, percent=percent, rest_envelope_mv=rest,
                         phases=phase_summary(recording.t_us, percent, recording.cues))


# ---------------------------------------------------------------------------
# save_mvc(result, folder, details) / load_mvc(path)
# What:  write the MVC to <folder>/mvc.json (with who/what details such as the
#        participant, channel and muscle name) and read it back. load_mvc accepts the
#        file or its folder.
# Time:  O(T) for T pushes.  Space: O(T).
# Data structures: dict (field name -> value) via dataclasses.asdict.
# Files used: writes / reads <MVC recording folder>/mvc.json.
# ---------------------------------------------------------------------------
def save_mvc(result: MvcResult, folder: Path, details: dict) -> Path:
    """Write mvc.json into the MVC recording folder."""
    contents = {**details, **asdict(result), "passed": result.passed}
    out_path = folder / "mvc.json"
    out_path.write_text(json.dumps(contents, indent=2))
    return out_path


# Time:  O(m) for an m-byte mvc.json.  Space: O(m).
# Data structures: a dict parsed from JSON.
def load_mvc(path: Path) -> dict:
    """Read mvc.json (give the file or the folder that holds it)."""
    json_path = path / "mvc.json" if path.is_dir() else path
    return json.loads(json_path.read_text())


# ---------------------------------------------------------------------------
# find_latest_mvc(participant_dir, passed_only)
# What:  the newest mvc.json among this participant's recording folders, so a set
#        can be normalised without typing the MVC path (MVC is redone every session).
#        With passed_only=True (the default) a session that failed TC02 is skipped:
#        a failed MVC (e.g. a saturated module, S20 on 17 Nov) would give nonsense %MVC.
# Time:  O(F log F) for F folders (sorting by modification time), plus reading
#        each candidate's small JSON file.
# Space: O(F).
# Data structures: list of Paths sorted by mtime, walked newest first.
# Files used: reads <participant_dir>/*/mvc.json.
# ---------------------------------------------------------------------------
def find_latest_mvc(participant_dir: Path, passed_only: bool = True) -> Path | None:
    """Newest <participant_dir>/*/mvc.json (that passed TC02, by default), or None."""
    candidates = sorted(participant_dir.glob("*/mvc.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    for candidate in candidates:
        if not passed_only or load_mvc(candidate).get("passed"):
            return candidate
    return None
