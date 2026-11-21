"""Signal-quality check for one sEMG recording: test TC01.

The ESP32 build's module outputs RAW EMG (measured 17 Nov, recordings S11-S13): the muscle
shows up as fast 20-250 Hz activity, on top of a DC offset and strong mains hum. So every
"how strong" number here is measured in the EMG band, i.e. 20 Hz up to Nyquist (250 Hz) with
the mains bands (50, 100, 150, 200 Hz +-2 Hz) left out, exactly what the Step 2 filters in
dsp.py keep. Measuring raw millivolts instead would mostly measure the hum and the offset.

Pass/fail checks (all must pass):
1. Baseline: the relaxed signal is steady (no drift, no burst of tension during the rest).
2. SNR:      muscle-band RMS while squeezing >= 3 x the muscle-band RMS at rest (9.5 dB).
3. Clipping: nothing reaches the ESP32 ceiling (~3100 mV).
Warnings (reported, do not fail TC01, but should be fixed in hardware):
4. 50 Hz hum: mains hum stronger than half the muscle signal (the notches remove it in
   software, but less hum means cleaner data).
5. ADC floor: many rest samples stuck at the ESP32's ~142 mV floor (a raw signal swings
   below that and its lower part is lost; a mid-range bias would fix it).

Data structures (this file): a Recording dataclass (NumPy int64 times, float64 signal, list of Cue); boolean NumPy
    masks for rest and active samples; lists of (start, end) windows; a QualityReport dataclass; a dict for
    quality.json.
Time complexity (this file): O(N + E log E) to load N samples and E cues; O(N) for the masks, clipping and drift;
    the direct DFTs of the rest (R samples) and active windows (A samples), O(R^2 + A^2), dominate TC01.
"""

from __future__ import annotations   # > type hints are stored as text, not run: allows forward references

import csv                            # > reads emg.csv and events.csv row by row
import json                           # > reads metadata.json, writes quality.json
import math                           # > sqrt and log10 on single numbers
from dataclasses import asdict, dataclass
from pathlib import Path              # > file paths that work on any operating system

import numpy as np                    # > fast arrays for the signal

from myomotion.semg.dsp import (DSP_CONFIG, DspConfig, emg_band_rms, emg_band_rms_from_spectrum,
                                mains_rms_from_spectrum, power_spectrum)

# ---------------------------------------------------------------------------
# Constants that never change: names used in events.csv and a unit conversion.
# ---------------------------------------------------------------------------
US_PER_S = 1_000_000                  # microseconds in one second (t_us is in microseconds)
CUE_SOURCE = "cue"                    # events.csv rows that come from the recorder's own phase cues
BASELINE_CUE = "BASELINE"             # the relaxed period at the start of the set
# > frozenset: an unchangeable set, so "event in ACTIVE_PHASES" is an O(1) lookup
ACTIVE_PHASES = frozenset({"ECCENTRIC", "ISOMETRIC_HOLD", "CONCENTRIC"})   # muscle working (any order)


# ---------------------------------------------------------------------------
# QualityConfig  (the CONFIG for this file)
# What:  every threshold the check uses, in one place (no magic numbers in the
#        functions). The frequencies of the EMG band come from dsp.py's DSP_CONFIG,
#        so TC01 measures exactly the band the Step 2 filters keep.
# Data structures: a frozen dataclass: named, typed, cannot be changed by accident.
# Values marked "provisional" are starting guesses: tune them on good recordings
# and write the final values in the report.
# Time:  O(1) to build (fixed fields).
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class QualityConfig:
    """Thresholds for the TC01 signal-quality check (millivolts are at the ESP32 pin)."""

    channel: str = "ch1"              # emg.csv column with the muscle: ch1 = VP (GPIO36), the only wired pin
    dsp: DspConfig = DSP_CONFIG       # EMG band and mains notches (shared with the Step 2 filters)
    # ESP32 ADC limits (analogReadMilliVolts)
    floor_mv: float = 150.0           # at or below this = stuck at the ADC floor (~142 mV)
    ceiling_mv: float = 3100.0        # top of the ADC range
    clip_margin_mv: float = 50.0      # within 50 mV of the ceiling counts as clipped
    max_clipped_samples: int = 0      # check 3: no clipping at all
    # Check 1, baseline (provisional)
    max_rest_drift_mv: float = 20.0   # the rest's mean must not wander (2nd half minus 1st half)
    max_rest_halves_ratio: float = 2.0  # muscle-band noise in one half of the rest at most 2x the other
    # Check 2, SNR
    min_snr_ratio: float = 3.0        # active / rest muscle-band RMS >= 3, i.e. 20*log10(3) = 9.54 dB
    reaction_delay_s: float = 0.3     # skip the first 0.3 s of each rep: you react after the cue
    # Warnings (provisional)
    max_hum_to_muscle: float = 0.5    # warn if hum RMS > half the muscle-band RMS while squeezing
    max_floor_share: float = 0.05     # warn if more than 5% of rest samples sit on the ADC floor


CONFIG = QualityConfig()

# > Rebuild of 20 Nov: a 10 kOhm bias to D32 lifts the module's 0 V to ~1.32 V and divides by 5. The module's
# > output tops out near +7 V (seen flat at 2320 mV on the old /3 divider, S20), which is now ~2710 mV at the pin,
# > below the ADC's 3100 mV: clipping must be judged against the module's limit, not the ADC's.
BIAS_CIRCUIT_MARK = "bias to D32"     # text record_session.py writes into metadata["values"] for this circuit
BIASED_CEILING_MV = 2710.0            # +7 V / 5 + 1.32 V


# ---------------------------------------------------------------------------
# ceiling_for(values_note)
# What:  the clipping ceiling (mV at the pin) for the circuit a recording was made
#        with: 2710 mV for the biased /5 circuit, else the ADC's 3100 mV.
# Time:  O(len(note)).  Space: O(1).
# Data structures: str search.  Files used: none (the caller reads metadata.json).
# ---------------------------------------------------------------------------
def ceiling_for(values_note: str) -> float:
    """Clipping ceiling in mV for this recording's circuit."""
    return BIASED_CEILING_MV if BIAS_CIRCUIT_MARK in (values_note or "") else CONFIG.ceiling_mv


# ---------------------------------------------------------------------------
# Cue
# What:  one phase cue from events.csv: when it started and which phase it is.
# Data structures: a frozen dataclass with two fields (a cue is a record of the
#        past and should never be edited).
# Time:  O(1) to build (fixed fields).
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Cue:
    """One phase cue: start time on the ESP32 clock and the phase name."""

    t_us: int        # when the phase started, in ESP32 micros() (an estimate, good to tens of ms)
    event: str       # BASELINE, CONCENTRIC, ISOMETRIC_HOLD, ECCENTRIC, REST or TAIL


# ---------------------------------------------------------------------------
# Recording
# What:  everything the check needs from one recording folder, held in memory.
# Data structures:
#   t_us       NumPy int64 array: one ESP32 timestamp per sample (the authoritative clock).
#   signal_mv  NumPy float64 array of the same length: the chosen channel in millivolts.
#              Two parallel arrays (index i = sample i), so NumPy can work on all at once.
#   cues       list of Cue, sorted by time (walked in order).
#   baseline_s seconds of real rest the recorder kept (the countdown is not included).
# Time:  O(1) to build (references to the arrays, no copies).
# ---------------------------------------------------------------------------
@dataclass
class Recording:
    """One recording: timestamps, one channel's signal, the phase cues and the baseline length."""

    t_us: np.ndarray
    signal_mv: np.ndarray
    cues: list[Cue]
    baseline_s: float


# ---------------------------------------------------------------------------
# QualityReport
# What:  all TC01 numbers, the three PASS/FAIL verdicts and the two warnings.
# Data structures: a dataclass (one named field per number, so the printout and
#        quality.json explain themselves); rep_ratios is a list with one value per
#        rep; `passed` is a computed property.
# Time:  O(R) to build for the R per-rep ratios; passed is O(1).
# ---------------------------------------------------------------------------
@dataclass
class QualityReport:
    """TC01 results for one recording."""

    rate_hz: float               # measured samples per second (from t_us)
    rest_mean_mv: float          # info: average raw level at rest (offset + hum + floor)
    rest_drift_mv: float         # check 1: 2nd-half mean minus 1st-half mean of the rest
    rest_halves_ratio: float     # check 1: larger / smaller muscle-band RMS of the two rest halves
    rest_emg_rms_mv: float       # check 2: muscle-band RMS at rest (the noise)
    active_emg_rms_mv: float     # check 2: muscle-band RMS while squeezing (all reps together)
    snr_ratio: float             # check 2: active / rest
    snr_db: float                # check 2: the same ratio in decibels
    rep_ratios: list[float]      # info: each rep's muscle-band RMS / rest (repeatability)
    clipped_samples: int         # check 3: samples within the margin of the ceiling
    longest_flat_top: int        # check 3: longest run of clipped samples in a row
    hum_rms_mv: float            # warning: RMS of the mains bands in the rest
    hum_share: float             # info: hum power / all rest power
    hum_to_muscle: float         # warning: hum RMS / active muscle-band RMS
    strongest_hz: float          # info: frequency with the most power in the rest
    rest_floor_share: float      # warning: fraction of rest samples on the ADC floor
    baseline_ok: bool
    snr_ok: bool
    clipping_ok: bool
    hum_warning: bool
    floor_warning: bool

    # Time:  O(1).
    # Data structures: three bools combined with `and`.
    @property
    def passed(self) -> bool:
        """TC01 passes if the three checks pass (warnings do not fail it)."""
        return self.baseline_ok and self.snr_ok and self.clipping_ok


# ---------------------------------------------------------------------------
# load_recording(folder, channel)
# What:  reads the three files record_session.py wrote and returns a Recording.
# Time:  O(N + E log E): one pass over the N rows of emg.csv, one pass over the E
#        rows of events.csv, then sorting the cues (E is about 20).
# Space: O(N + E).
# Data structures: Python lists while reading (append is O(1) on average, N is not
#        known in advance), then NumPy arrays; each csv row is a dict (name -> text).
# Files used (all inside `folder`, written by ~/Downloads/record_session.py):
#   emg.csv        columns pc_time_s, t_us, ch1, ch2, ch3: we read t_us (ESP32 clock)
#                  and the chosen channel (millivolts at the pin).
#   events.csv     columns pc_time_s, t_us, source, event, rep: we keep source == "cue"
#                  rows (the phases); SET_START, SET_END and LED_ON are markers.
#   metadata.json  we read baseline_s (seconds of real rest before the countdown).
# ---------------------------------------------------------------------------
def load_recording(folder: Path, channel: str = CONFIG.channel) -> Recording:
    """Load emg.csv, events.csv and metadata.json from one recording folder."""
    timestamps_us: list[int] = []
    samples_mv: list[float] = []
    # > skipinitialspace=True ignores a space after each comma, so "t_us" and " t_us" both work
    with (folder / "emg.csv").open(newline="") as emg_file:
        for row in csv.DictReader(emg_file, skipinitialspace=True):
            timestamps_us.append(int(row["t_us"]))
            samples_mv.append(float(row[channel]))

    cues: list[Cue] = []
    with (folder / "events.csv").open(newline="") as events_file:
        for row in csv.DictReader(events_file, skipinitialspace=True):
            if row["source"] == CUE_SOURCE:
                # > cue times are estimates and may be written as "12345.0", so go through float first
                cues.append(Cue(t_us=int(float(row["t_us"])), event=row["event"]))
    cues.sort(key=lambda cue: cue.t_us)                 # > the windows below assume time order

    metadata = json.loads((folder / "metadata.json").read_text())
    return Recording(
        t_us=np.array(timestamps_us, dtype=np.int64),   # > int64: micros() goes past the 32-bit signed limit
        signal_mv=np.array(samples_mv, dtype=np.float64),
        cues=cues,
        baseline_s=float(metadata["baseline_s"]),
    )


# ---------------------------------------------------------------------------
# rest_mask(t_us, cues, baseline_s)
# What:  marks the real rest: from the BASELINE cue for baseline_s seconds. The
#        3-2-1 countdown after it is left out (you may tense up while counting).
# Time:  O(N + E): one walk of the cues, one comparison per sample.
# Space: O(N): one True/False per sample.
# Data structures: a boolean NumPy mask; signal[mask] picks the rest samples at once.
# Files used: none.
# ---------------------------------------------------------------------------
def rest_mask(t_us: np.ndarray, cues: list[Cue], baseline_s: float) -> np.ndarray:
    """Boolean mask of the samples inside the relaxed baseline (countdown excluded)."""
    # > next(...) gives the first BASELINE cue's time, or None if there is none
    start_us = next((cue.t_us for cue in cues if cue.event == BASELINE_CUE), None)
    if start_us is None:
        raise ValueError("events.csv has no BASELINE cue, so the rest period is unknown")
    end_us = start_us + round(baseline_s * US_PER_S)    # > round(): whole microseconds, no float error
    return (t_us >= start_us) & (t_us < end_us)          # > & = element-wise AND of two boolean arrays


# ---------------------------------------------------------------------------
# rep_windows(cues, reaction_delay_s, last_t_us)
# What:  one (start_us, end_us) window per rep: from the rep's first active cue
#        (plus the reaction delay) to the end of its last active phase. Consecutive
#        active cues (SQUEEZE, HOLD, RELEASE or DOWN, HOLD, UP, in any order) form one
#        rep; a REST/TAIL cue ends it. Each window is one unbroken stretch of samples,
#        which the Fourier transform needs (gluing reps together would add jumps).
# Time:  O(E): one walk of the E cues.
# Space: O(R) for R reps.
# Data structures: a list of (start, end) tuples; tuples because a window never
#        changes once found.
# Files used: none.
# ---------------------------------------------------------------------------
def rep_windows(cues: list[Cue], reaction_delay_s: float, last_t_us: int) -> list[tuple[int, int]]:
    """(start_us, end_us) of every rep's active stretch, reaction delay skipped."""
    delay_us = round(reaction_delay_s * US_PER_S)
    windows: list[tuple[int, int]] = []
    start_us: int | None = None
    for index, cue in enumerate(cues):
        is_active = cue.event in ACTIVE_PHASES
        if is_active and start_us is None:                # > first active phase of a rep
            start_us = cue.t_us + delay_us
        has_next = index + 1 < len(cues)
        next_is_active = has_next and cues[index + 1].event in ACTIVE_PHASES
        if is_active and not next_is_active:              # > last active phase of the rep: close it
            end_us = cues[index + 1].t_us if has_next else last_t_us + 1
            windows.append((start_us, end_us))
            start_us = None
    return windows


# ---------------------------------------------------------------------------
# baseline_drift(rest_values)
# What:  mean of the second half of the rest minus mean of the first half: about 0
#        for a steady baseline, clearly + or - for a slow wander (loose electrode,
#        drying gel, sagging battery).
# Time:  O(n).  Space: O(1) extra (slices are views, not copies).
# Data structures: a 1-D NumPy array in, one float out.
# Files used: none.
# ---------------------------------------------------------------------------
def baseline_drift(rest_values: np.ndarray) -> float:
    """Second-half mean minus first-half mean of the rest samples (mV)."""
    half = len(rest_values) // 2
    if half == 0:
        raise ValueError("need at least 2 rest samples to measure drift")
    first_mean = float(np.sum(rest_values[:half])) / half
    second_mean = float(np.sum(rest_values[half:])) / (len(rest_values) - half)
    return second_mean - first_mean


# ---------------------------------------------------------------------------
# longest_true_run(flags)
# What:  length of the longest unbroken run of True values; on the "clipped" mask,
#        many clipped samples in a row = a flat top.
# Time:  O(n): one pass.  Space: O(1): two counters.
# Data structures: walks a boolean NumPy array with a plain loop.
# Files used: none.
# ---------------------------------------------------------------------------
def longest_true_run(flags: np.ndarray) -> int:
    """Length of the longest run of consecutive True values."""
    longest = 0
    current = 0
    for flag in flags:
        current = current + 1 if flag else 0             # > extend the run, or reset it on a False
        longest = max(longest, current)
    return longest


# ---------------------------------------------------------------------------
# measured_rate_hz(t_us)
# What:  real samples per second: (N - 1 gaps) / (time from first to last sample).
# Time:  O(1).  Space: O(1).
# Data structures: a NumPy int64 array in, one float out.
# Files used: none.
# ---------------------------------------------------------------------------
def measured_rate_hz(t_us: np.ndarray) -> float:
    """Samples per second measured from the first and last timestamp."""
    duration_s = (int(t_us[-1]) - int(t_us[0])) / US_PER_S
    if duration_s <= 0:
        raise ValueError("the timestamps do not increase, so the rate is unknown")
    return (len(t_us) - 1) / duration_s


# ---------------------------------------------------------------------------
# check_quality(recording, cfg)
# What:  runs TC01 and returns a QualityReport.
#   1. Baseline: |drift| <= max_rest_drift_mv, and the muscle-band RMS of the two
#      halves of the rest differ by at most max_rest_halves_ratio (a burst of
#      tension or movement during the rest makes one half much bigger).
#   2. SNR: active / rest muscle-band RMS >= 3. "Active" pools all reps: each rep's
#      mean square weighted by its length, then the square root, i.e. the RMS of
#      all active samples together. Each rep's own ratio is also reported.
#   3. Clipping: no sample within clip_margin_mv of the ceiling.
#   Warnings: hum RMS > max_hum_to_muscle x active muscle RMS; more than
#      max_floor_share of the rest on the ADC floor.
# Time:  O(N + R^2 + A^2): masks and clipping are O(N) for N samples; each DFT is
#        quadratic in its window (R = 2500 rest samples, A ~ 2350 per rep). About
#        half a second for a 3-rep set.
# Space: O(N).
# Data structures: boolean masks; a list of per-rep sample arrays; the rest's
#        spectrum is computed once and reused for the band, hum and peak values.
# Files used: none directly (load_recording reads the files; the maths is in
#        src/myomotion/semg/dsp.py: power_spectrum, emg_band_rms,
#        emg_band_rms_from_spectrum, mains_rms_from_spectrum).
# ---------------------------------------------------------------------------
def check_quality(recording: Recording, cfg: QualityConfig = CONFIG) -> QualityReport:
    """Run TC01 (baseline, SNR, clipping in the EMG band; hum and floor warnings) on one recording."""
    signal, t_us = recording.signal_mv, recording.t_us
    rate_hz = measured_rate_hz(t_us)
    rest = signal[rest_mask(t_us, recording.cues, recording.baseline_s)]
    windows = rep_windows(recording.cues, cfg.reaction_delay_s, int(t_us[-1]))
    reps = [signal[(t_us >= start) & (t_us < end)] for start, end in windows]
    reps = [rep for rep in reps if len(rep) >= 2]
    if len(rest) < 4 or not reps:
        raise ValueError("the rest or the reps are empty: check events.csv and baseline_s")

    # Rest: one spectrum, reused for the muscle band, the hum and the strongest frequency
    rest_freqs, rest_power = power_spectrum(rest, rate_hz)
    rest_emg = emg_band_rms_from_spectrum(rest_freqs, rest_power, rate_hz, cfg.dsp)
    hum_rms, hum_share = mains_rms_from_spectrum(rest_freqs, rest_power, rate_hz, cfg.dsp)
    strongest_hz = float(rest_freqs[int(np.argmax(rest_power))])   # > argmax = index of the largest power

    # 1. Baseline
    drift = baseline_drift(rest)
    half = len(rest) // 2
    half_rms = (emg_band_rms(rest[:half], rate_hz, cfg.dsp), emg_band_rms(rest[half:], rate_hz, cfg.dsp))
    halves_ratio = max(half_rms) / min(half_rms) if min(half_rms) > 0 else math.inf

    # 2. SNR in the muscle band
    rep_rms = [emg_band_rms(rep, rate_hz, cfg.dsp) for rep in reps]
    total_samples = sum(len(rep) for rep in reps)
    # > pool the reps: mean square of all active samples = length-weighted mean of each rep's mean square
    active_emg = math.sqrt(sum(value ** 2 * len(rep) for value, rep in zip(rep_rms, reps)) / total_samples)
    snr_ratio = active_emg / rest_emg if rest_emg > 0 else math.inf
    snr_db = 20 * math.log10(snr_ratio)                  # > 20, not 10: RMS is an amplitude, not a power
    rep_ratios = [value / rest_emg if rest_emg > 0 else math.inf for value in rep_rms]

    # 3. Clipping at the top
    clipped = signal >= cfg.ceiling_mv - cfg.clip_margin_mv
    clipped_samples = int(np.sum(clipped))

    # Warnings
    hum_to_muscle = hum_rms / active_emg if active_emg > 0 else math.inf
    floor_share = float(np.sum(rest <= cfg.floor_mv)) / len(rest)   # > True counts as 1 when summed

    return QualityReport(
        rate_hz=rate_hz,
        rest_mean_mv=float(np.sum(rest)) / len(rest),
        rest_drift_mv=drift,
        rest_halves_ratio=halves_ratio,
        rest_emg_rms_mv=rest_emg,
        active_emg_rms_mv=active_emg,
        snr_ratio=snr_ratio,
        snr_db=snr_db,
        rep_ratios=rep_ratios,
        clipped_samples=clipped_samples,
        longest_flat_top=longest_true_run(clipped),
        hum_rms_mv=hum_rms,
        hum_share=hum_share,
        hum_to_muscle=hum_to_muscle,
        strongest_hz=strongest_hz,
        rest_floor_share=floor_share,
        baseline_ok=abs(drift) <= cfg.max_rest_drift_mv and halves_ratio <= cfg.max_rest_halves_ratio,
        snr_ok=snr_ratio >= cfg.min_snr_ratio,
        clipping_ok=clipped_samples <= cfg.max_clipped_samples,
        hum_warning=hum_to_muscle > cfg.max_hum_to_muscle,
        floor_warning=floor_share > cfg.max_floor_share,
    )


# ---------------------------------------------------------------------------
# format_report(report, cfg)
# What:  turns a QualityReport into readable lines for the terminal.
# Time:  O(R) for R reps (the per-rep list); otherwise a fixed number of lines.
# Space: O(R).
# Data structures: a list of strings joined with newlines (cheaper than adding
#        strings together one by one).
# Files used: none (scripts/semg/check_quality.py prints the result).
# ---------------------------------------------------------------------------
def format_report(report: QualityReport, cfg: QualityConfig = CONFIG) -> str:
    """Readable TC01 summary: PASS/FAIL per check, OK/WARN per warning."""
    # Time:  O(1).
    # Data structures: a bool in, a str out.
    def verdict(ok: bool) -> str:
        return "PASS" if ok else "FAIL"

    # Time:  O(1).
    # Data structures: a bool in, a str out.
    def warn(flag: bool) -> str:
        return "WARN" if flag else "OK  "

    band = f"{cfg.dsp.highpass_hz:.0f}-{report.rate_hz / 2:.0f} Hz without mains"
    reps = ", ".join(f"{ratio:.1f}x" for ratio in report.rep_ratios)
    lines = [
        f"Sample rate    {report.rate_hz:.1f} samples/s; muscle band = {band}",
        f"1. Baseline    {verdict(report.baseline_ok)}: drift {report.rest_drift_mv:+.1f} mV "
        f"(max ±{cfg.max_rest_drift_mv:.0f}), rest halves differ {report.rest_halves_ratio:.2f}x "
        f"(max {cfg.max_rest_halves_ratio:.0f}x)",
        f"2. SNR         {verdict(report.snr_ok)}: muscle band active {report.active_emg_rms_mv:.1f} mV / "
        f"rest {report.rest_emg_rms_mv:.1f} mV = {report.snr_ratio:.2f} ({report.snr_db:.1f} dB, "
        f"need ≥ {cfg.min_snr_ratio:.0f}); per rep: {reps}",
        f"3. Clipping    {verdict(report.clipping_ok)}: {report.clipped_samples} samples near "
        f"{cfg.ceiling_mv:.0f} mV, longest flat top {report.longest_flat_top} samples",
        f"4. 50 Hz hum   {warn(report.hum_warning)}: {report.hum_rms_mv:.1f} mV RMS = "
        f"{report.hum_to_muscle:.2f}x the muscle signal (warn above {cfg.max_hum_to_muscle}x), "
        f"{report.hum_share:.0%} of rest power; strongest {report.strongest_hz:.1f} Hz",
        f"5. ADC floor   {warn(report.floor_warning)}: {report.rest_floor_share:.0%} of rest samples "
        f"at ≤ {cfg.floor_mv:.0f} mV (warn above {cfg.max_floor_share:.0%}); raw rest mean "
        f"{report.rest_mean_mv:.0f} mV",
        f"TC01           {verdict(report.passed)}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# save_report(report, folder)
# What:  writes the numbers to quality.json in the recording folder.
# Time:  O(R) for the per-rep list.  Space: O(R).
# Data structures: dataclasses.asdict turns the report into a dict that json can
#        write; the computed "passed" is added.
# Files used: writes <folder>/quality.json (next to emg.csv).
# ---------------------------------------------------------------------------
def save_report(report: QualityReport, folder: Path) -> Path:
    """Write the report to <folder>/quality.json and return its path."""
    contents = asdict(report)
    contents["passed"] = report.passed
    out_path = folder / "quality.json"
    out_path.write_text(json.dumps(contents, indent=2))
    return out_path
