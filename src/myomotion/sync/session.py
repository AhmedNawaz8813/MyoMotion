"""One recording folder + its video -> sync.json and merged.csv (one row per video frame).

Two ways to line up the ESP32 clock with the video clock:
  led     the sync LED's flashes, seen in the video and reported by the ESP32 (LED_ON events): works for any
          camera (phone or webcam), and measures TC03 (src/myomotion/sync/led_sync.py, align.py)
  clock   no LED: a webcam recording from scripts/sync/record_synced.py has the laptop's arrival time of every
          sample and frame, so both clocks are tied to the laptop clock (src/myomotion/sync/clock_sync.py)
  auto    (default) the LED when it is found, else the laptop clock; with both, the LED also re-measures the
          laptop clock's camera delay (sync.json "clock_check")

Inputs, all from one set:
  <folder>/emg.csv, events.csv, metadata.json   from record_session.py or record_synced.py (sEMG, cues, LED_ON)
  the phone video of the same set               default: <folder>/video.* (copy it there from the phone)
  mvc.json                                      optional: turns the envelope into %MVC
  Ahmad's deadlift summary JSON                 optional: rep phases seen in the video (cv_summary.json from
                                                src/myomotion/sync/cv_frames.py, or demo_deadlift.py's summary)
  a per-frame CV CSV with a t_s column          optional: its columns are joined onto each frame
                                                (cv_frames.csv from src/myomotion/sync/cv_frames.py)

Outputs, written into <folder>:
  merged.csv   frame, t_video_s, t_esp32_us, envelope_mv, percent_mvc, phase, rep [, cv_* columns]
               = X(t) and Y(t) side by side, one row per video frame (the dataset row format)
  sync.json    the method, the clock line, every flash on both clocks, the TC03 result, the per-phase %MVC

Data structures (this file): a SyncResult dataclass (NumPy arrays per sample and per frame, lists of intervals and
    statistics); a dict ms -> CV row for O(1) joins; a csv.writer; nested dicts and lists for sync.json.
Time complexity (this file): the LED search decodes the video (O(T * P)) and the envelope costs O(N * S) (N samples,
    S = 6 filter sections); the laptop-clock sync is O(N + F); merging is O(N + F) (two pointers) and writing
    merged.csv O(F * C) for C columns.
"""

from __future__ import annotations   # > type hints are stored as text, not run: allows forward references

import csv
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from myomotion.semg.dsp import emg_envelope
from myomotion.semg.mvc import PhaseStat, load_mvc, normalise_set
from myomotion.semg.quality import Recording, load_recording, measured_rate_hz
from myomotion.sync.align import (SYNC_CONFIG, ClockFit, PhaseInterval, SyncConfig, SyncReport, check_sync, fit_clock,
                                  interpolate_sorted, intervals_from_cues, intervals_from_cv, label_frames,
                                  phase_stats)
from myomotion.sync.clock_sync import CLOCK_CONFIG, ClockSyncConfig, LaptopClock, clock_check_ms, laptop_clock
from myomotion.sync.led_sync import LED_CONFIG, LedConfig, LedNotFound, LedTrace, find_led, median_step

VIDEO_SUFFIXES = (".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm")
BOARD_SOURCE = "board"               # events.csv rows written by the ESP32 itself
LED_EVENT = "LED_ON"                 # the ESP32's reply to "S": the moment the LED came on
MERGED_NAME = "merged.csv"
FRAMES_NAME = "video_frames.csv"     # written by the synced recorder: the camera's own time of every frame
SYNC_NAME = "sync.json"
SYNC_METHODS = ("auto", "led", "clock")
NOTES = (
    "sEMG rises about 50-100 ms before the movement is visible (electromechanical delay): physiology, not a "
    "sync error.",
    "Phase times from the CV summary are detection times; Ahmad's detector is causal and lags the real "
    "movement by a few hundred ms.",
)


# ---------------------------------------------------------------------------
# SyncResult
# What:  everything one sync run produced, for the script's printout and plot.
# Data structures: dataclass of NumPy arrays (per sample / per frame) and lists.
# Time:  O(1) to build (references to the arrays, no copies).
# ---------------------------------------------------------------------------
@dataclass
class SyncResult:
    """Result of syncing one recording with its video."""

    report: SyncReport
    fit: ClockFit
    led: LedTrace | None             # None when the laptop clock was used
    esp32_flash_us: list[int]
    sample_video_s: np.ndarray       # every sEMG sample's time on the video clock
    envelope_mv: np.ndarray          # per sample
    percent: np.ndarray | None       # per sample, None without an MVC
    intervals: list[PhaseInterval]
    phase_source: str                # "cv" (seen in the video) or "cues..." (what the recorder asked for)
    phases: list[PhaseStat] = field(default_factory=list)
    frames_with_emg: int = 0
    merged_path: Path | None = None
    json_path: Path | None = None
    method: str = "led"                                  # "led" or "clock"
    frame_times_s: np.ndarray | None = None              # every frame's time on the video clock
    frame_interval_s: float = 0.0
    clock: LaptopClock | None = None                     # the laptop-clock lines (webcam recordings)
    clock_check: tuple[float, ...] = ()                  # laptop-clock error at each LED flash, ms
    led_problem: str | None = None                       # why the LED was not used (auto fell back)


# ---------------------------------------------------------------------------
# find_video(folder)
# What:  the set's video inside its folder: video.<ext> first, else the only video file.
# Time:  O(F) for F files in the folder.  Space: O(F).
# Data structures: sorted list of Paths.  Files used: lists <folder>.
# ---------------------------------------------------------------------------
def find_video(folder: Path) -> Path | None:
    """The video file in a recording folder, or None."""
    videos = sorted(p for p in folder.iterdir() if p.suffix.lower() in VIDEO_SUFFIXES and p.is_file())
    named = [p for p in videos if p.stem.lower() == "video"]
    if named:
        return named[0]
    return videos[0] if len(videos) == 1 else None


# ---------------------------------------------------------------------------
# load_led_events(folder)
# What:  the ESP32's LED_ON times (unwrapped micros(), the same clock as emg.csv).
# Time:  O(E) for E events.  Space: O(L) flashes.
# Data structures: sorted list of ints.  Files used: reads <folder>/events.csv.
# ---------------------------------------------------------------------------
def load_led_events(folder: Path) -> list[int]:
    """ESP32 times (us) of every sync flash, in order."""
    with (folder / "events.csv").open(newline="") as events_file:
        rows = csv.DictReader(events_file, skipinitialspace=True)
        return sorted(int(float(row["t_us"])) for row in rows
                      if row["source"] == BOARD_SOURCE and row["event"] == LED_EVENT)


# ---------------------------------------------------------------------------
# load_frame_times(video)
# What:  the camera's timestamp of every frame (seconds from the first frame) when the
#        synced recorder saved them next to the video (video_frames.csv); else None and
#        the times are read from the video file itself.
# Time:  O(F).  Space: O(F).
# Data structures: NumPy array.  Files used: reads <video folder>/video_frames.csv if present.
# ---------------------------------------------------------------------------
def load_frame_times(video: Path) -> np.ndarray | None:
    """Frame times from video_frames.csv, or None."""
    path = Path(video).parent / FRAMES_NAME
    if not path.exists():
        return None
    with path.open(newline="") as frames_file:
        times = [float(row["t_camera_s"]) for row in csv.DictReader(frames_file)]
    return np.array(times) if len(times) >= 2 else None


# ---------------------------------------------------------------------------
# load_cv_frames(path)
# What:  a per-frame CV CSV (any columns, one of them t_s on the video clock), keyed
#        by its time in whole milliseconds so frames can be matched without float
#        rounding trouble.
# Time:  O(R) rows.  Space: O(R).
# Data structures: dict  ms -> row dict  (O(1) lookup per frame); column list.
# Files used: reads the given CSV.
# ---------------------------------------------------------------------------
def load_cv_frames(path: Path) -> tuple[list[str], dict[int, dict[str, str]]]:
    """(column names without t_s, {time in ms: row})."""
    with path.open(newline="") as cv_file:
        reader = csv.DictReader(cv_file, skipinitialspace=True)
        if not reader.fieldnames or "t_s" not in reader.fieldnames:
            raise ValueError(f"{path} has no t_s column")
        columns = [name for name in reader.fieldnames if name != "t_s"]
        return columns, {round(float(row["t_s"]) * 1000): row for row in reader}


# ---------------------------------------------------------------------------
# envelope_and_percent(recording, mvc_path)
# What:  the sEMG envelope per sample, and %MVC when an MVC is given (the set's own
#        baseline is its 0%, as in scripts/semg/percent_mvc.py).
# Time:  O(N * S) for the envelope filter.  Space: O(N).
# Data structures: NumPy arrays.
# Files used: src/myomotion/semg/mvc.py (load_mvc, normalise_set), src/myomotion/semg/dsp.py.
# ---------------------------------------------------------------------------
def envelope_and_percent(recording: Recording, mvc_path: Path | None) -> tuple[np.ndarray, np.ndarray | None]:
    """(envelope in mV, %MVC or None)."""
    if mvc_path is None:
        return emg_envelope(recording.signal_mv, measured_rate_hz(recording.t_us)), None
    normalised = normalise_set(recording, float(load_mvc(mvc_path)["mvc_above_rest_mv"]))
    return normalised.envelope_mv, normalised.percent


# ---------------------------------------------------------------------------
# write_merged(path, frame_times_s, fit, envelope_at, percent_at, phases, reps, cv)
# What:  merged.csv, one row per video frame. Cells are blank where there is no
#        value (a frame filmed before the sEMG started, no MVC, no CV row).
# Time:  O(F * C) for F frames and C columns.  Space: O(C) per row.
# Data structures: csv.writer; dict lookup for the CV rows.
# Files used: writes <folder>/merged.csv.
# ---------------------------------------------------------------------------
def write_merged(path: Path, frame_times_s: np.ndarray, fit: ClockFit, envelope_at: np.ndarray,
                 percent_at: np.ndarray | None, phases: list[str], reps: list[int],
                 cv: tuple[list[str], dict[int, dict[str, str]]] | None) -> None:
    """Write merged.csv."""
    cv_columns = cv[0] if cv else []
    with path.open("w", newline="") as merged_file:
        writer = csv.writer(merged_file)
        writer.writerow(["frame", "t_video_s", "t_esp32_us", "envelope_mv", "percent_mvc", "phase", "rep"]
                        + [f"cv_{name}" for name in cv_columns])
        esp32_us = fit.to_esp32_us(frame_times_s)
        for frame, t_s in enumerate(frame_times_s):
            has_emg = not math.isnan(envelope_at[frame])
            row = [frame, f"{t_s:.4f}", int(round(esp32_us[frame])),
                   f"{envelope_at[frame]:.3f}" if has_emg else "",
                   f"{percent_at[frame]:.2f}" if has_emg and percent_at is not None else "",
                   phases[frame], reps[frame] or ""]
            if cv:
                key = round(float(t_s) * 1000)
                match = cv[1].get(key) or cv[1].get(key - 1) or cv[1].get(key + 1)   # > allow 1 ms of rounding
                row += [match.get(name, "") if match else "" for name in cv_columns]
            writer.writerow(row)


# ---------------------------------------------------------------------------
# find_led_flashes(video, flash_us, frame_times, box_px, cfg)
# What:  the LED search for the ESP32's LED_ON events (at least 2 are needed).
# Time / Space: as find_led in src/myomotion/sync/led_sync.py.
# Data structures: LedTrace.  Files used: reads the video.
# ---------------------------------------------------------------------------
def find_led_flashes(video: Path, flash_us: list[int], frame_times: np.ndarray | None,
                     box_px: tuple[int, int, int, int] | None, cfg: LedConfig) -> LedTrace:
    """The LED's flashes in the video, matched to the LED_ON events."""
    if len(flash_us) < 2:
        raise ValueError(f"{len(flash_us)} LED_ON events; the LED sync needs at least 2 (the recorder flashes at "
                         "the start, middle and end of every set)")
    expected_gaps = [(b - a) / 1e6 for a, b in zip(flash_us, flash_us[1:])]
    return find_led(video, count=len(flash_us), expected_gaps_s=expected_gaps, box_px=box_px, cfg=cfg,
                    frame_times=frame_times)


# ---------------------------------------------------------------------------
# sync_recording(folder, video, cv_json, cv_frames, mvc_path, channel, box_px, method, ...)
# What:  the whole sync of one set:
#        clock line (LED flashes -> fit -> TC03, or the laptop clock) -> envelope / %MVC
#        on the video clock -> phases (CV or cues) -> merged.csv and sync.json.
# Time:  dominated by decoding the video twice (O(T * P)) and the envelope (O(N * S)).
# Space: O(T * B) for the LED search, O(N + F) for the signals.
# Data structures: see SyncResult.
# Files used: reads <folder>/emg.csv, events.csv, metadata.json, the video, mvc.json and the
#        CV files if given; writes <folder>/merged.csv and sync.json. Uses
#        src/myomotion/sync/led_sync.py, clock_sync.py, align.py and src/myomotion/semg/*.
# ---------------------------------------------------------------------------
def sync_recording(folder: Path, video: Path, cv_json: Path | None = None, cv_frames: Path | None = None,
                   mvc_path: Path | None = None, channel: str = "ch1",
                   box_px: tuple[int, int, int, int] | None = None, led_cfg: LedConfig = LED_CONFIG,
                   sync_cfg: SyncConfig = SYNC_CONFIG, method: str = "auto",
                   clock_cfg: ClockSyncConfig = CLOCK_CONFIG) -> SyncResult:
    """Sync one recording with its video and write merged.csv and sync.json."""
    if method not in SYNC_METHODS:
        raise ValueError(f"method must be one of {SYNC_METHODS}, not {method!r}")
    recording = load_recording(folder, channel)
    flash_us = load_led_events(folder)
    frame_times = load_frame_times(video)
    clock = None
    if frame_times is not None and method != "led":
        # > a webcam recording: both streams carry the laptop's arrival times
        try:
            clock = laptop_clock(folder, clock_cfg)
        except ValueError:
            if method == "clock":
                raise
    led, led_problem = None, None
    if method != "clock":
        try:
            led = find_led_flashes(video, flash_us, frame_times, box_px, led_cfg)
        except (LedNotFound, ValueError) as error:
            if clock is None:
                raise
            led_problem = str(error)

    if led is not None:
        video_flash_s = [flash.t_s for flash in led.flashes]
        fit = fit_clock(flash_us, video_flash_s)
        report = check_sync(flash_us, video_flash_s, fit, led.frame_interval_s, sync_cfg)
        frame_times_s, frame_interval_s, used = led.times_s, led.frame_interval_s, "led"
        check = clock_check_ms(clock, flash_us, video_flash_s) if clock is not None else ()
    else:
        fit = clock.to_clock_fit()
        frame_times_s, frame_interval_s, used, check = frame_times, median_step(frame_times), "clock", ()
        # > no flash to check against: TC03 is NOT MEASURED; the accuracy is the LED-checked one
        report = SyncReport(flashes=0, esp32_s_at_video_start=fit.esp32_s_at_video_start, drift_ppm=fit.drift_ppm,
                            check_errors_ms=(), max_check_error_ms=sync_cfg.max_check_error_ms,
                            drift_plausible=abs(fit.drift_ppm) <= sync_cfg.max_drift_ppm)

    envelope, percent = envelope_and_percent(recording, mvc_path)
    sample_video_s = np.asarray(fit.to_video_s(recording.t_us))
    intervals, source = [], "cues"
    if cv_json is not None:
        intervals, source = intervals_from_cv(json.loads(Path(cv_json).read_text())), "cv"
    if not intervals:
        # > no CV summary, or the CV counted no reps (e.g. a forearm test, or the legs out of view):
        # > fall back to what the recorder asked for, so merged.csv still has phases
        source = "cues" if cv_json is None else "cues (the CV counted no reps)"
        intervals = intervals_from_cues(recording.cues, int(recording.t_us[-1]), fit)

    envelope_at = interpolate_sorted(sample_video_s, envelope, frame_times_s)
    percent_at = interpolate_sorted(sample_video_s, percent, frame_times_s) if percent is not None else None
    phases, reps = label_frames(frame_times_s, intervals)
    cv = load_cv_frames(Path(cv_frames)) if cv_frames is not None else None
    merged_path = folder / MERGED_NAME
    write_merged(merged_path, frame_times_s, fit, envelope_at, percent_at, phases, reps, cv)

    result = SyncResult(report=report, fit=fit, led=led, esp32_flash_us=flash_us if led is not None else [],
                        sample_video_s=sample_video_s, envelope_mv=envelope, percent=percent, intervals=intervals,
                        phase_source=source,
                        phases=phase_stats(sample_video_s, percent, intervals) if percent is not None else [],
                        frames_with_emg=int(np.sum(~np.isnan(envelope_at))), merged_path=merged_path, method=used,
                        frame_times_s=frame_times_s, frame_interval_s=frame_interval_s, clock=clock,
                        clock_check=check, led_problem=led_problem)
    result.json_path = save_sync_json(result, folder, video, cv_json, cv_frames, mvc_path, channel)
    return result


# ---------------------------------------------------------------------------
# save_sync_json(result, folder, video, cv_json, cv_frames, mvc_path, channel)
# What:  sync.json: the inputs, the clock line, every flash on both clocks, TC03,
#        the LED box, frame counts and the per-phase %MVC.
# Time:  O(F + P).  Space: O(F + P).
# Data structures: nested dicts / lists -> JSON.  Files used: writes <folder>/sync.json.
# ---------------------------------------------------------------------------
def save_sync_json(result: SyncResult, folder: Path, video: Path, cv_json: Path | None, cv_frames: Path | None,
                   mvc_path: Path | None, channel: str) -> Path:
    """Write sync.json and return its path."""
    report, fit, led = result.report, result.fit, result.led
    contents = {
        "test": "TC03 video-sEMG sync",
        "method": result.method,
        "verdict": report.verdict,
        "inputs": {"emg_folder": str(folder), "video": str(video), "cv_json": str(cv_json) if cv_json else None,
                   "cv_frames": str(cv_frames) if cv_frames else None, "mvc": str(mvc_path) if mvc_path else None,
                   "channel": channel},
        "clock": {"formula": "t_video_s = video_ref_s + slope * (t_esp32_us - esp32_ref_us) / 1e6",
                  "video_ref_s": fit.video_ref_s, "esp32_ref_us": fit.esp32_ref_us, "slope": fit.slope,
                  "drift_ppm": fit.drift_ppm, "drift_ms_per_minute": fit.drift_ppm * 60e-3,
                  "esp32_s_at_video_start": fit.esp32_s_at_video_start},
        "flashes": [{"esp32_us": t_us, "video_s": flash.t_s, "frame": flash.frame, "brightness_jump": flash.size,
                     "fit_error_ms": error}
                    for t_us, flash, error in zip(result.esp32_flash_us, led.flashes, fit.residuals_ms)]
        if led is not None else [],
        "check_errors_ms": list(report.check_errors_ms),
        "limits": {"max_check_error_ms": report.max_check_error_ms},
        "drift_plausible": report.drift_plausible,
        "led": {"box_px": list(led.box_px), "pixels": led.led_pixels} if led is not None else None,
        "led_problem": result.led_problem,
        "laptop_clock": clock_summary(result),
        "video": {"frames": len(result.frame_times_s), "frame_interval_s": result.frame_interval_s,
                  "frames_with_emg": result.frames_with_emg},
        "phase_source": result.phase_source,
        "phases": [asdict(stat) for stat in result.phases],
        "notes": list(NOTES),
    }
    path = folder / SYNC_NAME
    path.write_text(json.dumps(contents, indent=2))
    return path


# ---------------------------------------------------------------------------
# clock_summary(result)
# What:  the laptop-clock numbers for sync.json: both drifts, typical delays, the camera
#        delay used and, when the LED was also seen, the error at each flash and the
#        camera delay that implies (used + mean error).
# Time / Space: O(L) flashes.  Data structures: dict.  Files used: none.
# ---------------------------------------------------------------------------
def clock_summary(result: SyncResult) -> dict | None:
    """Laptop-clock part of sync.json, or None for a recording without laptop times."""
    clock = result.clock
    if clock is None:
        return None
    check = list(result.clock_check)
    return {"used_for_sync": result.method == "clock",
            "esp32_drift_vs_laptop_ppm": clock.esp32.slope * 1e6,
            "camera_drift_vs_laptop_ppm": clock.camera.slope * 1e6,
            "esp32_typical_delay_ms": clock.esp32.median_delay_ms,
            "camera_typical_delay_ms": clock.camera.median_delay_ms,
            "camera_delay_ms_used": clock.camera_delay_ms,
            "check_errors_ms_at_led_flashes": check,
            "camera_delay_ms_measured": clock.camera_delay_ms + sum(check) / len(check) if check else None}


# ---------------------------------------------------------------------------
# format_report(result)
# What:  readable TC03 summary for the terminal.
# Time:  O(F + P).  Space: O(F + P).
# Data structures: list of strings joined with newlines.  Files used: none.
# ---------------------------------------------------------------------------
def format_report(result: SyncResult) -> str:
    """Human-readable TC03 result."""
    report, fit, led, clock = result.report, result.fit, result.led, result.clock
    if led is not None:
        x, y, w, h = led.box_px
        lines = [
            f"TC03 video-sEMG sync: {report.verdict}  (method: LED flashes)",
            f"  flashes        {report.flashes} LED_ON events, {len(led.flashes)} found in the video "
            f"(LED box x={x} y={y} {w}x{h} px, {led.led_pixels} LED pixels)",
        ]
    else:
        lines = [
            "Video-sEMG sync: done with the LAPTOP CLOCK (no LED needed). TC03 is only measured when the LED is "
            f"seen; checked against the LED this method was within about {CLOCK_CONFIG.expected_error_ms:.0f} ms.",
        ]
        if result.led_problem:
            lines.append(f"  LED            not used: {result.led_problem.split('. ')[0]}")
    lines.append(f"  clock          video started at ESP32 time {fit.esp32_s_at_video_start:.3f} s; drift "
                 f"{fit.drift_ppm:+.0f} ppm ({fit.drift_ppm * 60e-3:+.2f} ms per minute)"
                 + ("" if report.drift_plausible else "  <- implausible: check the recording"))
    for index, error in enumerate(report.check_errors_ms, start=2):
        lines.append(f"  check flash {index}  {abs(error):.1f} ms from the line through the first and last flashes "
                     f"(limit {report.max_check_error_ms:.0f} ms)")
    if led is not None and not report.check_errors_ms:
        lines.append("  check          not measured: only 2 flashes (the recorder's middle flash gives the check)")
    if clock is not None:
        lines.append(f"  laptop clock   Wi-Fi delay {clock.esp32.median_delay_ms:.1f} ms typical, camera read "
                     f"{clock.camera.median_delay_ms:.1f} ms typical, "
                     f"camera delay correction {clock.camera_delay_ms:.1f} ms")
    if result.clock_check:
        worst = max(abs(error) for error in result.clock_check)
        measured = clock.camera_delay_ms + sum(result.clock_check) / len(result.clock_check)
        lines.append(f"  clock vs LED   the laptop clock is within {worst:.1f} ms of the LED at every flash "
                     f"(camera delay measured {measured:.1f} ms)")
    lines.append(f"  frames         {len(result.frame_times_s)} at {1 / result.frame_interval_s:.1f} fps, "
                 f"{result.frames_with_emg} with sEMG")
    source = ("the CV summary (seen in the video)" if result.phase_source == "cv"
              else f"the recorder {result.phase_source}")
    lines.append(f"  phases from    {source}")
    if result.phases:
        lines.append("  rep  phase            seconds  mean %MVC  peak %MVC")
        for stat in result.phases:
            lines.append(f"  {stat.rep:>3}  {stat.phase:<15} {stat.seconds:>8.2f}  {stat.mean_percent:>9.1f}  "
                         f"{stat.peak_percent:>9.1f}")
    return "\n".join(lines)
