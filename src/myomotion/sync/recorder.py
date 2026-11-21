"""Webcam video and sEMG recorded at the same time, with a live view (video-sEMG sync, TC03).

One program does what the phone and record_session.py did together:
  * films with the laptop webcam and reads the ESP32's sEMG over Wi-Fi, side by side;
  * shows both live: the camera image with the current cue (RELAX, SQUEEZE, ...), a %MVC bar and a
    5-second trace (src/myomotion/semg/live.py does the %MVC, sample by sample);
  * flashes the sync LED at the start, after the middle rep and at the end of the set;
  * saves the usual set folder (emg.csv, events.csv, metadata.json) plus video.mp4 and
    video_frames.csv, so scripts/sync/record_synced.py can sync it straight away.

Two clocks, kept apart on purpose:
  ESP32 clock   every sample and every LED_ON reply carries the ESP32's own micros() (t_us).
  camera clock  every frame carries the webcam driver's timestamp of the moment it was captured
                (video_frames.csv). The video file itself is written at one nominal frame rate.
The LED flashes appear on both clocks; src/myomotion/sync/align.py fits the line between them, exactly
as for phone video. The laptop's own clock (pc_time_s) is stored too, but only to place the cues.

Data structures (this file): Step, SetPlan, RecordingResult and CameraSettings dataclasses; SharedData (lists and a
    deque under a lock) shared by the camera loop and the ESP32 reader thread; lists of frame rows and cues; a
    cv2.VideoWriter.
Time complexity (this file): real time: O(F + N) for F frames and N samples; O(S) per sample in the reader thread
    (live %MVC); O(P) per frame to write it and draw the live view (P pixels); O(1) per frame to hand it to the live
    CV.
"""

from __future__ import annotations   # > type hints are stored as text, not run: allows forward references

import csv
import json
import re
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

from myomotion.semg.acquisition import EspStream
from myomotion.semg.live import LIVE, LivePercentMvc

US_PER_S = 1_000_000


# ---------------------------------------------------------------------------
# RecorderConfig  (the CONFIG for this file)
# What:  every number the recorder uses: timing of the set, camera, display.
# Data structures: a frozen dataclass.
# Time:  O(1) to build (fixed fields).
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RecorderConfig:
    """Settings of the synced recorder."""

    baseline_s: float = 5.0                  # relaxed baseline (also the live meter's 0%)
    countdown_s: int = 3
    tail_s: float = 3.0                      # relaxed after the end flash
    rest_between_s: float = 1.0
    tempo_s: tuple[float, ...] = (2.0, 1.0, 2.0)
    first_data_timeout_s: float = 5.0
    after_end_s: float = 0.6                 # keep reading the ESP32 after the set: the last LED_ON reply
    sample_period_us: int = 2000             # 500 samples per second
    gap_factor: float = 1.5                  # a gap longer than 1.5 periods counts as missing samples
    trace_s: float = 5.0                     # length of the live trace
    trace_every: int = 10                    # keep every 10th sample for the trace (50 per second)
    flash_banner_s: float = 0.4              # how long "SYNC FLASH" shows on screen
    bar_max_percent: float = 120.0
    bar_width_px: int = 32                   # width of the live %MVC bar
    cue_strip_px: int = 54                   # live CV view: height of the cue strip above the frame
    cv_panel_px: int = 200                   # live CV view: width of the sEMG panel right of the frame
    values_note: str = ("millivolts at the ESP32 pin: 20k/10k divider + 10k bias to D32 (rebuild 20 Nov); "
                        "module volts = (mV - 1320) x 5 / 1000")


RECORDER_CONFIG = RecorderConfig()

# > (event written to events.csv, word on screen, instruction) for every phase, in the order they are done
HAND_PHASES = (("CONCENTRIC", "SQUEEZE", "make a fist"), ("ISOMETRIC_HOLD", "HOLD", "keep squeezing"),
               ("ECCENTRIC", "RELEASE", "open your hand slowly"))
SQUAT_PHASES = (("ECCENTRIC", "DOWN", "go down"), ("ISOMETRIC_HOLD", "HOLD", "hold"),
                ("CONCENTRIC", "UP", "stand up"))
DEADLIFT_PHASES = (("CONCENTRIC", "UP", "pull"), ("ISOMETRIC_HOLD", "HOLD", "lock out"),
                   ("ECCENTRIC", "DOWN", "lower the bar"))
CURL_PHASES = (("CONCENTRIC", "UP", "curl up"), ("ISOMETRIC_HOLD", "HOLD", "squeeze at the top"),
               ("ECCENTRIC", "DOWN", "lower slowly"))
CURL_EXERCISE_NAMES = frozenset({"curl", "biceps", "bicep_curl", "biceps_curl"})   # > as src/myomotion/sync/curl.py
# > the RDL starts standing, so it lowers first (as src/myomotion/sync/rdl.py counts it)
RDL_PHASES = (("ECCENTRIC", "DOWN", "hips back, dumbbells down your legs"),
              ("ISOMETRIC_HOLD", "HOLD", "pause, back flat"), ("CONCENTRIC", "UP", "stand tall"))
RDL_EXERCISE_NAMES = frozenset({"rdl", "romanian_deadlift", "dumbbell_rdl"})
HAND_EXERCISES = frozenset({"forearm", "grip", "hand"})


# ---------------------------------------------------------------------------
# phases_for(exercise)
# What:  the phase order of an exercise (hand tests SQUEEZE/HOLD/RELEASE; deadlift and
#        biceps curl concentric first; RDL, squat and bench eccentric first).
# Time:  O(1).  Space: O(1).
# Data structures: tuple of tuples.  Files used: none.
# ---------------------------------------------------------------------------
def phases_for(exercise: str) -> tuple[tuple[str, str, str], ...]:
    """Phases of one rep, in order."""
    if exercise in HAND_EXERCISES:
        return HAND_PHASES
    if exercise in CURL_EXERCISE_NAMES:
        return CURL_PHASES
    if exercise in RDL_EXERCISE_NAMES:
        return RDL_PHASES
    return DEADLIFT_PHASES if exercise == "deadlift" else SQUAT_PHASES


# ---------------------------------------------------------------------------
# Step
# What:  one stretch of the guided set: what the screen says, which cue is logged when
#        it starts, extra markers (SET_START, SYNC_MID, SET_END) and whether the LED
#        flashes at its start.
# Data structures: frozen dataclass.
# Time:  O(1) to build (fixed fields).
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Step:
    """One step of the set, on the set's own clock (seconds from the start)."""

    start_s: float
    end_s: float
    text: str
    event: str | None = None
    rep: int | str = ""
    markers: tuple[str, ...] = ()
    flash: bool = False


# ---------------------------------------------------------------------------
# build_schedule(reps, phases, tempo, cfg)
# What:  the whole guided set as a list of Steps: baseline, countdown, every phase of
#        every rep (with a rest between reps), then the tail. The LED flashes at the
#        first phase (SET_START), after the middle rep (SYNC_MID, sets of 2+ reps; the
#        independent TC03 check) and at the tail (SET_END): the same plan as
#        record_session.py.
# Time:  O(reps * phases).  Space: same.
# Data structures: list of Steps in time order.  Files used: none.
# ---------------------------------------------------------------------------
def build_schedule(reps: int, phases: tuple[tuple[str, str, str], ...], tempo: tuple[float, ...],
                   cfg: RecorderConfig = RECORDER_CONFIG) -> list[Step]:
    """The guided set, step by step."""
    steps = [Step(0.0, cfg.baseline_s, "RELAX, stay still", "BASELINE")]
    t = cfg.baseline_s
    for n in range(cfg.countdown_s, 0, -1):
        steps.append(Step(t, t + 1.0, f"get ready ... {n}"))
        t += 1.0
    middle_rep = (reps + 1) // 2 if reps >= 2 else 0
    mid_pending = False                              # > the middle flash is due at the next step
    for rep in range(1, reps + 1):
        for index, ((event, word, instruction), seconds) in enumerate(zip(phases, tempo)):
            markers, flash = (), False
            if rep == 1 and index == 0:
                markers, flash = ("SET_START",), True
            elif index == 0 and mid_pending:         # > no rest between reps: flash as the next rep starts
                markers, flash, mid_pending = ("SYNC_MID",), True, False
            steps.append(Step(t, t + seconds, f"{word}: {instruction}", event, rep, markers, flash))
            t += seconds
        if rep == middle_rep:
            mid_pending = True
        if rep < reps and cfg.rest_between_s > 0:
            markers, flash = (("SYNC_MID",), True) if mid_pending else ((), False)
            mid_pending = False
            steps.append(Step(t, t + cfg.rest_between_s, "rest", "REST", rep, markers, flash))
            t += cfg.rest_between_s
    steps.append(Step(t, t + cfg.tail_s, "RELAX, stay still", "TAIL", "", ("SET_END",), True))
    return steps


# ---------------------------------------------------------------------------
# step_at(steps, t_s)
# What:  index of the step running at time t_s, or None once the set is over.
# Time:  O(S).  Space: O(1).
# Data structures: walks the list.  Files used: none.
# ---------------------------------------------------------------------------
def step_at(steps: list[Step], t_s: float) -> int | None:
    """Which step is running now."""
    for index, step in enumerate(steps):
        if step.start_s <= t_s < step.end_s:
            return index
    return None


# ---------------------------------------------------------------------------
# next_set_folder(data_dir, participant, exercise, posture)
# What:  <data_dir>/<participant>/S<nn>_<exercise>_<posture>, numbered after every
#        existing S<nn>_ folder of that participant (the same numbering as
#        record_session.py, so both tools share one sequence). Not created here.
# Time:  O(F) folders.  Space: O(F).
# Data structures: regex over the folder names.  Files used: lists <data_dir>/<participant>.
# ---------------------------------------------------------------------------
def next_set_folder(data_dir: Path, participant: str, exercise: str, posture: str) -> Path:
    """The next free set folder."""
    base = Path(data_dir).expanduser() / participant
    numbers = [int(m.group(1)) for p in base.glob("S*") if (m := re.match(r"S(\d+)_", p.name))]
    return base / f"S{max(numbers, default=0) + 1:02d}_{exercise}_{posture}"


# ---------------------------------------------------------------------------
# SharedData
# What:  everything the ESP32 reader thread fills and the main (camera) thread reads.
#        One lock guards it; the main thread only takes short looks.
# Data structures: lists of rows (appended in time order), deque ring buffer for the
#        live trace (old points drop out by themselves), the latest meter reading.
# Time:  O(1) per append; the lock makes each update atomic between the two threads.
# ---------------------------------------------------------------------------
@dataclass
class SharedData:
    """sEMG side of the recording, shared between the two threads."""

    samples: list = field(default_factory=list)     # (pc_time_s, t_us, ch1, ch2, ch3)
    board_events: list = field(default_factory=list)  # (pc_time_s, t_us, "board", name, "")
    trace: deque = field(default_factory=lambda: deque(maxlen=250))   # (pc_time_s, value)
    latest_value: float | None = None
    latest_stage: str = ""
    last_sample: tuple[float, int] | None = None    # (pc_time_s, t_us) of the newest sample
    error: str = ""
    lock: threading.Lock = field(default_factory=threading.Lock)


# ---------------------------------------------------------------------------
# EspReader (thread)
# What:  reads the ESP32 stream as fast as it arrives, so the camera loop never makes
#        it wait: stores every sample, feeds the live %MVC meter, collects the LED_ON
#        replies. Stops on stop() or when the connection drops (error kept).
# Time:  O(1) per sample (meter step).  Space: O(N) for the stored samples.
# Data structures: SharedData; LivePercentMvc.  Files used: none.
# ---------------------------------------------------------------------------
class EspReader(threading.Thread):
    """Background reader of the ESP32 stream."""

    # Time:  O(1).  Space: O(1).
    # Data structures: references to the stream, meter and SharedData; ints.
    def __init__(self, stream: EspStream, meter: LivePercentMvc, channel_index: int, shared: SharedData,
                 start: float, show_percent: bool, cfg: RecorderConfig = RECORDER_CONFIG) -> None:
        super().__init__(daemon=True)
        self.stream, self.meter, self.channel, self.shared = stream, meter, channel_index, shared
        self.start_time, self.show_percent, self.cfg = start, show_percent, cfg
        self.running = True
        self._count = 0

    # Time:  O(n * S) per loop for the n samples that arrived (one filter step each, S sections); runs until stop().
    #        Space: O(n).
    # Data structures: SharedData lists and deques under its lock; a list of board events.
    def run(self) -> None:
        while self.running:
            try:
                new = self.stream.read()
            except (ConnectionError, OSError) as error:
                self.shared.error = str(error)
                return
            now = time.perf_counter() - self.start_time
            events = list(self.stream.parser.events)
            self.stream.parser.events.clear()
            with self.shared.lock:
                for sample in new:
                    reading = self.meter.step(sample.values_mv[self.channel])
                    self.shared.samples.append((now, sample.t_us, *sample.values_mv))
                    self.shared.last_sample = (now, sample.t_us)
                    value = reading.percent if self.show_percent else (
                        reading.envelope_mv - (reading.rest_mv or 0.0) if reading.stage == LIVE else None)
                    self.shared.latest_value, self.shared.latest_stage = value, reading.stage
                    self._count += 1
                    if value is not None and self._count % self.cfg.trace_every == 0:
                        self.shared.trace.append((now, value))
                for event in events:
                    self.shared.board_events.append((now, event.t_us, "board", event.name, ""))

    # Time:  O(1).
    # Data structures: a bool flag read by the thread.
    def stop(self) -> None:
        self.running = False


# ---------------------------------------------------------------------------
# esp32_now_us(shared, pc_now)
# What:  the ESP32 time "now", estimated from the newest sample: its t_us plus the
#        laptop time since it arrived. Used only to place the cues (good to tens of ms,
#        like record_session.py); the sync itself uses the LED.
# Time:  O(1).  Space: O(1).
# Data structures: none.  Files used: none.
# ---------------------------------------------------------------------------
def esp32_now_us(shared: SharedData, pc_now: float) -> int:
    """Estimated ESP32 micros() at laptop time pc_now."""
    with shared.lock:
        last = shared.last_sample
    if last is None:
        return 0
    return int(last[1] + (pc_now - last[0]) * US_PER_S)


# ---------------------------------------------------------------------------
# sample_quality(samples, cfg)
# What:  real sample rate and missing samples from the ESP32 timestamps (gaps longer
#        than 1.5 periods), as record_session.py reports them.
# Time:  O(N).  Space: O(N).
# Data structures: NumPy diff.  Files used: none.
# ---------------------------------------------------------------------------
def sample_quality(samples: list, cfg: RecorderConfig = RECORDER_CONFIG) -> dict:
    """Rate, duration and missing samples."""
    if len(samples) < 2:
        return {"samples": len(samples), "duration_s": 0, "rate_hz": 0, "missing_samples": 0}
    t_us = np.array([s[1] for s in samples], dtype=np.int64)
    gaps = np.diff(t_us)
    long_gaps = gaps[gaps > cfg.gap_factor * cfg.sample_period_us]
    missing = int(np.sum(np.round(long_gaps / cfg.sample_period_us) - 1))
    duration = (t_us[-1] - t_us[0]) / US_PER_S
    return {"samples": len(samples), "duration_s": round(float(duration), 3),
            "rate_hz": round((len(samples) - 1) / duration, 1) if duration > 0 else 0,
            "missing_samples": missing,
            "missing_percent": round(100 * missing / (len(samples) + missing), 3)}


# ---------------------------------------------------------------------------
# _draw_meter(view, left, top, bottom, value, unit, scale, cfg)
# What:  the %MVC (or mV) bar between rows top..bottom at column `left`, with the
#        100 %MVC line, the number under it and the unit above it.
# Time:  O(bar pixels).  Space: O(1) extra.
# Data structures: NumPy image (drawn in place).  Files used: none.
# ---------------------------------------------------------------------------
def _draw_meter(view: np.ndarray, left: int, top: int, bottom: int, value: float | None, unit: str, scale: float,
                cfg: RecorderConfig = RECORDER_CONFIG) -> None:
    """Bar + number + unit, drawn in place."""
    import cv2
    right = left + cfg.bar_width_px
    cv2.rectangle(view, (left, top), (right, bottom), (90, 90, 90), 1)
    if value is not None:
        level = int(bottom - (bottom - top) * min(max(value, 0.0), scale) / scale)
        cv2.rectangle(view, (left + 2, level), (right - 2, bottom - 1), (110, 42, 201), -1)
    if unit == "%MVC":
        y100 = int(bottom - (bottom - top) * 100 / scale)
        cv2.line(view, (left - 6, y100), (right + 6, y100), (220, 220, 220), 1)
        cv2.putText(view, "100", (left - 4, y100 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (220, 220, 220), 1)
    label = "--" if value is None else str(int(round(value)))   # > int(round(-0.3)) = 0: never "-0"
    cv2.putText(view, label, (left - 30, bottom + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
                cv2.LINE_AA)
    cv2.putText(view, unit, (left - 30, top - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# _draw_trace(view, box, trace, now, scale, cfg)
# What:  the last trace_s seconds of the live value as a line inside box (x0, x1, y0, y1).
# Time:  O(trace length).  Space: O(trace length).
# Data structures: list of (time, value) -> NumPy int32 points.  Files used: none.
# ---------------------------------------------------------------------------
def _draw_trace(view: np.ndarray, box: tuple[int, int, int, int], trace: list[tuple[float, float]], now: float,
                scale: float, cfg: RecorderConfig = RECORDER_CONFIG) -> None:
    """Trace line, drawn in place."""
    import cv2
    x0, x1, y0, y1 = box
    points = [(int(x1 - (now - t) / cfg.trace_s * (x1 - x0)), int(y1 - (y1 - y0) * min(max(v, 0.0), scale) / scale))
              for t, v in trace if now - t <= cfg.trace_s]
    if len(points) >= 2:
        cv2.polylines(view, [np.array(points, dtype=np.int32)], False, (180, 120, 255), 2, cv2.LINE_AA)


# Time:  O(L) for the L trace points (max).  Space: O(1).
# Data structures: a list of (time, value) tuples.
def _meter_scale(unit: str, trace: list[tuple[float, float]], cfg: RecorderConfig = RECORDER_CONFIG) -> float:
    """Top of the bar: 120 %MVC, or in mV the largest recent value (at least 50)."""
    return cfg.bar_max_percent if unit == "%MVC" else max(50.0, max((v for _, v in trace), default=50.0))


# ---------------------------------------------------------------------------
# draw_overlay(frame, ...)
# What:  the live view drawn on a COPY of the camera frame (the saved video stays
#        clean for the LED search and the pose model): the cue band on top, the %MVC
#        bar and number on the right, the 5-second trace at the bottom, a REC dot and
#        a SYNC FLASH banner.
# Time:  O(P) for the translucent bands (P pixels) + O(trace length).  Space: O(P).
# Data structures: NumPy image; list of trace points.  Files used: none.
# ---------------------------------------------------------------------------
def draw_overlay(frame: np.ndarray, step: Step | None, reps: int, seconds_left: float, value: float | None,
                 unit: str, trace: list[tuple[float, float]], now: float, flash_banner: bool, elapsed: float,
                 cfg: RecorderConfig = RECORDER_CONFIG) -> np.ndarray:
    """Camera frame with the live view on top."""
    import cv2
    view = frame.copy()
    h, w = view.shape[:2]
    band = view.copy()
    cv2.rectangle(band, (0, 0), (w, 54), (25, 25, 25), -1)
    cv2.rectangle(band, (0, h - 92), (w - 110, h), (25, 25, 25), -1)
    cv2.rectangle(band, (w - 104, 60), (w, h), (25, 25, 25), -1)
    view = cv2.addWeighted(band, 0.55, view, 0.45, 0)
    _draw_cue(view, w, step, reps, seconds_left, elapsed)
    scale = _meter_scale(unit, trace, cfg)
    _draw_meter(view, w - 70, 70, h - 40, value, unit, scale, cfg)
    _draw_trace(view, (10, w - 120, h - 84, h - 10), trace, now, scale, cfg)
    if flash_banner:
        cv2.putText(view, "SYNC FLASH", (w // 2 - 90, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 220, 255), 3,
                    cv2.LINE_AA)
    return view


# Time:  O(1): a fixed number of text and shape calls.  Space: O(1).
# Data structures: a NumPy image drawn in place; a Step dataclass.
def _draw_cue(view: np.ndarray, width: int, step: Step | None, reps: int, seconds_left: float, elapsed: float) -> None:
    """Cue text, rep counter and REC dot along the top 54 px of view (in place)."""
    import cv2
    text = step.text if step else "done"
    cv2.putText(view, text, (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    if step and step.rep != "":
        cv2.putText(view, f"rep {step.rep}/{reps}  {seconds_left:3.1f} s", (width - 230, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 230, 255), 1, cv2.LINE_AA)
    cv2.circle(view, (width - 20, 18), 7, (40, 40, 230), -1)
    cv2.putText(view, f"REC {elapsed:4.0f}s", (width - 110, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1,
                cv2.LINE_AA)


# ---------------------------------------------------------------------------
# compose_cv_view(annotated, ...)
# What:  the live view when Ahmad's CV runs: a cue strip across the top, his annotated
#        frame (skeleton, spine, phase, angles, form cues) untouched on the left, the
#        %MVC bar and trace in a panel on the right; so neither overlay covers the other.
# Time:  O(P) pixels of the canvas.  Space: O(P).
# Data structures: NumPy canvas assembled from three regions.  Files used: none.
# ---------------------------------------------------------------------------
def compose_cv_view(annotated: np.ndarray, step: Step | None, reps: int, seconds_left: float, value: float | None,
                    unit: str, trace: list[tuple[float, float]], now: float, flash_banner: bool, elapsed: float,
                    cfg: RecorderConfig = RECORDER_CONFIG) -> np.ndarray:
    """Cue strip + annotated frame + sEMG panel, as one image."""
    import cv2
    h, w = annotated.shape[:2]
    top, panel = cfg.cue_strip_px, cfg.cv_panel_px
    canvas = np.full((h + top, w + panel, 3), 25, np.uint8)
    canvas[top:, :w] = annotated
    _draw_cue(canvas, w + panel, step, reps, seconds_left, elapsed)
    scale = _meter_scale(unit, trace, cfg)
    _draw_meter(canvas, w + panel - 70, top + 40, top + h - 150, value, unit, scale, cfg)
    _draw_trace(canvas, (w + 10, w + panel - 10, top + h - 100, top + h - 10), trace, now, scale, cfg)
    cv2.putText(canvas, "sEMG", (w + 12, top + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
    if flash_banner:
        cv2.putText(canvas, "SYNC FLASH", ((w + panel) // 2 - 90, top + h // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (0, 220, 255), 3, cv2.LINE_AA)
    return canvas


# ---------------------------------------------------------------------------
# CameraSettings / camera_settings(platform)
# What:  how to open the laptop camera on this operating system:
#          Linux  V4L2: a fixed exposure can be set (units of 0.1 ms) and the driver
#                 stamps every frame (CAP_PROP_POS_MSEC);
#          macOS  AVFoundation: OpenCV cannot fix the exposure, and the frame time is
#                 taken from the laptop clock the moment the frame is read, the same
#                 clock for every frame (the laptop-clock sync only needs one steady
#                 clock; its constant camera delay is measured once with the LED);
#          other  OpenCV's default backend, laptop-clock frame times.
# Time / Space: O(1).  Data structures: frozen dataclass.  Files used: none.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CameraSettings:
    """What this computer's camera backend can do."""

    backend_name: str          # suffix of the cv2.CAP_* constant: "V4L2", "AVFOUNDATION" or "ANY"
    manual_exposure: bool      # OpenCV can set a fixed exposure
    driver_stamps: bool        # the driver gives a usable per-frame timestamp
    note: str                  # one line for the console and metadata.json


# Time:  O(1).  Space: O(1).
# Data structures: a str comparison; a CameraSettings dataclass out.
def camera_settings(platform: str = sys.platform) -> CameraSettings:
    """Camera settings for an operating system (sys.platform value)."""
    if platform.startswith("linux"):
        return CameraSettings("V4L2", True, True, "Linux camera (V4L2): fixed exposure, driver frame times")
    if platform == "darwin":
        return CameraSettings("AVFOUNDATION", False, False,
                              "macOS camera (AVFoundation): automatic exposure, frame times from the laptop clock")
    return CameraSettings("ANY", False, False, f"{platform} camera (OpenCV default): frame times from the laptop clock")


# ---------------------------------------------------------------------------
# Webcam
# What:  the laptop camera through OpenCV: fixed size and 30 fps; on Linux optionally a
#        fixed exposure (auto exposure drops to ~10 fps in a dim room, too slow for
#        sync) and the driver's timestamp of every frame; on macOS automatic exposure
#        and no driver times (camera_settings above). Auto exposure is restored on close.
# Data structures: cv2.VideoCapture; CameraSettings.
# Files used: /dev/video<index> on Linux; the built-in camera on macOS.
# Time:  O(P) per frame read for P pixels, plus the camera's own frame period (~33 ms at 30 fps).
# ---------------------------------------------------------------------------
class Webcam:
    """Laptop camera with per-frame timestamps where the driver gives them."""

    # Time:  O(1) (opens the device).  Space: O(1).
    # Data structures: a cv2.VideoCapture; a CameraSettings dataclass; a dict of camera info.
    def __init__(self, index: int = 0, width: int = 640, height: int = 480, exposure: float | None = 150.0,
                 platform: str = sys.platform) -> None:
        import cv2
        self.cv2 = cv2
        self.settings = camera_settings(platform)
        self.cap = cv2.VideoCapture(index, getattr(cv2, "CAP_" + self.settings.backend_name, cv2.CAP_ANY))
        if not self.cap.isOpened():
            hint = (" On macOS allow camera access for the Terminal: System Settings > Privacy & Security > Camera."
                    if platform == "darwin" else "")
            raise RuntimeError(f"camera {index} could not be opened (is another program using it?){hint}")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.fixed_exposure = exposure is not None and self.settings.manual_exposure
        if self.fixed_exposure:
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)      # > 1 = manual on V4L2
            self.cap.set(cv2.CAP_PROP_EXPOSURE, exposure)
        self.info = {"backend": self.settings.backend_name, "fixed_exposure": exposure if self.fixed_exposure else None,
                     "frame_times": "camera driver" if self.settings.driver_stamps else "laptop clock at read",
                     "platform": platform}

    # Time:  O(P) for a P-pixel frame (the driver copies it).  Space: O(P).
    # Data structures: a NumPy image; a float timestamp or None.
    def read(self) -> tuple[bool, np.ndarray | None, float | None]:
        """(ok, frame, driver timestamp in seconds, or None: the recorder then uses the laptop clock)."""
        ok, frame = self.cap.read()
        if not ok or not self.settings.driver_stamps:
            return ok, frame, None
        stamp = self.cap.get(self.cv2.CAP_PROP_POS_MSEC) / 1000.0
        return ok, frame, (stamp if stamp and stamp > 0 else None)

    # Time:  O(1).
    # Data structures: releases the cv2.VideoCapture.
    def close(self) -> None:
        if self.fixed_exposure:
            self.cap.set(self.cv2.CAP_PROP_AUTO_EXPOSURE, 3)  # > 3 = back to automatic
        self.cap.release()


# ---------------------------------------------------------------------------
# SetPlan / RecordingResult
# What:  what to record (exercise, reps, tempo, who) and what came out.
# Data structures: dataclasses.
# Time:  O(1) to build (fixed fields).
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SetPlan:
    """The set to record."""

    participant: str
    exercise: str
    posture: str
    reps: int
    tempo_s: tuple[float, ...]
    channels: tuple[str, ...] = ("forearm",)


# Time:  O(1) to build.
# Data structures: a dataclass: the folder Path, a dict of quality numbers, counts.
@dataclass
class RecordingResult:
    """What the recorder saved."""

    folder: Path
    aborted: bool
    quality: dict
    frames: int
    fps: float
    flashes: int


# ---------------------------------------------------------------------------
# record_set(camera, stream, plan, mvc_above_rest_mv, folder, ...)
# What:  the recording itself (see the file docstring). Waits for camera frames and
#        sEMG samples, measures the camera's real frame rate for 1 s, then runs the
#        schedule frame by frame: write the raw frame + its timestamps, log cues, flash
#        the LED, draw and show the live view. 'q' / Esc / Ctrl+C stop early and still
#        save. Then saves emg.csv, events.csv, metadata.json, video_frames.csv.
# Time:  O(F) frames + O(N) samples.  Space: O(N + F) rows (frames go straight to disk).
# Data structures: SharedData + EspReader thread; lists of rows; cv2.VideoWriter.
#        With live_cv (src/myomotion/sync/cv_frames.py LiveCv) every frame is also handed
#        to Ahmad's CV thread and the window shows his annotated frame next to the sEMG.
# Files used: writes <folder>/video.mp4, video_frames.csv, emg.csv, events.csv, metadata.json.
# ---------------------------------------------------------------------------
def record_set(camera, stream: EspStream, plan: SetPlan, mvc_above_rest_mv: float | None, folder: Path,
               channel_index: int = 0, display: bool = True, cfg: RecorderConfig = RECORDER_CONFIG,
               fps_probe_s: float = 1.0, live_cv=None) -> RecordingResult:
    """Record one synced set; returns where it went."""
    import cv2
    start = time.perf_counter()
    show_percent = mvc_above_rest_mv is not None
    meter = LivePercentMvc(mvc_above_rest_mv=mvc_above_rest_mv if show_percent else 1.0)
    shared = SharedData()
    reader = EspReader(stream, meter, channel_index, shared, start, show_percent, cfg)
    reader.start()
    deadline = time.perf_counter() + cfg.first_data_timeout_s
    while not shared.samples:
        if shared.error or time.perf_counter() > deadline:
            reader.stop()
            raise ConnectionError(shared.error or "no sEMG samples arrived from the ESP32")
        time.sleep(0.05)

    # > measure the camera's real frame rate (the video file needs one number)
    stamps = []
    probe_end = time.perf_counter() + fps_probe_s
    while time.perf_counter() < probe_end:
        ok, frame, stamp = camera.read()
        if ok:
            stamps.append(stamp if stamp is not None else time.perf_counter())
    if len(stamps) < 3:
        reader.stop()
        raise RuntimeError("the camera delivers no frames")
    fps = float(1.0 / np.median(np.diff(stamps)))
    height, width = frame.shape[:2]

    folder.mkdir(parents=True, exist_ok=False)
    writer = cv2.VideoWriter(str(folder / "video.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    schedule = build_schedule(plan.reps, phases_for(plan.exercise), plan.tempo_s, cfg)
    cues: list[tuple] = []
    frame_rows: list[tuple] = []
    first_stamp = None
    current = -1
    banner_until = 0.0
    aborted = False
    set_start = time.perf_counter()
    started_at = datetime.now().isoformat(timespec="seconds")
    try:
        while True:
            ok, frame, stamp = camera.read()
            pc_now = time.perf_counter() - start
            if not ok:
                aborted = True
                cues.append((pc_now, esp32_now_us(shared, pc_now), "cue", "ABORTED", "camera"))
                break
            stamp = stamp if stamp is not None else time.perf_counter()
            first_stamp = stamp if first_stamp is None else first_stamp
            writer.write(frame)
            frame_rows.append((len(frame_rows), round(stamp - first_stamp, 6), round(pc_now, 6)))
            if live_cv is not None:
                # > hand the frame to Ahmad's CV thread on the camera clock; returns at once
                live_cv.submit(frame, stamp - first_stamp)
            elapsed = time.perf_counter() - set_start
            index = step_at(schedule, elapsed)
            if index is None:
                break
            if index != current:
                current = index
                step = schedule[index]
                for marker in step.markers:
                    cues.append((pc_now, esp32_now_us(shared, pc_now), "cue", marker, ""))
                if step.flash:
                    stream.flash_sync()
                    banner_until = elapsed + cfg.flash_banner_s
                if step.event:
                    cues.append((pc_now, esp32_now_us(shared, pc_now), "cue", step.event, step.rep))
            if shared.error:
                aborted = True
                cues.append((pc_now, esp32_now_us(shared, pc_now), "cue", "ABORTED", "esp32"))
                break
            if display:
                step = schedule[current]
                with shared.lock:
                    value, trace = shared.latest_value, list(shared.trace)
                overlay_args = (step, plan.reps, step.end_s - elapsed, value, "%MVC" if show_percent else "mV", trace,
                                pc_now, elapsed < banner_until, elapsed, cfg)
                if live_cv is not None:
                    annotated = live_cv.latest()
                    # > until the CV's first frame is ready, show the plain camera frame in its place
                    left = annotated if annotated is not None and annotated.shape == frame.shape else frame
                    view = compose_cv_view(left, *overlay_args)
                else:
                    view = draw_overlay(frame, *overlay_args)
                cv2.imshow("MyoMotion: video + sEMG (q = stop)", view)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    aborted = True
                    cues.append((pc_now, esp32_now_us(shared, pc_now), "cue", "ABORTED", "key"))
                    break
    except KeyboardInterrupt:
        aborted = True
        pc_now = time.perf_counter() - start
        cues.append((pc_now, esp32_now_us(shared, pc_now), "cue", "ABORTED", "ctrl-c"))
    finally:
        writer.release()
        if display:
            cv2.destroyAllWindows()
    time.sleep(cfg.after_end_s)                       # > the end flash's LED_ON reply is still on its way
    reader.stop()
    reader.join(timeout=1.0)

    with shared.lock:
        samples, board = list(shared.samples), list(shared.board_events)
    quality = sample_quality(samples, cfg)
    names = list(plan.channels) + ["not_connected"] * (3 - len(plan.channels))
    metadata = {
        "participant": plan.participant, "set_folder": folder.name, "exercise": plan.exercise,
        "posture": plan.posture, "mode": "set", "reps": plan.reps,
        "tempo_s": {word.lower(): s for (_, word, _), s in zip(phases_for(plan.exercise), plan.tempo_s)},
        "phase_order": [event for event, _, _ in phases_for(plan.exercise)],
        "rest_between_reps_s": cfg.rest_between_s, "baseline_s": cfg.baseline_s, "started_at": started_at,
        "board": "esp32", "recorder": "scripts/sync/record_synced.py (webcam + sEMG)",
        "channels": {f"ch{i + 1}": names[i] for i in range(3)}, "channels_connected": len(plan.channels),
        "values": cfg.values_note, "aborted": aborted,
        "video": {"file": "video.mp4", "frames": len(frame_rows), "fps": round(fps, 3), "width": width,
                  "height": height, "frame_times": "video_frames.csv (t_camera_s: " + str(
                      getattr(camera, "info", {}).get("frame_times", "camera driver")) + ")",
                  "camera": getattr(camera, "info", None)},
        **quality,
    }
    if live_cv is not None:
        metadata["live_cv"] = {"frames_analysed": live_cv.analysed, "frames_skipped": live_cv.skipped,
                               "error": live_cv.error, "note": "live view only; the dataset rows come from the "
                               "offline pass over every frame (cv_frames.csv)"}
    _write_csv(folder / "emg.csv", ["pc_time_s", "t_us", "ch1", "ch2", "ch3"],
               [(f"{s[0]:.6f}", *s[1:]) for s in samples])
    _write_csv(folder / "events.csv", ["pc_time_s", "t_us", "source", "event", "rep"],
               [(f"{e[0]:.6f}", *e[1:]) for e in sorted(cues + board, key=lambda e: e[0])])
    _write_csv(folder / "video_frames.csv", ["frame", "t_camera_s", "pc_time_s"], frame_rows)
    (folder / "metadata.json").write_text(json.dumps(metadata, indent=2))
    flashes = sum(1 for e in board if e[3] == "LED_ON")
    return RecordingResult(folder=folder, aborted=aborted, quality=quality, frames=len(frame_rows), fps=fps,
                           flashes=flashes)


# Time:  O(R * C) for R rows of C columns.  Space: O(1) extra (rows stream to disk).
# Data structures: a list of row tuples; a csv.writer.
def _write_csv(path: Path, header: list[str], rows: list) -> None:
    """Write one CSV file."""
    with path.open("w", newline="") as out:
        writer = csv.writer(out)
        writer.writerow(header)
        writer.writerows(rows)
