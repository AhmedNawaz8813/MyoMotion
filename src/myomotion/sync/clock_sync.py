"""Video-sEMG sync WITHOUT the LED: both streams were timed by the same laptop clock (webcam recordings).

The synced recorder (src/myomotion/sync/recorder.py) writes, for every sEMG sample, the ESP32's own micros() and
the laptop time it arrived (emg.csv: t_us, pc_time_s), and for every video frame, the camera driver's time and the
laptop time it was read (video_frames.csv: t_camera_s, pc_time_s). Both laptop times come from one clock.

    arrival time = event time + a delay that is never negative (Wi-Fi batching, USB transfer, the read loop)

So the FASTEST arrivals in each second carry almost no delay: a straight line through the per-second minima of
(arrival - device time) maps each device's clock onto the laptop clock, offset and drift together. Chaining the
two lines gives video time -> laptop time -> ESP32 time. What no line can see is any delay that is ALWAYS there: the
camera's own delay between the light hitting the sensor and the driver's timestamp, and the shortest possible
Wi-Fi trip. Together they are one constant, measured once end to end against the LED:

    21 Nov 2026, TESTCV/S03_forearm_fist (LED found, TC03 PASS 2.9 ms): this method read the three flashes
    12.5, 8.6 and 11.0 ms late -> camera delay 10.7 ms, left over after subtracting it: within +-2 ms.

Every LED-synced webcam recording measures it again (sync.json "clock_check"), so the number can be kept honest.
Phone video has no laptop times: it still needs the LED (src/myomotion/sync/led_sync.py).

Data structures (this file): NumPy arrays of device and laptop times; a dict window -> fastest (time, delay) while
    scanning; EnvelopeLine and LaptopClock dataclasses (with a dict of arrays kept for the plot).
Time complexity (this file): O(N + F) for N sEMG samples and F frames (one pass to bucket the fastest arrivals, a
    least-squares line through the W window minima in O(W), one refit without outliers); O(L) to check L LED
    flashes.
"""

from __future__ import annotations   # > type hints are stored as text, not run: allows forward references

import csv
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from myomotion.sync.align import ClockFit

US_PER_S = 1_000_000
MS_PER_S = 1000.0
EMG_NAME = "emg.csv"
FRAMES_NAME = "video_frames.csv"


# ---------------------------------------------------------------------------
# ClockSyncConfig  (the CONFIG for this file)
# What:  the window for the fastest-arrival search and the camera delay.
# Data structures: frozen dataclass.
# Time:  O(1) to build (fixed fields).
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ClockSyncConfig:
    """Settings of the laptop-clock sync."""

    window_s: float = 1.0               # one fastest arrival per second of each stream
    min_windows: int = 3                # fewer than this = too short to fit a line
    # > a window whose fastest arrival is more than outlier_ms above the line is left out and the line refitted
    # > (a Wi-Fi stall lasting a whole window); on the real recordings every window was within 0.6 ms
    outlier_ms: float = 5.0
    camera_delay_ms: float = 10.7       # light -> driver timestamp, measured with the LED (21 Nov, S03; see above)
    expected_error_ms: float = 5.0      # what we claim after the correction (S03 left +-2 ms; margin for other days)


CLOCK_CONFIG = ClockSyncConfig()


# ---------------------------------------------------------------------------
# EnvelopeLine
# What:  delay = intercept + slope * t through the fastest arrivals of one stream, and
#        how far above it the typical arrival sits.
# Data structures: frozen dataclass.
# Time:  O(1) to build; at() is O(n) for n times.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class EnvelopeLine:
    """Laptop time minus device time, as a line in device time (seconds)."""

    slope: float                       # drift of the laptop clock against the device clock (1e-6 = 1 ppm)
    intercept_s: float                 # laptop minus device time at device time 0
    windows: int                       # how many per-second minima went into the line
    median_delay_ms: float             # typical arrival delay above the line (information)

    # Time:  O(n) for n times.  Space: O(n).
    # Data structures: a NumPy float64 array (or one float).
    def at(self, t_s: np.ndarray | float) -> np.ndarray | float:
        """Laptop minus device time at device time t_s."""
        return self.intercept_s + self.slope * np.asarray(t_s, dtype=np.float64)


# ---------------------------------------------------------------------------
# lower_envelope_line(t_s, delay_s, cfg)
# What:  our own fit: split the device time into windows, keep the smallest delay in
#        each (the arrival that waited least), then a least-squares line through those
#        minima:  slope = sum((x - mx)(y - my)) / sum((x - mx)^2),  intercept = my - slope*mx.
#        Windows whose minimum sits more than outlier_ms above that line (nothing arrived
#        quickly in them) are dropped and the line is fitted once more.
# Time:  O(N) for N arrivals (one pass to bucket, one per window).  Space: O(N / window).
# Data structures: NumPy arrays; dict window index -> (t, smallest delay) while scanning.
# Files used: none.
# ---------------------------------------------------------------------------
def lower_envelope_line(t_s: np.ndarray, delay_s: np.ndarray, cfg: ClockSyncConfig = CLOCK_CONFIG) -> EnvelopeLine:
    """Line through the smallest delay of every window."""
    t_s, delay_s = np.asarray(t_s, dtype=np.float64), np.asarray(delay_s, dtype=np.float64)
    if len(t_s) != len(delay_s) or len(t_s) == 0:
        raise ValueError("need matching, non-empty time and delay arrays")
    fastest: dict[int, tuple[float, float]] = {}
    first = float(t_s[0])
    for t, delay in zip(t_s.tolist(), delay_s.tolist()):
        window = int((t - first) // cfg.window_s)
        if window not in fastest or delay < fastest[window][1]:   # > O(1) dict update per arrival
            fastest[window] = (t, delay)
    if len(fastest) < cfg.min_windows:
        raise ValueError(f"only {len(fastest)} s of data: the laptop-clock sync needs at least {cfg.min_windows}")
    x = np.array([point[0] for point in fastest.values()])
    y = np.array([point[1] for point in fastest.values()])
    slope, intercept = _least_squares(x, y)
    keep = (y - (intercept + slope * x)) * MS_PER_S <= cfg.outlier_ms
    if not np.all(keep) and int(np.sum(keep)) >= cfg.min_windows:
        x, y = x[keep], y[keep]
        slope, intercept = _least_squares(x, y)
    above = delay_s - (intercept + slope * t_s)
    return EnvelopeLine(slope=slope, intercept_s=intercept, windows=len(x),
                        median_delay_ms=float(np.median(above)) * MS_PER_S)


# Time:  O(W) for the W window minima.  Space: O(W).
# Data structures: NumPy float64 arrays; a (slope, intercept) tuple out.
def _least_squares(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """(slope, intercept) of the least-squares line y = intercept + slope * x."""
    x_mean, y_mean = float(np.sum(x)) / len(x), float(np.sum(y)) / len(y)
    spread = float(np.sum((x - x_mean) ** 2))
    slope = float(np.sum((x - x_mean) * (y - y_mean))) / spread if spread > 0 else 0.0
    return slope, y_mean - slope * x_mean


# ---------------------------------------------------------------------------
# LaptopClock
# What:  both streams mapped onto the laptop clock, and the chain between them:
#            laptop(ESP32 u)   = u + esp32.at(u)          u = ESP32 s since its first sample
#            laptop(camera t)  = t + camera.at(t)         t = camera s since the first frame
#        A light switched on at ESP32 time u is stamped by the camera camera_delay later:
#            u + esp32.at(u) + delay = t + camera.at(t)
#        Solved for t this is a straight line in u, i.e. an ordinary ClockFit, so the rest
#        of the sync (session.py: interpolation, merged.csv, phases) works unchanged.
# Data structures: frozen dataclass; the raw delays kept for the plot.
# Time:  O(1) to build; to_clock_fit() is O(1).
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LaptopClock:
    """ESP32 and camera clocks, each tied to the laptop clock."""

    esp32: EnvelopeLine
    camera: EnvelopeLine
    esp32_origin_us: int               # ESP32 time of the first sample (u = 0)
    camera_delay_ms: float
    trace: dict = field(default_factory=dict, repr=False, compare=False)   # delays for sync.png

    # Time:  O(1).  Space: O(1).
    # Data structures: returns an align.ClockFit dataclass.
    def to_clock_fit(self) -> ClockFit:
        """The same line as an align.ClockFit: t_video = video_ref + slope * (t_esp32_us - ref) / 1e6."""
        delay_s = self.camera_delay_ms / MS_PER_S
        slope = (1.0 + self.esp32.slope) / (1.0 + self.camera.slope)
        video_ref = (self.esp32.intercept_s + delay_s - self.camera.intercept_s) / (1.0 + self.camera.slope)
        return ClockFit(slope=slope, esp32_ref_us=self.esp32_origin_us, video_ref_s=video_ref, residuals_ms=())

    # Time:  O(1).
    # Data structures: a float.
    @property
    def drift_ppm(self) -> float:
        """ESP32 clock against the camera clock, parts per million."""
        return (self.to_clock_fit().slope - 1.0) * 1e6


# ---------------------------------------------------------------------------
# load_arrivals(folder)
# What:  (ESP32 seconds since the first sample, laptop arrival s) for every sEMG sample and
#        (camera s, laptop read s) for every frame, plus the ESP32 time of the first sample.
# Time:  O(N + F).  Space: O(N + F).
# Data structures: NumPy arrays.  Files used: reads <folder>/emg.csv and video_frames.csv.
# ---------------------------------------------------------------------------
def load_arrivals(folder: Path) -> tuple[np.ndarray, np.ndarray, int, np.ndarray, np.ndarray]:
    """(esp32_s, esp32_arrival_s, esp32_origin_us, camera_s, camera_read_s)."""
    folder = Path(folder)
    frames_path = folder / FRAMES_NAME
    if not frames_path.exists():
        raise ValueError(f"{frames_path} is missing: the laptop-clock sync needs a recording made by "
                         "scripts/sync/record_synced.py (phone video needs the LED)")
    with (folder / EMG_NAME).open(newline="") as emg_file:
        rows = [(int(row["t_us"]), float(row["pc_time_s"])) for row in csv.DictReader(emg_file, skipinitialspace=True)]
    with frames_path.open(newline="") as frames_file:
        frames = [(float(row["t_camera_s"]), float(row["pc_time_s"])) for row in csv.DictReader(frames_file)]
    if not rows or not frames:
        raise ValueError(f"{folder} has no sEMG samples or no frames")
    t_us = np.array([row[0] for row in rows], dtype=np.int64)       # > already unwrapped by the recorder
    origin_us = int(t_us[0])
    esp32_s = (t_us - origin_us).astype(np.float64) / US_PER_S
    return (esp32_s, np.array([row[1] for row in rows]), origin_us,
            np.array([frame[0] for frame in frames]), np.array([frame[1] for frame in frames]))


# ---------------------------------------------------------------------------
# laptop_clock(folder, cfg)
# What:  fit both fastest-arrival lines for one recording.
# Time:  O(N + F).  Space: O(N + F).
# Data structures: LaptopClock.  Files used: as load_arrivals.
# ---------------------------------------------------------------------------
def laptop_clock(folder: Path, cfg: ClockSyncConfig = CLOCK_CONFIG) -> LaptopClock:
    """ESP32 -> laptop and camera -> laptop lines of one webcam recording."""
    esp32_s, esp32_pc, origin_us, camera_s, camera_pc = load_arrivals(folder)
    esp32 = lower_envelope_line(esp32_s, esp32_pc - esp32_s, cfg)
    camera = lower_envelope_line(camera_s, camera_pc - camera_s, cfg)
    trace = {"esp32_s": esp32_s, "esp32_delay_ms": (esp32_pc - esp32_s - esp32.at(esp32_s)) * MS_PER_S,
             "camera_s": camera_s, "camera_delay_ms": (camera_pc - camera_s - camera.at(camera_s)) * MS_PER_S}
    return LaptopClock(esp32=esp32, camera=camera, esp32_origin_us=origin_us, camera_delay_ms=cfg.camera_delay_ms,
                       trace=trace)


# ---------------------------------------------------------------------------
# clock_check_ms(clock, flash_us, flash_video_s)
# What:  when the LED WAS seen: how far the laptop-clock line puts each flash from where
#        the camera saw it, on the ESP32 clock (+ = the laptop clock says later). With
#        the camera delay right these sit around 0; their mean (plus the delay used) is a
#        fresh measurement of the camera delay.
# Time:  O(L) flashes.  Space: O(L).
# Data structures: tuple of floats.  Files used: none.
# ---------------------------------------------------------------------------
def clock_check_ms(clock: LaptopClock, flash_us: list[int], flash_video_s: list[float]) -> tuple[float, ...]:
    """Laptop-clock error at every LED flash, in ms."""
    fit = clock.to_clock_fit()
    predicted_us = np.asarray(fit.to_esp32_us(np.asarray(flash_video_s, dtype=np.float64)))
    return tuple(float(p - t) / 1000.0 for p, t in zip(predicted_us.tolist(), flash_us))
