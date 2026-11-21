"""Finding the sync LED's flashes in a video (video-sEMG sync, test TC03).

The ESP32 firmware (myomotion_emg_wifi.ino) lights an LED on pin D25 for 200 ms every time the recorder
sends "S", and writes that moment on the ESP32 clock into events.csv (source = board, event = LED_ON). The
recorder flashes at the start of the set, once in the middle and at the end. This file finds the same
flashes in the phone video, on the video clock; src/myomotion/sync/align.py then lines the two clocks up.

Finding the LED without being told where it is (two passes over the video):
  1. Coarse pass. Every frame is shrunk to a small brightness grid: each block holds the mean of its
     pixels' brightest colour channel, so a red, green or white LED all count. A small LED still lifts its
     block clearly, and a whole set fits in memory (frames x blocks bytes).
     In every block a "pulse" is a sharp rise followed by a sharp fall 0.06-0.6 s later (the LED is on
     for 0.2 s). A moving person makes pulses all over the place and of all sizes; the LED makes exactly
     one pulse per flash, all about the same size, and nothing else. So a block scores
         (the smallest of its `count` biggest pulses) - (its next biggest pulse),
     and the best blocks are tried in order until one shows flashes spaced like the ESP32's LED_ON events.
  2. Fine pass. Every pixel around the chosen block is kept; the pixels that brighten most during the
     flashes are the LED. Their mean brightness gives a clean on/off signal, and each flash time is where
     it passes halfway from off to on, interpolated between the two frames around it.

Frame times come from Ahmad's iter_video_frames (src/myomotion/vision/video_input.py), which reads them
from the video file itself: correct for phone video with an uneven frame rate, and exactly the times his
pose pipeline uses, so the LED and the pose share one clock.

Data structures (this file): NumPy uint8 / float32 arrays (frames x grid blocks, frames x box pixels); shifted array
    views for the pulse search; lists of (frame, size) tuples and LedFlash dataclasses; a LedTrace dataclass of per-
    frame arrays.
Time complexity (this file): two decodes of the video, O(T * P) each for T frames of P pixels (decoding dominates);
    pulse scoring O(T * B * K) for B grid blocks and K allowed on-times; flash timing O(T).
"""

from __future__ import annotations   # > type hints are stored as text, not run: allows forward references

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from myomotion.vision.video_input import iter_video_frames


# ---------------------------------------------------------------------------
# LedConfig  (the CONFIG for this file)
# What:  every tunable number of the LED search, in one place.
# Data structures: a frozen dataclass (named, typed, cannot change by accident).
# Time:  O(1) to build (fixed fields).
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LedConfig:
    """Settings for finding the sync LED."""

    grid_width: int = 128            # coarse pass: blocks across the frame (1920 px wide -> 15 x 15 px blocks)
    min_on_s: float = 0.06           # the LED is on 0.2 s; accept 0.06-0.6 s (motion blur, slow cameras)
    max_on_s: float = 0.6
    min_gap_s: float = 1.0           # two flashes are never closer than this
    min_pulse: float = 3.0           # grey levels (0-255) in a block mean: weaker pulses are noise
    candidates: int = 25             # blocks tried, best score first, until the flash spacing matches
    gap_frames_slack: int = 3        # flash spacing may differ from the ESP32's by 3 frames ...
    gap_ppm_slack: float = 2000.0    # ... plus 2000 ppm (clocks drift ~100 ppm; this only rejects wrong pairs)
    box_blocks: int = 1              # fine pass box = the chosen block plus 1 block on every side
    led_pixel_fraction: float = 0.5  # LED pixels = those that brighten by at least half the most any pixel does
    level_window_s: float = 0.5      # off level = median brightness in the 0.5 s before each flash
    crossing_fraction: float = 0.5   # flash time = where brightness passes halfway from off to on
    block_chunk: int = 512           # coarse scoring works on 512 blocks at a time (bounds the memory used)


LED_CONFIG = LedConfig()


# Time:  O(1) (an exception type).
# Data structures: a subclass of ValueError carrying the message.
class LedNotFound(ValueError):
    """The video does not show the expected flashes (the message says what to check)."""


# ---------------------------------------------------------------------------
# LedFlash
# What:  one flash found in the video.
# Data structures: a frozen dataclass (a found flash never changes).
# Time:  O(1) to build (fixed fields).
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LedFlash:
    """One LED flash on the video clock."""

    t_s: float                       # when the LED came on (video seconds, between two frames)
    frame: int                       # index of the first frame at least half lit
    size: float                      # on level - off level (grey levels): how clearly it was seen


# ---------------------------------------------------------------------------
# LedTrace
# What:  everything found about the LED: the frame times, the LED's brightness per
#        frame, where it is, its flashes, and one frame of the video for the check
#        picture.
# Data structures: dataclass holding NumPy arrays (one value per frame, parallel to
#        times_s) and a list of LedFlash in time order.
# Time:  O(1) to build (references to the arrays); frame_interval_s is O(T).
# ---------------------------------------------------------------------------
@dataclass
class LedTrace:
    """The LED as seen in the video."""

    times_s: np.ndarray              # every frame's time (video seconds), from the video file
    brightness: np.ndarray           # mean brightness of the LED pixels in every frame
    box_px: tuple[int, int, int, int]   # x, y, width, height of the searched box in the full frame
    led_pixels: int                  # how many pixels were taken as the LED
    flashes: list[LedFlash]
    snapshot: np.ndarray | None = field(default=None, repr=False)   # BGR frame during the first flash

    # Time:  O(T) for T frames (median of the steps).  Space: O(T).
    # Data structures: the NumPy array of frame times.
    @property
    def frame_interval_s(self) -> float:
        """Typical time between frames (1 / frame rate)."""
        return median_step(self.times_s)


# ---------------------------------------------------------------------------
# median_step(times_s)
# What:  the typical time between frames. The median ignores the odd long gap in
#        variable-frame-rate phone video.
# Time:  O(T log T) (sorting inside the median).  Space: O(T).
# Data structures: NumPy array of the T - 1 steps.  Files used: none.
# ---------------------------------------------------------------------------
def median_step(times_s: np.ndarray) -> float:
    """Median time between consecutive frames (seconds)."""
    if len(times_s) < 2:
        raise LedNotFound("the video has fewer than 2 frames")
    return float(np.median(np.diff(times_s)))


# ---------------------------------------------------------------------------
# frames_for(seconds, frame_interval_s)
# What:  a duration in whole frames (at least 1).
# Time:  O(1).  Space: O(1).
# Data structures: none.  Files used: none.
# ---------------------------------------------------------------------------
def frames_for(seconds: float, frame_interval_s: float) -> int:
    """How many frames last `seconds`."""
    return max(1, int(round(seconds / frame_interval_s)))


# ---------------------------------------------------------------------------
# brightest_channel(frame_bgr)
# What:  per pixel, the brightest of its blue, green and red values, so the LED
#        stands out whatever its colour (a red LED is bright only in red).
# Time:  O(P) for P pixels.  Space: O(P).
# Data structures: NumPy uint8 image (height x width).  Files used: none.
# ---------------------------------------------------------------------------
def brightest_channel(frame_bgr: np.ndarray) -> np.ndarray:
    """Max over the colour channels, one value per pixel."""
    return frame_bgr.max(axis=2) if frame_bgr.ndim == 3 else frame_bgr


# ---------------------------------------------------------------------------
# block_means(image, block_px)
# What:  shrinks an image to a grid: every block_px x block_px square becomes its
#        mean. The right and bottom edges that do not fill a whole block are dropped.
# Time:  O(P).  Space: O(P / block_px^2) for the grid (plus a float copy of the image).
# Data structures: NumPy reshape to (rows, block, cols, block), then a mean over the
#        two block axes: no Python loop over pixels.  Files used: none.
# ---------------------------------------------------------------------------
def block_means(image: np.ndarray, block_px: int) -> np.ndarray:
    """Mean of every block_px x block_px block (rows x cols grid)."""
    rows, cols = image.shape[0] // block_px, image.shape[1] // block_px
    cropped = image[:rows * block_px, :cols * block_px].astype(np.float32)
    # > view the image as rows x block x cols x block, then average each block's pixels at once
    return cropped.reshape(rows, block_px, cols, block_px).mean(axis=(1, 3))


# ---------------------------------------------------------------------------
# read_block_grid(video_path, cfg)
# What:  coarse pass: every frame's time and its brightness grid.
# Time:  O(T * P) for T frames of P pixels (decoding dominates).
# Space: O(T * B) bytes for B blocks (uint8: 1 byte per block per frame; a 2-minute
#        60 fps video with 128 x 72 blocks is about 66 MB).
# Data structures: Python lists while reading (unknown length), stacked once into a
#        NumPy uint8 array (frames x blocks).
# Files used: reads the video through src/myomotion/vision/video_input.py (iter_video_frames).
# ---------------------------------------------------------------------------
def read_block_grid(video_path: Path,
                    cfg: LedConfig = LED_CONFIG) -> tuple[np.ndarray, np.ndarray, int, tuple[int, int]]:
    """(frame times, grid of shape frames x rows x cols, block size in px, (width, height))."""
    times: list[float] = []
    grids: list[np.ndarray] = []
    block_px = 0
    size = (0, 0)
    for t_s, frame in iter_video_frames(str(video_path)):
        if not grids:
            size = (frame.shape[1], frame.shape[0])
            block_px = max(1, frame.shape[1] // cfg.grid_width)
        times.append(t_s)
        grids.append(np.clip(np.rint(block_means(brightest_channel(frame), block_px)), 0, 255).astype(np.uint8))
    if len(grids) < 2:
        raise LedNotFound(f"could not read frames from {video_path}")
    return np.array(times), np.stack(grids), block_px, size


# ---------------------------------------------------------------------------
# read_box_pixels(video_path, box_px, keep_frame)
# What:  fine pass: every pixel (brightest channel) inside the box, for every frame,
#        and a copy of one full frame for the check picture.
# Time:  O(T * P) decoding.  Space: O(T * box pixels) bytes.
# Data structures: list of flattened uint8 boxes, stacked into frames x pixels.
# Files used: reads the video through src/myomotion/vision/video_input.py (iter_video_frames).
# ---------------------------------------------------------------------------
def read_box_pixels(video_path: Path, box_px: tuple[int, int, int, int],
                    keep_frame: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """(frame times, pixels of shape frames x box pixels, the kept full frame or None)."""
    x, y, w, h = box_px
    times: list[float] = []
    boxes: list[np.ndarray] = []
    kept: np.ndarray | None = None
    for index, (t_s, frame) in enumerate(iter_video_frames(str(video_path))):
        times.append(t_s)
        boxes.append(brightest_channel(frame[y:y + h, x:x + w]).reshape(-1).copy())
        if index == keep_frame:
            kept = frame.copy()
    if len(boxes) < 2 or boxes[0].size == 0:
        raise LedNotFound(f"could not read the box {box_px} from {video_path}")
    return np.array(times), np.stack(boxes), kept


# ---------------------------------------------------------------------------
# pulse_strength(series, frame_interval_s, cfg)
# What:  for every frame t and every column: how strongly a flash could start at t.
#        rise[t]  = brightness[t] - brightness[t - 2]   (over two frames, so a flash
#                   that starts halfway through a frame's exposure is caught whole)
#        fall[t]  = the biggest drop k frames later, for k across the allowed on-time
#        pulse[t] = min(rise, fall): big only when the block lights up AND goes dark
#                   again soon after, which is what the LED does.
# Time:  O(T * C * K) for T frames, C columns and K allowed on-times (about 20 at 30 fps).
# Space: O(T * C).
# Data structures: NumPy float32 arrays (frames x columns), shifted views (no copies
#        per shift).  Files used: none.
# ---------------------------------------------------------------------------
def pulse_strength(series: np.ndarray, frame_interval_s: float, cfg: LedConfig = LED_CONFIG) -> np.ndarray:
    """Flash-likeness of every frame (rows) in every column."""
    values = series.astype(np.float32).reshape(len(series), -1)
    step = np.zeros_like(values)
    step[2:] = values[2:] - values[:-2]
    rise = np.maximum(step, 0.0)
    fall = np.zeros_like(values)
    k_min = max(1, int(cfg.min_on_s / frame_interval_s))
    k_max = max(k_min, int(np.ceil(cfg.max_on_s / frame_interval_s)))
    for k in range(k_min, min(k_max, len(values) - 1) + 1):
        # > fall[t] keeps the biggest drop seen k frames after t (step[t + k] < 0 is a drop)
        np.maximum(fall[:-k], -step[k:], out=fall[:-k])
    return np.minimum(rise, fall)


# ---------------------------------------------------------------------------
# strongest_pulses(pulses, count, gap_frames)
# What:  the `count` biggest pulses of one series, at least gap_frames apart (pick
#        the biggest, blank out its neighbourhood, repeat), returned in time order.
# Time:  O(count * T).  Space: O(T) (one working copy).
# Data structures: NumPy copy that gets blanked; list of (frame, size) tuples.
# Files used: none.
# ---------------------------------------------------------------------------
def strongest_pulses(pulses: np.ndarray, count: int, gap_frames: int) -> list[tuple[int, float]]:
    """[(frame, size), ...] of the biggest well-separated pulses, in time order."""
    remaining = np.asarray(pulses, dtype=np.float64).copy()
    picks: list[tuple[int, float]] = []
    for _ in range(count):
        frame = int(np.argmax(remaining))
        if remaining[frame] <= 0:
            break
        picks.append((frame, float(remaining[frame])))
        remaining[max(0, frame - gap_frames):frame + gap_frames + 1] = 0.0   # > one flash counts once
    return sorted(picks)


# ---------------------------------------------------------------------------
# score_blocks(grid, count, frame_interval_s, cfg)
# What:  coarse pass score of every block (see the file docstring):
#        smallest of its `count` biggest pulses minus its next biggest pulse.
#        Blocks whose flashes are too weak score -inf.
# Time:  O(T * B * K) for the pulses + O(B * count * gap) for the blanking.
# Space: O(T * chunk): blocks are scored block_chunk at a time.
# Data structures: NumPy arrays; argmax per column picks every block's best frame
#        at once; a small Python loop blanks each pick.  Files used: none.
# ---------------------------------------------------------------------------
def score_blocks(grid: np.ndarray, count: int, frame_interval_s: float, cfg: LedConfig = LED_CONFIG) -> np.ndarray:
    """One score per block (flattened rows x cols order)."""
    flat = grid.reshape(len(grid), -1)
    gap = frames_for(cfg.min_gap_s, frame_interval_s)
    scores = np.full(flat.shape[1], -np.inf)
    for start in range(0, flat.shape[1], cfg.block_chunk):
        pulses = pulse_strength(flat[:, start:start + cfg.block_chunk], frame_interval_s, cfg)
        columns = np.arange(pulses.shape[1])
        best = np.zeros((count + 1, pulses.shape[1]), dtype=np.float32)
        for rank in range(count + 1):
            frames = pulses.argmax(axis=0)                      # > each block's biggest remaining pulse
            best[rank] = pulses[frames, columns]
            for column, frame in enumerate(frames):
                pulses[max(0, frame - gap):frame + gap + 1, column] = 0.0
        weakest_flash = best[:count].min(axis=0)
        chunk_scores = weakest_flash - best[count]
        chunk_scores[weakest_flash < cfg.min_pulse] = -np.inf
        scores[start:start + len(chunk_scores)] = chunk_scores
    return scores


# ---------------------------------------------------------------------------
# gaps_match(flash_times_s, expected_gaps_s, frame_interval_s, cfg)
# What:  True when the flashes are spaced like the ESP32's LED_ON events (within a
#        few frames plus a generous drift). This is what tells the LED apart from a
#        light that happens to blink the right number of times.
# Time:  O(F) for F flashes.  Space: O(F).
# Data structures: lists.  Files used: none.
# ---------------------------------------------------------------------------
def gaps_match(flash_times_s: list[float], expected_gaps_s: list[float] | None, frame_interval_s: float,
               cfg: LedConfig = LED_CONFIG) -> bool:
    """Do the video flashes have the expected spacing?"""
    if expected_gaps_s is None:
        return True
    if len(flash_times_s) != len(expected_gaps_s) + 1:
        return False
    for index, expected in enumerate(expected_gaps_s):
        seen = flash_times_s[index + 1] - flash_times_s[index]
        allowed = cfg.gap_frames_slack * frame_interval_s + cfg.gap_ppm_slack * 1e-6 * expected
        if abs(seen - expected) > allowed:
            return False
    return True


# ---------------------------------------------------------------------------
# flash_times(times_s, brightness, count, cfg)
# What:  fine timing on the LED's own brightness: finds the `count` flashes, then for
#        each one the moment brightness passes halfway between its off level (median
#        of the 0.5 s before) and its on level (brightest frame of the flash), by
#        straight-line interpolation between the frame below and the frame above.
# Time:  O(T) for the pulses + O(count * window) for the levels.  Space: O(T).
# Data structures: NumPy arrays; list of LedFlash.  Files used: none.
# ---------------------------------------------------------------------------
def flash_times(times_s: np.ndarray, brightness: np.ndarray, count: int, cfg: LedConfig = LED_CONFIG) -> list[LedFlash]:
    """The flashes in a 1-D LED brightness signal, timed between frames."""
    interval = median_step(times_s)
    pulses = pulse_strength(brightness, interval, cfg)[:, 0]
    picks = strongest_pulses(pulses, count, frames_for(cfg.min_gap_s, interval))
    on_frames = frames_for(cfg.max_on_s, interval)
    level_frames = frames_for(cfg.level_window_s, interval)
    flashes: list[LedFlash] = []
    for rise_frame, _ in picks:
        before = brightness[max(0, rise_frame - 2 - level_frames):max(1, rise_frame - 2)]
        off = float(np.median(before))
        on = float(np.max(brightness[rise_frame:rise_frame + on_frames + 1]))
        half = off + cfg.crossing_fraction * (on - off)
        # > walk forward from just before the rise to the first frame at or above halfway
        first = max(1, rise_frame - 2)
        while first < len(brightness) - 1 and brightness[first] < half:
            first += 1
        below, above = float(brightness[first - 1]), float(brightness[first])
        fraction = (half - below) / (above - below) if above > below else 1.0
        fraction = min(1.0, max(0.0, fraction))
        t_s = float(times_s[first - 1] + fraction * (times_s[first] - times_s[first - 1]))
        flashes.append(LedFlash(t_s=t_s, frame=first, size=on - off))
    return flashes


# ---------------------------------------------------------------------------
# led_box(block_index, grid_shape, block_px, frame_size, cfg)
# What:  the fine-pass box in pixels: the chosen block plus box_blocks blocks on
#        every side, kept inside the frame.
# Time:  O(1).  Space: O(1).
# Data structures: a tuple (x, y, width, height).  Files used: none.
# ---------------------------------------------------------------------------
def led_box(block_index: int, grid_shape: tuple[int, int], block_px: int, frame_size: tuple[int, int],
            cfg: LedConfig = LED_CONFIG) -> tuple[int, int, int, int]:
    """Pixel box around a block of the coarse grid."""
    row, col = divmod(block_index, grid_shape[1])
    x0 = max(0, (col - cfg.box_blocks) * block_px)
    y0 = max(0, (row - cfg.box_blocks) * block_px)
    x1 = min(frame_size[0], (col + cfg.box_blocks + 1) * block_px)
    y1 = min(frame_size[1], (row + cfg.box_blocks + 1) * block_px)
    return x0, y0, x1 - x0, y1 - y0


# ---------------------------------------------------------------------------
# use_frame_times(times, frame_times)
# What:  swaps the frame times read from the video file for better ones when the
#        recorder saved them (video_frames.csv: the webcam driver's own timestamp of
#        every frame). Webcam files are written at one nominal frame rate, so their
#        container times are only approximate. Returns the times and how many frames
#        both sources share (extra frames on either side are dropped).
# Time:  O(T).  Space: O(T).
# Data structures: NumPy arrays.  Files used: none.
# ---------------------------------------------------------------------------
def use_frame_times(times: np.ndarray, frame_times: np.ndarray | None) -> tuple[np.ndarray, int]:
    """(frame times to use, number of frames)."""
    if frame_times is None:
        return times, len(times)
    count = min(len(times), len(frame_times))
    return np.asarray(frame_times[:count], dtype=np.float64), count


# ---------------------------------------------------------------------------
# coarse_search(video_path, count, expected_gaps_s, cfg)
# What:  coarse pass: scores every block, then tries the best ones until one shows
#        `count` flashes with the expected spacing. Returns that block's pixel box and
#        its rough flash frames.
# Time:  O(T * P) decoding + O(T * B * K) scoring + O(candidates * T) checking.
# Space: O(T * B) bytes for the grid.
# Data structures: NumPy grid; argsort for the candidate order.
# Files used: reads the video (read_block_grid).
# ---------------------------------------------------------------------------
def coarse_search(video_path: Path, count: int, expected_gaps_s: list[float] | None,
                  cfg: LedConfig = LED_CONFIG,
                  frame_times: np.ndarray | None = None) -> tuple[tuple[int, int, int, int], list[int]]:
    """(pixel box around the LED, rough flash frames)."""
    times, grid, block_px, size = read_block_grid(video_path, cfg)
    times, frames = use_frame_times(times, frame_times)
    grid = grid[:frames]
    interval = median_step(times)
    scores = score_blocks(grid, count, interval, cfg)
    flat = grid.reshape(len(grid), -1)
    gap = frames_for(cfg.min_gap_s, interval)
    for block in np.argsort(-scores)[:cfg.candidates]:
        if not np.isfinite(scores[block]):
            break
        picks = strongest_pulses(pulse_strength(flat[:, block], interval, cfg)[:, 0], count, gap)
        if len(picks) == count and gaps_match([float(times[f]) for f, _ in picks], expected_gaps_s, interval, cfg):
            return led_box(int(block), grid.shape[1:], block_px, size, cfg), [f for f, _ in picks]
    raise LedNotFound(
        f"no spot in the video flashes {count} times with the spacing of the ESP32's LED_ON events. "
        "Check that the LED is in view and not covered, that filming started before the first flash and "
        "stopped after the last, and that the video belongs to this recording.")


# ---------------------------------------------------------------------------
# find_led(video_path, count, expected_gaps_s, box_px, cfg)
# What:  the whole search: coarse pass (unless box_px is given), then the fine pass:
#        keeps every pixel in the box, takes as the LED the pixels that brighten most
#        during the flashes, and times each flash on their mean brightness.
# Time:  two decodes of the video, O(T * P) each, plus the scoring above.
# Space: O(T * B) for the coarse grid, O(T * box pixels) for the fine pass.
# Data structures: see LedTrace.  Files used: reads the video twice
#        (src/myomotion/vision/video_input.py through read_block_grid / read_box_pixels).
# ---------------------------------------------------------------------------
def find_led(video_path: Path, count: int, expected_gaps_s: list[float] | None = None,
             box_px: tuple[int, int, int, int] | None = None, cfg: LedConfig = LED_CONFIG,
             frame_times: np.ndarray | None = None) -> LedTrace:
    """Find the sync LED and time its `count` flashes on the video clock (frame_times: see use_frame_times)."""
    if count < 2:
        raise LedNotFound("sync needs at least 2 flashes (start and end of the set)")
    rough_frames: list[int] | None = None
    if box_px is None:
        box_px, rough_frames = coarse_search(Path(video_path), count, expected_gaps_s, cfg, frame_times)

    # > keep one frame for the check picture: during the first flash if we know it, else the first frame
    keep = rough_frames[0] + 1 if rough_frames else 0
    times, pixels, snapshot = read_box_pixels(Path(video_path), box_px, keep_frame=keep)
    times, frames = use_frame_times(times, frame_times)
    pixels = pixels[:frames]
    interval = median_step(times)
    if rough_frames is None:                                     # > a box was given: find the flashes in it
        box_mean = pixels.astype(np.float32).mean(axis=1)
        rough_frames = [f for f, _ in strongest_pulses(pulse_strength(box_mean, interval, cfg)[:, 0], count,
                                                       frames_for(cfg.min_gap_s, interval))]
        if len(rough_frames) < count:
            raise LedNotFound(f"found only {len(rough_frames)} of {count} flashes inside the box {box_px}")

    # > LED pixels: the ones that got much brighter during the flashes than just before them
    on_rows = [min(len(pixels) - 1, f + 1) for f in rough_frames]
    off_rows = [max(0, f - 3) for f in rough_frames]
    brighter = pixels[on_rows].astype(np.float32).mean(axis=0) - pixels[off_rows].astype(np.float32).mean(axis=0)
    if brighter.max() <= 0:
        raise LedNotFound("the LED box does not get brighter during the flashes")
    led_pixels = brighter >= cfg.led_pixel_fraction * brighter.max()
    brightness = pixels[:, led_pixels].astype(np.float64).mean(axis=1)

    flashes = flash_times(times, brightness, count, cfg)
    if len(flashes) != count or not gaps_match([f.t_s for f in flashes], expected_gaps_s, interval, cfg):
        raise LedNotFound("the LED's flashes could not be timed consistently; check the video and the LED")
    return LedTrace(times_s=times, brightness=brightness, box_px=box_px, led_pixels=int(led_pixels.sum()),
                    flashes=flashes, snapshot=snapshot)
