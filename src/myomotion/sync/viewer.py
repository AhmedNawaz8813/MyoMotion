"""synced_view.mp4: the set video with the synced sEMG next to it, one output frame per video frame.

Left: the set video, with Ahmad's overlay (skeleton, spine, his phase, angles, cues) when cv_annotated.mp4 exists,
else the plain video. Right: a panel from merged.csv for the SAME frame: the phase (our names, the UI's colours),
the rep, the %MVC (or envelope in mV) as a bar and a number, and the last few seconds as a trace.
So anyone can see the two pipelines line up: the bar rises as the pull starts (about 50-100 ms before the
movement is visible: electromechanical delay).

Data structures (this file): a MergedRows dataclass of NumPy arrays; a slice of the value array per trace; NumPy
    images; a cv2.VideoCapture and cv2.VideoWriter.
Time complexity (this file): O(F) to load merged.csv; O(F * (P + K)) to render F frames with a P-pixel panel and K
    trace points each (the trace window is found by binary search, O(log F)).
"""

from __future__ import annotations   # > type hints are stored as text, not run: allows forward references

import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from myomotion.sync.cv_frames import CV_VIDEO_NAME

SYNCED_VIDEO_NAME = "synced_view.mp4"


# ---------------------------------------------------------------------------
# ViewerConfig
# What:  panel size, colours (BGR, from the team's UI design board) and scales.
# Data structures: frozen dataclass; dict phase -> colour.
# Time:  O(1) to build (fixed fields).
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ViewerConfig:
    """Look of the synced video."""

    panel_width: int = 300
    background: tuple[int, int, int] = (37, 25, 21)          # > #151925 dark video
    text: tuple[int, int, int] = (240, 240, 240)
    dim: tuple[int, int, int] = (160, 160, 160)
    activation: tuple[int, int, int] = (110, 42, 201)        # > #C92A6E activation magenta
    phase_colours: tuple[tuple[str, tuple[int, int, int]], ...] = (
        ("CONCENTRIC", (224, 79, 31)),                         # > #1F4FE0 "Up"
        ("ISOMETRIC_HOLD", (222, 210, 205)),                   # > #CDD2DE "Hold"
        ("ECCENTRIC", (255, 191, 169)),                        # > #A9BFFF "Down"
    )
    percent_scale: float = 120.0                             # > bar top = 120 %MVC (a push can pass 100)
    min_mv_scale: float = 50.0                               # > bar top in mV mode is at least this
    trace_s: float = 5.0
    fourcc: str = "mp4v"


VIEWER_CONFIG = ViewerConfig()


# ---------------------------------------------------------------------------
# MergedRows
# What:  the columns of merged.csv the panel needs, as NumPy arrays (NaN = blank).
# Data structures: dataclass of arrays + list of phase names.
# Time:  O(1) to build (references to the arrays).
# ---------------------------------------------------------------------------
@dataclass
class MergedRows:
    """merged.csv, the columns the viewer draws."""

    t_s: np.ndarray
    value: np.ndarray            # %MVC, or the envelope in mV when there was no MVC
    unit: str                    # "%MVC" or "mV"
    phase: list[str]
    rep: list[str]


# ---------------------------------------------------------------------------
# load_merged(path)
# What:  read merged.csv; %MVC when the column has values, else the envelope in mV.
# Time:  O(F) rows.  Space: O(F).
# Data structures: lists -> NumPy arrays.  Files used: reads <folder>/merged.csv.
# ---------------------------------------------------------------------------
def load_merged(path: Path) -> MergedRows:
    """merged.csv as arrays."""
    # Time:  O(1).
    # Data structures: a str in, a float out (NaN for a blank cell).
    def number(text: str) -> float:
        return float(text) if text not in ("", None) else math.nan

    with Path(path).open(newline="") as merged_file:
        rows = list(csv.DictReader(merged_file))
    percent = np.array([number(row["percent_mvc"]) for row in rows])
    use_percent = bool(np.any(~np.isnan(percent)))
    value = percent if use_percent else np.array([number(row["envelope_mv"]) for row in rows])
    return MergedRows(t_s=np.array([float(row["t_video_s"]) for row in rows]), value=value,
                      unit="%MVC" if use_percent else "mV", phase=[row["phase"] for row in rows],
                      rep=[row["rep"] for row in rows])


# ---------------------------------------------------------------------------
# draw_panel(height, rows, index, scale, cfg)
# What:  the right-hand panel for frame `index`: phase name in its colour, rep, the value
#        as a bar (100 %MVC line) and number, the last trace_s seconds as a line.
# Time:  O(K) for the K frames inside the trace window.  Space: O(height * panel_width).
# Data structures: NumPy image; a slice of the value array for the trace.
# Files used: none.
# ---------------------------------------------------------------------------
def draw_panel(height: int, rows: MergedRows, index: int, scale: float,
               cfg: ViewerConfig = VIEWER_CONFIG) -> np.ndarray:
    """One panel image (height x panel_width, BGR)."""
    import cv2
    width = cfg.panel_width
    panel = np.zeros((height, width, 3), np.uint8)
    panel[:] = cfg.background
    font = cv2.FONT_HERSHEY_SIMPLEX
    phase, rep = rows.phase[index], rows.rep[index]
    colour = dict(cfg.phase_colours).get(phase, cfg.dim)
    cv2.putText(panel, "sEMG (synced)", (12, 26), font, 0.55, cfg.dim, 1, cv2.LINE_AA)
    cv2.putText(panel, phase or "-", (12, 58), font, 0.75, colour, 2, cv2.LINE_AA)
    cv2.putText(panel, f"rep {rep}" if rep else "", (12, 86), font, 0.6, cfg.text, 1, cv2.LINE_AA)
    cv2.putText(panel, f"t = {rows.t_s[index]:6.2f} s", (12, 112), font, 0.5, cfg.dim, 1, cv2.LINE_AA)

    # > the bar: from the bottom up to the value, with a line at 100 %MVC
    top, bottom, left, right = 130, height - 120, width - 70, width - 30
    cv2.rectangle(panel, (left, top), (right, bottom), (90, 90, 90), 1)
    value = rows.value[index]
    if not math.isnan(value):
        level = int(bottom - (bottom - top) * min(max(value, 0.0), scale) / scale)
        cv2.rectangle(panel, (left + 2, level), (right - 2, bottom - 1), cfg.activation, -1)
    if rows.unit == "%MVC":
        y100 = int(bottom - (bottom - top) * 100.0 / scale)
        cv2.line(panel, (left - 6, y100), (right + 6, y100), (220, 220, 220), 1)
        cv2.putText(panel, "100", (left - 40, y100 + 4), font, 0.4, (220, 220, 220), 1, cv2.LINE_AA)
    label = "--" if math.isnan(value) else str(int(round(value)))   # > never "-0"
    cv2.putText(panel, label, (12, bottom - 10), font, 1.4, cfg.text, 3, cv2.LINE_AA)
    cv2.putText(panel, rows.unit, (12, bottom + 18), font, 0.55, cfg.dim, 1, cv2.LINE_AA)

    # > the trace: frames from (now - trace_s) to now, newest on the right
    x0, x1, y0, y1 = 10, width - 10, height - 95, height - 12
    cv2.rectangle(panel, (x0, y0), (x1, y1), (70, 70, 70), 1)
    now = rows.t_s[index]
    first = int(np.searchsorted(rows.t_s, now - cfg.trace_s))          # > binary search: O(log F)
    window_t, window_v = rows.t_s[first:index + 1], rows.value[first:index + 1]
    keep = ~np.isnan(window_v)
    if int(np.sum(keep)) >= 2:
        xs = x1 - (now - window_t[keep]) / cfg.trace_s * (x1 - x0)
        ys = y1 - (y1 - y0) * np.clip(window_v[keep], 0.0, scale) / scale
        cv2.polylines(panel, [np.stack([xs, ys], axis=1).astype(np.int32)], False, (180, 120, 255), 2, cv2.LINE_AA)
    return panel


# ---------------------------------------------------------------------------
# render_synced_video(folder, video, out, cfg)
# What:  write synced_view.mp4: every frame of cv_annotated.mp4 (or the plain video)
#        side by side with its panel. Frame i of the video is row i of merged.csv
#        (both are the video's frames in order).
# Time:  O(F * (P + K)) for F frames, P panel pixels, K trace points.
# Space: O(one frame) + O(F) for the merged arrays.
# Data structures: MergedRows; cv2.VideoCapture / VideoWriter.
# Files used: reads <folder>/merged.csv and <folder>/cv_annotated.mp4 (else the video);
#        writes <folder>/synced_view.mp4.
# ---------------------------------------------------------------------------
def render_synced_video(folder: Path, video: Path | None = None, out: Path | None = None,
                        cfg: ViewerConfig = VIEWER_CONFIG) -> Path:
    """Write the side-by-side synced video; returns its path."""
    import cv2
    folder = Path(folder)
    rows = load_merged(folder / "merged.csv")
    annotated = folder / CV_VIDEO_NAME
    source = annotated if annotated.exists() else (Path(video) if video else folder / "video.mp4")
    out = Path(out) if out else folder / SYNCED_VIDEO_NAME
    finite = rows.value[~np.isnan(rows.value)]
    largest = float(finite.max()) if finite.size else 0.0
    scale = cfg.percent_scale if rows.unit == "%MVC" else max(cfg.min_mv_scale, largest)
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise OSError(f"could not open video {source}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    writer = None
    try:
        for index in range(len(rows.t_s)):
            ok, frame = capture.read()
            if not ok:
                break
            canvas = np.hstack([frame, draw_panel(frame.shape[0], rows, index, scale, cfg)])
            if writer is None:
                writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*cfg.fourcc), fps,
                                         (canvas.shape[1], canvas.shape[0]))
                if not writer.isOpened():
                    raise OSError(f"could not open a video writer for {out}")
            writer.write(canvas)
    finally:
        capture.release()
        if writer is not None:
            writer.release()
    return out
