"""Sync one sEMG recording with its phone video (test TC03) and write merged.csv.

Usage, from the repo root:
    python scripts/sync/sync_session.py ~/Downloads/data/P01/S01_deadlift_normal
    python scripts/sync/sync_session.py <folder> --video clip.mp4 --cv-json output/clip.json --mvc <mvc folder>

The folder comes from record_session.py (emg.csv, events.csv with the ESP32's LED_ON events). Put the phone
video in it as video.mp4, or pass --video. Optional:
    --cv         run Ahmad's deadlift CV over every frame first (src/myomotion/sync/cv_frames.py): writes
                 cv_frames.csv, cv_summary.json and cv_annotated.mp4 into the folder (needs the pose model)
    --cv-json    Ahmad's summary JSON (default: the folder's cv_summary.json if there is one): rep phases
                 seen in the video; without it the recorder's cues are used
    --cv-frames  a per-frame CV CSV with a t_s column (default: the folder's cv_frames.csv), joined onto
                 merged.csv as cv_* columns
    --mvc        an MVC folder or mvc.json (default: the participant's newest MVC that passed TC02)
    --led-box    x,y,w,h of the LED in pixels, if the automatic search picks the wrong spot

Thin command-line wrapper: the logic lives in src/myomotion/sync/ (led_sync.py, align.py, session.py).
Writes merged.csv, sync.json, sync.png and synced_view.mp4 (the video, with Ahmad's overlay when there is a
cv_annotated.mp4, beside the synced %MVC) into the folder. Exit code 0 = PASS or NOT MEASURED, 1 = FAIL
or the sync could not be done.

Data structures (this file): an argparse Namespace; a SyncResult; a matplotlib figure; dicts of NumPy arrays for the
    laptop-clock plot.
Time complexity (this file): decoding the video dominates (O(T * P) for T frames of P pixels, twice when the LED is
    searched); the envelope O(N * S) (N samples, S = 6 filter sections); with --cv the CV pass O(T * P).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

# > This file is scripts/sync/sync_session.py, so parents[2] is the repo root; adding <repo>/src to
# > sys.path lets `import myomotion` work without `pip install -e .` (as in Ahmad's scripts).
_SRC = str(Path(__file__).resolve().parents[2] / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from myomotion.semg.mvc import find_latest_mvc  # noqa: E402
from myomotion.sync.clock_sync import CLOCK_CONFIG  # noqa: E402
from myomotion.sync.cv_frames import CV_CONFIG, analyze_video, find_cv_outputs, run_report  # noqa: E402
from myomotion.sync.led_sync import LedNotFound  # noqa: E402
from myomotion.sync.session import (SYNC_METHODS, SyncResult, find_video, format_report,  # noqa: E402
                                    load_frame_times, sync_recording)
from myomotion.sync.viewer import render_synced_video  # noqa: E402

# > phase colours from the team's UI design board (Up = concentric, Hold, Down = eccentric)
PLOT_CONFIG = {
    "figure_size": (13, 7),
    "dpi": 150,
    "colours": {"CONCENTRIC": "#1F4FE0", "ISOMETRIC_HOLD": "#CDD2DE", "ECCENTRIC": "#A9BFFF"},
    "flash_colour": "#378A58",
    "predicted_colour": "#EF4136",
    "signal_colour": "#C92A6E",
    "max_points": 3000,                 # laptop-clock plot: at most this many sEMG points
    "delay_ylim_ms": 60,
}


# ---------------------------------------------------------------------------
# parse_box(text)
# What:  "x,y,w,h" -> a tuple of 4 ints (for --led-box).
# Time:  O(1).  Space: O(1).
# Data structures: tuple.  Files used: none.
# ---------------------------------------------------------------------------
def parse_box(text: str) -> tuple[int, int, int, int]:
    """Parse --led-box."""
    parts = [int(value) for value in text.split(",")]
    if len(parts) != 4 or parts[2] <= 0 or parts[3] <= 0:
        raise argparse.ArgumentTypeError("--led-box needs x,y,width,height in pixels, e.g. 1500,80,60,60")
    return parts[0], parts[1], parts[2], parts[3]


# ---------------------------------------------------------------------------
# plot_sync(result, out_path)
# What:  sync.png: (1) LED method: the LED's brightness with the flashes found in the
#        video and where the ESP32 says they were (after the clock fit they should
#        coincide); laptop-clock method: how long each sample and frame took to reach
#        the laptop (the fastest ones fix the clocks), (2) the video frame during the
#        first flash with the LED box, or the clock numbers, (3) %MVC (or the envelope)
#        on the video clock with the rep phases shaded.
# Time:  O(N + F).  Space: O(N + F).
# Data structures: matplotlib figure (Agg backend: no window needed).
# Files used: writes <folder>/sync.png.
# ---------------------------------------------------------------------------
def plot_sync(result: SyncResult, out_path: Path) -> None:
    """Draw sync.png."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure = plt.figure(figsize=PLOT_CONFIG["figure_size"])
    grid = figure.add_gridspec(2, 2, width_ratios=[4, 1], height_ratios=[1, 1.3])
    led_axis = figure.add_subplot(grid[0, 0])
    snap_axis = figure.add_subplot(grid[0, 1])
    signal_axis = figure.add_subplot(grid[1, :], sharex=led_axis)

    if result.led is not None:
        _plot_led(led_axis, snap_axis, result)
    else:
        _plot_laptop_clock(led_axis, snap_axis, result)

    values = result.percent if result.percent is not None else result.envelope_mv
    signal_axis.plot(result.sample_video_s, values, color=PLOT_CONFIG["signal_colour"], linewidth=0.7)
    for interval in result.intervals:
        signal_axis.axvspan(interval.start_s, interval.end_s, color=PLOT_CONFIG["colours"].get(interval.phase, "#eee"),
                            alpha=0.35, linewidth=0)
    signal_axis.set_ylabel("%MVC" if result.percent is not None else "envelope (mV)")
    signal_axis.set_xlabel("video time (s)")
    source = "CV summary" if result.phase_source == "cv" else "recorder cues"
    signal_axis.set_title(f"sEMG on the video clock; shaded = rep phases from the {source} "
                          "(dark blue concentric, grey hold, light blue eccentric)", fontsize=10)
    signal_axis.set_xlim(float(result.frame_times_s[0]), float(result.frame_times_s[-1]))
    figure.tight_layout()
    figure.savefig(out_path, dpi=PLOT_CONFIG["dpi"])
    plt.close(figure)


# Time:  O(T + L) for T frames of LED brightness and L flashes.  Space: O(T).
# Data structures: NumPy arrays from the LedTrace; matplotlib axes; a Rectangle patch for the LED box.
def _plot_led(led_axis, snap_axis, result: SyncResult) -> None:
    """Top row for the LED method: brightness + flashes, and the LED box."""
    from matplotlib.patches import Rectangle
    led = result.led
    led_axis.plot(led.times_s, led.brightness, color="#333333", linewidth=0.8, label="LED brightness")
    for index, flash in enumerate(led.flashes):
        led_axis.axvline(flash.t_s, color=PLOT_CONFIG["flash_colour"], linewidth=2,
                         label="flash found in the video" if index == 0 else None)
    for index, t_us in enumerate(result.esp32_flash_us):
        led_axis.axvline(float(result.fit.to_video_s(t_us)), color=PLOT_CONFIG["predicted_colour"], linestyle="--",
                         linewidth=1, label="ESP32 LED_ON after the clock fit" if index == 0 else None)
    report = result.report
    check = (f"middle flash {max(abs(e) for e in report.check_errors_ms):.1f} ms off" if report.check_errors_ms
             else "no middle flash")
    led_axis.set_title(f"TC03 {report.verdict}: video starts at ESP32 {report.esp32_s_at_video_start:.3f} s, "
                       f"drift {report.drift_ppm:+.0f} ppm, {check} (limit {report.max_check_error_ms:.0f} ms)")
    led_axis.set_ylabel("brightness (0-255)")
    led_axis.legend(loc="upper right", fontsize=8)

    if led.snapshot is not None:
        snap_axis.imshow(led.snapshot[:, :, ::-1])                      # > OpenCV frames are BGR, matplotlib wants RGB
        x, y, w, h = led.box_px
        snap_axis.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor=PLOT_CONFIG["predicted_colour"],
                                      linewidth=2))
        snap_axis.set_title("LED box (first flash)", fontsize=9)
    snap_axis.axis("off")


# Time:  O(N / step + F): at most max_points sEMG delays and F frame delays are drawn.  Space: O(N / step + F).
# Data structures: the LaptopClock's trace (dict of NumPy arrays), thinned with a slice step; matplotlib axes.
def _plot_laptop_clock(clock_axis, text_axis, result: SyncResult) -> None:
    """Top row for the laptop-clock method: arrival delays above the fitted lines, and the numbers."""
    clock, fit = result.clock, result.fit
    trace = clock.trace
    step = max(1, len(trace["esp32_s"]) // PLOT_CONFIG["max_points"])      # > thin out 15 000 samples for the plot
    esp32_video_s = fit.to_video_s(clock.esp32_origin_us + trace["esp32_s"][::step] * 1e6)
    clock_axis.plot(esp32_video_s, trace["esp32_delay_ms"][::step], ".", markersize=1.5,
                    color=PLOT_CONFIG["signal_colour"], label="sEMG sample: Wi-Fi delay above the fastest")
    clock_axis.plot(trace["camera_s"], trace["camera_delay_ms"], ".", markersize=2.5, color="#333333",
                    label="video frame: read delay above the fastest")
    clock_axis.set_ylim(-2, PLOT_CONFIG["delay_ylim_ms"])
    clock_axis.set_ylabel("arrival delay (ms)")
    clock_axis.set_title(f"Laptop-clock sync (no LED): video starts at ESP32 {fit.esp32_s_at_video_start:.3f} s, "
                         f"drift {fit.drift_ppm:+.0f} ppm; the fastest arrivals (0 ms) tie both clocks to the laptop")
    clock_axis.legend(loc="upper right", fontsize=8, markerscale=6)
    lines = ["Laptop clock", f"ESP32 vs laptop {clock.esp32.slope * 1e6:+.0f} ppm",
             f"camera vs laptop {clock.camera.slope * 1e6:+.0f} ppm",
             f"camera delay {clock.camera_delay_ms:.1f} ms", "(measured with the LED)"]
    if result.led_problem:
        lines += ["", "LED not found"]
    text_axis.text(0.02, 0.95, "\n".join(lines), va="top", fontsize=9, family="monospace")
    text_axis.axis("off")


# ---------------------------------------------------------------------------
# main()
# What:  parse the arguments, find the video and MVC, run the sync, print TC03,
#        draw sync.png.
# Time / Space: see sync_recording in src/myomotion/sync/session.py.
# Data structures: argparse.Namespace.
# Files used: the recording folder, its video, mvc.json, the CV files (see the docstring).
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Sync an sEMG recording with its video (TC03)")
    parser.add_argument("folder", type=Path, help="recording folder from record_session.py")
    parser.add_argument("--video", type=Path, default=None, help="the set's video (default: video.* in the folder)")
    parser.add_argument("--cv-json", type=Path, default=None, help="Ahmad's deadlift summary JSON for this video")
    parser.add_argument("--cv-frames", type=Path, default=None, help="per-frame CV CSV with a t_s column")
    parser.add_argument("--mvc", type=Path, default=None, help="MVC folder or mvc.json (default: newest passed)")
    parser.add_argument("--channel", default="ch1", help="sEMG channel: ch1, ch2 or ch3 (default ch1)")
    parser.add_argument("--led-box", type=parse_box, default=None, help="x,y,w,h of the LED (skip the search)")
    parser.add_argument("--no-plot", action="store_true", help="do not write sync.png")
    parser.add_argument("--cv", action="store_true", help="run Ahmad's deadlift CV over the video first")
    parser.add_argument("--spine", default="auto", choices=["auto", "spinepose", "contour", "proxy"],
                        help="with --cv: spine source (auto = SpinePose if installed, else the back contour)")
    parser.add_argument("--no-view", action="store_true", help="do not write synced_view.mp4")
    parser.add_argument("--camera-delay-ms", type=float, default=CLOCK_CONFIG.camera_delay_ms,
                        help="the camera's delay for the laptop-clock sync (default: the Linux webcam's)")
    parser.add_argument("--sync", default="auto", choices=list(SYNC_METHODS),
                        help="led = the LED flashes; clock = the laptop clock (webcam recordings, no LED needed); "
                             "auto = the LED if it is found, else the laptop clock")
    args = parser.parse_args(argv)

    folder = args.folder.expanduser()
    video = args.video.expanduser() if args.video else find_video(folder)
    if video is None or not video.exists():
        print(f"No video found. Copy the phone video into {folder} as video.mp4, or pass --video PATH.")
        return 1
    mvc = args.mvc.expanduser() if args.mvc else find_latest_mvc(folder.parent)
    if mvc is None:
        print("No MVC that passed TC02 found for this participant: showing the envelope in mV, not %MVC.")

    if args.cv:
        try:
            run_cv(folder, video, args.spine)
        except (FileNotFoundError, RuntimeError, ImportError, OSError) as error:
            print(f"Ahmad's CV failed: {error}")
            return 1
    found_json, found_frames = find_cv_outputs(folder)
    cv_json = args.cv_json.expanduser() if args.cv_json else found_json
    cv_frames = args.cv_frames.expanduser() if args.cv_frames else found_frames
    if cv_json or cv_frames:
        print(f"CV:    {cv_json or '-'}  {cv_frames or '-'}")
    try:
        result = sync_recording(folder, video, cv_json=cv_json, cv_frames=cv_frames, mvc_path=mvc,
                                channel=args.channel, box_px=args.led_box, method=args.sync,
                                clock_cfg=replace(CLOCK_CONFIG, camera_delay_ms=args.camera_delay_ms))
    except (LedNotFound, ValueError) as error:
        print(f"Sync failed: {error}")
        return 1

    print(f"Video: {video}")
    print(f"MVC:   {mvc if mvc else 'none'}")
    print(format_report(result))
    print(f"Saved {result.merged_path} and {result.json_path}")
    if not args.no_plot:
        plot_path = folder / "sync.png"
        plot_sync(result, plot_path)
        print(f"Saved {plot_path}")
    if not args.no_view:
        print(f"Saved {render_synced_video(folder, video)}")
    return 1 if result.report.verdict == "FAIL" else 0


# ---------------------------------------------------------------------------
# recorded_exercise(folder)
# What:  the exercise written in the set's metadata.json (default: deadlift).
# Time / Space: O(1).  Data structures: dict.  Files used: reads <folder>/metadata.json.
# ---------------------------------------------------------------------------
def recorded_exercise(folder: Path) -> str:
    """The set's exercise name."""
    path = folder / "metadata.json"
    return str(json.loads(path.read_text()).get("exercise", "deadlift")) if path.exists() else "deadlift"


# ---------------------------------------------------------------------------
# run_cv(folder, video, spine)
# What:  Ahmad's deadlift CV over every frame of the set's video, on the clock the LED
#        search uses (video_frames.csv from the synced recorder, else the file's own
#        timestamps); writes cv_frames.csv, cv_summary.json, cv_annotated.mp4.
# Time:  ~35-40 ms per frame on this laptop.  Space: O(frames) rows.
# Data structures: CvRun.  Files used: src/myomotion/sync/cv_frames.py; the set folder.
# ---------------------------------------------------------------------------
def run_cv(folder: Path, video: Path, spine: str) -> None:
    """Run the CV and print what it found."""
    cfg = replace(CV_CONFIG, spine=spine)
    exercise = recorded_exercise(folder)
    print(f"CV on every frame ({exercise}; loading MediaPipe) ...")
    run = analyze_video(video, folder, frame_times=load_frame_times(video), cfg=cfg, exercise=exercise)
    print(f"{run_report(run, exercise)}\n  ({run.spine_note})")


if __name__ == "__main__":
    sys.exit(main())
