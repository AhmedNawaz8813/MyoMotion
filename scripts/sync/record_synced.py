"""Record webcam video and sEMG at the same time, with a live view, then sync them (TC03).

Usage, from the repo root (the repo's .venv: it has OpenCV):
    source .venv/bin/activate
    python scripts/sync/record_synced.py --participant TEST --exercise forearm --posture fist --reps 3
    python scripts/sync/record_synced.py --participant P01 --exercise deadlift --posture normal --reps 5 \\
        --channels erector_spinae --mvc ~/Downloads/data/P01/S01_deadlift_mvc --cv

--cv adds Ahmad's deadlift CV (MediaPipe pose + SpinePose spine + his rep phases and form rules, all from
src/myomotion/, unchanged): his skeleton live in the window next to the %MVC, then after the set his analysis of
EVERY frame (cv_frames.csv, cv_summary.json, cv_annotated.mp4), joined to the sEMG in merged.csv, and
synced_view.mp4 (his annotated video beside the synced %MVC). Needs models/pose_landmarker_heavy.task.

Before: laptop on the MyoMotion-EMG Wi-Fi, ESP32 on the power bank, no other program connected to it
(close record_session.py and live_mvc.py), the webcam pointed at you. The sync LED does NOT have to be in the
picture: the sync uses the laptop's clock (src/myomotion/sync/clock_sync.py); when the LED is seen it is used and
also re-checks the laptop clock.
The window shows the camera, the cue, the live %MVC (or the envelope in mV without an MVC) and a trace;
press q (or Esc) in it to stop early (everything recorded so far is still saved).
Afterwards the set is synced at once (src/myomotion/sync/session.py): merged.csv, sync.json, sync.png
(+ the CV files above with --cv).

Thin command-line wrapper: the recording is src/myomotion/sync/recorder.py (record_set).
Exit code 0 = recorded and synced (TC03 PASS or NOT MEASURED), 1 = something failed.

Data structures (this file): an argparse Namespace; SetPlan, RecordingResult, CvRun and SyncResult dataclasses; the
    recorder's threads and buffers (see recorder.py).
Time complexity (this file): recording is real time, O(F + N) for F frames and N samples; afterwards the CV pass O(F
    * P) for P pixels, the sync O(N + F) (laptop clock) or O(F * P) (LED search), the envelope O(N * S) (N samples,
    S = 6 filter sections) and the synced video O(F * P).
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

# > This file is scripts/sync/record_synced.py, so parents[2] is the repo root; adding <repo>/src to
# > sys.path lets `import myomotion` work without `pip install -e .` (as in Ahmad's scripts).
_SRC = str(Path(__file__).resolve().parents[2] / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from myomotion.semg.acquisition import AcquisitionConfig, EspStream  # noqa: E402
from myomotion.semg.mvc import find_latest_mvc, load_mvc  # noqa: E402
from myomotion.sync.clock_sync import CLOCK_CONFIG  # noqa: E402
from myomotion.sync.cv_frames import (CV_CONFIG, LIVE_POSE_MODEL, POSE_MODELS, CvEngine, LiveCv,  # noqa: E402
                                      analyze_video, pose_model_path, run_report)
from myomotion.sync.led_sync import LedNotFound  # noqa: E402
from myomotion.sync.recorder import SetPlan, Webcam, next_set_folder, record_set  # noqa: E402
from myomotion.sync.session import SYNC_METHODS, format_report, load_frame_times, sync_recording  # noqa: E402
from myomotion.sync.viewer import render_synced_video  # noqa: E402
from sync_session import plot_sync  # noqa: E402  (same folder: scripts/sync/sync_session.py)

# > exposure 300 = 30 ms (V4L2 units of 0.1 ms): the longest that still fits 30 frames per second
DEFAULTS = {"data_dir": "~/Downloads/data", "camera": 0, "exposure": "300", "width": 640, "height": 480}


# ---------------------------------------------------------------------------
# parse_args(argv)
# What:  what to record and how (see the docstring).
# Time / Space: O(1).
# Data structures: argparse.Namespace.  Files used: none.
# ---------------------------------------------------------------------------
def parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Command-line options."""
    p = argparse.ArgumentParser(description="Record webcam video + sEMG together, with a live view, then sync")
    p.add_argument("--participant", default="TEST")
    p.add_argument("--exercise", default="forearm", help="forearm/grip/hand, curl, rdl, squat, bench or deadlift")
    p.add_argument("--posture", default="fist")
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--tempo", default="2,1,2", help="seconds per phase in the exercise's own order")
    p.add_argument("--channels", nargs="+", default=["forearm"], help="names of the WIRED channels, from ch1")
    p.add_argument("--channel", default="ch1", choices=["ch1", "ch2", "ch3"], help="channel shown live")
    p.add_argument("--mvc", type=Path, default=None, help="MVC folder or mvc.json (default: newest passed)")
    p.add_argument("--camera", type=int, default=DEFAULTS["camera"])
    p.add_argument("--exposure", default=DEFAULTS["exposure"],
                   help="Linux only: fixed exposure (keeps 30 fps in a dim room), or 'auto'; a Mac always uses auto")
    p.add_argument("--width", type=int, default=DEFAULTS["width"])
    p.add_argument("--height", type=int, default=DEFAULTS["height"])
    p.add_argument("--host", default=AcquisitionConfig.host)
    p.add_argument("--port", type=int, default=AcquisitionConfig.port)
    p.add_argument("--secure", action="store_true", help="the ESP32 runs the encrypted firmware")
    p.add_argument("--keys", type=Path, default=Path("~/.config/myomotion/keys.json"))
    p.add_argument("--out", type=Path, default=Path(DEFAULTS["data_dir"]))
    p.add_argument("--cv-json", type=Path, default=None, help="Ahmad's summary JSON for the sync (optional)")
    p.add_argument("--cv", action="store_true",
                   help="run the CV (curl tracker for --exercise curl, else Ahmad's deadlift analysis): live "
                        "skeleton + per-frame analysis joined into merged.csv")
    p.add_argument("--no-live-cv", action="store_true", help="with --cv: skip the live skeleton (slow laptop)")
    p.add_argument("--live-pose", default=LIVE_POSE_MODEL, choices=list(POSE_MODELS),
                   help="with --cv: MediaPipe model for the LIVE view (the dataset pass always uses heavy)")
    p.add_argument("--spine", default="auto", choices=["auto", "spinepose", "contour", "proxy"],
                   help="with --cv: spine source (auto = SpinePose if installed, else the back contour)")
    p.add_argument("--no-sync", action="store_true", help="record only")
    p.add_argument("--camera-delay-ms", type=float, default=CLOCK_CONFIG.camera_delay_ms,
                   help="this camera's delay for the laptop-clock sync (default: the Linux webcam's, measured 21 Nov); "
                        "a set where the LED is seen prints the measured value")
    p.add_argument("--sync", default="auto", choices=list(SYNC_METHODS),
                   help="clock = the laptop clock, no LED needed; led = the LED flashes; auto (default) = the LED "
                        "if the camera saw it, else the laptop clock")
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# main(argv)
# What:  load the MVC -> open the camera -> connect to the ESP32 -> record the set with
#        the live view -> print the recording summary -> sync and draw sync.png.
# Time / Space: as record_set and sync_recording.
# Data structures: SetPlan, RecordingResult, SyncResult.
# Files used: the MVC's mvc.json; writes the set folder (see the docstring).
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    args = parse_args(argv)
    out = args.out.expanduser()
    mvc_path = args.mvc.expanduser() if args.mvc else find_latest_mvc(out / args.participant)
    mvc_above_rest = None
    if mvc_path is not None:
        mvc = load_mvc(mvc_path)
        mvc_above_rest = float(mvc["mvc_above_rest_mv"])
        print(f"MVC: {mvc_path} ({'passed' if mvc.get('passed') else 'did NOT pass'} TC02)")
    else:
        print("No MVC found for this participant: the live view shows the envelope in mV, not %MVC.")
    plan = SetPlan(participant=args.participant, exercise=args.exercise, posture=args.posture, reps=args.reps,
                   tempo_s=tuple(float(x) for x in args.tempo.split(",")), channels=tuple(args.channels))

    try:
        camera = Webcam(args.camera, args.width, args.height,
                        None if args.exposure == "auto" else float(args.exposure))
    except RuntimeError as error:
        print(f"Camera: {error}")
        return 1
    live = None
    if args.cv and not args.no_live_cv:
        try:
            live = start_live_cv(args.spine, args.live_pose, args.exercise)
        except (FileNotFoundError, RuntimeError, ImportError) as error:
            camera.close()
            print(f"Ahmad's CV could not start: {error}")
            return 1
    try:
        keys = None
        if args.secure:
            from myomotion.semg.secure_link import load_keys
            keys = load_keys(args.keys)
        stream = EspStream(AcquisitionConfig(host=args.host, port=args.port), keys=keys)
    except (ConnectionError, FileNotFoundError) as error:
        camera.close()
        if live is not None:
            live.stop()
            live.engine.close()
        print(error)
        return 1

    folder = next_set_folder(out, args.participant, args.exercise, args.posture)
    print(camera.settings.note)
    print(f"Recording into {folder}\nGet yourself in the camera's view (the LED does not need to be). Follow the "
          "cues in the window; press q there to stop early.")
    try:
        result = record_set(camera, stream, plan, mvc_above_rest, folder, int(args.channel[2]) - 1, live_cv=live)
    except (ConnectionError, RuntimeError) as error:
        print(f"Recording failed: {error}")
        return 1
    finally:
        camera.close()
        stream.close()
        if live is not None:
            live.stop()

    q = result.quality
    print(f"\nSaved {q['samples']} samples over {q['duration_s']} s = {q['rate_hz']} per second; missing "
          f"{q['missing_samples']}. Video: {result.frames} frames at {result.fps:.1f} fps. LED flashes "
          f"answered by the ESP32: {result.flashes}." + ("  (stopped early)" if result.aborted else ""))
    if live is not None:
        print(f"Live CV: {live.analysed} frames analysed, {live.skipped} skipped (the live view only; the dataset "
              "comes from the pass below)." + (f"  ERROR: {live.error}" if live.error else ""))
    cv_json, cv_frames = args.cv_json, None
    if args.cv:
        try:
            cv_json, cv_frames = run_cv(folder, args.spine, live, args.exercise)
        except (FileNotFoundError, RuntimeError, ImportError, OSError) as error:
            print(f"Ahmad's CV failed: {error}")
            return 1
    elif live is not None:
        live.engine.close()
    if args.no_sync:
        return 0
    try:
        synced = sync_recording(folder, folder / "video.mp4", cv_json=cv_json, cv_frames=cv_frames,
                                mvc_path=mvc_path, channel=args.channel, method=args.sync,
                                clock_cfg=replace(CLOCK_CONFIG, camera_delay_ms=args.camera_delay_ms))
    except (LedNotFound, ValueError) as error:
        print(f"Sync failed: {error}\nYou can retry with scripts/sync/sync_session.py {folder} --sync clock "
              "(or --led-box x,y,w,h if the LED is in the picture)")
        return 1
    print(format_report(synced))
    plot_sync(synced, folder / "sync.png")
    print(f"Saved {synced.merged_path}, {synced.json_path} and {folder / 'sync.png'}")
    if args.cv:
        print(f"Saved {render_synced_video(folder)} (Ahmad's annotated video beside the synced sEMG)")
    return 1 if synced.report.verdict == "FAIL" else 0


# ---------------------------------------------------------------------------
# start_live_cv(spine, pose_size)
# What:  load Ahmad's CV (MediaPipe + the spine model) and start the live thread. The
#        live view uses the faster `full` pose model by default (14 vs 48 ms a frame).
# Time:  model loading, ~1-12 s (SpinePose is slowest the first time).  Space: the models.
# Data structures: CvEngine inside a LiveCv thread.
# Files used: src/myomotion/sync/cv_frames.py; models/pose_landmarker_heavy.task.
# ---------------------------------------------------------------------------
def start_live_cv(spine: str, pose_size: str = LIVE_POSE_MODEL, exercise: str = "deadlift") -> LiveCv:
    """A started LiveCv thread."""
    model, used = pose_model_path(pose_size)
    print(f"Loading the CV for the live view (MediaPipe {used}, exercise {exercise}) ...")
    engine = CvEngine(replace(CV_CONFIG, spine=spine, model_path=model), exercise=exercise)
    print(f"  {engine.spine_note}")
    live = LiveCv(engine)
    live.start()
    return live


# ---------------------------------------------------------------------------
# run_cv(folder, spine, live)
# What:  after the set, Ahmad's CV over EVERY frame of video.mp4 on the camera's clock
#        (video_frames.csv): writes cv_frames.csv, cv_summary.json, cv_annotated.mp4.
#        Reuses the live engine's SpinePose network (no second load) with a fresh
#        MediaPipe tracker and analyser (a new video starts at time 0).
# Time:  ~35-40 ms per frame on this laptop (~1 min for a 45 s set).  Space: O(frames) rows.
# Data structures: CvEngine, CvRun.  Files used: src/myomotion/sync/cv_frames.py; the set folder.
# ---------------------------------------------------------------------------
def run_cv(folder: Path, spine: str, live: LiveCv | None, exercise: str = "deadlift") -> tuple[Path, Path]:
    """(cv_summary.json, cv_frames.csv) of the recorded set."""
    cfg = replace(CV_CONFIG, spine=spine)
    if live is not None:
        live.engine.close(keep_spine=True)
        engine = CvEngine(cfg, spine_tracker=live.engine.spine_tracker, spine_mode=live.engine.spine_mode,
                          exercise=exercise)
    else:
        engine = CvEngine(cfg, exercise=exercise)
    print(f"\nCV on every frame (MediaPipe heavy, {exercise}, {engine.spine_note}) ...")
    try:
        run = analyze_video(folder / "video.mp4", folder, frame_times=load_frame_times(folder / "video.mp4"),
                            engine=engine, cfg=cfg)
    finally:
        engine.close()
    print(run_report(run, exercise))
    print(f"Saved {run.frames_path}, {run.summary_path}" + (f" and {run.video_path}" if run.video_path else ""))
    return run.summary_path, run.frames_path


if __name__ == "__main__":
    sys.exit(main())
