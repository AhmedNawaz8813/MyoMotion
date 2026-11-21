"""Ahmad's deadlift CV run on a synced recording: one row per video frame, on the same clock as the sEMG.

The bridge between his vision and biomechanics code (src/myomotion/vision/, src/myomotion/biomechanics/,
src/myomotion/coaching/overlay.py: USED here, never changed) and the sync (src/myomotion/sync/session.py):

  CvEngine        one frame -> MediaPipe pose -> SpinePose / back contour -> DeadliftAnalyzer -> FrameResult,
                  the same steps as his scripts/demo_deadlift.py, as a class the recorder can call
  analyze_video   every frame of a saved set video -> cv_frames.csv (one row per frame: angles, spine regions,
                  phase, rep), cv_summary.json (his session summary: reps, phase changes, form events) and
                  cv_annotated.mp4 (his overlay drawn on every frame)
  LiveCv          a background thread that analyses the NEWEST camera frame and hands back that frame with his
                  overlay drawn on it; the recording loop never waits for it (the CV needs ~35-40 ms a frame on
                  this laptop, the camera delivers one every 33 ms)

Frame times: the camera's own times from video_frames.csv (the clock the LED search uses) when the synced
recorder wrote one, else the video file's timestamps (his iter_video_frames). The same clock as the LED search,
so cv_frames.csv joins merged.csv by time and his rep times are already on the video clock.

Data structures (this file): Ahmad's PoseTracker / SpinePoseTracker / DeadliftAnalyzer objects behind a CvEngine;
    CurlAnalyzer or RdlCounter for the demo exercises; a dict column -> text per CSV row; a one-slot mailbox under a
    threading.Condition (LiveCv); a CvRun dataclass; a dict phase -> frame count.
Time complexity (this file): O(F * P) for F frames of P pixels: the networks dominate (MediaPipe 14-61 ms +
    SpinePose 15-20 ms per frame on this laptop); the analysers are O(1) per frame; writing the CSV O(F * K) for K
    columns; LiveCv.submit O(1).
"""

from __future__ import annotations   # > type hints are stored as text, not run: allows forward references

import csv
import json
import statistics
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np

from myomotion.biomechanics import DeadliftAnalyzer, DeadliftRuleConfig, FrameResult, Phase
from myomotion.sync.curl import CURL_COLUMNS, CURL_EXERCISES, CurlAnalyzer, curl_row, draw_curl
from myomotion.sync.rdl import PHASE_LABELS as RDL_PHASE_LABELS
from myomotion.sync.rdl import RDL_EXERCISES, RdlCounter, draw_rdl_hud

CV_FRAMES_NAME = "cv_frames.csv"       # one row per video frame (joined onto merged.csv as cv_* columns)
CV_SUMMARY_NAME = "cv_summary.json"    # his myomotion.deadlift_session summary (reps -> phases in merged.csv)
CV_VIDEO_NAME = "cv_annotated.mp4"     # the set video with his overlay (skeleton, spine, phase, cues)
SPINE_MODES = ("auto", "spinepose", "contour", "proxy")
# > MediaPipe pose model sizes. Measured on this laptop (20 Nov, 640x360, person in view): lite 10 ms, full 14 ms,
# > heavy 48 ms per frame; heavy found the person in 115 of 300 frames, full 94, lite 95. So the live view uses
# > full (speed) and the dataset pass after the set uses heavy (accuracy).
POSE_MODELS = ("lite", "full", "heavy")
LIVE_POSE_MODEL = "full"

# > His six phases -> the four of our protocol (events.csv cues, merged.csv, the UI's Up / Hold / Down).
# > The deadlift is concentric first: the pull is CONCENTRIC, the lockout the ISOMETRIC_HOLD.
PHASE_TO_OURS: dict[Phase, str] = {
    Phase.CONCENTRIC_PULL: "CONCENTRIC",
    Phase.LOCKOUT: "ISOMETRIC_HOLD",
    Phase.ECCENTRIC_DESCENT: "ECCENTRIC",
    Phase.SETUP: "REST",
    Phase.REP_COMPLETE: "REST",
    Phase.STANDBY: "REST",
}

CV_COLUMNS = (
    "frame", "t_s", "person", "tracked", "back_only", "view", "phase_label", "phase", "rep_count",
    "hip_deg", "hip_signed_deg", "knee_deg", "trunk_incl_deg", "tibia_incl_deg", "trunk_tibia_deg",
    "spine_source", "spine_curve_visible", "lumbar_deg", "thoracic_deg", "cervical_deg", "pelvic_deg",
    "spine_confidence", "bias", "flags", "process_ms",
)


# ---------------------------------------------------------------------------
# CvConfig
# What:  every setting of the CV run in one place (no magic numbers below).
# Data structures: frozen dataclass.
# Time:  O(1) to build (fixed fields).
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CvConfig:
    """Settings of the CV run."""

    spine: str = "auto"                 # as demo_deadlift.py --spine: SpinePose if installed, else back contour
    spine_size: str = "medium"          # his measured best: separated rounded from neutral backs (G-34)
    spine_every: int = 1                # run SpinePose every N analysed frames (reuse the points in between)
    bridge_max_error: float = 0.35      # SpinePose and MediaPipe must agree on the body (his G-01 check)
    model_path: str | None = None       # None = models/pose_landmarker_heavy.task (src/myomotion/vision/pose.py)
    lumbar_cues: bool = False           # his default: lumbar cue off (noisy from 2-D keypoints, G-34)
    annotated_fourcc: str = "mp4v"      # codec of cv_annotated.mp4 (OpenCV on Linux cannot write H.264)
    progress_every_s: float = 5.0       # analyze_video prints progress this often (seconds of video)
    decimals: int = 2                   # angles in cv_frames.csv


CV_CONFIG = CvConfig()


# ---------------------------------------------------------------------------
# pose_model_path(size)
# What:  models/pose_landmarker_<size>.task, next to the heavy model his PoseTracker uses
#        by default; the heavy one if that size was not downloaded.
# Time / Space: O(1).  Data structures: Path.  Files used: checks models/*.task.
# ---------------------------------------------------------------------------
def pose_model_path(size: str) -> tuple[str, str]:
    """(model path, the size actually used)."""
    from myomotion.vision.pose import DEFAULT_MODEL_PATH
    if size not in POSE_MODELS:
        raise ValueError(f"pose model must be one of {POSE_MODELS}, not {size!r}")
    path = Path(DEFAULT_MODEL_PATH).with_name(f"pose_landmarker_{size}.task")
    return (str(path), size) if path.exists() else (DEFAULT_MODEL_PATH, "heavy")


# ---------------------------------------------------------------------------
# resolve_spine(cfg)
# What:  which spine source to use and its tracker, like his demo's resolve_spine but
#        without printing or exiting: "auto" = SpinePose when installed, else the back
#        contour; "spinepose" demanded but missing -> RuntimeError.
# Time / Space: O(1) (+ loading the SpinePose network once, ~1-10 s).
# Data structures: tuple (mode, tracker or None, note).
# Files used: src/myomotion/vision/spine_model.py (SpinePoseTracker, spinepose_available).
# ---------------------------------------------------------------------------
def resolve_spine(cfg: CvConfig = CV_CONFIG) -> tuple[str, Any, str]:
    """(mode, SpinePoseTracker or None, a sentence saying what was chosen)."""
    if cfg.spine not in SPINE_MODES:
        raise ValueError(f"spine must be one of {SPINE_MODES}, not {cfg.spine!r}")
    if cfg.spine in ("contour", "proxy"):
        return cfg.spine, None, f"spine: {cfg.spine} (asked for)"
    from myomotion.vision.spine_model import INSTALL_HINT, SpinePoseTracker, SpinePoseUnavailable, spinepose_available
    if not spinepose_available():
        if cfg.spine == "spinepose":
            raise RuntimeError(f"SpinePose is not installed. {INSTALL_HINT}")
        return "contour", None, "spine: back contour (SpinePose is not installed)"
    try:
        return "spinepose", SpinePoseTracker(size=cfg.spine_size), f"spine: SpinePose ({cfg.spine_size})"
    except (SpinePoseUnavailable, OSError, RuntimeError) as error:
        if cfg.spine == "spinepose":
            raise RuntimeError(f"could not start SpinePose: {error}") from error
        return "contour", None, f"spine: back contour (SpinePose failed to start: {error})"


# ---------------------------------------------------------------------------
# CvEngine
# What:  one frame in, one result out: pose (MediaPipe) -> spine points (SpinePose,
#        checked against the pose) -> his DeadliftAnalyzer (angles, phases, reps, form
#        rules); for exercise "rdl" the same, with the reps counted by the RDL counter
#        (src/myomotion/sync/rdl.py); for "curl", pose -> the curl tracker (src/myomotion/
#        sync/curl.py: elbow angle, phases, reps, cues; no spine model needed). Keeps the state
#        of one continuous video (MediaPipe's VIDEO mode and the state machines need
#        increasing times), so use a new engine per video.
# Time:  per frame ~14 ms pose + ~20 ms SpinePose + < 1 ms analyzer (measured on this
#        laptop, 20 Nov: MediaPipe heavy, SpinePose medium, CPU).
# Space: O(1) per frame; his analyzer keeps per-rep records and a spine log.
# Data structures: his PoseTracker / SpinePoseTracker / DeadliftAnalyzer / OverlayRenderer.
# Files used: src/myomotion/vision/pose.py, vision/spine_model.py, biomechanics/, coaching/overlay.py
#        (all Ahmad's, unchanged); reads models/pose_landmarker_heavy.task.
# ---------------------------------------------------------------------------
class CvEngine:
    """Per-frame exercise analysis behind one call (Ahmad's deadlift, or the curl tracker)."""

    # Time:  O(1) here, plus loading the models (MediaPipe about 1 s, SpinePose 1-10 s).  Space: the models.
    # Data structures: PoseTracker, SpinePoseTracker, DeadliftAnalyzer or CurlAnalyzer, RdlCounter.
    def __init__(self, cfg: CvConfig = CV_CONFIG, pose_tracker: Any = None, spine_tracker: Any = None,
                 spine_mode: str | None = None, exercise: str = "deadlift") -> None:
        self.cfg = cfg
        self.exercise = exercise
        self.is_curl = exercise in CURL_EXERCISES
        self.rdl = RdlCounter() if exercise in RDL_EXERCISES else None
        if self.is_curl:
            spine_mode, spine_tracker, self.spine_note = "none", None, "spine: not used for the curl"
        elif spine_mode is None:
            spine_mode, spine_tracker, self.spine_note = resolve_spine(cfg)
        else:
            self.spine_note = f"spine: {spine_mode}"
        self.spine_mode = spine_mode
        self.spine_tracker = spine_tracker
        if pose_tracker is None:
            from myomotion.vision.pose import DEFAULT_MODEL_PATH, PoseTracker
            # > the person mask is only needed for the back-contour spine (and costs time otherwise)
            pose_tracker = PoseTracker(cfg.model_path or DEFAULT_MODEL_PATH, masks=spine_mode == "contour")
        self.pose = pose_tracker
        self.analyzer = (CurlAnalyzer() if self.is_curl
                         else DeadliftAnalyzer(rule_cfg=DeadliftRuleConfig(lumbar_cues=cfg.lumbar_cues)))
        self.columns = CURL_COLUMNS if self.is_curl else CV_COLUMNS
        self._renderer = None
        self._spine_points: np.ndarray | None = None
        self.frames = 0
        self.spine_rejected = 0           # > SpinePose points dropped because they disagreed with the pose

    # Time:  O(P) per frame for the P pixels the networks read (MediaPipe 14-61 ms + SpinePose 15-20 ms on this
    #        laptop); the analysers are O(1).  Space: O(P).
    # Data structures: a NumPy image; [33, 3] landmark arrays; a FrameResult / CurlFrame / RdlFrame out.
    def process(self, image: np.ndarray, t_s: float) -> FrameResult:
        """Analyse one BGR frame taken at t_s seconds (strictly increasing)."""
        height, width = image.shape[:2]
        detection = self.pose.detect_full(image, t_s)
        landmarks = detection.image if detection is not None else None
        if self.is_curl:
            self.frames += 1
            return self.analyzer.process(t_s, landmarks)
        if self.spine_tracker is not None:
            if landmarks is None:
                self._spine_points = None
            elif self.frames % max(1, self.cfg.spine_every) == 0:
                self._spine_points = self._detect_spine(image, landmarks, (width, height))
        mask = detection.mask if detection is not None and self.spine_mode == "contour" else None
        result = self.analyzer.process(t_s, landmarks, image_wh=(width, height),
                                       world=detection.world if detection is not None else None,
                                       spine_points=self._spine_points, mask=mask)
        self.frames += 1
        return self.rdl.update(t_s, result) if self.rdl is not None else result

    # Time:  O(P) for the SpinePose network on the person box.  Space: O(P).
    # Data structures: a landmark array; a 4-value box; a SpineDetection with a [9, 3] array.
    def _detect_spine(self, image: np.ndarray, landmarks: np.ndarray, image_wh: tuple[int, int]) -> np.ndarray | None:
        """SpinePose's nine spine points inside the person's box, or None when they disagree with MediaPipe."""
        from myomotion.vision.spine_model import bridge_error, person_bbox
        box = person_bbox(landmarks, image_wh)
        found = self.spine_tracker.detect(image, box) if box is not None else None
        if found is None:
            return None
        error = bridge_error(found, landmarks)
        # > his G-01 check: both models must put the shoulders and hips in the same places
        if error is None or error < self.cfg.bridge_max_error:
            return found.points
        self.spine_rejected += 1
        return None

    # Time:  O(P) for the image pixels the overlay touches.  Space: O(1) extra (in place).
    # Data structures: a NumPy image.
    def draw(self, image: np.ndarray, result: FrameResult) -> np.ndarray:
        """The overlay (his for the deadlift: skeleton, spine, phase, angles, cues) drawn onto image, in place."""
        if self.is_curl:
            return draw_curl(image, result, self.analyzer)
        if self._renderer is None:
            from myomotion.coaching.overlay import OverlayRenderer
            self._renderer = OverlayRenderer()
        if self.rdl is not None:                                 # > his overlay, then the RDL phase and count on top
            self._renderer.render(image, result.deadlift, self.analyzer)
            return draw_rdl_hud(image, result)
        return self._renderer.render(image, result, self.analyzer)

    # Time:  O(K) for the K columns.  Space: O(K).
    # Data structures: a dict column -> text.
    def row(self, frame: int, t_s: float, result: Any, process_ms: float) -> dict[str, str]:
        """One cv_frames.csv row for this exercise."""
        if self.is_curl:
            return curl_row(frame, t_s, result, process_ms)
        if self.rdl is not None:                                 # > his columns; phase and reps from the RDL counter
            return {**frame_row(frame, t_s, result.deadlift, process_ms, self.cfg), "phase": result.phase,
                    "phase_label": RDL_PHASE_LABELS[result.phase], "rep_count": str(result.rep_count)}
        return frame_row(frame, t_s, result, process_ms, self.cfg)

    # Time:  O(R + T) for R reps and T phase changes.  Space: O(R + T).
    # Data structures: a dict (JSON-ready).
    def summary(self, meta: dict[str, Any]) -> dict[str, Any]:
        """The session summary (his schema myomotion.deadlift_session/1, or the curl's) with our meta."""
        if self.is_curl:
            return self.analyzer.summary({**meta, "exercise": self.exercise})
        summary = self.analyzer.summary({**meta, "exercise": self.exercise, "spine_mode": self.spine_mode,
                                         "spine_rejected_by_bridge_check": self.spine_rejected})
        return self.rdl.summary(summary) if self.rdl is not None else summary

    # Time:  O(1).
    # Data structures: releases the model objects.
    def close(self, keep_spine: bool = False) -> None:
        """Release MediaPipe (and SpinePose unless keep_spine: it can be handed to the next engine)."""
        self.pose.close()
        if self.spine_tracker is not None and not keep_spine:
            self.spine_tracker.close()


# ---------------------------------------------------------------------------
# frame_row(frame, t_s, result, cv_ms, cfg)
# What:  one FrameResult flattened into one cv_frames.csv row. Blank cells where the
#        value does not exist this frame (no person, legs out of view, spine curve not
#        visible from this angle) so nobody trains on a fake 0.
# Time / Space: O(1).
# Data structures: dict column -> text.  Files used: none.
# ---------------------------------------------------------------------------
def frame_row(frame: int, t_s: float, result: FrameResult, cv_ms: float, cfg: CvConfig = CV_CONFIG) -> dict[str, str]:
    """One row of cv_frames.csv."""
    # Time:  O(1).
    # Data structures: a float (or None) in, a str out.
    def number(value: float | None) -> str:
        return "" if value is None or not np.isfinite(value) else f"{value:.{cfg.decimals}f}"

    angles = result.angles
    spine = angles.spine if angles is not None else result.partial_spine
    curve = spine is not None and spine.curvature_visible
    row = {
        "frame": str(frame), "t_s": f"{t_s:.6f}", "person": str(int(result.landmarks is not None)),
        "tracked": str(int(angles is not None)), "back_only": str(int(result.back_only)), "view": result.view,
        "phase_label": result.phase.value, "phase": PHASE_TO_OURS[result.phase], "rep_count": str(result.rep_count),
        "hip_deg": number(angles.hip_deg if angles else None),
        "hip_signed_deg": number(angles.hip_signed_deg if angles else None),
        "knee_deg": number(angles.knee_deg if angles else None),
        "trunk_incl_deg": number(result.trunk_incl_deg),
        "tibia_incl_deg": number(angles.tibia_incl_deg if angles else None),
        "trunk_tibia_deg": number(angles.trunk_tibia_deg if angles else None),
        "spine_source": spine.source if spine is not None else "",
        "spine_curve_visible": str(int(curve)) if spine is not None else "",
        "lumbar_deg": number(spine.lumbar_deg if curve else None),
        "thoracic_deg": number(spine.thoracic_deg if curve else None),
        "cervical_deg": number(spine.cervical_deg if curve else None),
        "pelvic_deg": number(spine.pelvic_deg if curve else None),
        "spine_confidence": number(spine.confidence if spine is not None else None),
        "bias": result.bias.value if result.bias is not None else "",
        # > sorted: the same flags always give the same text (a set has no order)
        "flags": "|".join(sorted(flag.value for flag in result.active_flags)),
        "process_ms": f"{cv_ms:.1f}",
    }
    return row


# ---------------------------------------------------------------------------
# video_frames(video, frame_times)
# What:  (index, time, frame) for every frame of a video. Times from frame_times (the
#        camera's own, from video_frames.csv) when given, else the file's timestamps
#        via his iter_video_frames (correct for variable-frame-rate phone video).
# Time:  O(T) decoding.  Space: O(one frame).
# Data structures: generator.  Files used: reads the video; src/myomotion/vision/video_input.py.
# ---------------------------------------------------------------------------
def video_frames(video: Path, frame_times: np.ndarray | None) -> Iterator[tuple[int, float, np.ndarray]]:
    """Yield (frame index, t_s, BGR frame)."""
    if frame_times is None:
        from myomotion.vision.video_input import iter_video_frames
        for index, (t_s, frame) in enumerate(iter_video_frames(str(video))):
            yield index, float(t_s), frame
        return
    import cv2
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise OSError(f"could not open video {video}")
    try:
        # > one camera time per saved frame; stop at whichever runs out first
        for index in range(len(frame_times)):
            ok, frame = capture.read()
            if not ok:
                break
            yield index, float(frame_times[index]), frame
    finally:
        capture.release()


# ---------------------------------------------------------------------------
# CvRun
# What:  what one analyze_video call produced.
# Data structures: dataclass.
# Time:  O(P) to build for the P phase counts.
# ---------------------------------------------------------------------------
@dataclass
class CvRun:
    """Result of analysing one video."""

    frames: int
    rep_count: int
    frames_tracked: int
    frames_back_only: int
    frames_with_person: int
    median_ms: float
    spine_note: str
    summary: dict[str, Any]
    frames_path: Path
    summary_path: Path
    video_path: Path | None = None
    phase_frames: dict[str, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# analyze_video(video, out_folder, frame_times, engine, annotate, cfg, progress)
# What:  run CvEngine on EVERY frame of a saved video (the live view skips frames; the
#        dataset must not) and write cv_frames.csv, cv_summary.json and, if annotate,
#        cv_annotated.mp4 (his overlay on every frame, same frame count as the video).
# Time:  O(T) frames x ~35-40 ms each (pose + SpinePose) on this laptop.
# Space: O(T) rows (small dicts); frames go straight to disk.
# Data structures: list of row dicts; cv2.VideoWriter; his summary dict.
# Files used: reads the video; writes <out_folder>/cv_frames.csv, cv_summary.json,
#        cv_annotated.mp4. Uses CvEngine (Ahmad's modules, see above).
# ---------------------------------------------------------------------------
def analyze_video(video: Path, out_folder: Path, frame_times: np.ndarray | None = None,
                  engine: CvEngine | None = None, annotate: bool = True, cfg: CvConfig = CV_CONFIG,
                  progress: Callable[[str], None] | None = print, exercise: str = "deadlift") -> CvRun:
    """Analyse every frame of video; write the CV files into out_folder."""
    video, out_folder = Path(video), Path(out_folder)
    own_engine = engine is None
    engine = engine or CvEngine(cfg, exercise=exercise)
    rows: list[dict[str, str]] = []
    timings: list[float] = []
    writer = None
    next_report = cfg.progress_every_s
    try:
        for index, t_s, frame in video_frames(video, frame_times):
            started = time.perf_counter()
            result = engine.process(frame, t_s)
            cv_ms = 1000.0 * (time.perf_counter() - started)
            timings.append(cv_ms)
            rows.append(engine.row(index, t_s, result, cv_ms))
            if annotate:
                if writer is None:
                    writer = _open_writer(out_folder / CV_VIDEO_NAME, video, frame, cfg)
                writer.write(engine.draw(frame, result))
            if progress is not None and t_s >= next_report:
                progress(f"  CV: {t_s:5.1f} s of video analysed, reps so far {result.rep_count}")
                next_report += cfg.progress_every_s
        meta = {"tool": "myomotion.sync.cv_frames", "source": str(video), "frames": len(rows),
                "frame_times": "video_frames.csv (camera)" if frame_times is not None else "video file",
                "median_cv_ms": round(statistics.median(timings), 1) if timings else None,
                "input": "synced_recording"}
        summary = engine.summary(meta)
    finally:
        if writer is not None:
            writer.release()
        if own_engine:
            engine.close()

    out_folder.mkdir(parents=True, exist_ok=True)
    frames_path, summary_path = out_folder / CV_FRAMES_NAME, out_folder / CV_SUMMARY_NAME
    with frames_path.open("w", newline="") as out:
        csv_writer = csv.DictWriter(out, fieldnames=list(engine.columns))
        csv_writer.writeheader()
        csv_writer.writerows(rows)
    summary_path.write_text(json.dumps(summary, indent=2))
    phase_frames: dict[str, int] = {}
    for row in rows:                                   # > O(T) count with a dict: phase -> frames
        phase_frames[row["phase"]] = phase_frames.get(row["phase"], 0) + 1
    return CvRun(frames=len(rows), rep_count=int(summary["rep_count"]), frames_tracked=int(summary["frames_tracked"]),
                 frames_back_only=int(summary.get("frames_back_monitored", 0)),
                 frames_with_person=sum(row["person"] == "1" for row in rows),
                 median_ms=statistics.median(timings) if timings else 0.0, spine_note=engine.spine_note,
                 summary=summary, frames_path=frames_path, summary_path=summary_path,
                 video_path=(out_folder / CV_VIDEO_NAME) if writer is not None else None, phase_frames=phase_frames)


GOOD_TRACKING_SHARE = 0.9          # below this share of frames the summary says how to film better


# ---------------------------------------------------------------------------
# run_report(run, exercise)
# What:  the printed summary of a CV pass, in the exercise's own words: for the curl
#        "arm seen" (shoulder, elbow, wrist), for the deadlift and RDL "whole body"
#        (shoulder to ankle) and "back only"; plus what to fix when tracking was poor.
# Time / Space: O(P) phases.  Data structures: list of lines.  Files used: none.
# ---------------------------------------------------------------------------
def run_report(run: CvRun, exercise: str) -> str:
    """Summary lines of one analyze_video run."""
    phases = ", ".join(f"{name} {count}" for name, count in sorted(run.phase_frames.items()))
    share = run.frames_tracked / run.frames if run.frames else 0.0
    if exercise in CURL_EXERCISES:
        lines = [f"CV: {run.frames} frames ({run.median_ms:.0f} ms each); arm seen (shoulder, elbow, wrist) in "
                 f"{run.frames_tracked} ({share:.0%}); reps {run.rep_count}; frames per phase: {phases}"]
        if share < GOOD_TRACKING_SHARE:
            lines.append("  The arm was out of view in many frames: sit about 2 m away, side-on, with your head, "
                         "hip and hand inside the picture, in a lit room.")
        if run.rep_count == 0 and run.frames_tracked:
            lines.append("  No rep counted: let the arm hang straight (elbow open past 130 deg) between curls, and "
                         "curl up to the shoulder (below 80 deg).")
    else:
        lines = [f"CV: {run.frames} frames ({run.median_ms:.0f} ms each); person seen in {run.frames_with_person}; "
                 f"whole body (shoulder to ankle) in {run.frames_tracked} ({share:.0%}), back only in "
                 f"{run.frames_back_only}; reps {run.rep_count}; frames per phase: {phases}"]
        if run.frames_tracked == 0:
            lines.append("  No frame showed the whole body: reps and joint angles need it side-on, 2.5-3 m away, in a "
                         "lit room.")
    return "\n".join(lines)


# Time:  O(1) (opens the file; the caller writes the frames).  Space: O(1).
# Data structures: a cv2.VideoCapture (to read the frame rate) and a cv2.VideoWriter.
def _open_writer(path: Path, video: Path, frame: np.ndarray, cfg: CvConfig):
    """A VideoWriter for cv_annotated.mp4 at the source video's nominal frame rate."""
    import cv2
    capture = cv2.VideoCapture(str(video))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    capture.release()
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frame.shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*cfg.annotated_fourcc), fps, (width, height))
    if not writer.isOpened():
        raise OSError(f"could not open a video writer for {path}")
    return writer


# ---------------------------------------------------------------------------
# find_cv_outputs(folder)
# What:  the CV files analyze_video left in a set folder, if any.
# Time / Space: O(1).  Data structures: tuple of Paths.  Files used: checks <folder>/cv_*.
# ---------------------------------------------------------------------------
def find_cv_outputs(folder: Path) -> tuple[Path | None, Path | None]:
    """(cv_summary.json or None, cv_frames.csv or None)."""
    summary, frames = Path(folder) / CV_SUMMARY_NAME, Path(folder) / CV_FRAMES_NAME
    return (summary if summary.exists() else None), (frames if frames.exists() else None)


# ---------------------------------------------------------------------------
# LiveCv
# What:  his CV for the LIVE view, in a background thread. submit() puts the newest
#        camera frame in a one-slot mailbox (replacing a frame still waiting, which is
#        counted as skipped) and returns at once; the thread analyses whatever is
#        newest and publishes that frame with his overlay. So the recording loop keeps
#        the camera's full frame rate while the skeleton updates as fast as the CPU
#        allows. The saved video is never drawn on.
# Time:  submit O(1); the thread ~35-40 ms per analysed frame.
# Space: O(2 frames): the one waiting + the last annotated.
# Data structures: threading.Condition guarding a one-slot mailbox (newest wins).
# Files used: none (CvEngine reads the models).
# ---------------------------------------------------------------------------
class LiveCv(threading.Thread):
    """Analyse the newest camera frame in the background; hand back the annotated frame."""

    # Time:  O(1).  Space: O(1).
    # Data structures: a threading.Condition; a one-slot mailbox (tuple or None).
    def __init__(self, engine: CvEngine) -> None:
        super().__init__(daemon=True)
        self.engine = engine
        self._cond = threading.Condition()
        self._pending: tuple[np.ndarray, float] | None = None
        self._latest: np.ndarray | None = None
        self._last_t: float | None = None
        self._stopping = False
        self.analysed = 0
        self.skipped = 0
        self.error: str | None = None

    # Time:  O(1): never waits for the CV.  Space: O(1).
    # Data structures: the one-slot mailbox under the Condition (the newest frame replaces a waiting one).
    def submit(self, frame: np.ndarray, t_s: float) -> None:
        """Offer the newest frame (never blocks). Times must increase."""
        with self._cond:
            if self._pending is not None:
                self.skipped += 1                     # > the older waiting frame is dropped: newest wins
            self._pending = (frame, t_s)
            self._cond.notify()

    # Time:  O(1).
    # Data structures: the last annotated NumPy image (returned by reference).
    def latest(self) -> np.ndarray | None:
        """The most recent frame with the overlay drawn on it, or None before the first one."""
        with self._cond:
            return self._latest

    # Time:  O(P) per analysed frame (CvEngine.process + draw); otherwise it waits.  Space: O(P) for one image copy.
    # Data structures: the Condition-guarded mailbox; a NumPy image copy.
    def run(self) -> None:
        while True:
            with self._cond:
                while self._pending is None and not self._stopping:
                    self._cond.wait()
                if self._stopping:
                    return
                frame, t_s = self._pending
                self._pending = None
            if self._last_t is not None and t_s <= self._last_t:
                continue                              # > MediaPipe's VIDEO mode needs increasing times
            try:
                result = self.engine.process(frame, t_s)
                view = self.engine.draw(frame.copy(), result)   # > a copy: never draw on the recorded frame
            except Exception as error:               # noqa: BLE001  (shown to the user, recording goes on)
                self.error = f"{type(error).__name__}: {error}"
                return
            self._last_t = t_s
            with self._cond:
                self._latest = view
                self.analysed += 1

    # Time:  O(1), plus waiting for the thread (at most timeout_s).
    # Data structures: Condition.notify; a bool flag.
    def stop(self, timeout_s: float = 2.0) -> None:
        """Stop the thread and wait for it."""
        with self._cond:
            self._stopping = True
            self._cond.notify()
        if self.is_alive():
            self.join(timeout=timeout_s)
