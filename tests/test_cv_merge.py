"""Tests for the CV + sEMG merge (src/myomotion/sync/cv_frames.py, viewer.py), without a camera or the pose model.

Run from the repo root:  python -m pytest tests/test_cv_merge.py -q
Ahmad's pipeline (src/myomotion/biomechanics/, coaching/overlay.py) runs for real. Only MediaPipe is replaced by a
fake that returns his synthetic lifter (src/myomotion/biomechanics/synthetic.py), timed to match the fake sEMG
recording of tests/test_sync.py, so the reps, phases and %MVC are all known in advance.
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from myomotion.biomechanics import DeadliftAnalyzer, Phase
from myomotion.biomechanics.synthetic import Body, ClipSpec, generate_clip_3d
from myomotion.sync.cv_frames import (CV_COLUMNS, CV_FRAMES_NAME, CV_SUMMARY_NAME, CV_VIDEO_NAME, PHASE_TO_OURS,
                                      CvEngine, LiveCv, analyze_video, find_cv_outputs, frame_row)
from myomotion.sync.recorder import RECORDER_CONFIG, Step, compose_cv_view
from myomotion.sync.session import sync_recording
from myomotion.sync.viewer import SYNCED_VIDEO_NAME, load_merged, render_synced_video
from myomotion.vision.pose import DEFAULT_MODEL_PATH, PoseDetection
from test_sync import FPS, SECONDS, SIZE, make_recording, write_video

# > the same rep timing as tests/test_sync.py: pulls at 2-3 s and 7-8 s, holds 0.5 s, lowers 1 s
SPEC = ClipSpec(fps=FPS, n_reps=2, pre_s=2.0, up_s=1.0, hold_s=0.5, down_s=1.0, post_s=2.5, noise_px=0.5, seed=3)
BODY = Body().scaled(0.3, floor_y=SIZE[1] - 15, x0=SIZE[0] * 0.45)


def synthetic_clip() -> list[tuple[float, np.ndarray, np.ndarray]]:
    """His synthetic lifter, 2 reps, sized for the 640 x 360 test video."""
    return list(generate_clip_3d("good", SPEC, BODY, yaw_deg=0.0, cx=SIZE[0] * 0.45, floor_y=SIZE[1] - 15))


class FakePose:
    """Stands in for MediaPipe: the synthetic lifter's landmarks for the i-th frame it is asked about."""

    def __init__(self, clip: list[tuple[float, np.ndarray, np.ndarray]]) -> None:
        self.clip = clip
        self.calls = 0
        self.closed = False

    def detect_full(self, image: np.ndarray, t_s: float) -> PoseDetection | None:
        index = self.calls
        self.calls += 1
        if index >= len(self.clip):
            return None
        _, landmarks, world = self.clip[index]
        return PoseDetection(image=landmarks.copy(), world=world.copy())

    def close(self) -> None:
        self.closed = True


def fake_engine() -> CvEngine:
    """Ahmad's CvEngine with the fake pose and his geometric proxy spine (no network)."""
    return CvEngine(pose_tracker=FakePose(synthetic_clip()), spine_mode="proxy")


def test_every_phase_of_his_maps_to_one_of_ours() -> None:
    assert set(PHASE_TO_OURS) == set(Phase)
    assert set(PHASE_TO_OURS.values()) == {"CONCENTRIC", "ISOMETRIC_HOLD", "ECCENTRIC", "REST"}
    assert PHASE_TO_OURS[Phase.LOCKOUT] == "ISOMETRIC_HOLD"         # > deadlift: the lockout is the hold


def test_frame_row_leaves_cells_blank_when_nobody_is_seen() -> None:
    result = DeadliftAnalyzer().process(0.0, None)
    row = frame_row(0, 0.0, result, 12.3)
    assert list(row) == list(CV_COLUMNS)
    assert row["person"] == "0" and row["tracked"] == "0" and row["phase"] == "REST"
    assert row["hip_deg"] == "" and row["lumbar_deg"] == "" and row["process_ms"] == "12.3"


def test_analyze_video_counts_the_synthetic_reps(tmp_path: Path) -> None:
    video = write_video(tmp_path / "video.mp4")
    run = analyze_video(video, tmp_path, engine=fake_engine(), progress=None)
    assert run.frames == int(SECONDS * FPS) and run.rep_count == 2
    assert run.frames_tracked == run.frames                          # > the whole body is always in view
    assert {"CONCENTRIC", "ISOMETRIC_HOLD", "ECCENTRIC", "REST"} <= set(run.phase_frames)
    assert find_cv_outputs(tmp_path) == (tmp_path / CV_SUMMARY_NAME, tmp_path / CV_FRAMES_NAME)
    with (tmp_path / CV_FRAMES_NAME).open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == run.frames and float(rows[75]["hip_deg"]) > 0
    assert rows[75]["phase"] == "CONCENTRIC" and rows[75]["phase_label"] == "Concentric Pull"   # > t = 2.5 s
    summary = json.loads((tmp_path / CV_SUMMARY_NAME).read_text())
    assert summary["schema"] == "myomotion.deadlift_session/1" and summary["meta"]["spine_mode"] == "proxy"
    capture = cv2.VideoCapture(str(tmp_path / CV_VIDEO_NAME))
    assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == run.frames  # > his overlay on every frame
    capture.release()


def test_analyze_video_uses_the_camera_times_when_given(tmp_path: Path) -> None:
    video = write_video(tmp_path / "video.mp4")
    camera_times = np.arange(int(SECONDS * FPS)) / FPS + 0.004       # > camera clock: 4 ms after the file's
    analyze_video(video, tmp_path, frame_times=camera_times, engine=fake_engine(), annotate=False, progress=None)
    with (tmp_path / CV_FRAMES_NAME).open() as f:
        times = [float(row["t_s"]) for row in csv.DictReader(f)]
    assert times == pytest.approx(list(camera_times), abs=1e-6)
    assert not (tmp_path / CV_VIDEO_NAME).exists()


@pytest.fixture()
def synced_set(tmp_path: Path) -> Path:
    """A fake sEMG set folder (tests/test_sync.py) with its LED video, plus an MVC that passed."""
    folder = tmp_path / "P99" / "S01_deadlift_test"
    make_recording(folder)
    write_video(folder / "video.mp4")
    mvc_folder = tmp_path / "P99" / "S00_deadlift_mvc"
    mvc_folder.mkdir()
    (mvc_folder / "mvc.json").write_text(json.dumps({"mvc_above_rest_mv": 150.0, "passed": True}))
    return folder


def test_cv_and_semg_end_up_in_one_row_per_frame(synced_set: Path) -> None:
    run = analyze_video(synced_set / "video.mp4", synced_set, engine=fake_engine(), progress=None)
    result = sync_recording(synced_set, synced_set / "video.mp4", cv_json=run.summary_path,
                            cv_frames=run.frames_path, mvc_path=synced_set.parent / "S00_deadlift_mvc")
    assert result.report.verdict == "PASS" and result.phase_source == "cv"
    with result.merged_path.open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == int(SECONDS * FPS)
    pull = rows[75]                                                  # > t = 2.5 s: first pull, seen by both
    assert pull["phase"] == "CONCENTRIC" and pull["rep"] == "1" and float(pull["percent_mvc"]) > 70
    assert pull["cv_phase"] == "CONCENTRIC" and float(pull["cv_hip_deg"]) > 0 and pull["cv_tracked"] == "1"
    standing = rows[30]                                              # > t = 1.0 s: set-up, relaxed
    assert standing["phase"] == "REST" and abs(float(standing["percent_mvc"])) < 10
    concentric = [stat for stat in result.phases if stat.phase == "CONCENTRIC"]
    assert len(concentric) == 2 and all(stat.mean_percent > 60 for stat in concentric)

    out = render_synced_video(synced_set)
    assert out.name == SYNCED_VIDEO_NAME
    capture = cv2.VideoCapture(str(out))
    frames, width = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    capture.release()
    assert frames == len(rows) and width > SIZE[0]                   # > the video plus the sEMG panel
    assert load_merged(result.merged_path).unit == "%MVC"


def test_sync_falls_back_to_the_cues_when_the_cv_counted_no_reps(synced_set: Path, tmp_path: Path) -> None:
    empty = tmp_path / "no_reps.json"
    empty.write_text(json.dumps({"reps": []}))
    result = sync_recording(synced_set, synced_set / "video.mp4", cv_json=empty, mvc_path=None)
    assert result.phase_source.startswith("cues") and result.intervals


class SlowEngine:
    """A CvEngine stand-in that takes 40 ms a frame (like the real one on this laptop)."""

    def __init__(self) -> None:
        self.times: list[float] = []

    def process(self, frame: np.ndarray, t_s: float) -> float:
        time.sleep(0.04)
        self.times.append(t_s)
        return t_s

    def draw(self, frame: np.ndarray, result: float) -> np.ndarray:
        frame[:4, :4] = 255
        return frame


def test_live_cv_never_holds_up_the_camera_and_analyses_the_newest_frame() -> None:
    engine = SlowEngine()
    live = LiveCv(engine)          # type: ignore[arg-type]
    live.start()
    frame = np.zeros((48, 64, 3), np.uint8)
    longest = 0.0
    try:
        for index in range(30):                                     # > 30 frames at 100 per second
            started = time.perf_counter()
            live.submit(frame, index / 100)
            longest = max(longest, time.perf_counter() - started)
            time.sleep(0.01)
        time.sleep(0.1)
    finally:
        live.stop()
    assert longest < 0.005                                          # > submit returns at once
    assert 0 < live.analysed < 30 and live.skipped > 0 and live.error is None
    assert engine.times == sorted(engine.times) and engine.times[-1] == pytest.approx(0.29)   # > newest frame won
    shown = live.latest()
    assert shown is not None and shown[0, 0, 0] == 255 and frame[0, 0, 0] == 0   # > drawn on a copy only


def test_live_view_puts_his_frame_and_the_semg_side_by_side() -> None:
    annotated = np.zeros((480, 640, 3), np.uint8)
    step = Step(0.0, 2.0, "RELEASE: open your hand slowly", "ECCENTRIC", 1)
    view = compose_cv_view(annotated, step, 3, 1.5, 42.0, "%MVC", [(0.0, 10.0), (0.5, 42.0)], 0.5, True, 3.0)
    assert view.shape == (480 + RECORDER_CONFIG.cue_strip_px, 640 + RECORDER_CONFIG.cv_panel_px, 3)
    assert np.array_equal(view[RECORDER_CONFIG.cue_strip_px + 100, 300], annotated[100, 300])   # > untouched


@pytest.mark.skipif(not Path(DEFAULT_MODEL_PATH).exists(), reason="models/pose_landmarker_heavy.task not downloaded")
def test_the_real_pose_model_runs_through_the_engine(tmp_path: Path) -> None:
    from dataclasses import replace

    from myomotion.sync.cv_frames import CV_CONFIG
    video = write_video(tmp_path / "video.mp4")
    run = analyze_video(video, tmp_path, cfg=replace(CV_CONFIG, spine="proxy"), annotate=False, progress=None)
    assert run.frames == int(SECONDS * FPS) and run.frames_with_person == 0   # > a moving box is not a person


def test_pose_model_sizes_fall_back_to_heavy_when_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import myomotion.vision.pose as pose
    from myomotion.sync.cv_frames import pose_model_path
    heavy = tmp_path / "pose_landmarker_heavy.task"
    (tmp_path / "pose_landmarker_full.task").write_bytes(b"x")
    monkeypatch.setattr(pose, "DEFAULT_MODEL_PATH", str(heavy))
    assert pose_model_path("full") == (str(tmp_path / "pose_landmarker_full.task"), "full")
    assert pose_model_path("lite") == (str(heavy), "heavy")
    with pytest.raises(ValueError):
        pose_model_path("huge")
