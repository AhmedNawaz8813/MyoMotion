"""Tests for the Romanian deadlift (src/myomotion/sync/rdl.py, angle_cycle.py) in the CV + sEMG pipeline.

Run from the repo root:  python -m pytest tests/test_rdl.py -q
Ahmad's real DeadliftAnalyzer runs on his own synthetic lifter (src/myomotion/biomechanics/synthetic.py), moved
like an RDL: start standing, lower, pause at the bottom, stand up, pause, repeat. Only MediaPipe is faked.
His floor-start rep counter misses one of these; the RDL counter must count every rep, lowering first.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from myomotion.biomechanics import DeadliftAnalyzer
from myomotion.biomechanics.geometry import joint_angle
from myomotion.biomechanics.synthetic import SCENARIOS, Body, build_landmarks_3d, deadlift_pose, smoothstep
from myomotion.sync.align import intervals_from_cv
from myomotion.sync.angle_cycle import HOLD, REST
from myomotion.sync.cv_frames import CV_COLUMNS, CvEngine, analyze_video
from myomotion.sync.rdl import CONCENTRIC, ECCENTRIC, RdlCounter
from myomotion.sync.recorder import RDL_PHASES, phases_for
from myomotion.sync.session import sync_recording
from myomotion.vision.pose import PoseDetection
from test_sync import FPS, SECONDS, SIZE, make_recording, write_video

BODY = Body().scaled(0.3, floor_y=SIZE[1] - 15, x0=SIZE[0] * 0.45)


def rdl_segments(reps: int, stand: float = 2.0, down: float = 2.0, bottom: float = 0.5, up: float = 2.0,
                 top: float = 1.0, end: float = 2.0) -> list[tuple[str, float]]:
    """(kind, seconds) of an RDL set: it starts and ends standing."""
    return [("stand", stand)] + [("down", down), ("bottom", bottom), ("up", up), ("top", top)] * reps + [("stand", end)]


def pose_at(t: float, segments: list[tuple[str, float]]) -> tuple[float, bool]:
    """(bar progress s: 1 standing, 0 at the bottom; moving up?) at time t."""
    t0 = 0.0
    for kind, seconds in segments:
        if t < t0 + seconds:
            u = (t - t0) / seconds
            return {"stand": (1.0, False), "top": (1.0, False), "down": (1.0 - smoothstep(u), False),
                    "bottom": (0.0, True), "up": (smoothstep(u), True)}[kind]
        t0 += seconds
    return 1.0, False


def starts(segments: list[tuple[str, float]], kind: str) -> list[float]:
    """Start time of every segment of one kind."""
    out, t0 = [], 0.0
    for name, seconds in segments:
        if name == kind:
            out.append(t0)
        t0 += seconds
    return out


def landmarks_at(t: float, segments: list[tuple[str, float]]) -> tuple[np.ndarray, np.ndarray]:
    """Ahmad's synthetic lifter (image and world landmarks), side-on, sized for the 640 x 360 test video."""
    s, ascending = pose_at(t, segments)
    return build_landmarks_3d(deadlift_pose(s, SCENARIOS["good"], ascending), BODY, 0.0, SIZE[0] * 0.45, SIZE[1] - 15)


def hip_onsets(segments: list[tuple[str, float]], change_deg: float = 1.0) -> tuple[list[float], list[float]]:
    """When the RAW hip angle really starts to close (each lowering) and to open (each rise). In Ahmad's synthetic
    body the hips start to bend a little after the bar starts down, so the segment start is not the truth."""
    downs, ups = starts(segments, "down"), starts(segments, "up")
    hip = [(i / FPS, joint_angle(*(landmarks_at(i / FPS, segments)[0][j, :2] for j in (12, 24, 26))))
           for i in range(int(sum(seconds for _, seconds in segments) * FPS))]
    close = [next(t for t, a in hip if t >= d and a < hip[0][1] - change_deg) for d in downs]
    lowest = [min(a for t, a in hip if d <= t < u) for d, u in zip(downs, ups)]
    open_ = [next(t for t, a in hip if t >= u - 0.5 and a > low + change_deg) for u, low in zip(ups, lowest)]
    return close, open_


def run(segments: list[tuple[str, float]]) -> tuple[DeadliftAnalyzer, RdlCounter]:
    analyzer, counter = DeadliftAnalyzer(), RdlCounter()
    total = sum(seconds for _, seconds in segments)
    for i in range(int(total * FPS)):
        t = i / FPS
        image, world = landmarks_at(t, segments)
        counter.update(t, analyzer.process(t, image, image_wh=SIZE, world=world))
    return analyzer, counter


def test_every_rdl_rep_is_counted_lowering_first() -> None:
    segments = rdl_segments(3)
    analyzer, counter = run(segments)
    assert counter.cycle.rep_count == 3
    assert [change["to"] for change in counter.cycle.transitions] == [ECCENTRIC, HOLD, CONCENTRIC, REST] * 3
    summary = counter.summary(analyzer.summary({}))
    assert summary["rep_count"] == 3 and summary["schema"] == "myomotion.rdl_session/1"
    assert summary["deadlift_rep_count"] is not None                 # > his own count is kept for comparison
    hip_moves_down, hip_moves_up = hip_onsets(segments)
    for rep, down_t, up_t in zip(summary["reps"], hip_moves_down, hip_moves_up):
        names = [part["phase"] for part in rep["phases"]]
        assert names == [ECCENTRIC, HOLD, CONCENTRIC]
        # > the lowering starts when the hips really start to bend, the rise when they really start to open
        # > (within Ahmad's landmark smoothing, ~0.1-0.2 s)
        assert rep["phases"][0]["start_s"] == pytest.approx(down_t, abs=0.3)
        assert rep["phases"][2]["start_s"] == pytest.approx(up_t, abs=0.3)
        assert rep["max_trunk_incl_deg"] > 30                         # > a real hinge


def test_standing_still_is_not_a_rep() -> None:
    _, counter = run([("stand", 6.0)])
    assert counter.cycle.rep_count == 0 and counter.cycle.phase == REST


def test_the_recorder_cues_an_rdl_down_first() -> None:
    assert phases_for("rdl") == RDL_PHASES and RDL_PHASES[0][0] == "ECCENTRIC"


def test_phase_lists_keep_each_exercises_order() -> None:
    summary = {"reps": [{"rep_index": 1, "phases": [{"phase": "ECCENTRIC", "start_s": 1.0, "end_s": 2.0},
                                                    {"phase": "ISOMETRIC_HOLD", "start_s": 2.0, "end_s": 2.5},
                                                    {"phase": "CONCENTRIC", "start_s": 2.5, "end_s": 4.0}]}]}
    assert [(i.phase, i.start_s) for i in intervals_from_cv(summary)] == [
        ("ECCENTRIC", 1.0), ("ISOMETRIC_HOLD", 2.0), ("CONCENTRIC", 2.5)]


class FakePose:
    """MediaPipe stand-in: the synthetic RDL lifter for the i-th frame it is asked about."""

    def __init__(self, segments: list[tuple[str, float]]) -> None:
        self.segments, self.calls = segments, 0

    def detect_full(self, image: np.ndarray, t_s: float) -> PoseDetection:
        landmarks, world = landmarks_at(self.calls / FPS, self.segments)
        self.calls += 1
        return PoseDetection(image=landmarks, world=world)

    def close(self) -> None:
        pass


def test_rdl_cv_and_semg_end_up_in_one_row_per_frame(tmp_path: Path) -> None:
    # > 12 s like the fake sEMG set of tests/test_sync.py: 2 RDLs, lowering at 1.5 s and 5.5 s
    segments = rdl_segments(2, stand=1.5, down=1.5, bottom=0.5, up=1.5, top=0.5, end=2.5)
    folder = tmp_path / "P99" / "S01_rdl_test"
    make_recording(folder)
    write_video(folder / "video.mp4")
    engine = CvEngine(pose_tracker=FakePose(segments), spine_mode="proxy", exercise="rdl")
    run_ = analyze_video(folder / "video.mp4", folder, engine=engine, progress=None)
    assert run_.rep_count == 2 and run_.frames == int(SECONDS * FPS) and engine.columns == CV_COLUMNS
    summary = json.loads(run_.summary_path.read_text())
    assert summary["meta"]["exercise"] == "rdl" and summary["meta"]["spine_mode"] == "proxy"

    result = sync_recording(folder, folder / "video.mp4", cv_json=run_.summary_path, cv_frames=run_.frames_path)
    assert result.phase_source == "cv"
    with result.merged_path.open() as f:
        rows = list(csv.DictReader(f))
    lowering, rising = rows[int(2.3 * FPS)], rows[int(4.3 * FPS)]   # > middle of rep 1's lowering and standing up
    assert lowering["phase"] == "ECCENTRIC" and rising["phase"] == "CONCENTRIC" and lowering["rep"] == "1"
    assert float(lowering["cv_hip_signed_deg"]) < float(rows[int(1.0 * FPS)]["cv_hip_signed_deg"])
    assert rows[int(1.0 * FPS)]["cv_phase_label"] == "Standing"
