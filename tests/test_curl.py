"""Tests for the biceps-curl tracker (src/myomotion/sync/curl.py) and its place in the CV + sEMG pipeline.

Run from the repo root:  python -m pytest tests/test_curl.py -q
A synthetic side-on arm (MediaPipe's 33 landmarks, right side visible) curls with a known elbow angle over time,
so the reps, the phase times and the form cues are all known in advance. No camera or model is needed.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from myomotion.sync.curl import (CONCENTRIC, CUE_ELBOW, CUE_RANGE, CUE_SWING, CURL_COLUMNS, ECCENTRIC, HOLD, REST,
                                 CurlAnalyzer, curl_row)
from myomotion.sync.cv_frames import CvEngine, analyze_video
from myomotion.sync.recorder import CURL_PHASES, phases_for
from myomotion.sync.session import sync_recording
from myomotion.vision.pose import PoseDetection
from test_sync import FPS, SECONDS, make_recording, write_video

STRAIGHT, CURLED = 170.0, 45.0


@dataclass(frozen=True)
class CurlSpec:
    """Timing of the synthetic set (seconds)."""

    reps: int = 3
    rest_first_s: float = 1.5
    up_s: float = 1.5
    hold_s: float = 1.0
    down_s: float = 1.5
    rest_s: float = 1.5
    top_deg: float = CURLED
    elbow_drift_deg: float = 0.0      # upper arm swinging forward at the top
    trunk_swing_deg: float = 0.0      # trunk leaning back at the top
    fps: float = FPS


def smoothstep(x: float) -> float:
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def truth(spec: CurlSpec, rep: int) -> dict[str, float]:
    """True start of the curl, of the hold, of the lowering and end of the lowering of rep `rep` (0-based)."""
    cycle = spec.up_s + spec.hold_s + spec.down_s + spec.rest_s
    t0 = spec.rest_first_s + rep * cycle
    return {"up": t0, "top": t0 + spec.up_s, "down": t0 + spec.up_s + spec.hold_s,
            "end": t0 + spec.up_s + spec.hold_s + spec.down_s}


def progress(spec: CurlSpec, t: float) -> float:
    """0 = arm straight, 1 = fully curled."""
    cycle = spec.up_s + spec.hold_s + spec.down_s + spec.rest_s
    if t < spec.rest_first_s or t >= spec.rest_first_s + spec.reps * cycle:
        return 0.0
    u = (t - spec.rest_first_s) % cycle
    if u < spec.up_s:
        return smoothstep(u / spec.up_s)
    if u < spec.up_s + spec.hold_s:
        return 1.0
    if u < spec.up_s + spec.hold_s + spec.down_s:
        return 1.0 - smoothstep((u - spec.up_s - spec.hold_s) / spec.down_s)
    return 0.0


def landmarks_at(spec: CurlSpec, t: float, rng: np.random.Generator) -> np.ndarray:
    """33 landmarks (x_px, y_px, visibility); the person faces image-right, right side to the camera."""
    s = progress(spec, t)
    elbow_angle = math.radians(STRAIGHT + (spec.top_deg - STRAIGHT) * s)
    lean = math.radians(spec.trunk_swing_deg * s)                # > trunk tips back (shoulder moves left)
    drift = math.radians(spec.elbow_drift_deg * s)               # > upper arm swings forward
    hip = np.array([300.0, 330.0])
    shoulder = hip + 180.0 * np.array([-math.sin(lean), -math.cos(lean)])
    upper_dir = np.array([math.sin(drift - lean), math.cos(drift - lean)])   # > down along the trunk, + drift
    elbow = shoulder + 110.0 * upper_dir
    # > the forearm turns away from the upper arm's "up" direction by the elbow angle (towards the front)
    up = -upper_dir
    forward = np.array([-up[1], up[0]])                          # > up turned 90 deg towards the front (+x)
    forearm = math.cos(elbow_angle) * up + math.sin(elbow_angle) * forward
    wrist = elbow + 100.0 * forearm
    lm = np.zeros((33, 3))
    lm[:, 2] = 0.1
    for index, point in ((12, shoulder), (14, elbow), (16, wrist), (24, hip)):
        lm[index, :2] = point + rng.normal(0.0, 0.7, 2)
        lm[index, 2] = 0.95
    return lm


def run(spec: CurlSpec, seed: int = 0) -> tuple[CurlAnalyzer, list]:
    rng = np.random.default_rng(seed)
    analyzer = CurlAnalyzer()
    cycle = spec.up_s + spec.hold_s + spec.down_s + spec.rest_s
    duration = spec.rest_first_s + spec.reps * cycle + 1.0
    frames = [analyzer.process(i / spec.fps, landmarks_at(spec, i / spec.fps, rng))
              for i in range(int(duration * spec.fps))]
    return analyzer, frames


def test_the_synthetic_arm_has_the_elbow_angle_it_was_given() -> None:
    spec = CurlSpec()
    _, frames = run(spec)
    assert frames[10].elbow_deg == pytest.approx(STRAIGHT, abs=2.0)             # > resting, arm straight
    top = int((truth(spec, 0)["top"] + 0.5) * FPS)
    assert frames[top].elbow_deg == pytest.approx(CURLED, abs=3.0)


def test_three_curls_are_counted_with_their_phases_in_order() -> None:
    spec = CurlSpec()
    analyzer, frames = run(spec)
    assert analyzer.rep_count == 3 and len(analyzer.reps) == 3
    order = [change["to"] for change in analyzer.transitions]
    assert order == [CONCENTRIC, HOLD, ECCENTRIC, REST] * 3
    for index, rep in enumerate(analyzer.reps):
        true = truth(spec, index)
        # > the curl's start is moved back to when the arm was last straight; the lowering to the top turning point
        assert rep["t_concentric_start_s"] == pytest.approx(true["up"], abs=0.25)
        assert true["up"] < rep["t_lockout_s"] <= true["top"] + 0.1
        assert rep["t_descent_start_s"] == pytest.approx(true["down"], abs=0.25)
        assert true["down"] < rep["t_end_s"] <= true["end"] + 0.1
        assert rep["min_elbow_deg"] == pytest.approx(CURLED, abs=3.0) and rep["cues"] == []
    assert {frame.phase for frame in frames} == {REST, CONCENTRIC, HOLD, ECCENTRIC}


def test_curls_without_straightening_the_arm_count_with_a_cue() -> None:
    spec = CurlSpec()
    rng = np.random.default_rng(6)
    analyzer = CurlAnalyzer()
    bent = 140.0                                                  # > S06: the arm only opened to ~140 between curls
    duration = spec.rest_first_s + spec.reps * (spec.up_s + spec.hold_s + spec.down_s + spec.rest_s) + 1.0
    for i in range(int(duration * FPS)):
        t = i / FPS
        analyzer.process(t, arm_at_angle(bent + (CURLED - bent) * progress(spec, t), rng))
    assert analyzer.rep_count == 3
    assert all(CUE_RANGE in rep["cues"] and rep["start_elbow_deg"] < 145 for rep in analyzer.reps)


def arm_at_angle(elbow_deg: float, rng: np.random.Generator) -> np.ndarray:
    """A still, upright arm with the given elbow angle (right side to the camera)."""
    shoulder, elbow = np.array([300.0, 150.0]), np.array([300.0, 260.0])
    a = math.radians(elbow_deg)
    wrist = elbow + 100.0 * np.array([math.sin(a), -math.cos(a)])
    lm = np.zeros((33, 3))
    lm[:, 2] = 0.1
    for index, point in ((12, shoulder), (14, elbow), (16, wrist), (24, np.array([300.0, 330.0]))):
        lm[index, :2] = point + rng.normal(0.0, 0.7, 2)
        lm[index, 2] = 0.95
    return lm


def test_a_half_curl_is_not_a_rep() -> None:
    analyzer, frames = run(CurlSpec(reps=2, top_deg=100.0))
    assert analyzer.rep_count == 0 and frames[-1].phase == REST


def test_elbow_drifting_forward_raises_the_elbow_cue() -> None:
    analyzer, frames = run(CurlSpec(elbow_drift_deg=45.0))
    assert analyzer.rep_count == 3
    assert all(CUE_ELBOW in rep["cues"] for rep in analyzer.reps)
    assert any(CUE_ELBOW in frame.cues for frame in frames)


def test_swinging_the_body_raises_the_swing_cue() -> None:
    analyzer, _ = run(CurlSpec(trunk_swing_deg=20.0))
    assert analyzer.rep_count == 3 and all(CUE_SWING in rep["cues"] for rep in analyzer.reps)


def test_an_arm_held_bent_is_not_a_curl() -> None:
    analyzer = CurlAnalyzer()
    rng = np.random.default_rng(2)
    held = CurlSpec(reps=1, rest_first_s=0.0, up_s=0.5, hold_s=30.0, top_deg=110.0)   # > bent at 110 for 30 s
    phases = [analyzer.process(i / FPS, landmarks_at(held, min(i / FPS, 10.0), rng)).phase
              for i in range(int(12 * FPS))]
    assert analyzer.rep_count == 0 and phases[-1] == REST             # > never counted, never stuck in a phase


def test_a_rep_needs_a_straight_arm_first() -> None:
    analyzer = CurlAnalyzer()
    rng = np.random.default_rng(3)
    spec = CurlSpec(reps=1, rest_first_s=0.0)
    bent_start = landmarks_at(CurlSpec(reps=1, rest_first_s=0.0, up_s=0.1, hold_s=5.0, top_deg=120.0), 1.0, rng)
    for i in range(10):                                                # > starts with the arm bent at 120
        analyzer.process(i / FPS, bent_start)
    assert analyzer.phase == REST
    for i in range(10, int(10 * FPS)):                                 # > then curls properly
        analyzer.process(i / FPS, landmarks_at(spec, (i - 10) / FPS, rng))
    assert analyzer.rep_count == 1


def test_no_arm_no_rep_and_blank_cells() -> None:
    analyzer = CurlAnalyzer()
    frame = analyzer.process(0.0, None)
    row = curl_row(0, 0.0, frame, 9.5)
    assert list(row) == list(CURL_COLUMNS) and row["tracked"] == "0" and row["elbow_deg"] == ""
    assert analyzer.summary()["rep_count"] == 0


def test_the_recorder_cues_a_curl_up_first() -> None:
    assert phases_for("curl") == CURL_PHASES and CURL_PHASES[0][0] == "CONCENTRIC"


class FakePose:
    """MediaPipe stand-in: the synthetic arm for the i-th frame it is asked about."""

    def __init__(self, spec: CurlSpec) -> None:
        self.spec, self.calls, self.rng = spec, 0, np.random.default_rng(5)

    def detect_full(self, image: np.ndarray, t_s: float) -> PoseDetection:
        lm = landmarks_at(self.spec, self.calls / self.spec.fps, self.rng)
        self.calls += 1
        return PoseDetection(image=lm, world=None)

    def close(self) -> None:
        pass


# > the same timing as the fake sEMG set of tests/test_sync.py: curls at 2-3 s and 7-8 s
SYNC_SPEC = CurlSpec(reps=2, rest_first_s=2.0, up_s=1.0, hold_s=0.5, down_s=1.0, rest_s=2.5)


def test_curl_cv_and_semg_end_up_in_one_row_per_frame(tmp_path: Path) -> None:
    folder = tmp_path / "P99" / "S01_curl_test"
    make_recording(folder)
    write_video(folder / "video.mp4")
    engine = CvEngine(pose_tracker=FakePose(SYNC_SPEC), exercise="curl")
    assert engine.spine_tracker is None and engine.columns == CURL_COLUMNS
    run_ = analyze_video(folder / "video.mp4", folder, engine=engine, progress=None)
    assert run_.rep_count == 2 and run_.frames == int(SECONDS * FPS) and run_.frames_tracked == run_.frames
    summary = json.loads(run_.summary_path.read_text())
    assert summary["schema"] == "myomotion.curl_session/1" and summary["meta"]["exercise"] == "curl"

    result = sync_recording(folder, folder / "video.mp4", cv_json=run_.summary_path, cv_frames=run_.frames_path)
    assert result.phase_source == "cv" and result.method == "led"
    with result.merged_path.open() as f:
        rows = list(csv.DictReader(f))
    pull = rows[75]                                               # > t = 2.5 s: the first curl, seen by both
    assert pull["phase"] == "CONCENTRIC" and pull["rep"] == "1"
    assert rows[80]["cv_phase"] == "CONCENTRIC"                   # > the live label flips a little later (causal)
    assert 60 < float(pull["cv_elbow_deg"]) < 160
    assert rows[30]["phase"] == "REST" and float(rows[30]["cv_elbow_deg"]) > 160      # > t = 1 s, arm down
    # > merged.csv's 7 own columns + every curl column as cv_* (except t_s, which is the join key)
    assert len(rows[0]) == 7 + len(CURL_COLUMNS) - 1


def test_the_cv_summary_speaks_about_the_arm_for_a_curl() -> None:
    from myomotion.sync.cv_frames import CvRun, run_report
    run = CvRun(frames=670, rep_count=0, frames_tracked=307, frames_back_only=0, frames_with_person=670,
                median_ms=48.0, spine_note="", summary={}, frames_path=Path("a"), summary_path=Path("b"),
                phase_frames={"REST": 670})
    text = run_report(run, "curl")
    assert "arm seen" in text and "ankle" not in text and "2 m away" in text and "hang straight" in text
    assert "whole body" in run_report(run, "rdl")
