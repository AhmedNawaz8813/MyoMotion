"""Tests for src/myomotion/sync/ (video-sEMG sync, TC03) and scripts/sync/sync_session.py, without hardware.

Run from the repo root:  python -m pytest tests/test_sync.py -q
A synthetic phone video is written with OpenCV: a dark noisy room, a bright "person" moving left and
right (to tempt the LED search), and a small red LED that flashes three times for 0.2 s. The flash moments
sit at different points between frames (a quarter, a half and three quarters of a frame), so the timing
error of each is known in advance.
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from myomotion.sync.align import (ClockFit, PhaseInterval, check_sync, fit_clock, interpolate_sorted,
                                  intervals_from_cv, label_frames)
from myomotion.sync.led_sync import LedNotFound, find_led
from myomotion.sync.session import sync_recording

REPO = Path(__file__).resolve().parents[1]
FPS = 30.0
SECONDS = 12.0
SIZE = (640, 360)
LED_XY = (560, 50)
FLASHES_S = [1.0 + 0.25 / FPS, 6.0 + 0.5 / FPS, 10.5 + 0.75 / FPS]   # > true flash moments, video clock
HALF_FRAME_MS = 1000 / FPS / 2
SLOPE = 1 + 50e-6                                                    # > video clock runs 50 ppm fast
OFFSET_S = -812.0                                                    # > ESP32 had run 812 s at video time 0
CONCENTRIC = [(2.0, 3.0), (7.0, 8.0)]                                # > rep phases on the video clock
HOLD = [(3.0, 3.5), (8.0, 8.5)]
ECCENTRIC = [(3.5, 4.5), (8.5, 9.5)]


def write_video(path: Path) -> Path:
    """The synthetic phone video described in the module docstring."""
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, SIZE)
    rng = np.random.default_rng(0)
    for index in range(int(SECONDS * FPS)):
        t_s = index / FPS
        image = np.full((SIZE[1], SIZE[0], 3), 40, np.uint8) + rng.integers(0, 6, (SIZE[1], SIZE[0], 3), dtype=np.uint8)
        x = int(300 + 200 * np.sin(2 * np.pi * t_s / 1.5))
        cv2.rectangle(image, (x, 100), (x + 60, 300), (230, 230, 230), -1)          # > the moving "person"
        if any(start <= t_s < start + 0.2 for start in FLASHES_S):
            cv2.circle(image, LED_XY, 4, (40, 40, 255), -1)                          # > the red LED, on
        writer.write(image)
    writer.release()
    return path


@pytest.fixture(scope="module")
def video(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return write_video(tmp_path_factory.mktemp("video") / "video.mp4")


def esp32_us(t_video_s: float) -> int:
    """Video time -> ESP32 micros() for the made-up clock line."""
    return int(round((t_video_s - OFFSET_S) / SLOPE * 1e6))


def test_find_led_picks_the_led_not_the_moving_person(video: Path) -> None:
    gaps = [b - a for a, b in zip(FLASHES_S, FLASHES_S[1:])]
    trace = find_led(video, count=3, expected_gaps_s=gaps)
    x, y, w, h = trace.box_px
    assert x <= LED_XY[0] < x + w and y <= LED_XY[1] < y + h
    for flash, true_s in zip(trace.flashes, FLASHES_S):
        assert abs(flash.t_s - true_s) * 1000 <= HALF_FRAME_MS + 1          # > within half a frame
    assert trace.snapshot is not None and len(trace.times_s) == int(SECONDS * FPS)


def test_find_led_inside_a_given_box(video: Path) -> None:
    trace = find_led(video, count=3, box_px=(540, 30, 40, 40))
    assert [round(f.t_s, 1) for f in trace.flashes] == [round(t, 1) for t in FLASHES_S]


def test_find_led_explains_a_spacing_that_does_not_match(video: Path) -> None:
    with pytest.raises(LedNotFound, match="spacing"):
        find_led(video, count=3, expected_gaps_s=[2.0, 2.0])


def test_fit_clock_recovers_offset_and_drift() -> None:
    video_s = [1.0, 6.0, 10.5]
    fit = fit_clock([esp32_us(t) for t in video_s], video_s)
    assert fit.slope == pytest.approx(SLOPE, abs=1e-7)                 # > 0.1 ppm: micros() are whole numbers
    assert fit.esp32_s_at_video_start == pytest.approx(esp32_us(0.0) / 1e6, abs=1e-6)
    assert fit.drift_ppm == pytest.approx(50.0, abs=0.01)
    assert max(abs(r) for r in fit.residuals_ms) < 0.01
    assert float(fit.to_video_s(fit.to_esp32_us(4.2))) == pytest.approx(4.2)
    with pytest.raises(ValueError):
        fit_clock([1, 2], [1.0])


def test_check_sync_pass_fail_and_not_measured() -> None:
    flash_us = [esp32_us(t) for t in (1.0, 6.0, 10.5)]
    good = [1.0, 6.010, 10.5]                              # > middle flash 10 ms off the line
    bad = [1.0, 6.050, 10.5]                               # > middle flash 50 ms off the line
    assert check_sync(flash_us, good, fit_clock(flash_us, good), 1 / FPS).verdict == "PASS"
    assert check_sync(flash_us, bad, fit_clock(flash_us, bad), 1 / FPS).verdict == "FAIL"
    two = check_sync(flash_us[::2], good[::2], fit_clock(flash_us[::2], good[::2]), 1 / FPS)
    assert two.verdict == "NOT MEASURED"


def test_interpolate_sorted_matches_numpy() -> None:
    rng = np.random.default_rng(1)
    x = np.cumsum(rng.uniform(0.001, 0.003, 500))
    y = rng.standard_normal(500)
    targets = np.linspace(x[0] - 0.1, x[-1] + 0.1, 333)
    ours = interpolate_sorted(x, y, targets)
    inside = (targets >= x[0]) & (targets <= x[-1])
    assert np.allclose(ours[inside], np.interp(targets[inside], x, y))
    assert np.isnan(ours[~inside]).all()


def test_cv_reps_become_phase_labels_per_frame() -> None:
    summary = {"reps": [{"rep_index": 1, "t_concentric_start_s": 2.0, "t_lockout_s": 3.0, "t_descent_start_s": 3.5,
                         "t_end_s": 4.5}]}
    intervals = intervals_from_cv(summary)
    assert [i.phase for i in intervals] == ["CONCENTRIC", "ISOMETRIC_HOLD", "ECCENTRIC"]
    phases, reps = label_frames(np.array([1.0, 2.5, 3.2, 4.0, 5.0]), intervals)
    assert phases == ["REST", "CONCENTRIC", "ISOMETRIC_HOLD", "ECCENTRIC", "REST"]
    assert reps == [0, 1, 1, 1, 0]
    assert label_frames(np.array([0.0]), [PhaseInterval(1, "CONCENTRIC", 0.0, 1.0)])[0] == ["CONCENTRIC"]


def make_recording(folder: Path) -> None:
    """A fake record_session.py folder whose LED_ON events match the video's flashes through the clock line."""
    folder.mkdir(parents=True)
    rng = np.random.default_rng(2)
    t_us = np.arange(esp32_us(-3.0), esp32_us(12.5), 2000, dtype=np.int64)          # > 500 samples per second
    t_video = OFFSET_S + SLOPE * t_us / 1e6
    size = np.full(len(t_us), 5.0)
    for windows, mv in ((CONCENTRIC, 150.0), (HOLD, 60.0), (ECCENTRIC, 90.0)):
        for start, end in windows:
            size[(t_video >= start) & (t_video < end)] = mv
    signal = 600 + 80 * np.sin(2 * np.pi * 50 * t_us / 1e6) + size * rng.standard_normal(len(t_us))
    with (folder / "emg.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["pc_time_s", "t_us", "ch1", "ch2", "ch3"])
        for t, value in zip(t_us, signal):
            writer.writerow([0, int(t), f"{value:.1f}", 0, 0])
    events = [(esp32_us(-3.0), "cue", "BASELINE", ""), (esp32_us(0.9), "cue", "SET_START", "")]
    for rep, ((c0, c1), (_, h1), (_, e1)) in enumerate(zip(CONCENTRIC, HOLD, ECCENTRIC), start=1):
        events += [(esp32_us(c0), "cue", "CONCENTRIC", rep), (esp32_us(c1), "cue", "ISOMETRIC_HOLD", rep),
                   (esp32_us(h1), "cue", "ECCENTRIC", rep), (esp32_us(e1), "cue", "REST", rep)]
    events += [(esp32_us(t), "board", "LED_ON", "") for t in FLASHES_S]
    with (folder / "events.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["pc_time_s", "t_us", "source", "event", "rep"])
        for t, source, event, rep in sorted(events):
            writer.writerow([0, t, source, event, rep])
    (folder / "metadata.json").write_text(json.dumps({"baseline_s": 2.5, "rate_hz": 500.0}))


@pytest.fixture()
def session(tmp_path: Path, video: Path) -> Path:
    """P99/S01_deadlift_test (recording + video.mp4) next to P99/S00_forearm_mvc/mvc.json."""
    folder = tmp_path / "P99" / "S01_deadlift_test"
    make_recording(folder)
    (folder / "video.mp4").write_bytes(video.read_bytes())
    mvc_folder = tmp_path / "P99" / "S00_forearm_mvc"
    mvc_folder.mkdir()
    (mvc_folder / "mvc.json").write_text(json.dumps({"mvc_above_rest_mv": 150.0, "passed": True}))
    return folder


def test_sync_recording_end_to_end_with_cv_phases(session: Path, tmp_path: Path) -> None:
    lag = 0.2                                                     # > CV detection lag: phases a little late
    cv_json = tmp_path / "cv.json"
    cv_json.write_text(json.dumps({"reps": [
        {"rep_index": i, "t_concentric_start_s": c0 + lag, "t_lockout_s": c1 + lag, "t_descent_start_s": h1 + lag,
         "t_end_s": e1 + lag}
        for i, ((c0, c1), (_, h1), (_, e1)) in enumerate(zip(CONCENTRIC, HOLD, ECCENTRIC), start=1)]}))
    cv_frames = tmp_path / "frames.csv"
    with cv_frames.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["t_s", "hip_deg"])
        for index in range(int(SECONDS * FPS)):
            writer.writerow([f"{index / FPS:.4f}", 100 + index])

    result = sync_recording(session, session / "video.mp4", cv_json=cv_json, cv_frames=cv_frames,
                            mvc_path=session.parent / "S00_forearm_mvc")
    assert result.report.verdict == "PASS"
    # > the video's start is placed on the ESP32 clock within half a frame (plus the slope error over 1 s)
    assert result.fit.esp32_s_at_video_start == pytest.approx(esp32_us(0.0) / 1e6, abs=HALF_FRAME_MS / 1000 + 0.004)
    assert result.frames_with_emg == int(SECONDS * FPS)
    assert result.phase_source == "cv"

    with result.merged_path.open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == int(SECONDS * FPS)
    assert rows[75]["phase"] == "CONCENTRIC" and float(rows[75]["percent_mvc"]) > 70   # > t = 2.5 s, first pull
    assert rows[30]["phase"] == "REST" and abs(float(rows[30]["percent_mvc"])) < 10      # > t = 1.0 s, standing
    assert rows[75]["cv_hip_deg"] == "175" and rows[75]["rep"] == "1"
    concentric = [s for s in result.phases if s.phase == "CONCENTRIC"]
    assert len(concentric) == 2 and all(s.mean_percent > 60 for s in concentric)
    saved = json.loads(result.json_path.read_text())
    assert saved["verdict"] == "PASS" and len(saved["flashes"]) == 3 and saved["phase_source"] == "cv"


def test_sync_script_runs_with_recorder_cues_and_finds_the_mvc(session: Path) -> None:
    run = subprocess.run([sys.executable, str(REPO / "scripts" / "sync" / "sync_session.py"), str(session)],
                         capture_output=True, text=True, timeout=300)
    assert run.returncode == 0, run.stdout + run.stderr
    assert "TC03 video-sEMG sync: PASS" in run.stdout
    assert "S00_forearm_mvc" in run.stdout                       # > found the participant's MVC by itself
    assert (session / "sync.png").exists() and (session / "merged.csv").exists()
    phases = json.loads((session / "sync.json").read_text())["phases"]
    assert phases[0]["phase"] == "CONCENTRIC" and phases[0]["mean_percent"] > 70


def test_sync_script_explains_a_missing_video(tmp_path: Path) -> None:
    folder = tmp_path / "P99" / "S02_empty"
    make_recording(folder)
    run = subprocess.run([sys.executable, str(REPO / "scripts" / "sync" / "sync_session.py"), str(folder)],
                         capture_output=True, text=True, timeout=60)
    assert run.returncode == 1 and "No video found" in run.stdout


def test_clockfit_round_trip_is_exact() -> None:
    fit = ClockFit(slope=SLOPE, esp32_ref_us=esp32_us(1.0), video_ref_s=1.0, residuals_ms=())
    t_us = np.array([esp32_us(0.0), esp32_us(5.0)])
    assert np.allclose(fit.to_video_s(t_us), [0.0, 5.0], atol=1e-6)
