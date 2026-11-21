"""Tests for the LED-free sync (src/myomotion/sync/clock_sync.py and the 'clock' method of session.py).

Run from the repo root:  python -m pytest tests/test_clock_sync.py -q
A made-up recording with KNOWN clocks: the ESP32 runs 40 ppm slow and the camera 15 ppm fast against the laptop,
every sample reaches the laptop after a random Wi-Fi delay (the newest sample of each batch almost at once), every
frame is read a little after its driver timestamp, and the camera stamps a frame 10.7 ms after the light hit it.
The sync must find where any ESP32 moment appears in the video to well under a millisecond.
"""

from __future__ import annotations

import csv
import json
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest

from myomotion.sync.clock_sync import (CLOCK_CONFIG, clock_check_ms, laptop_clock, load_arrivals,
                                       lower_envelope_line)
from myomotion.sync.session import sync_recording

ESP32_ORIGIN_US = 812_000_000                 # > the ESP32 had been on for 812 s
ESP32_LAPTOP = (3.25, -40e-6)                 # > laptop time = 3.25 s + (1 - 40 ppm) * ESP32 seconds
CAMERA_LAPTOP = (4.10, 15e-6)                 # > laptop time = 4.10 s + (1 + 15 ppm) * camera seconds
CAMERA_DELAY_S = 0.0107
FPS, SECONDS, RATE = 30.0, 20.0, 500
VIDEO_SECONDS = 18.0                          # > the video starts 0.85 s after the sEMG and ends before it
SIZE = (320, 240)


def laptop_from_esp32(u_s: np.ndarray | float) -> np.ndarray | float:
    return ESP32_LAPTOP[0] + (1 + ESP32_LAPTOP[1]) * np.asarray(u_s)


def laptop_from_camera(t_s: np.ndarray | float) -> np.ndarray | float:
    return CAMERA_LAPTOP[0] + (1 + CAMERA_LAPTOP[1]) * np.asarray(t_s)


def video_time_of_esp32(u_s: float) -> float:
    """Where a light switched on at ESP32 time u_s shows up: the frame stamped CAMERA_DELAY later."""
    return (laptop_from_esp32(u_s) + CAMERA_DELAY_S - CAMERA_LAPTOP[0]) / (1 + CAMERA_LAPTOP[1])


def make_webcam_recording(folder: Path, seed: int = 4) -> None:
    """emg.csv + video_frames.csv + events.csv + metadata.json + a plain video (no LED anywhere)."""
    folder.mkdir(parents=True)
    rng = np.random.default_rng(seed)
    u = np.arange(int(SECONDS * RATE)) / RATE
    # > Wi-Fi: samples go out in batches of 10; the newest of a batch can arrive almost at once, older ones waited
    # > longer. (A delay that is NEVER shorter than some floor cannot be seen by any clock method; on real data it is
    # > part of the camera_delay_ms constant, which was measured end to end against the LED.)
    batch_age = (9 - np.arange(len(u)) % 10) / RATE
    arrival = laptop_from_esp32(u) + batch_age + rng.uniform(0.0, 0.0002, len(u)) + rng.exponential(0.004, len(u))
    signal = 1320 + 5 * rng.standard_normal(len(u))
    active = (u > 6) & (u < 8)
    signal[active] += 150 * np.abs(rng.standard_normal(int(np.sum(active))))
    with (folder / "emg.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["pc_time_s", "t_us", "ch1", "ch2", "ch3"])
        for pc, t, value in zip(arrival, u, signal):
            writer.writerow([f"{pc:.6f}", ESP32_ORIGIN_US + int(round(t * 1e6)), f"{value:.0f}", 142, 142])
    camera_t = np.arange(int(VIDEO_SECONDS * FPS)) / FPS
    read = laptop_from_camera(camera_t) + rng.exponential(0.002, len(camera_t))
    with (folder / "video_frames.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "t_camera_s", "pc_time_s"])
        for index, (t, pc) in enumerate(zip(camera_t, read)):
            writer.writerow([index, f"{t:.6f}", f"{pc:.6f}"])
    cues = [(0.0, "BASELINE", ""), (6.0, "CONCENTRIC", 1), (7.0, "ISOMETRIC_HOLD", 1), (7.5, "ECCENTRIC", 1),
            (8.0, "REST", 1)]
    with (folder / "events.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["pc_time_s", "t_us", "source", "event", "rep"])
        for t, event, rep in cues:
            writer.writerow([0, ESP32_ORIGIN_US + int(t * 1e6), "cue", event, rep])
        for t in (5.0, 10.0, 15.0):                               # > the ESP32 still answers the flashes
            writer.writerow([0, ESP32_ORIGIN_US + int(t * 1e6), "board", "LED_ON", ""])
    (folder / "metadata.json").write_text(json.dumps({"baseline_s": 5.0, "rate_hz": 500.0}))
    writer = cv2.VideoWriter(str(folder / "video.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), FPS, SIZE)
    for _ in camera_t:
        writer.write(np.full((SIZE[1], SIZE[0], 3), 40, np.uint8) + rng.integers(0, 4, (SIZE[1], SIZE[0], 3),
                                                                                 dtype=np.uint8))
    writer.release()


def test_lower_envelope_finds_offset_and_drift_through_random_delays() -> None:
    rng = np.random.default_rng(0)
    t = np.arange(0, 30, 0.002)
    delay = 0.8 + 30e-6 * t + rng.exponential(0.01, len(t))
    delay[::25] = 0.8 + 30e-6 * t[::25]                     # > some arrivals are (almost) instant
    line = lower_envelope_line(t, delay)
    assert line.slope == pytest.approx(30e-6, abs=1e-6)
    assert line.intercept_s == pytest.approx(0.8, abs=1e-4)
    assert line.windows == 30


def test_a_whole_slow_window_is_left_out() -> None:
    rng = np.random.default_rng(1)
    t = np.arange(0, 30, 0.002)
    delay = 0.5 + rng.exponential(0.01, len(t))
    delay[::25] = 0.5
    stall = (t >= 12) & (t < 13)
    delay[stall] += 0.08                                     # > nothing arrived quickly for a whole second
    line = lower_envelope_line(t, delay)
    assert line.windows == 29 and line.intercept_s == pytest.approx(0.5, abs=2e-4) and abs(line.slope) < 2e-5


def test_too_short_a_recording_is_refused() -> None:
    with pytest.raises(ValueError, match="at least"):
        lower_envelope_line(np.array([0.0, 0.5, 1.2]), np.array([0.1, 0.1, 0.1]))


def test_laptop_clock_puts_esp32_moments_on_the_right_video_time(tmp_path: Path) -> None:
    folder = tmp_path / "S01"
    make_webcam_recording(folder)
    clock = laptop_clock(folder, replace(CLOCK_CONFIG, camera_delay_ms=CAMERA_DELAY_S * 1000))
    # > drift within 5 ppm (= 0.1 ms over the 20 s); what matters is where moments land, checked below
    assert clock.esp32.slope == pytest.approx(-40e-6, abs=5e-6) and clock.camera.slope == pytest.approx(15e-6, abs=5e-6)
    fit = clock.to_clock_fit()
    for u in (2.0, 9.5, 17.25):
        assert float(fit.to_video_s(ESP32_ORIGIN_US + u * 1e6)) == pytest.approx(video_time_of_esp32(u), abs=0.5e-3)
    # > the LED check: flashes the camera "saw" where they really were -> errors about 0
    flashes_us = [ESP32_ORIGIN_US + int(u * 1e6) for u in (5.0, 10.0, 15.0)]
    errors = clock_check_ms(clock, flashes_us, [video_time_of_esp32(u) for u in (5.0, 10.0, 15.0)])
    assert max(abs(e) for e in errors) < 0.5


def test_load_arrivals_explains_a_phone_recording(tmp_path: Path) -> None:
    folder = tmp_path / "S02"
    make_webcam_recording(folder)
    (folder / "video_frames.csv").unlink()
    with pytest.raises(ValueError, match="needs the LED"):
        load_arrivals(folder)


def test_sync_falls_back_to_the_laptop_clock_when_no_led_is_seen(tmp_path: Path) -> None:
    folder = tmp_path / "S03"
    make_webcam_recording(folder)
    cfg = replace(CLOCK_CONFIG, camera_delay_ms=CAMERA_DELAY_S * 1000)
    result = sync_recording(folder, folder / "video.mp4", clock_cfg=cfg)              # > method "auto"
    assert result.method == "clock" and result.led is None and result.led_problem
    assert result.report.verdict == "NOT MEASURED"
    assert result.frames_with_emg == int(VIDEO_SECONDS * FPS)
    with result.merged_path.open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == int(VIDEO_SECONDS * FPS)
    row = rows[300]                                                                    # > video time 10 s
    true_us = ESP32_ORIGIN_US + 1e6 * ((laptop_from_camera(10.0) - CAMERA_DELAY_S - ESP32_LAPTOP[0])
                                       / (1 + ESP32_LAPTOP[1]))
    assert int(row["t_esp32_us"]) == pytest.approx(true_us, abs=500)
    saved = json.loads(result.json_path.read_text())
    assert saved["method"] == "clock" and saved["laptop_clock"]["used_for_sync"] is True
    assert saved["flashes"] == [] and saved["led"] is None


def test_led_only_still_refuses_without_the_led(tmp_path: Path) -> None:
    folder = tmp_path / "S04"
    make_webcam_recording(folder)
    with pytest.raises(ValueError):
        sync_recording(folder, folder / "video.mp4", method="led")


def test_clock_only_skips_the_led_search(tmp_path: Path) -> None:
    folder = tmp_path / "S05"
    make_webcam_recording(folder)
    result = sync_recording(folder, folder / "video.mp4", method="clock")
    assert result.method == "clock" and result.led_problem is None
