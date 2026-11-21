"""Tests for the synced webcam + sEMG recorder (src/myomotion/sync/recorder.py), without hardware.

Run from the repo root:  python -m pytest tests/test_record_synced.py -q
The end-to-end test plays both devices: a fake ESP32 (a TCP server streaming 500 samples per second and
answering "S" with E,<t_us>,LED_ON) and a fake camera that draws a lit LED for 0.2 s after every flash.
Both run on this computer's clock, so the sync must come out right.
"""

from __future__ import annotations

import csv
import json
import socket
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from myomotion.semg.acquisition import AcquisitionConfig, EspStream, LineParser
from myomotion.sync.cv_frames import LiveCv
from myomotion.sync.recorder import (HAND_PHASES, RecorderConfig, SetPlan, build_schedule, next_set_folder,
                                     record_set, step_at)
from myomotion.sync.session import sync_recording

FAST = RecorderConfig(baseline_s=2.0, countdown_s=1, tail_s=1.2, rest_between_s=0.4, tempo_s=(0.4, 0.4, 0.4),
                      first_data_timeout_s=3.0, after_end_s=0.4)


def test_schedule_flashes_at_start_middle_and_end() -> None:
    steps = build_schedule(3, HAND_PHASES, (1.0, 1.0, 1.0), FAST)
    flashes = [(s.markers, s.event, s.rep) for s in steps if s.flash]
    assert flashes == [(("SET_START",), "CONCENTRIC", 1), (("SYNC_MID",), "REST", 2), (("SET_END",), "TAIL", "")]
    assert all(a.end_s == b.start_s for a, b in zip(steps, steps[1:]))       # > no gaps, no overlaps
    assert step_at(steps, 0.0) == 0 and step_at(steps, steps[-1].end_s) is None


def test_schedule_without_rest_flashes_as_the_next_rep_starts() -> None:
    cfg = RecorderConfig(rest_between_s=0.0)
    steps = build_schedule(2, HAND_PHASES, (1.0, 1.0, 1.0), cfg)
    mid = [s for s in steps if "SYNC_MID" in s.markers]
    assert len(mid) == 1 and mid[0].rep == 2 and mid[0].event == "CONCENTRIC"
    assert sum(s.flash for s in steps) == 3


def test_line_parser_keeps_board_events() -> None:
    parser = LineParser()
    samples = parser.feed(b"D,1000,1,2,3\nE,1500,LED_ON\nD,3000,1,2,3\n")
    assert [s.t_us for s in samples] == [1000, 3000]
    assert [(e.t_us, e.name) for e in parser.events] == [(1500, "LED_ON")]


def test_next_set_folder_continues_the_numbering(tmp_path: Path) -> None:
    (tmp_path / "P01" / "S07_forearm_mvc").mkdir(parents=True)
    (tmp_path / "P01" / "S39_forearm_fist").mkdir()
    assert next_set_folder(tmp_path, "P01", "forearm", "fist").name == "S40_forearm_fist"
    assert next_set_folder(tmp_path, "P02", "squat", "deep").name == "S01_squat_deep"


class FakeEsp32:
    """TCP server on 127.0.0.1: 500 samples/s of noisy 'rest' around 1320 mV; answers S with LED_ON."""

    def __init__(self) -> None:
        self.server = socket.socket()
        self.server.bind(("127.0.0.1", 0))
        self.server.listen(1)
        self.port = self.server.getsockname()[1]
        self.t0 = time.perf_counter()
        self.flash_times: list[float] = []
        self.running = True
        self.thread = threading.Thread(target=self._serve, daemon=True)
        self.thread.start()

    def t_us(self, pc: float) -> int:
        return int((pc - self.t0) * 1e6) + 500_000_000

    def _serve(self) -> None:
        connection, _ = self.server.accept()
        connection.setblocking(False)
        rng = np.random.default_rng(0)
        sent = 0
        try:
            while self.running:
                now = time.perf_counter()
                due = int((now - self.t0) * 500)
                lines = []
                while sent < due:
                    value = 1320 + 5 * rng.standard_normal()
                    lines.append(f"D,{(500_000_000 + sent * 2000) % 2 ** 32},{value:.0f},142,142\n")
                    sent += 1
                try:
                    if b"S" in connection.recv(64):
                        self.flash_times.append(now)
                        lines.append(f"E,{self.t_us(now) % 2 ** 32},LED_ON\n")
                except BlockingIOError:
                    pass
                if lines:
                    connection.sendall("".join(lines).encode())
                time.sleep(0.01)
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            connection.close()

    def stop(self) -> None:
        self.running = False
        self.server.close()


class FakeCvEngine:
    """Stands in for Ahmad's CV in the live view: 40 ms a frame, marks the corner of what it drew on."""

    def __init__(self) -> None:
        self.times: list[float] = []

    def process(self, frame: np.ndarray, t_s: float) -> float:
        time.sleep(0.04)
        self.times.append(t_s)
        return t_s

    def draw(self, frame: np.ndarray, result: float) -> np.ndarray:
        frame[:4, :4] = 255
        return frame


class FakeCamera:
    """30 frames per second; a small red LED is lit for 0.2 s after every flash of the fake ESP32."""

    def __init__(self, esp: FakeEsp32) -> None:
        self.esp = esp
        self.next_time = time.perf_counter()
        self.rng = np.random.default_rng(1)

    def read(self):
        self.next_time += 1 / 30
        delay = self.next_time - time.perf_counter()
        if delay > 0:
            time.sleep(delay)
        now = time.perf_counter()
        frame = np.full((240, 320, 3), 35, np.uint8) + self.rng.integers(0, 6, (240, 320, 3), dtype=np.uint8)
        if any(0 <= now - f < 0.2 for f in self.esp.flash_times):
            frame[26:34, 246:254] = (40, 40, 255)
        return True, frame, now

    def close(self) -> None:
        pass


def test_record_set_end_to_end_then_sync(tmp_path: Path) -> None:
    esp = FakeEsp32()
    stream = EspStream(AcquisitionConfig(host="127.0.0.1", port=esp.port, read_timeout_s=0.02))
    folder = tmp_path / "TEST" / "S01_forearm_fist"
    plan = SetPlan("TEST", "forearm", "fist", reps=3, tempo_s=(0.4, 0.4, 0.4))
    cv_engine = FakeCvEngine()
    live = LiveCv(cv_engine)  # type: ignore[arg-type]
    live.start()
    try:
        result = record_set(FakeCamera(esp), stream, plan, mvc_above_rest_mv=100.0, folder=folder, display=False,
                            cfg=FAST, fps_probe_s=0.5, live_cv=live)
    finally:
        live.stop()
        stream.close()
        esp.stop()
    assert not result.aborted and result.flashes == 3
    assert result.quality["missing_samples"] == 0 and 28 < result.fps < 32
    for name in ("video.mp4", "video_frames.csv", "emg.csv", "events.csv", "metadata.json"):
        assert (folder / name).exists(), name
    with (folder / "events.csv").open() as f:
        events = [row["event"] for row in csv.DictReader(f)]
    assert events.count("LED_ON") == 3 and "SYNC_MID" in events and events[0] == "BASELINE"
    metadata = json.loads((folder / "metadata.json").read_text())
    assert metadata["video"]["frames"] == result.frames
    # > the CV thread ran beside the recording without slowing it (fps checked above), on the camera clock
    assert metadata["live_cv"]["frames_analysed"] > 10 and metadata["live_cv"]["frames_skipped"] > 0
    assert cv_engine.times == sorted(cv_engine.times) and cv_engine.times[0] >= 0.0

    synced = sync_recording(folder, folder / "video.mp4", mvc_path=None)
    assert synced.report.verdict == "PASS", synced.report
    assert synced.method == "led" and max(abs(e) for e in synced.report.check_errors_ms) < 33
    assert len(synced.clock_check) == 3          # > the laptop clock was checked against every flash too


class DarkCamera(FakeCamera):
    """The same camera, but the LED is never in the picture."""

    def read(self):
        self.next_time += 1 / 30
        delay = self.next_time - time.perf_counter()
        if delay > 0:
            time.sleep(delay)
        frame = np.full((240, 320, 3), 35, np.uint8) + self.rng.integers(0, 6, (240, 320, 3), dtype=np.uint8)
        return True, frame, time.perf_counter()


def test_record_set_without_the_led_syncs_on_the_laptop_clock(tmp_path: Path) -> None:
    esp = FakeEsp32()
    stream = EspStream(AcquisitionConfig(host="127.0.0.1", port=esp.port, read_timeout_s=0.02))
    folder = tmp_path / "TEST" / "S01_forearm_fist"
    plan = SetPlan("TEST", "forearm", "fist", reps=2, tempo_s=(0.4, 0.4, 0.4))
    try:
        result = record_set(DarkCamera(esp), stream, plan, mvc_above_rest_mv=100.0, folder=folder, display=False,
                            cfg=FAST, fps_probe_s=0.5)
    finally:
        stream.close()
        esp.stop()
    synced = sync_recording(folder, folder / "video.mp4", mvc_path=None)
    assert synced.method == "clock" and synced.report.verdict == "NOT MEASURED"
    assert synced.frames_with_emg >= result.frames - 2 and abs(synced.fit.drift_ppm) < 1000


def test_camera_settings_per_operating_system() -> None:
    from myomotion.sync.recorder import camera_settings
    linux, mac, other = camera_settings("linux"), camera_settings("darwin"), camera_settings("win32")
    assert (linux.backend_name, linux.manual_exposure, linux.driver_stamps) == ("V4L2", True, True)
    assert (mac.backend_name, mac.manual_exposure, mac.driver_stamps) == ("AVFOUNDATION", False, False)
    assert (other.backend_name, other.driver_stamps) == ("ANY", False)


class FakeCapture:
    """Stands in for cv2.VideoCapture: remembers how it was opened and what was set."""

    opened_with: list = []

    def __init__(self, index, backend) -> None:
        FakeCapture.opened_with.append(backend)
        self.props: dict = {}

    def isOpened(self) -> bool:
        return True

    def set(self, prop, value) -> bool:
        self.props[prop] = value
        return True

    def get(self, prop) -> float:
        return 1234.0                                  # > a driver time that macOS must ignore

    def read(self):
        return True, np.zeros((4, 4, 3), np.uint8)

    def release(self) -> None:
        pass


def test_the_webcam_on_macos_uses_avfoundation_auto_exposure_and_laptop_times(monkeypatch) -> None:
    import cv2
    from myomotion.sync.recorder import Webcam
    monkeypatch.setattr(cv2, "VideoCapture", FakeCapture)
    mac = Webcam(0, exposure=300.0, platform="darwin")
    assert FakeCapture.opened_with[-1] == cv2.CAP_AVFOUNDATION
    assert cv2.CAP_PROP_EXPOSURE not in mac.cap.props and not mac.fixed_exposure
    assert mac.read()[2] is None                       # > the recorder then stamps the frame with the laptop clock
    assert mac.info["frame_times"] == "laptop clock at read"
    linux = Webcam(0, exposure=300.0, platform="linux")
    assert FakeCapture.opened_with[-1] == cv2.CAP_V4L2 and linux.cap.props[cv2.CAP_PROP_EXPOSURE] == 300.0
    assert linux.read()[2] == pytest.approx(1.234)


def test_record_synced_takes_a_camera_delay() -> None:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "sync"))
    import record_synced
    assert record_synced.parse_args(["--camera-delay-ms", "25"]).camera_delay_ms == 25.0
    assert record_synced.parse_args([]).camera_delay_ms == pytest.approx(10.7)
