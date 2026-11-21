"""Tests for src/myomotion/semg/acquisition.py and live.py (Step 5), without an ESP32.

Run from the repo root:  python -m pytest tests/test_semg_live.py -q
The stream test starts a fake ESP32: a small TCP server on this computer.
"""

from __future__ import annotations

import socket
import threading

import numpy as np
import pytest

from myomotion.semg.acquisition import AcquisitionConfig, EspStream, LineParser, parse_line
from myomotion.semg.dsp import emg_envelope
from myomotion.semg.live import LIVE, REST, SETTLING, LivePercentMvc

RATE_HZ = 500.0


# ---------------------------------------------------------------------------
# raw_signal(seconds_and_sizes, seed)
# What:  a raw-EMG-like test signal: 600 mV offset, 50 Hz hum, and broadband
#        "muscle" noise whose size changes per stretch [(seconds, size_mv), ...].
# Time:  O(N) for N samples.  Space: O(N).
# Data structures: NumPy arrays.  Files used: none.
# ---------------------------------------------------------------------------
def raw_signal(seconds_and_sizes: list[tuple[float, float]], seed: int = 0) -> np.ndarray:
    """Concatenated stretches of muscle noise on an offset with mains hum."""
    rng = np.random.default_rng(seed)
    parts = [size * rng.standard_normal(round(seconds * RATE_HZ)) for seconds, size in seconds_and_sizes]
    noise = np.concatenate(parts)
    t_s = np.arange(len(noise)) / RATE_HZ
    return 600 + 80 * np.sin(2 * np.pi * 50 * t_s) + noise


def test_parse_line_reads_samples_and_skips_the_rest() -> None:
    sample = parse_line("D,1000,150,900,800")
    assert sample is not None and sample.t_us == 1000 and sample.values_mv == (150.0, 900.0, 800.0)
    assert parse_line("E,2000,LED_ON") is None
    assert parse_line("MyoMotion-EMG ready") is None
    assert parse_line("D,12x,150") is None


def test_line_parser_joins_lines_split_across_reads() -> None:
    """TCP can cut a line anywhere: the parser waits for the rest of it."""
    parser = LineParser()
    assert parser.feed(b"D,1000,15") == []
    samples = parser.feed(b"0,900,800\r\nE,1500,LED_ON\r\nD,3000,160,900,800\r\nD,50")
    assert [s.t_us for s in samples] == [1000, 3000]
    assert samples[0].values_mv[0] == 150.0
    assert parser.feed(b"00,170,900,800\r\nD,bad\r\n")[0].t_us == 5000
    assert parser.bad_lines == 1


def test_line_parser_unwraps_the_32_bit_clock() -> None:
    """micros() restarts at 0 after 2^32 us; the parser keeps the time increasing."""
    parser = LineParser()
    samples = parser.feed(b"D,4294966000,1\nD,704,1\n")
    assert samples[1].t_us == 4294967296 + 704
    assert samples[1].t_us - samples[0].t_us == 2000


def test_meter_goes_settling_rest_live() -> None:
    meter = LivePercentMvc(mvc_above_rest_mv=100.0)
    stages = [meter.step(value).stage for value in raw_signal([(5.0, 5.0)])]
    assert stages[0] == SETTLING and stages[499] == SETTLING
    assert stages[500] == REST and stages[1998] == REST
    assert stages[1999] == LIVE and stages[-1] == LIVE
    assert meter.seconds_left() == 0.0


def test_meter_reads_zero_at_rest_and_the_right_percent_when_working() -> None:
    """MVC = the envelope of a 200 mV burst; a 100 mV burst should read about 50%, rest about 0%."""
    mvc_burst = emg_envelope(raw_signal([(4.0, 200.0)], seed=1), RATE_HZ)[500:].mean()
    rest_level = emg_envelope(raw_signal([(4.0, 5.0)], seed=2), RATE_HZ)[500:].mean()
    meter = LivePercentMvc(mvc_above_rest_mv=float(mvc_burst - rest_level))
    readings = [meter.step(value) for value in raw_signal([(4.0, 5.0), (2.0, 5.0), (3.0, 100.0)], seed=3)]
    relaxed = [r.percent for r in readings[2100:3000]]             # > 4.2-6 s: live, still relaxed
    working = [r.percent for r in readings[3200:4500]]             # > 6.4-9 s: the 100 mV burst
    assert abs(np.mean(relaxed)) < 3
    assert 40 < np.mean(working) < 60


def test_live_envelope_matches_the_offline_envelope() -> None:
    """Sample by sample (live) and whole array (offline) give the same envelope once settled."""
    signal = raw_signal([(2.0, 5.0), (3.0, 150.0)], seed=4)
    meter = LivePercentMvc(mvc_above_rest_mv=100.0)
    live = np.array([meter.step(value).envelope_mv for value in signal])
    offline = emg_envelope(signal, RATE_HZ)
    assert np.allclose(live[1000:], offline[1000:], rtol=0.02, atol=0.5)


def test_meter_rejects_an_mvc_below_rest() -> None:
    with pytest.raises(ValueError):
        LivePercentMvc(mvc_above_rest_mv=-5.0)


def test_esp_stream_reads_from_a_fake_esp32() -> None:
    """A local TCP server plays the ESP32: two sample lines, one sync reply, then it hangs up."""
    server = socket.socket()
    server.bind(("127.0.0.1", 0))                      # > port 0: let the system pick a free port
    server.listen(1)
    port = server.getsockname()[1]

    def serve() -> None:
        connection, _ = server.accept()
        connection.sendall(b"D,1000,150,900,800\r\nE,1500,LED_ON\r\nD,3000,16")
        connection.sendall(b"0,900,800\r\n")
        connection.close()

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    samples = []
    with EspStream(AcquisitionConfig(host="127.0.0.1", port=port, read_timeout_s=0.5)) as stream:
        with pytest.raises(ConnectionError):
            for _ in range(10):                        # > read until the fake ESP32 hangs up
                samples += stream.read()
    thread.join(timeout=2)
    server.close()
    assert [s.t_us for s in samples] == [1000, 3000]
    assert samples[1].values_mv == (160.0, 900.0, 800.0)


def test_esp_stream_explains_a_failed_connection() -> None:
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    port = server.getsockname()[1]
    server.close()                                     # > nothing listens on this port any more
    with pytest.raises(ConnectionError, match="MyoMotion-EMG"):
        EspStream(AcquisitionConfig(host="127.0.0.1", port=port, connect_timeout_s=1.0))
