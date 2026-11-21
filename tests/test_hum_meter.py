"""Tests for the live hum meter (scripts/semg/hum_meter.py): its numbers on signals with a known hum.

Run from the repo root:  python -m pytest tests/test_hum_meter.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "semg"))
from hum_meter import format_reading, measure_window  # noqa: E402

RATE = 500.0
T = np.arange(1000) / RATE                         # > one 2-second window


def test_a_known_100_hz_hum_is_measured() -> None:
    rng = np.random.default_rng(0)
    signal = 1400 + 200 * np.sin(2 * np.pi * 100 * T) + 3 * rng.standard_normal(len(T))
    reading = measure_window(signal)
    assert reading.hum_mv == pytest.approx(200 / np.sqrt(2), rel=0.02)          # > RMS of a sine = amplitude / sqrt 2
    assert reading.hum_share > 0.95 and reading.level_mv == pytest.approx(1400, abs=2)
    assert reading.verdict.startswith("HIGH")


def test_a_quiet_relaxed_muscle_is_good() -> None:
    rng = np.random.default_rng(1)
    signal = 1350 + 5 * np.sin(2 * np.pi * 50 * T) + 4 * rng.standard_normal(len(T))
    reading = measure_window(signal)
    assert reading.hum_mv < 10 and reading.verdict == "GOOD"
    assert reading.muscle_band_mv == pytest.approx(4 * np.sqrt(0.92 * 0.8), rel=0.3)   # > white noise in 20-250 Hz


def test_the_line_flags_clipping() -> None:
    signal = 1400 + 1400 * np.sin(2 * np.pi * 100 * T)    # > sampled peak 1400 + 1400 * sin(72 deg) = 2731 mV
    assert "NEAR CLIPPING" in format_reading(4.0, measure_window(signal))
