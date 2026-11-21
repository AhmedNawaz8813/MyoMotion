"""Tests for src/myomotion/semg/dsp.py, using synthetic signals (no ESP32 needed).

Run from the repo root:  python -m pytest tests/test_semg_dsp.py -q
pytest finds `myomotion` because pyproject.toml sets pythonpath = ["src"].
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from myomotion.semg.dsp import (EmgFilter, RunningRms, emg_band_rms, emg_envelope, mains_rms_from_spectrum,
                                mean_and_sd, moving_average, moving_rms, power_spectrum, rectify, rms)

RATE_HZ = 500.0
SETTLE = 250                     # skip the first 0.5 s of filter output when the test needs it settled


# ---------------------------------------------------------------------------
# sine(freq_hz, amplitude, count)
# What:  a test sine wave sampled at RATE_HZ.
# Time:  O(count).  Space: O(count).
# Data structures: NumPy float64 array.  Files used: none.
# ---------------------------------------------------------------------------
def sine(freq_hz: float, amplitude: float, count: int) -> np.ndarray:
    """amplitude * sin(2 pi f t) for `count` samples at RATE_HZ."""
    return amplitude * np.sin(2 * np.pi * freq_hz * np.arange(count) / RATE_HZ)


def test_rms_of_sine_is_amplitude_over_root_two() -> None:
    """RMS of a sine of amplitude A over whole cycles is A / sqrt(2)."""
    assert rms(sine(50.0, 2.0, 1000)) == pytest.approx(2.0 / math.sqrt(2))


def test_mean_and_sd_match_hand_calculation() -> None:
    """[2, 4, 4, 4, 5, 5, 7, 9]: mean 5, squared differences add up to 32, sample SD = sqrt(32 / 7)."""
    mean, sd = mean_and_sd(np.array([2, 4, 4, 4, 5, 5, 7, 9], dtype=float))
    assert mean == 5.0
    assert sd == pytest.approx(math.sqrt(32 / 7))


def test_power_spectrum_matches_numpy_fft() -> None:
    """Our O(N^2) DFT must give the same powers and frequencies as NumPy's O(N log N) FFT."""
    values = np.random.default_rng(1).standard_normal(200)
    freqs_hz, power = power_spectrum(values, RATE_HZ)
    expected = np.abs(np.fft.rfft(values - values.mean())) ** 2 / len(values) ** 2
    expected[1:-1] *= 2                                  # 200 is even: double every bin except 0 and N/2
    assert np.allclose(power, expected)
    assert np.allclose(freqs_hz, np.fft.rfftfreq(len(values), d=1 / RATE_HZ))


def test_power_adds_up_to_variance() -> None:
    """Parseval: the spectrum's powers add up to the signal's variance."""
    values = np.random.default_rng(2).standard_normal(301)   # odd length: there is no Nyquist bin
    _, power = power_spectrum(values, RATE_HZ)
    assert float(np.sum(power)) == pytest.approx(float(np.var(values)))


def test_emg_band_rms_keeps_the_muscle_band_only() -> None:
    """Offset, a 5 Hz wobble and 50 Hz hum are left out; an 80 Hz 'muscle' sine of amplitude 10 is kept."""
    values = 600 + sine(5.0, 100.0, 2500) + sine(50.0, 100.0, 2500) + sine(80.0, 10.0, 2500)
    assert emg_band_rms(values, RATE_HZ) == pytest.approx(10.0 / math.sqrt(2), rel=1e-6)


def test_mains_rms_measures_the_hum() -> None:
    """A pure 50 Hz sine of amplitude 10 mV is hum of RMS 10/sqrt(2) and 100% of the power."""
    freqs_hz, power = power_spectrum(142.0 + sine(50.0, 10.0, 2500), RATE_HZ)
    hum_rms, hum_share = mains_rms_from_spectrum(freqs_hz, power, RATE_HZ)
    assert hum_rms == pytest.approx(10.0 / math.sqrt(2))
    assert hum_share == pytest.approx(1.0)


def test_filter_removes_offset_drift_and_hum_but_keeps_muscle_band() -> None:
    """After settling, only the 80 Hz part (RMS 7.07 mV) is left of offset + 2 Hz + 50/150 Hz + 80 Hz."""
    values = (600 + sine(2.0, 100.0, 5000) + sine(50.0, 200.0, 5000) + sine(150.0, 100.0, 5000)
              + sine(80.0, 10.0, 5000))
    cleaned = EmgFilter(RATE_HZ).filter(values)
    assert rms(cleaned[SETTLE:]) == pytest.approx(10.0 / math.sqrt(2), rel=0.05)


def test_whole_array_filter_equals_sample_by_sample() -> None:
    """filter() on an array and step() one sample at a time (the live form) give the same output."""
    values = np.random.default_rng(3).standard_normal(1000) * 50 + 600
    whole = EmgFilter(RATE_HZ).filter(values, settle=False)
    live_filter = EmgFilter(RATE_HZ)
    live = np.array([live_filter.step(float(x)) for x in values])
    assert np.allclose(whole, live)


def test_constant_input_gives_no_start_jump() -> None:
    """A recording that starts at 600 mV and stays there gives 0 out, from the very first sample."""
    cleaned = EmgFilter(RATE_HZ).filter(np.full(1000, 600.0))
    assert np.max(np.abs(cleaned)) < 1e-6


def test_settle_padding_stops_start_ringing() -> None:
    """Pure hum: with padding the start is quiet; without it the notches ring for a while."""
    hum_only = 300 + sine(50.0, 200.0, 2000) + sine(150.0, 100.0, 2000)
    padded = EmgFilter(RATE_HZ).filter(hum_only)
    cold = EmgFilter(RATE_HZ).filter(hum_only, settle=False)
    assert np.max(np.abs(padded[:SETTLE])) < 5.0
    assert np.max(np.abs(cold[:SETTLE])) > 50.0


def test_time_domain_filter_agrees_with_spectral_band_rms() -> None:
    """The two ways of measuring the muscle band (filters vs spectrum) agree within 10% on white noise."""
    noise = np.random.default_rng(4).standard_normal(5000) * 20
    filtered = EmgFilter(RATE_HZ).filter(noise)
    assert rms(filtered[SETTLE:]) == pytest.approx(emg_band_rms(noise[SETTLE:], RATE_HZ), rel=0.10)


def test_moving_average_and_rms_match_brute_force() -> None:
    """Compare the O(n) running-total versions with a plain loop over every window."""
    values = np.random.default_rng(5).standard_normal(40)
    window = 5
    for index in range(len(values)):
        chunk = values[max(0, index - window + 1): index + 1]
        assert moving_average(values, window)[index] == pytest.approx(chunk.mean())
        assert moving_rms(values, window)[index] == pytest.approx(math.sqrt(np.mean(chunk ** 2)))


def test_running_rms_matches_moving_rms() -> None:
    """The live ring-buffer version gives the same numbers as the whole-array version."""
    values = np.random.default_rng(6).standard_normal(200)
    running = RunningRms(window=7)
    live = np.array([running.step(float(x)) for x in values])
    assert np.allclose(live, moving_rms(values, 7))
    expected_sum = sum(float(x) ** 2 for x in values[-7:])
    running.recompute()
    assert running.sum_of_squares == pytest.approx(expected_sum)


def test_rectify_and_window_checks() -> None:
    assert rectify(np.array([-3.0, 0.0, 2.0])).tolist() == [3.0, 0.0, 2.0]
    with pytest.raises(ValueError):
        moving_average(np.ones(3), 0)
    with pytest.raises(ValueError):
        RunningRms(window=0)


def test_envelope_rises_during_a_burst() -> None:
    """Raw-EMG-like input (offset + hum + noise, a burst from 2 to 3 s): the envelope is >10x higher in it."""
    rng = np.random.default_rng(7)
    count = 2500
    values = 600 + sine(50.0, 150.0, count) + 5 * rng.standard_normal(count)
    values[1000:1500] += 100 * rng.standard_normal(500)  # > 2-3 s: the muscle works
    envelope = emg_envelope(values, RATE_HZ)
    assert np.median(envelope[1100:1450]) > 10 * np.median(envelope[300:900])
