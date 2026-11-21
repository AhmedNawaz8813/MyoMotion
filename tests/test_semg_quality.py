"""Tests for src/myomotion/semg/quality.py (TC01), using synthetic raw-EMG recordings (no ESP32 needed).

Run from the repo root:  python -m pytest tests/test_semg_quality.py -q
pytest finds `myomotion` because pyproject.toml sets pythonpath = ["src"].
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from myomotion.semg.quality import (Cue, Recording, baseline_drift, check_quality, load_recording,
                                    longest_true_run, rep_windows, rest_mask, save_report)

RATE_HZ = 500.0
PERIOD_US = 2000                 # 500 Hz = one sample every 2000 microseconds
FLOOR_MV = 142.0                 # the ESP32 cannot read lower than this
CEILING_MV = 3100.0              # nor higher than this
BASELINE_S = 5.0                 # real rest the recorder keeps (countdown excluded)
ACTIVE = {"ECCENTRIC", "ISOMETRIC_HOLD", "CONCENTRIC"}
# The hand test as the recorder runs it: (cue, seconds). BASELINE = 5 s rest + 3 s countdown.
HAND_PLAN = ([("BASELINE", 8.0)]
             + [("CONCENTRIC", 2.0), ("ISOMETRIC_HOLD", 1.0), ("ECCENTRIC", 2.0), ("REST", 1.0)] * 3
             + [("TAIL", 3.0)])


# ---------------------------------------------------------------------------
# make_recording(...)
# What:  a fake hand-test recording shaped like the real ESP32 data: a DC offset,
#        small broadband noise at rest, a much bigger broadband "muscle" burst in
#        every active phase (from 0.2 s after the rep's first cue, like a real
#        reaction), optional mains hum and extra effects, then clamped to the
#        ESP32's floor and ceiling as the real ADC would.
# Time:  O(N * P): N samples (14,500) compared once per phase P (14).
# Space: O(N).
# Data structures: NumPy arrays for time and signal; a list of Cue; `extra` is an
#        optional array added on top (used for drift and rest bursts).
# Files used: none.
# ---------------------------------------------------------------------------
def make_recording(burst_mv: float = 100.0, hum_mv: float = 0.0, rest_mv: float = 600.0, noise_mv: float = 5.0,
                   plan: list[tuple[str, float]] | None = None, extra: np.ndarray | None = None,
                   seed: int = 0) -> Recording:
    """Synthetic raw-EMG recording that follows `plan` (the hand test by default)."""
    plan = plan or HAND_PLAN
    rng = np.random.default_rng(seed)                    # > fixed seed: the same "random" noise every run
    total_s = sum(seconds for _, seconds in plan)
    t_us = np.arange(0, round(total_s * 1_000_000), PERIOD_US, dtype=np.int64)
    t_s = t_us / 1_000_000
    signal = rest_mv + noise_mv * rng.standard_normal(len(t_us)) + hum_mv * np.sin(2 * np.pi * 50.0 * t_s)
    cues: list[Cue] = []
    start_s = 0.0
    previous_active = False
    for event, seconds in plan:
        cues.append(Cue(t_us=round(start_s * 1_000_000), event=event))
        if event in ACTIVE:
            reaction_s = 0.0 if previous_active else 0.2  # > you react only at the start of a rep
            in_phase = (t_s >= start_s + reaction_s) & (t_s < start_s + seconds)
            signal[in_phase] += burst_mv * rng.standard_normal(int(np.sum(in_phase)))
        previous_active = event in ACTIVE
        start_s += seconds
    if extra is not None:
        signal += extra
    signal = np.clip(signal, FLOOR_MV, CEILING_MV)       # > the ADC cannot report outside these
    return Recording(t_us=t_us, signal_mv=signal, cues=cues, baseline_s=BASELINE_S)


def test_longest_true_run() -> None:
    flags = np.array([False, True, True, False, True, True, True, False])
    assert longest_true_run(flags) == 3


def test_baseline_drift_is_second_half_minus_first() -> None:
    assert baseline_drift(np.array([10.0, 10.0, 30.0, 30.0])) == 20.0


def test_rest_mask_keeps_baseline_and_drops_countdown() -> None:
    """5 s of rest at 500 Hz = 2500 samples; the 3 s countdown is left out."""
    recording = make_recording()
    assert int(np.sum(rest_mask(recording.t_us, recording.cues, recording.baseline_s))) == 2500


@pytest.mark.parametrize("first, last", [("CONCENTRIC", "ECCENTRIC"), ("ECCENTRIC", "CONCENTRIC")])
def test_rep_windows_one_per_rep_in_either_phase_order(first: str, last: str) -> None:
    """Hand test (SQUEEZE first) and squat (DOWN first): 3 windows of 5 s - 0.3 s reaction = 4.7 s each."""
    plan = [("BASELINE", 8.0)] + [(first, 2.0), ("ISOMETRIC_HOLD", 1.0), (last, 2.0), ("REST", 1.0)] * 3
    windows = rep_windows(make_recording(plan=plan).cues, reaction_delay_s=0.3, last_t_us=26_000_000)
    assert len(windows) == 3
    assert all(end - start == 4_700_000 for start, end in windows)
    assert windows[0][0] == 8_300_000


def test_clean_recording_passes_without_warnings() -> None:
    report = check_quality(make_recording())
    assert report.passed
    assert report.rate_hz == pytest.approx(RATE_HZ)
    assert report.snr_ratio > 3
    assert len(report.rep_ratios) == 3 and min(report.rep_ratios) > 3
    assert not report.hum_warning and not report.floor_warning


def test_weak_bursts_fail_snr_only() -> None:
    report = check_quality(make_recording(burst_mv=8.0))
    assert not report.snr_ok
    assert report.baseline_ok and report.clipping_ok


def test_strong_hum_warns_but_does_not_hide_the_muscle() -> None:
    """300 mV of 50 Hz hum: the muscle band leaves it out, so SNR still passes; the hum is flagged."""
    report = check_quality(make_recording(hum_mv=300.0))
    assert report.snr_ok and report.passed
    assert report.hum_warning
    assert report.strongest_hz == pytest.approx(50.0)


def test_signal_sitting_on_the_floor_warns() -> None:
    """A rest level right at the ADC floor: about half the rest samples are cut off."""
    report = check_quality(make_recording(rest_mv=FLOOR_MV))
    assert report.floor_warning


def test_clipping_is_caught() -> None:
    """Huge bursts on a high offset reach the 3100 mV ceiling: flat tops."""
    report = check_quality(make_recording(burst_mv=1500.0, rest_mv=2000.0))
    assert not report.clipping_ok
    assert report.longest_flat_top >= 2


def test_tension_during_the_rest_fails_the_baseline() -> None:
    """A burst in the second half of the rest makes the two halves very different."""
    count = round(sum(seconds for _, seconds in HAND_PLAN) * RATE_HZ)
    extra = np.zeros(count)
    extra[1500:2000] = 60 * np.random.default_rng(9).standard_normal(500)   # > 3-4 s: tensed while "relaxed"
    report = check_quality(make_recording(extra=extra))
    assert report.rest_halves_ratio > 2
    assert not report.baseline_ok


def test_slow_drift_fails_the_baseline() -> None:
    """The rest level climbs 60 mV over the 5 s baseline: about +30 mV between the halves."""
    count = round(sum(seconds for _, seconds in HAND_PLAN) * RATE_HZ)
    extra = np.zeros(count)
    extra[:2500] = np.linspace(0.0, 60.0, 2500)
    report = check_quality(make_recording(extra=extra))
    assert report.rest_drift_mv == pytest.approx(30.0, abs=3.0)
    assert not report.baseline_ok


def test_no_reps_is_an_error() -> None:
    with pytest.raises(ValueError):
        check_quality(make_recording(plan=[("BASELINE", 8.0), ("TAIL", 3.0)]))


def test_load_and_save_round_trip(tmp_path: Path) -> None:
    """Writes the three files the recorder makes, loads them back, then saves quality.json."""
    (tmp_path / "emg.csv").write_text(
        "pc_time_s, t_us, ch1, ch2, ch3\n"
        "0.000, 1000, 150, 900, 800\n"
        "0.002, 3000, 160, 900, 800\n"
        "0.004, 5000, 170, 900, 800\n"
    )
    (tmp_path / "events.csv").write_text(
        "pc_time_s,t_us,source,event,rep\n"
        "0.0,1000,cue,BASELINE,\n"
        "0.1,4000.0,board,LED_ON,\n"
        "0.2,5000,cue,CONCENTRIC,1\n"
    )
    (tmp_path / "metadata.json").write_text(json.dumps({"baseline_s": 5.0}))
    recording = load_recording(tmp_path)
    assert recording.t_us.tolist() == [1000, 3000, 5000]
    assert recording.signal_mv.tolist() == [150.0, 160.0, 170.0]
    assert [cue.event for cue in recording.cues] == ["BASELINE", "CONCENTRIC"]   # the board's LED_ON is left out
    assert recording.baseline_s == 5.0

    saved = json.loads(save_report(check_quality(make_recording()), tmp_path).read_text())
    assert saved["passed"] is True
    assert len(saved["rep_ratios"]) == 3
