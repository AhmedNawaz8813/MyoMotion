"""Tests for src/myomotion/semg/mvc.py (MVC, TC02 and %MVC), using synthetic raw-EMG recordings.

Run from the repo root:  python -m pytest tests/test_semg_mvc.py -q
pytest finds `myomotion` because pyproject.toml sets pythonpath = ["src"].
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pytest

from myomotion.semg.mvc import (Cue, MvcResult, best_window_mean, cue_windows, find_latest_mvc, load_mvc,
                                measure_mvc, normalise_set, percent_mvc, phase_summary, save_mvc)
from myomotion.semg.quality import Recording

PERIOD_US = 2000                 # 500 Hz
REST_NOISE_MV = 5.0              # broadband noise at rest
ACTIVE = {"MVC", "CONCENTRIC", "ISOMETRIC_HOLD", "ECCENTRIC"}


# ---------------------------------------------------------------------------
# synthetic_recording(plan, burst_levels)
# What:  a fake recording that follows `plan` [(cue, seconds), ...]: a 600 mV
#        offset plus small broadband noise, and in every active cue a broadband
#        "muscle" burst whose size comes from burst_levels in order (one value per
#        active cue). A burst starts 0.3 s after its cue only at the start of a rep or
#        push (reaction time); later phases of the same rep continue straight on.
# Time:  O(N * P) for N samples and P cues.  Space: O(N).
# Data structures: NumPy arrays; list of Cue; burst_levels is a list consumed in order.
# Files used: none.
# ---------------------------------------------------------------------------
def synthetic_recording(plan: list[tuple[str, float]], burst_levels: list[float], seed: int = 0) -> Recording:
    """Synthetic raw-EMG recording; burst sizes (mV) given per active cue, in order."""
    rng = np.random.default_rng(seed)
    total_s = sum(seconds for _, seconds in plan)
    t_us = np.arange(0, round(total_s * 1_000_000), PERIOD_US, dtype=np.int64)
    t_s = t_us / 1_000_000
    signal = 600 + REST_NOISE_MV * rng.standard_normal(len(t_us))
    levels = iter(burst_levels)
    cues: list[Cue] = []
    start_s = 0.0
    previous_active = False
    for event, seconds in plan:
        cues.append(Cue(t_us=round(start_s * 1_000_000), event=event))
        if event in ACTIVE:
            reaction_s = 0.0 if previous_active else 0.3
            in_burst = (t_s >= start_s + reaction_s) & (t_s < start_s + seconds)
            signal[in_burst] += next(levels) * rng.standard_normal(int(np.sum(in_burst)))
        previous_active = event in ACTIVE
        start_s += seconds
    signal = np.clip(signal, 142.0, 3100.0)              # > the ESP32 cannot report outside its floor and ceiling
    return Recording(t_us=t_us, signal_mv=signal, cues=cues, baseline_s=5.0)


def mvc_session(levels: list[float], push_s: float = 5.0) -> Recording:
    """Baseline 8 s, then one push per level with 3 s rests between (short rests keep the test fast)."""
    plan: list[tuple[str, float]] = [("BASELINE", 8.0)]
    for index in range(len(levels)):
        plan.append(("MVC", push_s))
        if index < len(levels) - 1:
            plan.append(("REST", 3.0))
    plan.append(("TAIL", 3.0))
    return synthetic_recording(plan, levels)


HAND_SET = ([("BASELINE", 8.0)]
            + [("CONCENTRIC", 2.0), ("ISOMETRIC_HOLD", 1.0), ("ECCENTRIC", 2.0), ("REST", 1.0)] * 3
            + [("TAIL", 3.0)])


def test_cue_windows_run_to_the_next_cue() -> None:
    cues = [Cue(0, "BASELINE"), Cue(100, "MVC"), Cue(250, "REST"), Cue(400, "MVC")]
    assert cue_windows(cues, "MVC", last_t_us=999) == [(100, 250), (400, 1000)]


def test_best_window_mean_finds_the_strongest_second() -> None:
    mean, start = best_window_mean(np.array([0.0, 0.0, 5.0, 5.0, 5.0, 0.0]), window=3)
    assert (mean, start) == (5.0, 2)
    with pytest.raises(ValueError):
        best_window_mean(np.ones(2), window=3)


def test_three_good_pushes_pass_tc02() -> None:
    """Pushes of 200, 190 and 180 mV: all found, all strong, best two 5% apart."""
    result = measure_mvc(mvc_session([200.0, 190.0, 180.0]))
    assert result.passed
    assert len(result.trial_peaks_mv) == 3
    assert result.trial_peaks_mv[0] > result.trial_peaks_mv[2]          # > order of the pushes is kept
    assert result.mvc_envelope_mv == max(result.trial_peaks_mv)
    assert result.mvc_above_rest_mv == pytest.approx(result.mvc_envelope_mv - result.rest_envelope_mv)
    assert result.rest_envelope_mv < 10                                 # > noise floor of 5 mV broadband


def test_pushes_that_disagree_fail_consistency() -> None:
    result = measure_mvc(mvc_session([200.0, 120.0, 100.0]))
    assert result.best_two_gap > 0.3
    assert not result.consistency_ok and not result.passed


def test_a_weak_push_fails_strength() -> None:
    result = measure_mvc(mvc_session([200.0, 195.0, 8.0]))
    assert not result.strength_ok and not result.passed


def test_a_clipped_push_fails_tc02() -> None:
    """Pushes so strong that the raw signal hits the 3100 mV ceiling: the MVC cannot be trusted."""
    result = measure_mvc(mvc_session([2000.0, 2000.0, 2000.0]))
    assert result.clipped_samples > 0
    assert not result.clipping_ok and not result.passed


def test_a_push_shorter_than_one_second_is_skipped() -> None:
    """An aborted 0.6 s push cannot give a 1-second mean: only the other two count."""
    plan = [("BASELINE", 8.0), ("MVC", 5.0), ("REST", 3.0), ("MVC", 5.0), ("REST", 3.0), ("MVC", 0.6)]
    result = measure_mvc(synthetic_recording(plan, [200.0, 195.0, 200.0]))
    assert len(result.trial_peaks_mv) == 2
    assert not result.trials_ok


def test_no_push_is_an_error() -> None:
    with pytest.raises(ValueError):
        measure_mvc(synthetic_recording([("BASELINE", 8.0), ("TAIL", 3.0)], []))


def test_percent_mvc_formula() -> None:
    envelope = np.array([10.0, 60.0, 110.0])
    assert percent_mvc(envelope, rest_mv=10.0, mvc_above_rest_mv=100.0).tolist() == [0.0, 50.0, 100.0]
    with pytest.raises(ValueError):
        percent_mvc(envelope, rest_mv=10.0, mvc_above_rest_mv=0.0)


def test_phase_summary_numbers_reps_and_phases() -> None:
    """Two reps of (CONCENTRIC, ISOMETRIC_HOLD) with known %MVC levels."""
    cues = [Cue(0, "BASELINE"), Cue(10, "CONCENTRIC"), Cue(20, "ISOMETRIC_HOLD"), Cue(30, "REST"),
            Cue(40, "CONCENTRIC"), Cue(50, "ISOMETRIC_HOLD"), Cue(60, "TAIL")]
    t_us = np.arange(0, 70)
    percent = np.zeros(70)
    percent[10:20], percent[20:30], percent[40:50], percent[50:60] = 40.0, 60.0, 30.0, 50.0
    stats = phase_summary(t_us, percent, cues)
    assert [(s.rep, s.phase, s.mean_percent) for s in stats] == [
        (1, "CONCENTRIC", 40.0), (1, "ISOMETRIC_HOLD", 60.0), (2, "CONCENTRIC", 30.0), (2, "ISOMETRIC_HOLD", 50.0)]


def test_set_at_half_the_mvc_reads_about_fifty_percent() -> None:
    """MVC session with 200 mV pushes; a hand set with 100 mV bursts reads ~50% MVC, rest ~0%."""
    mvc = measure_mvc(mvc_session([200.0, 200.0, 200.0]))
    normalised = normalise_set(synthetic_recording(HAND_SET, [100.0] * 9, seed=1), mvc.mvc_above_rest_mv)
    assert len(normalised.phases) == 9 and normalised.phases[-1].rep == 3
    holds = [stat.mean_percent for stat in normalised.phases if stat.phase == "ISOMETRIC_HOLD"]
    assert all(40 < value < 60 for value in holds)
    rest = normalised.percent[500:2500]                    # > 1-5 s of the baseline
    assert abs(float(np.mean(rest))) < 3


def test_save_load_and_find_latest(tmp_path: Path) -> None:
    result = MvcResult(rate_hz=500.0, rest_envelope_mv=5.0, trial_peaks_mv=[200.0, 190.0], trial_peak_start_us=[1, 2],
                       mvc_envelope_mv=200.0, mvc_above_rest_mv=195.0, best_two_gap=0.05, clipped_samples=0,
                       trials_ok=False, strength_ok=True, consistency_ok=True, clipping_ok=True)
    older, newer = tmp_path / "S01_forearm_mvc", tmp_path / "S03_forearm_mvc"
    older.mkdir()
    newer.mkdir()
    save_mvc(result, older, {"folder": older.name, "channel": "ch1", "muscle": "forearm"})
    save_mvc(result, newer, {"folder": newer.name, "channel": "ch1", "muscle": "forearm"})
    old_time = time.time() - 100
    os.utime(older / "mvc.json", (old_time, old_time))     # > make S01 clearly older
    assert find_latest_mvc(tmp_path, passed_only=False) == newer / "mvc.json"
    loaded = load_mvc(newer)
    assert loaded["mvc_above_rest_mv"] == 195.0 and loaded["passed"] is False and loaded["muscle"] == "forearm"
    assert find_latest_mvc(tmp_path) is None              # > neither session passed TC02
    result.trials_ok = True                               # > now the OLDER session passes
    save_mvc(result, older, {"folder": older.name, "channel": "ch1", "muscle": "forearm"})
    os.utime(older / "mvc.json", (old_time, old_time))
    assert find_latest_mvc(tmp_path) == older / "mvc.json"   # > the newer one failed, so it is skipped
    assert find_latest_mvc(tmp_path / "S01_forearm_mvc") is None
