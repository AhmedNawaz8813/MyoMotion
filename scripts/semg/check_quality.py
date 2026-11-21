"""Run the TC01 signal-quality check on one recording folder, and plot it.

Usage, from the repo root:
    python scripts/semg/check_quality.py ~/Downloads/data/TEST/S13_forearm_fist
    python scripts/semg/check_quality.py <folder> --channel ch1 --no-plot

Thin command-line wrapper only: the logic lives in src/myomotion/semg/quality.py (TC01)
and src/myomotion/semg/dsp.py (filters and envelope).
Writes quality.json (the numbers) and quality.png (raw / cleaned / envelope) into the folder.
Exit code 0 = TC01 passed, 1 = failed.

Data structures (this file): a Recording (NumPy arrays and cues), a QualityReport dataclass, a dict for
    quality.json, a matplotlib figure.
Time complexity (this file): as check_quality in quality.py: O(N + R^2 + A^2), dominated by the direct DFTs; O(N)
    more to plot.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

# > This file is scripts/semg/check_quality.py, so parents[2] is the repo root and the
# > package is in <repo>/src. Adding it to sys.path lets `import myomotion` work without
# > `pip install -e .` (the same trick as Ahmad's scripts/demo_deadlift.py).
_SRC = str(Path(__file__).resolve().parents[2] / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from myomotion.semg.dsp import EmgFilter, emg_envelope  # noqa: E402
from myomotion.semg.quality import (CONFIG, Recording, ceiling_for, check_quality, format_report,  # noqa: E402
                                    load_recording, measured_rate_hz, save_report)

# > Plot settings (the plot's own CONFIG): phase colours match the recorder's plot.png.
PLOT_CONFIG = {
    "phase_colours": {"ECCENTRIC": "#fde68a", "ISOMETRIC_HOLD": "#fecaca", "CONCENTRIC": "#bfdbfe"},
    "figure_size": (12, 8),
    "dpi": 150,
    "file_name": "quality.png",
}


# ---------------------------------------------------------------------------
# parse_args(argv)
# What:  reads the command line: the recording folder, the channel, and --no-plot.
# Time:  O(a) for a arguments.  Space: O(a).
# Data structures: argparse.Namespace (one attribute per option).
# Files used: none.
# ---------------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Read the recording folder, the channel name and the plot switch from the command line."""
    parser = argparse.ArgumentParser(description="TC01 signal-quality check for one sEMG recording")
    parser.add_argument("folder", type=Path, help="recording folder with emg.csv, events.csv and metadata.json")
    parser.add_argument("--channel", default=CONFIG.channel, help="emg.csv column to check (default: ch1)")
    parser.add_argument("--no-plot", action="store_true", help="skip quality.png")
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# save_plot(recording, folder)
# What:  three stacked panels sharing one time axis, phases shaded in each:
#        1. raw mV from the ESP32 (offset + hum + muscle),
#        2. after the Step 2 filter (20 Hz high-pass + mains notches): the muscle alone,
#        3. the 100 ms RMS envelope: muscle activation over time.
#        Time is taken from t_us (the ESP32 clock), starting at 0.
# Time:  O(N * S) for the filter (S = 6 sections) plus O(N) for the envelope and
#        drawing.
# Space: O(N).
# Data structures: NumPy arrays for time and signals; the cue list for shading.
# Files used: src/myomotion/semg/dsp.py (EmgFilter, emg_envelope);
#        writes <folder>/quality.png. matplotlib is imported here only, so the
#        check itself still runs where matplotlib is missing.
# ---------------------------------------------------------------------------
def save_plot(recording: Recording, folder: Path) -> Path | None:
    """Save quality.png (raw, cleaned, envelope); return its path, or None without matplotlib."""
    try:
        import matplotlib
        matplotlib.use("Agg")                            # > draw to a file, no window needed
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib not installed: skipped quality.png)")
        return None
    rate_hz = measured_rate_hz(recording.t_us)
    start_us = int(recording.t_us[0])
    time_s = (recording.t_us - start_us) / 1_000_000
    cleaned = EmgFilter(rate_hz).filter(recording.signal_mv)
    envelope = emg_envelope(recording.signal_mv, rate_hz)
    panels = [
        (recording.signal_mv, "raw (mV at the ESP32 pin)"),
        (cleaned, "filtered: 20 Hz high-pass + 50/100/150/200 Hz notches (mV)"),
        (envelope, "RMS envelope, 100 ms (mV)"),
    ]
    fig, axes = plt.subplots(len(panels), 1, figsize=PLOT_CONFIG["figure_size"], sharex=True)
    cue_times_s = [(cue.t_us - start_us) / 1_000_000 for cue in recording.cues] + [float(time_s[-1])]
    for axis, (values, label) in zip(axes, panels):
        for cue, begin_s, end_s in zip(recording.cues, cue_times_s, cue_times_s[1:]):
            colour = PLOT_CONFIG["phase_colours"].get(cue.event)
            if colour:
                axis.axvspan(begin_s, end_s, color=colour, alpha=0.6, lw=0)
        axis.plot(time_s, values, lw=0.6)
        axis.set_ylabel(label, fontsize=8)
    axes[-1].set_xlabel("time since start (s)")
    axes[0].set_title("blue = concentric (SQUEEZE / UP), red = hold, yellow = eccentric (RELEASE / DOWN)")
    fig.tight_layout()
    out_path = folder / PLOT_CONFIG["file_name"]
    fig.savefig(out_path, dpi=PLOT_CONFIG["dpi"])
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# main(argv)
# What:  load -> check -> print -> save quality.json (and quality.png) -> exit code.
# Time:  O(N + R^2 + A^2), as check_quality (N samples, DFTs on the rest and each rep).
# Space: O(N).
# Data structures: a Recording and a QualityReport (see quality.py).
# Files used:
#   src/myomotion/semg/quality.py   imported: load_recording, check_quality,
#                                   format_report, save_report, measured_rate_hz, CONFIG.
#   src/myomotion/semg/dsp.py       imported: EmgFilter, emg_envelope (for the plot).
#   <folder>/emg.csv, events.csv, metadata.json   read through load_recording.
#   <folder>/quality.json, quality.png            written.
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    """Check one recording, print the result, save quality.json and quality.png; 0 if TC01 passed."""
    args = parse_args(argv)
    folder = args.folder.expanduser()                    # > turns "~" into /home/<you>
    recording = load_recording(folder, args.channel)
    # > the biased /5 circuit (rebuild of 20 Nov) clips at the module's limit, ~2710 mV, not the ADC's 3100 mV
    values_note = json.loads((folder / "metadata.json").read_text()).get("values", "")
    cfg = replace(CONFIG, ceiling_mv=ceiling_for(values_note))
    report = check_quality(recording, cfg)
    print(format_report(report, cfg))
    print(f"\nSaved {save_report(report, folder)}")
    if not args.no_plot:
        plot_path = save_plot(recording, folder)
        if plot_path is not None:
            print(f"Saved {plot_path}")
    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
