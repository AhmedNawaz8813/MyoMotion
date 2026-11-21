"""Measure the MVC from an MVC session and run test TC02.

Usage, from the repo root:
    python scripts/semg/compute_mvc.py ~/Downloads/data/TEST/S14_forearm_mvc

The folder comes from:  python record_session.py --board esp32 --participant TEST --exercise forearm --mvc
                                                 --channels forearm
Thin command-line wrapper: the logic lives in src/myomotion/semg/mvc.py.
Writes mvc.json (the numbers) and mvc.png (envelope, pushes, best 1-second windows) into the folder.
Exit code 0 = TC02 passed, 1 = failed.

Data structures (this file): a Recording, an MvcResult dataclass, a dict for mvc.json, a matplotlib figure.
Time complexity (this file): O(N * S) for the envelope (N samples, S = 6 filter sections), O(N) for the push windows
    and the plot.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
import sys
from pathlib import Path

# > This file is scripts/semg/compute_mvc.py, so parents[2] is the repo root; adding <repo>/src to
# > sys.path lets `import myomotion` work without `pip install -e .` (as in Ahmad's scripts).
_SRC = str(Path(__file__).resolve().parents[2] / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from myomotion.semg.dsp import emg_envelope  # noqa: E402
from myomotion.semg.mvc import MVC_CONFIG, MVC_EVENT, MvcResult, cue_windows, measure_mvc, save_mvc  # noqa: E402
from myomotion.semg.mvc import MvcConfig  # noqa: E402
from myomotion.semg.quality import Recording, ceiling_for, load_recording  # noqa: E402

PLOT_CONFIG = {"figure_size": (12, 4.5), "dpi": 150, "push_colour": "#bbf7d0", "best_colour": "#16a34a"}


# ---------------------------------------------------------------------------
# format_result(result)
# What:  readable TC02 summary for the terminal.
# Time:  O(T) for T pushes.  Space: O(T).
# Data structures: list of strings joined with newlines.
# Files used: none.
# ---------------------------------------------------------------------------
def format_result(result: MvcResult, cfg: MvcConfig = MVC_CONFIG) -> str:
    """PASS/FAIL per TC02 check, with the numbers."""
    # Time:  O(1).  Space: O(1).
    # Data structures: a bool in, a str out.
    def verdict(ok: bool) -> str:
        return "PASS" if ok else "FAIL"

    peaks = ", ".join(f"{peak:.1f}" for peak in result.trial_peaks_mv)
    weakest = min(result.trial_peaks_mv) if result.trial_peaks_mv else 0.0
    # > a rest of exactly 0 mV (relaxed signal flat on the ADC floor, e.g. S35) would divide by zero
    strength = (f"{weakest / result.rest_envelope_mv:.1f}x rest" if result.rest_envelope_mv > 0
                else f"{weakest:.1f} mV above a rest of 0 mV (relaxed signal flat on the ADC floor)")
    lines = [
        f"Rest envelope    {result.rest_envelope_mv:.1f} mV (relaxed baseline)",
        f"Push peaks       {peaks} mV (highest 1-second mean of each push)",
        f"MVC              {result.mvc_envelope_mv:.1f} mV; above rest: {result.mvc_above_rest_mv:.1f} mV = 100% MVC",
        f"1. Pushes found  {verdict(result.trials_ok)}: {len(result.trial_peaks_mv)} of {cfg.expected_trials}",
        f"2. Strength      {verdict(result.strength_ok)}: weakest push "
        f"{strength} (need ≥ {cfg.min_peak_to_rest:.0f}x)",
        f"3. Consistency   {verdict(result.consistency_ok)}: best two differ by {result.best_two_gap:.0%} "
        f"(max {cfg.max_best_two_gap:.0%}; more means a push was not maximal: redo)",
        f"4. No clipping   {verdict(result.clipping_ok)}: {result.clipped_samples} samples near "
        f"{cfg.ceiling_mv:.0f} mV during the pushes (clipped = MVC too small: turn the gain down a little)",
        f"TC02             {verdict(result.passed)}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# save_plot(recording, result, folder)
# What:  the envelope over time with each push shaded green, each push's best
#        1-second window marked darker, and dashed lines at the rest level and the MVC.
# Time:  O(N * S) for the envelope.  Space: O(N).
# Data structures: NumPy arrays; cue windows from mvc.cue_windows.
# Files used: src/myomotion/semg/dsp.py (emg_envelope); writes <folder>/mvc.png.
# ---------------------------------------------------------------------------
def save_plot(recording: Recording, result: MvcResult, folder: Path) -> Path | None:
    """Save mvc.png; return its path, or None without matplotlib."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib not installed: skipped mvc.png)")
        return None
    start_us = int(recording.t_us[0])
    time_s = (recording.t_us - start_us) / 1_000_000
    envelope = emg_envelope(recording.signal_mv, result.rate_hz)
    fig, axis = plt.subplots(figsize=PLOT_CONFIG["figure_size"])
    for begin_us, end_us in cue_windows(recording.cues, MVC_EVENT, int(recording.t_us[-1])):
        axis.axvspan((begin_us - start_us) / 1e6, (end_us - start_us) / 1e6, color=PLOT_CONFIG["push_colour"], lw=0)
    for best_us in result.trial_peak_start_us:
        best_s = (best_us - start_us) / 1e6
        axis.axvspan(best_s, best_s + MVC_CONFIG.peak_window_s, color=PLOT_CONFIG["best_colour"], alpha=0.35, lw=0)
    axis.plot(time_s, envelope, lw=0.8)
    axis.axhline(result.rest_envelope_mv, color="grey", ls="--", lw=1, label="rest")
    axis.axhline(result.mvc_envelope_mv, color="black", ls="--", lw=1, label="MVC")
    axis.set_xlabel("time since start (s)")
    axis.set_ylabel("RMS envelope, 100 ms (mV)")
    axis.set_title("green = push, dark green = best 1 s of each push")
    axis.legend(loc="upper right")
    fig.tight_layout()
    out_path = folder / "mvc.png"
    fig.savefig(out_path, dpi=PLOT_CONFIG["dpi"])
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# main(argv)
# What:  load -> measure -> print -> save mvc.json and mvc.png -> exit code.
# Time:  O(N * S).  Space: O(N).
# Data structures: Recording, MvcResult, the metadata dict.
# Files used:
#   src/myomotion/semg/quality.py   load_recording (reads emg.csv, events.csv, metadata.json).
#   src/myomotion/semg/mvc.py       measure_mvc, save_mvc.
#   <folder>/metadata.json          read again for the participant and muscle name.
#   <folder>/mvc.json, mvc.png      written.
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    """Measure the MVC of one session; return 0 if TC02 passed."""
    parser = argparse.ArgumentParser(description="MVC from a record_session.py --mvc recording (TC02)")
    parser.add_argument("folder", type=Path, help="MVC recording folder (S<nn>_<exercise>_mvc)")
    parser.add_argument("--channel", default=MVC_CONFIG.channel, help="emg.csv column (default: ch1)")
    parser.add_argument("--no-plot", action="store_true", help="skip mvc.png")
    args = parser.parse_args(argv)
    folder = args.folder.expanduser()
    recording = load_recording(folder, args.channel)
    metadata = json.loads((folder / "metadata.json").read_text())
    # > the biased /5 circuit (rebuild of 20 Nov) clips at the module's limit, ~2710 mV, not the ADC's 3100 mV
    cfg = replace(MVC_CONFIG, ceiling_mv=ceiling_for(metadata.get("values", "")))
    result = measure_mvc(recording, cfg)
    details = {"participant": metadata.get("participant"), "folder": folder.name, "channel": args.channel,
               "muscle": metadata.get("channels", {}).get(args.channel, args.channel)}
    print(format_result(result, cfg))
    print(f"\nSaved {save_mvc(result, folder, details)}")
    if not args.no_plot:
        plot_path = save_plot(recording, result, folder)
        if plot_path is not None:
            print(f"Saved {plot_path}")
    return 0 if result.passed else 1


if __name__ == "__main__":
    sys.exit(main())
