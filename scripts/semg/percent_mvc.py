"""Convert an exercise set to %MVC and summarise it per rep phase.

Usage, from the repo root:
    python scripts/semg/percent_mvc.py ~/Downloads/data/TEST/S15_forearm_fist
    python scripts/semg/percent_mvc.py <set folder> --mvc ~/Downloads/data/TEST/S14_forearm_mvc

Without --mvc it uses the newest mvc.json in the same participant folder (MVC is redone every
session, so the newest one belongs to the current session).
Thin command-line wrapper: the logic lives in src/myomotion/semg/mvc.py.
Writes into the set folder:
    percent_mvc.csv   t_us, envelope_mv, percent_mvc   (one row per sample: the measured Y(t))
    percent_mvc.json  the MVC used and the per-phase summary
    percent_mvc.png   %MVC over time with the phases shaded

Data structures (this file): a Recording, a NormalisedSet (NumPy arrays and a list of PhaseStat), a matplotlib
    figure.
Time complexity (this file): O(N * S) for the envelope (N samples, S = 6 filter sections) plus O(N * P) for the per-
    phase summary of P phase windows.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path

# > This file is scripts/semg/percent_mvc.py, so parents[2] is the repo root; adding <repo>/src to
# > sys.path lets `import myomotion` work without `pip install -e .` (as in Ahmad's scripts).
_SRC = str(Path(__file__).resolve().parents[2] / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from myomotion.semg.mvc import MVC_CONFIG, NormalisedSet, find_latest_mvc, load_mvc, normalise_set  # noqa: E402
from myomotion.semg.quality import Recording, load_recording  # noqa: E402

PLOT_CONFIG = {
    "phase_colours": {"ECCENTRIC": "#fde68a", "ISOMETRIC_HOLD": "#fecaca", "CONCENTRIC": "#bfdbfe"},
    "figure_size": (12, 4.5),
    "dpi": 150,
}
PHASE_WORDS = {"CONCENTRIC": "squeeze/up", "ISOMETRIC_HOLD": "hold", "ECCENTRIC": "release/down"}


# ---------------------------------------------------------------------------
# format_phases(normalised, mvc)
# What:  a small table: one line per rep phase with its length, mean and peak %MVC.
# Time:  O(P) for P phases.  Space: O(P).
# Data structures: list of strings joined with newlines.
# Files used: none.
# ---------------------------------------------------------------------------
def format_phases(normalised: NormalisedSet, mvc: dict) -> str:
    """Per-phase %MVC table."""
    lines = [
        f"MVC used: {mvc['folder']} ({mvc['muscle']}), 100% = {mvc['mvc_above_rest_mv']:.1f} mV above rest; "
        f"TC02 {'PASS' if mvc['passed'] else 'FAIL'}",
        f"This set's rest envelope: {normalised.rest_envelope_mv:.1f} mV (= 0%)",
        "",
        "rep  phase           seconds   mean %MVC   peak %MVC",
    ]
    for stat in normalised.phases:
        lines.append(f"{stat.rep:3d}  {PHASE_WORDS.get(stat.phase, stat.phase):14s}  {stat.seconds:6.1f}   "
                     f"{stat.mean_percent:9.1f}   {stat.peak_percent:9.1f}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# save_outputs(normalised, mvc, folder)
# What:  writes percent_mvc.csv (the time series) and percent_mvc.json (summary).
# Time:  O(N) for N samples.  Space: O(P) beyond the arrays already in memory.
# Data structures: csv.writer streams one row per sample; the summary is a dict.
# Files used: writes <folder>/percent_mvc.csv and <folder>/percent_mvc.json.
# ---------------------------------------------------------------------------
def save_outputs(normalised: NormalisedSet, mvc: dict, folder: Path) -> list[Path]:
    """Save the %MVC time series and the per-phase summary."""
    csv_path = folder / "percent_mvc.csv"
    with csv_path.open("w", newline="") as out_file:
        writer = csv.writer(out_file)
        writer.writerow(["t_us", "envelope_mv", "percent_mvc"])
        for t_us, envelope, percent in zip(normalised.t_us, normalised.envelope_mv, normalised.percent):
            writer.writerow([int(t_us), f"{envelope:.3f}", f"{percent:.2f}"])
    json_path = folder / "percent_mvc.json"
    summary = {"mvc_folder": mvc["folder"], "muscle": mvc["muscle"], "mvc_above_rest_mv": mvc["mvc_above_rest_mv"],
               "mvc_passed": mvc["passed"], "rest_envelope_mv": normalised.rest_envelope_mv,
               "phases": [asdict(stat) for stat in normalised.phases]}
    json_path.write_text(json.dumps(summary, indent=2))
    return [csv_path, json_path]


# ---------------------------------------------------------------------------
# save_plot(recording, normalised, folder)
# What:  %MVC over time with the phases shaded and dashed lines at 0% and 100%.
# Time:  O(N).  Space: O(N).
# Data structures: NumPy arrays; the cue list for shading.
# Files used: writes <folder>/percent_mvc.png.
# ---------------------------------------------------------------------------
def save_plot(recording: Recording, normalised: NormalisedSet, folder: Path) -> Path | None:
    """Save percent_mvc.png; return its path, or None without matplotlib."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib not installed: skipped percent_mvc.png)")
        return None
    start_us = int(recording.t_us[0])
    time_s = (recording.t_us - start_us) / 1_000_000
    cue_times_s = [(cue.t_us - start_us) / 1e6 for cue in recording.cues] + [float(time_s[-1])]
    fig, axis = plt.subplots(figsize=PLOT_CONFIG["figure_size"])
    for cue, begin_s, end_s in zip(recording.cues, cue_times_s, cue_times_s[1:]):
        colour = PLOT_CONFIG["phase_colours"].get(cue.event)
        if colour:
            axis.axvspan(begin_s, end_s, color=colour, alpha=0.6, lw=0)
    axis.plot(time_s, normalised.percent, lw=0.8)
    axis.axhline(0, color="grey", ls="--", lw=1)
    axis.axhline(100, color="black", ls="--", lw=1)
    axis.set_xlabel("time since start (s)")
    axis.set_ylabel("%MVC")
    axis.set_title("blue = concentric (SQUEEZE / UP), red = hold, yellow = eccentric (RELEASE / DOWN)")
    fig.tight_layout()
    out_path = folder / "percent_mvc.png"
    fig.savefig(out_path, dpi=PLOT_CONFIG["dpi"])
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# main(argv)
# What:  find the MVC -> load the set -> %MVC -> print the table -> save files.
# Time:  O(N * S) for the envelope (N samples, S filter sections).  Space: O(N).
# Data structures: Recording, NormalisedSet, the mvc.json dict.
# Files used:
#   src/myomotion/semg/quality.py   load_recording (emg.csv, events.csv, metadata.json of the set).
#   src/myomotion/semg/mvc.py       find_latest_mvc, load_mvc, normalise_set.
#   <MVC folder>/mvc.json           read (written by scripts/semg/compute_mvc.py).
#   <set folder>/percent_mvc.*      written.
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    """Convert one set to %MVC; return 0 on success."""
    parser = argparse.ArgumentParser(description="Convert an exercise set to %MVC, per rep phase")
    parser.add_argument("folder", type=Path, help="exercise set folder (S<nn>_<exercise>_<posture>)")
    parser.add_argument("--mvc", type=Path, default=None, help="mvc.json or its folder (default: newest in the "
                                                               "same participant folder)")
    parser.add_argument("--channel", default=MVC_CONFIG.channel, help="emg.csv column (default: ch1)")
    parser.add_argument("--no-plot", action="store_true", help="skip percent_mvc.png")
    args = parser.parse_args(argv)
    folder = args.folder.expanduser()
    mvc_path = args.mvc.expanduser() if args.mvc else find_latest_mvc(folder.parent)
    if mvc_path is None:
        sys.exit(f"No MVC that passed TC02 in {folder.parent}: record one (record_session.py --mvc) and run "
                 "scripts/semg/compute_mvc.py on it first.")
    mvc = load_mvc(mvc_path)
    if not mvc.get("passed"):
        print(f"Warning: {mvc.get('folder')} FAILED TC02; its %MVC values are not trustworthy.")
    if not mvc.get("mvc_above_rest_mv", 0) > 0:
        sys.exit(f"{mvc.get('folder')}: the MVC is not above the rest level, so %MVC cannot be computed.")
    if mvc.get("channel") != args.channel:
        print(f"Warning: the MVC was measured on {mvc.get('channel')}, this set uses {args.channel}.")
    recording = load_recording(folder, args.channel)
    normalised = normalise_set(recording, mvc["mvc_above_rest_mv"])
    print(format_phases(normalised, mvc))
    for path in save_outputs(normalised, mvc, folder):
        print(f"Saved {path}")
    if not args.no_plot:
        plot_path = save_plot(recording, normalised, folder)
        if plot_path is not None:
            print(f"Saved {plot_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
