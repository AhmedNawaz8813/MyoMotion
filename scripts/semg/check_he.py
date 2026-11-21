"""Check the homomorphic envelope of an encrypted recording against the plain one.

Usage, from the repo root:
    python scripts/semg/check_he.py ~/Downloads/data/TEST/S40_forearm_fist
    python scripts/semg/check_he.py <folder> --channel ch1 --keys ~/.config/myomotion/keys.json

The folder comes from  record_session.py --board esp32 --secure  (encrypted firmware
hardware/myomotion_emg_secure). It holds emg.csv (decrypted raw millivolts), he.csv (homomorphic
ciphertexts, still encrypted) and link.json (session and key id). Two roles, kept apart:
  1. Aggregator, NO key: adds the ciphertexts of every 100 ms window (src/myomotion/semg/secure_link.py
     aggregate_windows). It only ever sees ciphertexts.
  2. Key holder: removes the keys from each window sum and takes the square root -> RMS envelope
     (decrypt_windows).
The result must equal the envelope computed in the clear from emg.csv with the same filter
(plain_envelope). Writes he.json and he.png into the folder. Exit code 0 = match, 1 = mismatch.

Data structures (this file): NumPy uint64 arrays for the ciphertexts and the encrypted window sums; NumPy float64
    arrays for the envelopes; a dict for he.json; a matplotlib figure.
Time complexity (this file): O(N * C) for N samples and C channels (sliding sums, one keystream block per sample and
    channel), plus O(N * S) for the plain envelope it is compared with (N samples, S = 6 filter sections).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

# > This file is scripts/semg/check_he.py, so parents[2] is the repo root; adding <repo>/src to
# > sys.path lets `import myomotion` work without `pip install -e .` (as in Ahmad's scripts).
_SRC = str(Path(__file__).resolve().parents[2] / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from myomotion.semg.secure_link import (SECURE_CONFIG, aggregate_windows, decrypt_windows,  # noqa: E402
                                        load_keys, plain_envelope)

CHECK_CONFIG = {"max_difference_mv": 0.01,     # rounding to thousandths of mV^2 gives ~1e-5 mV; 0.01 is generous
                "figure_size": (12, 5), "dpi": 150, "plain_colour": "#1F4FE0", "he_colour": "#C92A6E"}


# ---------------------------------------------------------------------------
# load_he(folder)
# What:  he.csv -> sample numbers, ESP32 times and the ciphertext matrix.
# Time:  O(N).  Space: O(N * channels).
# Data structures: lists while reading, then NumPy arrays (uint64 for the ciphertexts).
# Files used: reads <folder>/he.csv.
# ---------------------------------------------------------------------------
def load_he(folder: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(n, t_us, ciphertexts N x 3)."""
    numbers, times, ciphertexts = [], [], []
    with (folder / "he.csv").open(newline="") as he_file:
        for row in csv.DictReader(he_file):
            numbers.append(int(row["n"]))
            times.append(int(row["t_us"]))
            ciphertexts.append([int(row[f"c{c}"], 16) for c in (1, 2, 3)])
    return np.array(numbers), np.array(times, dtype=np.int64), np.array(ciphertexts, dtype=np.uint64)


# ---------------------------------------------------------------------------
# load_emg(folder, channel)
# What:  emg.csv -> ESP32 times and one channel's raw millivolts (the decrypted D lines).
# Time:  O(N).  Space: O(N).
# Data structures: NumPy arrays.  Files used: reads <folder>/emg.csv.
# ---------------------------------------------------------------------------
def load_emg(folder: Path, channel: str) -> tuple[np.ndarray, np.ndarray]:
    """(t_us, millivolts)."""
    times, values = [], []
    with (folder / "emg.csv").open(newline="") as emg_file:
        for row in csv.DictReader(emg_file):
            times.append(int(row["t_us"]))
            values.append(float(row[channel]))
    return np.array(times, dtype=np.int64), np.array(values)


# ---------------------------------------------------------------------------
# first_gap(numbers)
# What:  index of the first missing sample number (a rejected batch), or len(numbers).
#        After a gap the ESP32's filter has seen samples the laptop never got, so the
#        comparison stops there.
# Time:  O(N).  Space: O(N).
# Data structures: NumPy diff.  Files used: none.
# ---------------------------------------------------------------------------
def first_gap(numbers: np.ndarray) -> int:
    """Length of the gap-free start of the sample numbers."""
    jumps = np.nonzero(np.diff(numbers) != 1)[0]
    return int(jumps[0]) + 1 if len(jumps) else len(numbers)


# ---------------------------------------------------------------------------
# plot_check(times_s, plain, homomorphic, out_path)
# What:  he.png: both envelopes on top of each other, and their difference.
# Time:  O(N).  Space: O(N).
# Data structures: matplotlib figure (Agg backend).  Files used: writes he.png.
# ---------------------------------------------------------------------------
def plot_check(times_s: np.ndarray, plain: np.ndarray, homomorphic: np.ndarray, out_path: Path) -> None:
    """Draw he.png."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, (top, bottom) = plt.subplots(2, 1, figsize=CHECK_CONFIG["figure_size"], sharex=True,
                                         gridspec_kw={"height_ratios": [3, 1]})
    top.plot(times_s, plain, color=CHECK_CONFIG["plain_colour"], linewidth=2.5, label="plain (from decrypted emg.csv)")
    top.plot(times_s, homomorphic, color=CHECK_CONFIG["he_colour"], linewidth=0.9, linestyle="--",
             label="homomorphic (summed encrypted, then decrypted)")
    top.set_ylabel("RMS envelope (mV)")
    top.legend(loc="upper right")
    top.set_title("Homomorphic envelope vs plain envelope (100 ms RMS)")
    bottom.plot(times_s, homomorphic - plain, color="#333333", linewidth=0.8)
    bottom.set_ylabel("difference (mV)")
    bottom.set_xlabel("time (s)")
    figure.tight_layout()
    figure.savefig(out_path, dpi=CHECK_CONFIG["dpi"])
    plt.close(figure)


# ---------------------------------------------------------------------------
# main()
# What:  load -> aggregate without the key -> decrypt with the key -> compare with the
#        plain envelope -> print, he.json, he.png.
# Time:  O(N * S) for the plain filter, O(N * channels) for the homomorphic part.
# Space: O(N * channels).
# Data structures: NumPy arrays.
# Files used: <folder>/emg.csv, he.csv, link.json; the key file; writes he.json, he.png.
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Check the homomorphic envelope of an encrypted recording")
    parser.add_argument("folder", type=Path, help="recording folder from record_session.py --secure")
    parser.add_argument("--channel", default="ch1", choices=["ch1", "ch2", "ch3"])
    parser.add_argument("--keys", type=Path, default=SECURE_CONFIG.key_file, help="laptop key file")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args(argv)

    folder = args.folder.expanduser()
    if not (folder / "he.csv").exists():
        print(f"{folder} has no he.csv: record with  record_session.py --board esp32 --secure")
        return 1
    link = json.loads((folder / "link.json").read_text())
    keys = load_keys(args.keys)
    if link["key_id"] != keys.key_id:
        print(f"The recording used keys {link['key_id']}, this key file is {keys.key_id}.")
        return 1
    channel_index = int(args.channel[2]) - 1
    numbers, he_times, ciphertexts = load_he(folder)
    emg_times, values = load_emg(folder, args.channel)
    usable = min(first_gap(numbers), len(values))
    if usable < SECURE_CONFIG.window or not np.array_equal(he_times[:usable], emg_times[:usable]):
        print("he.csv and emg.csv do not line up (or are too short): cannot compare.")
        return 1

    encrypted_sums = aggregate_windows(ciphertexts[:usable])                       # > role 1: no key used
    homomorphic = decrypt_windows(encrypted_sums, keys.he_key, bytes.fromhex(link["session"]),
                                  int(numbers[0]), usable)[:, channel_index]   # > role 2: key holder
    plain = plain_envelope(values[:usable])
    difference = np.abs(homomorphic - plain)
    passed = bool(difference.max() <= CHECK_CONFIG["max_difference_mv"])
    times_s = (emg_times[SECURE_CONFIG.window - 1:usable] - emg_times[0]) / 1e6

    summary = {
        "test": "homomorphic envelope = plain envelope",
        "verdict": "PASS" if passed else "FAIL",
        "channel": args.channel,
        "samples": usable,
        "windows": len(plain),
        "window_samples": SECURE_CONFIG.window,
        "max_difference_mv": float(difference.max()),
        "mean_difference_mv": float(difference.mean()),
        "limit_mv": CHECK_CONFIG["max_difference_mv"],
        "envelope_range_mv": [float(plain.min()), float(plain.max())],
        "link": link,
    }
    (folder / "he.json").write_text(json.dumps(summary, indent=2))
    print(f"Homomorphic envelope check: {summary['verdict']}")
    print(f"  link        {link['batches_verified']} batches verified, {link['batches_rejected']} rejected "
          f"(key id {link['key_id']})")
    print(f"  aggregator  added {len(plain)} windows of {SECURE_CONFIG.window} ciphertexts WITHOUT the key")
    print(f"  key holder  envelope {plain.min():.1f} to {plain.max():.1f} mV; largest difference from the plain "
          f"envelope {difference.max():.6f} mV (limit {CHECK_CONFIG['max_difference_mv']} mV)")
    if usable < len(numbers):
        print(f"  note        a batch was missing after sample {usable}: compared up to there")
    if not args.no_plot:
        plot_check(times_s, plain, homomorphic, folder / "he.png")
        print(f"Saved {folder / 'he.json'} and {folder / 'he.png'}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
