"""Live hum meter: how much mains hum is on the sEMG right now, one line every 2 seconds.

For finding a hum problem on the hardware: keep the muscle relaxed and watch the number while you push the
electrode plug in, wiggle a wire, swap a pad, move a cable. The wire that changes the number is the culprit.

Usage, from the repo root (the repo's .venv), with nothing else connected to the ESP32:
    python scripts/semg/hum_meter.py --secure            # the ESP32 runs the encrypted firmware
    python scripts/semg/hum_meter.py                     # plain firmware
Ctrl+C stops.

Each line: the raw level at the pin (mV), the hum (RMS of the 50 Hz bands and their harmonics, mV), the
muscle-band noise left without the hum (mV), the highest sample (clipping check) and a verdict.
For reference (21 Nov): TESTCV/S03 (good) had hum 6.8 mV; TEST/S42 and P01/S02 (bad contact) about 160-280 mV.

Data structures (this file): a Python list collecting one 2-second window (1000 samples); NumPy arrays for the
    spectrum; a HumReading dataclass per line.
Time complexity (this file): O(W^2) per window for the direct DFT of W = 1000 samples (a few tens of ms), once every
    2 s; O(b) to read b bytes from the ESP32.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# > This file is scripts/semg/hum_meter.py, so parents[2] is the repo root; adding <repo>/src to
# > sys.path lets `import myomotion` work without `pip install -e .` (as in Ahmad's scripts).
_SRC = str(Path(__file__).resolve().parents[2] / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from myomotion.semg.acquisition import AcquisitionConfig, EspStream  # noqa: E402
from myomotion.semg.dsp import emg_band_rms_from_spectrum, mains_rms_from_spectrum, power_spectrum  # noqa: E402

# > the CONFIG of this script: window, verdict limits and the module's ceiling at the pin
CONFIG = {
    "window_s": 2.0,            # one line per 2 s of samples (1000 samples at 500 per second)
    "rate_hz": 500.0,
    "good_hum_mv": 25.0,        # below this: clean (S03 had 6.8 mV)
    "high_hum_mv": 100.0,       # at or above this: fix the hardware before recording
    "ceiling_mv": 2650.0,       # close to the module's saturation (~2710 mV at the pin with the bias circuit)
}


# ---------------------------------------------------------------------------
# HumReading
# What:  the numbers for one window of samples.
# Data structures: frozen dataclass.
# Time:  O(1) to build (five floats); verdict is O(1).
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class HumReading:
    """One window's hum numbers (mV)."""

    level_mv: float           # mean raw level at the pin
    hum_mv: float             # RMS of the mains bands (50, 100, 150, 200 Hz)
    hum_share: float          # hum power / all power in the window
    muscle_band_mv: float     # RMS of 20-250 Hz without the mains bands (what is left when relaxed)
    peak_mv: float            # highest sample (clipping check)

    # Time:  O(1).  Space: O(1).
    # Data structures: two float comparisons; a str out.
    @property
    def verdict(self) -> str:
        """GOOD / OK / HIGH from the hum alone."""
        if self.hum_mv < CONFIG["good_hum_mv"]:
            return "GOOD"
        return "OK" if self.hum_mv < CONFIG["high_hum_mv"] else "HIGH: fix before recording"


# ---------------------------------------------------------------------------
# measure_window(values_mv, rate_hz)
# What:  hum and noise of one window, with the same maths as TC01 (src/myomotion/semg/
#        quality.py): our own DFT (dsp.power_spectrum), then the power in the mains bands
#        and in the rest of the 20-250 Hz band.
# Time:  O(N^2) for the direct DFT of N samples (1000: a few tens of ms).  Space: O(N).
# Data structures: NumPy arrays.  Files used: src/myomotion/semg/dsp.py.
# ---------------------------------------------------------------------------
def measure_window(values_mv: np.ndarray, rate_hz: float = CONFIG["rate_hz"]) -> HumReading:
    """Hum numbers of one window of raw samples."""
    values = np.asarray(values_mv, dtype=np.float64)
    freqs_hz, power = power_spectrum(values, rate_hz)
    hum_mv, hum_share = mains_rms_from_spectrum(freqs_hz, power, rate_hz)
    return HumReading(level_mv=float(np.mean(values)), hum_mv=hum_mv, hum_share=hum_share,
                      muscle_band_mv=emg_band_rms_from_spectrum(freqs_hz, power, rate_hz),
                      peak_mv=float(np.max(values)))


# Time:  O(1): one formatted line.  Space: O(1).
# Data structures: a HumReading dataclass in, a str out.
def format_reading(elapsed_s: float, reading: HumReading) -> str:
    """One printed line."""
    clip = "  NEAR CLIPPING" if reading.peak_mv >= CONFIG["ceiling_mv"] else ""
    return (f"{elapsed_s:6.1f} s   level {reading.level_mv:6.0f} mV   hum {reading.hum_mv:6.1f} mV "
            f"({reading.hum_share:4.0%})   noise {reading.muscle_band_mv:5.1f} mV   peak {reading.peak_mv:5.0f} mV   "
            f"{reading.verdict}{clip}")


# ---------------------------------------------------------------------------
# main()
# What:  connect to the ESP32, collect window_s of one channel, print its hum line,
#        repeat until Ctrl+C.
# Time:  O(N^2) per window (the DFT).  Space: O(N) for one window.
# Data structures: a Python list collecting one window of samples.
# Files used: reads the laptop's link keys with --secure (~/.config/myomotion/keys.json).
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Live mains-hum meter for the sEMG channel")
    parser.add_argument("--secure", action="store_true", help="the ESP32 runs the encrypted firmware")
    parser.add_argument("--keys", type=Path, default=Path("~/.config/myomotion/keys.json"))
    parser.add_argument("--channel", default="ch1", choices=["ch1", "ch2", "ch3"])
    parser.add_argument("--host", default=AcquisitionConfig.host)
    parser.add_argument("--port", type=int, default=AcquisitionConfig.port)
    args = parser.parse_args(argv)

    keys = None
    if args.secure:
        from myomotion.semg.secure_link import load_keys
        keys = load_keys(args.keys.expanduser())
    try:
        stream = EspStream(AcquisitionConfig(host=args.host, port=args.port), keys=keys)
    except (ConnectionError, OSError) as error:
        print(f"Could not connect to the ESP32: {error}\nIs the laptop on the MyoMotion-EMG Wi-Fi, and is no other "
              "program (record_session.py, live_mvc.py) connected?")
        return 1
    channel = int(args.channel[2]) - 1
    needed = int(CONFIG["window_s"] * CONFIG["rate_hz"])
    print(f"Hum meter on {args.channel}: keep the muscle RELAXED. GOOD < {CONFIG['good_hum_mv']:.0f} mV, "
          f"HIGH >= {CONFIG['high_hum_mv']:.0f} mV. Ctrl+C stops.")
    start = time.perf_counter()
    window: list[float] = []
    try:
        while True:
            for sample in stream.read():
                window.append(sample.values_mv[channel])
            if len(window) >= needed:
                print(format_reading(time.perf_counter() - start, measure_window(np.array(window[:needed]))))
                window = window[needed:]
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        stream.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
