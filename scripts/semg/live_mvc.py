"""Live %MVC on screen while you move (Step 5).

Usage, from the repo root (laptop on the MyoMotion-EMG Wi-Fi; record_session.py must NOT be
running, the ESP32 serves one connection at a time):
    python scripts/semg/live_mvc.py --mvc ~/Downloads/data/TEST/S34_forearm_mvc
    python scripts/semg/live_mvc.py --mvc <folder or mvc.json> --text          # terminal only
    python scripts/semg/live_mvc.py --mvc <...> --seconds 60                    # stop after 60 s

What you see: 1 s "settling", 3 s "RELAX" (measures this session's rest level = 0%), then live
%MVC. The window shows the last 10 s as a graph plus a bar with the current value; without a
window toolkit it falls back to a text bar in the terminal (for the window on the repo's Python:
`sudo apt install python3-tk` once). Close the window or press Ctrl+C to stop.

Thin command-line wrapper: the logic lives in src/myomotion/semg/acquisition.py (reading the
ESP32) and src/myomotion/semg/live.py (filter, envelope, %MVC).

Data structures (this file): a LivePercentMvc (filter chain and deque ring buffer); two deques as ring buffers for
    the plot; LiveReading dataclasses.
Time complexity (this file): O(S) per sample (S = 6 filter sections, O(1) RMS); O(n) per redraw for the n plotted
    points.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from pathlib import Path

# > This file is scripts/semg/live_mvc.py, so parents[2] is the repo root; adding <repo>/src to
# > sys.path lets `import myomotion` work without `pip install -e .` (as in Ahmad's scripts).
_SRC = str(Path(__file__).resolve().parents[2] / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from myomotion.semg.acquisition import AcquisitionConfig, EspStream  # noqa: E402
from myomotion.semg.live import REST, LiveConfig, LivePercentMvc, LiveReading  # noqa: E402
from myomotion.semg.mvc import load_mvc  # noqa: E402

DISPLAY_CONFIG = {
    "window_s": 10.0,                 # seconds of history in the graph
    "fps": 20,                        # screen refreshes per second
    "bar_width": 40,                  # characters in the terminal bar (= 100% MVC)
    "y_max_percent": 120.0,           # top of the graph and the bar
    "figure_size": (12, 5),
}


# ---------------------------------------------------------------------------
# parse_args(argv)
# What:  reads the command line.
# Time:  O(a) for a arguments.  Space: O(a).
# Data structures: argparse.Namespace (one attribute per option).
# Files used: none.
# ---------------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Read the MVC path, channel, connection and display options."""
    parser = argparse.ArgumentParser(description="Live %MVC from the ESP32 sEMG stream")
    parser.add_argument("--mvc", type=Path, required=True,
                        help="mvc.json or the MVC recording folder (from scripts/semg/compute_mvc.py)")
    parser.add_argument("--channel", default="ch1", help="channel to show: ch1, ch2 or ch3 (default ch1)")
    parser.add_argument("--host", default=AcquisitionConfig.host, help="ESP32 address (default 192.168.4.1)")
    parser.add_argument("--port", type=int, default=AcquisitionConfig.port, help="ESP32 TCP port (default 3333)")
    parser.add_argument("--text", action="store_true", help="terminal display only, no window")
    parser.add_argument("--seconds", type=float, default=0.0, help="stop after this many seconds (0 = until Ctrl+C)")
    parser.add_argument("--secure", action="store_true",
                        help="the ESP32 runs the encrypted firmware (hardware/myomotion_emg_secure)")
    parser.add_argument("--keys", type=Path, default=Path("~/.config/myomotion/keys.json"),
                        help="key file from scripts/semg/make_link_keys.py (default ~/.config/myomotion/keys.json)")
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# channel_index(name)
# What:  "ch1" -> 0, "ch2" -> 1, "ch3" -> 2 (position in each sample's values).
# Time:  O(1).  Space: O(1).
# Data structures: str slicing.  Files used: none.
# ---------------------------------------------------------------------------
def channel_index(name: str) -> int:
    """Index of a channel name in the sample's values."""
    if not (name.startswith("ch") and name[2:].isdigit() and int(name[2:]) >= 1):
        raise ValueError(f"channel must look like ch1, ch2, ch3; got {name!r}")
    return int(name[2:]) - 1


# ---------------------------------------------------------------------------
# gui_available()
# What:  True if matplotlib can open a window here (it needs the tkinter toolkit).
# Time:  O(1).  Space: O(1).
# Data structures: none.  Files used: none.
# ---------------------------------------------------------------------------
def gui_available() -> bool:
    """Can a matplotlib window be opened with this Python?"""
    try:
        import tkinter  # noqa: F401  # > only checking that it exists
    except ImportError:
        return False
    return True


# ---------------------------------------------------------------------------
# TextDisplay
# What:  one terminal line, redrawn in place: the stage while starting up, then a
#        bar of '#' (bar_width characters = 100% MVC) and the number.
# Data structures: none beyond the settings dict.
# Time:  O(1) per update (one printed line).  Space: O(1).
# ---------------------------------------------------------------------------
class TextDisplay:
    """Live %MVC as a text bar in the terminal."""

    # Time:  O(1) (nothing is stored).  Space: O(1).
    # Data structures: none.
    def add(self, t_us: int, reading: LiveReading) -> None:
        """Nothing to store for the text display."""

    # -----------------------------------------------------------------------
    # update(reading, seconds_left)
    # What:  rewrites the line ('\r' returns to the start of the same line).
    # Time:  O(w) for the bar width w.  Space: O(w).
    # Data structures: strings.  Files used: none.
    # -----------------------------------------------------------------------
    def update(self, reading: LiveReading, seconds_left: float) -> None:
        """Redraw the terminal line."""
        if reading.percent is None:
            word = "RELAX: measuring your rest level" if reading.stage == REST else "starting up (filters settling)"
            line = f"{word} ... {seconds_left:3.1f} s"
        else:
            width = DISPLAY_CONFIG["bar_width"]
            filled = int(round(min(max(reading.percent, 0.0), DISPLAY_CONFIG["y_max_percent"]) / 100 * width))
            line = f"%MVC {reading.percent:6.1f}  |{'#' * filled}{'.' * max(0, width - filled)}|"
        sys.stdout.write("\r" + line.ljust(DISPLAY_CONFIG["bar_width"] + 30))
        sys.stdout.flush()

    # Time:  O(1).
    # Data structures: a bool.
    def closed(self) -> bool:
        """The terminal display is only closed by Ctrl+C."""
        return False

    # Time:  O(1).
    # Data structures: none.
    def close(self) -> None:
        """End the line so the shell prompt starts on a new one."""
        print()


# ---------------------------------------------------------------------------
# PlotDisplay
# What:  a matplotlib window: left, %MVC over the last window_s seconds; right, a
#        bar with the current value and the number above it.
# Data structures: two deques with maxlen = window_s * rate as ring buffers (time
#        and %MVC): appending past the limit drops the oldest point in O(1), so the
#        graph always shows the latest window_s seconds.
# Files used: none (matplotlib draws on screen).
# Time:  O(1) per add; O(n) per redraw for the n points in the window.  Space: O(n).
# ---------------------------------------------------------------------------
class PlotDisplay:
    """Live %MVC graph and bar in a window."""

    # -----------------------------------------------------------------------
    # __init__(rate_hz, title)
    # What:  opens the window and creates the empty graph and bar.
    # Time:  O(1).  Space: O(n) for the n = window_s * rate stored points.
    # Data structures: two deques (ring buffers).  Files used: none.
    # -----------------------------------------------------------------------
    def __init__(self, rate_hz: float, title: str) -> None:
        """Open the live window."""
        import matplotlib
        matplotlib.use("TkAgg")
        import matplotlib.pyplot as plt
        self.plt = plt
        points = int(DISPLAY_CONFIG["window_s"] * rate_hz)
        self.times_s: deque = deque(maxlen=points)
        self.percents: deque = deque(maxlen=points)
        self.start_us: int | None = None
        plt.ion()                                        # > interactive mode: drawing does not block
        self.fig, (self.trace_axis, self.bar_axis) = plt.subplots(
            1, 2, figsize=DISPLAY_CONFIG["figure_size"], gridspec_kw={"width_ratios": [5, 1]})
        self.fig.suptitle(title)
        self.line, = self.trace_axis.plot([], [], lw=1.2)
        for level in (0.0, 100.0):
            self.trace_axis.axhline(level, color="grey", ls="--", lw=0.8)
        self.trace_axis.set_ylim(-10, DISPLAY_CONFIG["y_max_percent"])
        self.trace_axis.set_xlabel("time (s)")
        self.trace_axis.set_ylabel("%MVC")
        self.bar = self.bar_axis.bar([0], [0.0], width=0.6)[0]
        self.bar_axis.set_ylim(0, DISPLAY_CONFIG["y_max_percent"])
        self.bar_axis.set_xticks([])
        self.value_text = self.bar_axis.text(0, DISPLAY_CONFIG["y_max_percent"] * 0.92, "", ha="center",
                                             fontsize=22, fontweight="bold")
        self.fig.tight_layout()

    # -----------------------------------------------------------------------
    # add(t_us, reading)
    # What:  stores one point (only once %MVC exists).
    # Time:  O(1).  Space: O(1) amortised (the deques are already sized).
    # Data structures: the two deques.  Files used: none.
    # -----------------------------------------------------------------------
    def add(self, t_us: int, reading: LiveReading) -> None:
        """Store one %MVC point."""
        if self.start_us is None:
            self.start_us = t_us
        if reading.percent is not None:
            self.times_s.append((t_us - self.start_us) / 1e6)
            self.percents.append(reading.percent)

    # -----------------------------------------------------------------------
    # update(reading, seconds_left)
    # What:  moves the graph to the newest window_s seconds, sets the bar and the
    #        number, and lets the window redraw.
    # Time:  O(n) for the n points handed to the line.  Space: O(n).
    # Data structures: lists made from the deques.  Files used: none.
    # -----------------------------------------------------------------------
    def update(self, reading: LiveReading, seconds_left: float) -> None:
        """Redraw the graph, the bar and the number."""
        if reading.percent is None:
            word = "RELAX" if reading.stage == REST else "wait"
            self.value_text.set_text(f"{word}\n{seconds_left:.1f} s")
        else:
            self.value_text.set_text(f"{reading.percent:.0f}%")
            self.bar.set_height(max(0.0, reading.percent))
        if self.times_s:
            self.line.set_data(list(self.times_s), list(self.percents))
            newest = self.times_s[-1]
            self.trace_axis.set_xlim(max(0.0, newest - DISPLAY_CONFIG["window_s"]),
                                     max(DISPLAY_CONFIG["window_s"], newest))
        self.fig.canvas.draw_idle()
        self.plt.pause(0.001)                            # > lets the window process events and repaint

    # Time:  O(1).
    # Data structures: a lookup in matplotlib's list of open figures.
    def closed(self) -> bool:
        """True once the user has closed the window."""
        return not self.plt.fignum_exists(self.fig.number)

    # Time:  O(1).
    # Data structures: the matplotlib figure.
    def close(self) -> None:
        """Close the window."""
        self.plt.close(self.fig)


# ---------------------------------------------------------------------------
# run(stream, meter, display, index, seconds)
# What:  the live loop: read new samples -> meter -> display. The screen is redrawn
#        at most fps times a second, however many samples arrive (500 per second),
#        so drawing never falls behind the data.
# Time:  O(1) per sample (meter) plus O(n) per redraw.  Space: O(n) in the display.
# Data structures: the latest LiveReading; time stamps for the redraw rate.
# Files used: none.
# ---------------------------------------------------------------------------
def run(stream: EspStream, meter: LivePercentMvc, display: TextDisplay | PlotDisplay, index: int,
        seconds: float) -> None:
    """Read, process and show until stopped."""
    started = time.perf_counter()
    last_draw = 0.0
    latest: LiveReading | None = None
    while True:
        for sample in stream.read():
            latest = meter.step(sample.values_mv[index])
            display.add(sample.t_us, latest)
        now = time.perf_counter()
        if latest is not None and now - last_draw >= 1 / DISPLAY_CONFIG["fps"]:
            display.update(latest, meter.seconds_left())
            last_draw = now
        if display.closed() or (seconds > 0 and now - started >= seconds):
            return


# ---------------------------------------------------------------------------
# main(argv)
# What:  load the MVC -> connect -> choose the display -> run until stopped.
# Time:  as run().  Space: O(n).
# Data structures: the mvc.json dict, an EspStream, a LivePercentMvc, a display.
# Files used:
#   <MVC folder>/mvc.json                   read (written by scripts/semg/compute_mvc.py).
#   src/myomotion/semg/acquisition.py       EspStream (TCP to the ESP32).
#   src/myomotion/semg/live.py              LivePercentMvc.
#   src/myomotion/semg/mvc.py               load_mvc.
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    """Show live %MVC; return 0 when stopped normally."""
    args = parse_args(argv)
    mvc = load_mvc(args.mvc.expanduser())
    if not mvc.get("passed"):
        print(f"Note: {mvc.get('folder')} did not pass TC02; using it anyway (provisional MVC).")
    if mvc.get("channel") != args.channel:
        print(f"Warning: the MVC was measured on {mvc.get('channel')}, showing {args.channel}.")
    meter = LivePercentMvc(mvc_above_rest_mv=float(mvc["mvc_above_rest_mv"]), cfg=LiveConfig())
    title = f"Live %MVC: {mvc.get('muscle', args.channel)} (MVC from {mvc.get('folder')})"
    use_window = not args.text and gui_available()
    if not args.text and not use_window:
        print("(No window toolkit for this Python: using the terminal. For the window: sudo apt install python3-tk)")
    cfg = AcquisitionConfig(host=args.host, port=args.port)
    try:
        keys = None
        if args.secure:                                  # > encrypted firmware: decrypt with the laptop's keys
            from myomotion.semg.secure_link import load_keys
            keys = load_keys(args.keys)
        stream = EspStream(cfg, keys=keys)
    except (ConnectionError, FileNotFoundError) as error:   # > a clear message instead of a traceback
        print(error, file=sys.stderr)
        return 1
    with stream:
        display = PlotDisplay(meter.cfg.rate_hz, title) if use_window else TextDisplay()
        print("Connected. Relax your arm for the first 4 seconds. Ctrl+C (or close the window) to stop.")
        try:
            run(stream, meter, display, channel_index(args.channel), args.seconds)
        except KeyboardInterrupt:
            pass
        except ValueError as error:                      # > KeyMismatch: board and laptop keys differ
            print(f"\n{error}", file=sys.stderr)
            return 1
        finally:
            display.close()
    print(f"Stopped. Rest level this session: {meter.rest_mv or 0:.1f} mV; "
          f"damaged lines skipped: {stream.parser.bad_lines}")
    if stream.decoder is not None:
        print(f"Encrypted link: {stream.decoder.frames_ok} batches verified, "
              f"{stream.decoder.frames_rejected} rejected (changed, replayed or wrong key)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
