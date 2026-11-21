"""Live %MVC, one sample at a time (Step 5).

Every sample from the ESP32 goes through the same chain as the offline analysis, but causally
(using only the past, so it can run while you move):

    raw mV -> EmgFilter.step (20 Hz high-pass + 50/100/150/200 Hz notches)
           -> RunningRms.step (100 ms window)              = envelope in mV
           -> %MVC = 100 * (envelope - rest) / (MVC - rest_mvc)

It runs in three stages:
  1. SETTLING  (1 s):  the filters start cold and ring briefly on the mains hum; ignored.
  2. REST      (3 s):  you relax; the mean envelope becomes this session's rest level (= 0%).
                       Measured live because the noise floor changes with every pad placement.
  3. LIVE:             %MVC for every sample.
The 100% step (MVC above rest) comes from mvc.json, measured by scripts/semg/compute_mvc.py in
the same session (same electrodes, same divider).

Data structures (this file): an EmgFilter (list of biquads with state) and a RunningRms (deque ring buffer);
    LiveReading dataclasses.
Time complexity (this file): O(S) per sample for the S = 6 filter sections; O(1) for the running RMS and the %MVC
    arithmetic.
"""

from __future__ import annotations   # > type hints are stored as text, not run: allows forward references

from dataclasses import dataclass, field

from myomotion.semg.dsp import DSP_CONFIG, DspConfig, EmgFilter, RunningRms

SETTLING = "SETTLING"
REST = "REST"
LIVE = "LIVE"
PERCENT = 100.0


# ---------------------------------------------------------------------------
# LiveConfig  (the CONFIG for this file)
# What:  timing of the three stages and the filter settings.
# Data structures: a frozen dataclass.
# Time:  O(1) to build (fixed fields).
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LiveConfig:
    """Settings for the live %MVC meter."""

    rate_hz: float = 500.0            # ESP32 sample rate (2 ms per sample)
    settle_s: float = 1.0             # ignore the filters' start-up ringing (~0.4 s on strong hum)
    rest_s: float = 3.0               # relaxed calibration that sets 0%
    dsp: DspConfig = DSP_CONFIG       # same filters and 100 ms envelope as the offline analysis


LIVE_CONFIG = LiveConfig()


# ---------------------------------------------------------------------------
# LiveReading
# What:  what the meter knows after one sample: its stage, the envelope, and the
#        %MVC (None until the rest level is known).
# Data structures: a frozen dataclass (one reading never changes).
# Time:  O(1) to build (fixed fields).
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LiveReading:
    """Meter output for one sample."""

    stage: str                        # SETTLING, REST or LIVE
    envelope_mv: float                # 100 ms RMS of the cleaned signal
    percent: float | None             # %MVC, or None before the rest level is known
    rest_mv: float | None             # this session's rest envelope, once measured


# ---------------------------------------------------------------------------
# LivePercentMvc
# What:  the live meter: feed raw millivolts one sample at a time, get LiveReadings.
# Data structures: an EmgFilter (list of biquads with state) and a RunningRms (deque
#        ring buffer), so each sample costs the same small amount of work no matter
#        how long the session runs; two counters and a running sum for the rest level.
# Time:  O(S) per sample for the S = 6 filter sections; O(1) for the RMS.  Space: O(window).
# ---------------------------------------------------------------------------
@dataclass
class LivePercentMvc:
    """Raw mV in, live %MVC out (causal: uses only past samples)."""

    mvc_above_rest_mv: float
    cfg: LiveConfig = LIVE_CONFIG
    samples_seen: int = field(default=0, init=False)
    rest_mv: float | None = field(default=None, init=False)
    _rest_sum: float = field(default=0.0, init=False, repr=False)
    _rest_count: int = field(default=0, init=False, repr=False)

    # -----------------------------------------------------------------------
    # __post_init__()
    # What:  checks the MVC and builds the filter and the running RMS.
    # Time:  O(w) for the RMS window w (50 samples).  Space: O(w).
    # Data structures: EmgFilter, RunningRms.  Files used: src/myomotion/semg/dsp.py.
    # -----------------------------------------------------------------------
    def __post_init__(self) -> None:
        """Validate the MVC and create the filter chain."""
        if not self.mvc_above_rest_mv > 0:
            raise ValueError("the MVC must be above the rest level")
        self.filter = EmgFilter(self.cfg.rate_hz, self.cfg.dsp)
        self.rms = RunningRms(window=max(1, round(self.cfg.dsp.envelope_window_s * self.cfg.rate_hz)))
        self.settle_samples = round(self.cfg.settle_s * self.cfg.rate_hz)
        self.rest_samples = round(self.cfg.rest_s * self.cfg.rate_hz)

    # -----------------------------------------------------------------------
    # step(value_mv)
    # What:  one raw sample -> filter -> RMS envelope -> stage logic -> LiveReading.
    #        The first sample primes the filter (as if the signal had always been at
    #        that level), so the DC offset causes no start-up jump.
    # Time:  O(S) for the S = 6 filter sections; O(1) for the RMS and the rest maths.
    # Space: O(1).
    # Data structures: see the class.  Files used: none.
    # -----------------------------------------------------------------------
    def step(self, value_mv: float) -> LiveReading:
        """Process one raw sample (mV at the ESP32 pin)."""
        if self.samples_seen == 0:
            self.filter.prime(value_mv)
        self.samples_seen += 1
        envelope = self.rms.step(self.filter.step(value_mv))
        if self.samples_seen <= self.settle_samples:
            return LiveReading(SETTLING, envelope, None, None)
        if self.rest_mv is None:
            self._rest_sum += envelope
            self._rest_count += 1
            if self._rest_count < self.rest_samples:
                return LiveReading(REST, envelope, None, None)
            self.rest_mv = self._rest_sum / self._rest_count   # > rest finished: this is 0%
        percent = PERCENT * (envelope - self.rest_mv) / self.mvc_above_rest_mv
        return LiveReading(LIVE, envelope, percent, self.rest_mv)

    # -----------------------------------------------------------------------
    # seconds_left()
    # What:  how long the current stage still lasts (0 once LIVE), for the display.
    # Time:  O(1).  Space: O(1).
    # Data structures: ints.  Files used: none.
    # -----------------------------------------------------------------------
    def seconds_left(self) -> float:
        """Seconds until the meter goes LIVE."""
        remaining = self.settle_samples + self.rest_samples - self.samples_seen
        return max(0.0, remaining / self.cfg.rate_hz)
