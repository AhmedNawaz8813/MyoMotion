"""Signal processing for the raw sEMG from the ESP32 build (Step 2).

Measured on 17 Nov (recordings S11-S13): the AD8226 module's SIG output behaves like RAW
EMG, not a smooth envelope. When the muscle works, the fast 20-250 Hz content rises 3-7x,
while mains hum (50 Hz and its multiples) is large all the time. So this file follows the
standard pipeline from the EMG lecture notes:

    raw mV --> high-pass 20 Hz --> notch 50/100/150/200 Hz --> rectify / square --> RMS envelope
               (removes the DC     (removes mains hum)          (makes everything     (~100 ms window:
                offset, slow drift                                positive)             muscle activation
                and movement)                                                         over time)

At 500 samples per second the highest frequency we can see is 250 Hz (Nyquist), so there is
no digital low-pass: the "band-pass" is the 20 Hz high-pass plus the Nyquist limit.

Two kinds of tool live here:
  * frequency domain: our own DFT (power_spectrum) and band RMS values from it (used by the
    TC01 check in quality.py);
  * time domain: IIR biquad filters and RMS envelopes that work on a whole array OR one sample
    at a time (Step 5 will use the sample-by-sample form for live %MVC).
All core maths is written out here; NumPy is used only for arrays and element-wise arithmetic.

Data structures (this file): NumPy float64 arrays for signals; Biquad dataclasses (five coefficients, two state
    values); a list of Biquads as the filter chain (EmgFilter); a deque with maxlen as a ring buffer (RunningRms);
    boolean NumPy masks over the spectrum bins.
Time complexity (this file): filtering and the envelope O(N * S) for N samples, S = 6 filter sections; RMS and
    moving average O(N); power_spectrum O(N^2) (our own direct DFT, fine for N <= 2500); live use O(S) per sample
    and O(1) for the running RMS.
"""

from __future__ import annotations   # > type hints are stored as text, not run: allows forward references

import math                           # > sin, cos, sqrt on single numbers
from collections import deque         # > ring buffer for the live RMS (RunningRms)
from dataclasses import dataclass, field

import numpy as np                    # > arrays for the signal

# > The two Q values of a 4th-order Butterworth filter built from two biquads:
# > Q_k = 1 / (2 cos(theta_k)) with theta = pi/8 and 3*pi/8. Together they give the flattest
# > possible pass band (no ripple) for a 4th-order filter.
BUTTERWORTH_4_QS = (1 / (2 * math.cos(math.pi / 8)), 1 / (2 * math.cos(3 * math.pi / 8)))


# ---------------------------------------------------------------------------
# DspConfig  (the CONFIG for this file)
# What:  every frequency and window length the processing uses, in one place.
# Data structures: a frozen dataclass, so a value cannot be changed by accident and
#        a typo in a field name fails at once.
# Time:  O(1) to build (fixed fields).
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DspConfig:
    """Frequencies and windows for the raw-EMG pipeline."""

    highpass_hz: float = 20.0          # bottom of the EMG band; below it: DC offset, drift, movement
    mains_hz: float = 50.0             # mains frequency in Pakistan
    mains_harmonics: int = 4           # notch 50, 100, 150 and 200 Hz (all below Nyquist = 250 Hz)
    notch_bandwidth_hz: float = 4.0    # each notch removes about +-2 Hz (mains drifts a little)
    envelope_window_s: float = 0.1     # RMS envelope window: 100 ms (lecture notes: ~100 ms)
    settle_pad_s: float = 0.5          # offline only: padding so the filters have settled at t = 0


DSP_CONFIG = DspConfig()


# ---------------------------------------------------------------------------
# rms(values)
# What:  Root Mean Square = sqrt( (x1^2 + x2^2 + ... + xn^2) / n ).
#        How strong a signal is on average, whatever its sign. For a sine of
#        amplitude A it is A / sqrt(2); for a constant c it is c.
# Time:  O(n): squares and adds each sample once.
# Space: O(n): NumPy makes one temporary array of the squares.
# Data structures: a 1-D NumPy float64 array in, one Python float out.
# Files used: none (pure maths).
# ---------------------------------------------------------------------------
def rms(values: np.ndarray) -> float:
    """Root mean square, written out from the formula."""
    if len(values) == 0:
        raise ValueError("RMS of an empty window is undefined")
    return math.sqrt(float(np.sum(values * values)) / len(values))


# ---------------------------------------------------------------------------
# mean_and_sd(values)
# What:  mean = (x1 + ... + xn) / n
#        SD   = sqrt( sum((xi - mean)^2) / (n - 1) )   (the "sample" SD, Bessel's
#        correction: the mean came from the same samples, which makes the spread
#        look slightly too small; n - 1 corrects for that). Used later for the
#        onset threshold T = mean_rest + 3 * SD_rest.
# Time:  O(n): one pass to add up, one pass for the squared differences.
# Space: O(n): the temporary array of squared differences.
# Data structures: a 1-D NumPy array in; a tuple (mean, sd) of two floats out.
# Files used: none (pure maths).
# ---------------------------------------------------------------------------
def mean_and_sd(values: np.ndarray) -> tuple[float, float]:
    """Mean and sample standard deviation, written out from the formulas."""
    count = len(values)
    if count < 2:
        raise ValueError("need at least 2 samples for a standard deviation")
    mean = float(np.sum(values)) / count
    squared_differences = (values - mean) ** 2           # > NumPy subtracts and squares every sample in one go
    return mean, math.sqrt(float(np.sum(squared_differences)) / (count - 1))


# ---------------------------------------------------------------------------
# power_spectrum(values, rate_hz)
# What:  our own Discrete Fourier Transform (DFT): how much power sits at each
#        frequency. For N samples x[0..N-1], bin k (k = 0 .. N/2) is
#            X[k] = sum over n of  x[n] * e^(-j * 2*pi * k * n / N)
#        e^(-j*theta) = cos(theta) - j*sin(theta) goes round the unit circle, so X[k]
#        measures how well x matches a wave of k cycles per window. Bin k is the
#        frequency f_k = k * rate_hz / N (5 s at 500 Hz: N = 2500, bins 0.2 Hz apart,
#        top bin = Nyquist = 250 Hz).
#        Power is scaled so the bins add up to the variance (Parseval's theorem):
#            P[k] = 2 * |X[k]|^2 / N^2  for 0 < k < N/2   (x2: negative frequencies mirror these)
#            P[k] =     |X[k]|^2 / N^2  for k = 0 and k = N/2
#        so a sine of amplitude A gives A^2 / 2 = (its RMS)^2, and sqrt(sum of a band)
#        is the RMS of the signal inside that band. The mean is removed first so the
#        DC level does not swamp bin 0.
# Time:  O(N^2): N/2 + 1 bins, each a sum over all N samples (N = 2500: ~3 million
#        multiplications, well under a second). The FFT (np.fft.rfft) gives the same
#        numbers in O(N log N); the tests check ours against it.
# Space: O(N): one array of "twiddle factors" at a time plus the two outputs.
# Data structures: NumPy arrays: sample index 0..N-1, the complex twiddle factors,
#        and two float64 outputs of length N/2 + 1 (frequencies, powers).
# Files used: none (pure maths).
# ---------------------------------------------------------------------------
def power_spectrum(values: np.ndarray, rate_hz: float) -> tuple[np.ndarray, np.ndarray]:
    """One-sided power spectrum by a direct DFT: returns (frequencies in Hz, power in mV^2)."""
    count = len(values)
    centred = values - float(np.sum(values)) / count     # > remove the mean (the DC level)
    sample_index = np.arange(count)                       # > n = 0, 1, ..., N-1
    n_bins = count // 2 + 1                               # > bins 0 .. N/2: the one-sided spectrum
    freqs_hz = np.zeros(n_bins)
    power = np.zeros(n_bins)
    for k in range(n_bins):
        twiddle = np.exp(-2j * np.pi * k * sample_index / count)   # > e^(-j 2 pi k n / N) for every n
        coefficient = np.sum(centred * twiddle)                    # > X[k]: one complex number
        is_edge_bin = k == 0 or (count % 2 == 0 and k == count // 2)
        power[k] = (1.0 if is_edge_bin else 2.0) * abs(coefficient) ** 2 / count ** 2
        freqs_hz[k] = k * rate_hz / count
    return freqs_hz, power


# ---------------------------------------------------------------------------
# mains_band_mask(freqs_hz, rate_hz, cfg)
# What:  marks the spectrum bins within notch_bandwidth_hz / 2 of 50 Hz and its
#        multiples (100, 150, 200 Hz): the frequencies the notch filters remove.
# Time:  O(K * H): K bins compared once per harmonic H (4).
# Space: O(K): one boolean mask.
# Data structures: a boolean NumPy mask over the bins, OR-ed per harmonic.
# Files used: none.
# ---------------------------------------------------------------------------
def mains_band_mask(freqs_hz: np.ndarray, rate_hz: float, cfg: DspConfig = DSP_CONFIG) -> np.ndarray:
    """Boolean mask of the bins near 50 Hz and its harmonics below Nyquist."""
    in_band = np.zeros(len(freqs_hz), dtype=bool)
    for harmonic in range(1, cfg.mains_harmonics + 1):
        centre_hz = harmonic * cfg.mains_hz
        if centre_hz >= rate_hz / 2:                     # > at or above Nyquist: cannot be seen
            break
        in_band |= np.abs(freqs_hz - centre_hz) <= cfg.notch_bandwidth_hz / 2
    return in_band


# ---------------------------------------------------------------------------
# emg_band_rms_from_spectrum(freqs_hz, power, rate_hz, cfg)
# What:  RMS of the muscle signal: sqrt of the power from highpass_hz (20 Hz) up to
#        Nyquist, leaving out the mains bands. Thanks to Parseval this equals the RMS
#        the signal would have after the high-pass and notch filters below, without
#        their start-up ringing. Leaving out 4 x 4 Hz of the ~230 Hz band loses ~7% of
#        the muscle power, the same for rest and activity, so ratios are unaffected.
# Time:  O(K): a few comparisons and one sum over the K bins.
# Space: O(K): the boolean selection mask.
# Data structures: boolean NumPy masks combined with & and ~ (AND, NOT).
# Files used: none.
# ---------------------------------------------------------------------------
def emg_band_rms_from_spectrum(freqs_hz: np.ndarray, power: np.ndarray, rate_hz: float,
                               cfg: DspConfig = DSP_CONFIG) -> float:
    """RMS (mV) of the 20 Hz .. Nyquist band with the mains bands removed."""
    in_emg_band = (freqs_hz >= cfg.highpass_hz) & ~mains_band_mask(freqs_hz, rate_hz, cfg)
    return math.sqrt(float(np.sum(power[in_emg_band])))


# ---------------------------------------------------------------------------
# mains_rms_from_spectrum(freqs_hz, power, rate_hz, cfg)
# What:  RMS of the mains hum alone (sqrt of the power in the mains bands) and its
#        share of all the power (hum power / total power, 0 when there is no power).
# Time:  O(K) over the K bins.
# Space: O(K): the mask.
# Data structures: boolean NumPy mask; returns a tuple (hum_rms_mv, hum_share).
# Files used: none.
# ---------------------------------------------------------------------------
def mains_rms_from_spectrum(freqs_hz: np.ndarray, power: np.ndarray, rate_hz: float,
                            cfg: DspConfig = DSP_CONFIG) -> tuple[float, float]:
    """RMS (mV) and power share of the mains hum (50 Hz and harmonics)."""
    hum_power = float(np.sum(power[mains_band_mask(freqs_hz, rate_hz, cfg)]))
    total_power = float(np.sum(power))
    return math.sqrt(hum_power), (hum_power / total_power if total_power > 0 else 0.0)


# ---------------------------------------------------------------------------
# emg_band_rms(values, rate_hz, cfg)
# What:  convenience wrapper: spectrum first, then the EMG-band RMS.
# Time:  O(N^2), from power_spectrum.
# Space: O(N).
# Data structures: NumPy arrays (see power_spectrum).
# Files used: none.
# ---------------------------------------------------------------------------
def emg_band_rms(values: np.ndarray, rate_hz: float, cfg: DspConfig = DSP_CONFIG) -> float:
    """RMS (mV) of the muscle band (20 Hz .. Nyquist, mains removed) of one window."""
    freqs_hz, power = power_spectrum(values, rate_hz)
    return emg_band_rms_from_spectrum(freqs_hz, power, rate_hz, cfg)


# ---------------------------------------------------------------------------
# Biquad
# What:  one second-order IIR ("infinite impulse response") filter section. Each
#        output depends on the last two inputs and the last two outputs:
#            y[n] = b0 x[n] + b1 x[n-1] + b2 x[n-2] - a1 y[n-1] - a2 y[n-2]
#        Written in "Direct Form II Transposed", which keeps just two state numbers
#        (z1, z2) and is numerically well behaved:
#            y  = b0 * x + z1
#            z1 = b1 * x - a1 * y + z2
#            z2 = b2 * x - a2 * y
#        High-pass and notch filters are both biquads; only the five coefficients
#        differ (see highpass_biquad and notch_biquad). Chaining biquads makes
#        steeper filters.
# Data structures: a dataclass holding five float coefficients and two float
#        states. The state is what lets it run one sample at a time (live) and
#        continue exactly where it left off.
# Time:  O(1) per sample (step); O(n) to filter n samples.  Space: O(1).
# ---------------------------------------------------------------------------
@dataclass
class Biquad:
    """Second-order IIR filter section (Direct Form II Transposed), usable sample by sample."""

    b0: float
    b1: float
    b2: float
    a1: float
    a2: float
    z1: float = 0.0          # filter memory: what the previous samples still add to the next output
    z2: float = 0.0

    # -----------------------------------------------------------------------
    # step(x)
    # What:  filters one sample and updates the memory (the three equations above).
    # Time:  O(1): five multiplications and four additions.
    # Space: O(1).
    # Data structures: two float state variables.
    # Files used: none.
    # -----------------------------------------------------------------------
    def step(self, x: float) -> float:
        """Filter one sample."""
        y = self.b0 * x + self.z1
        self.z1 = self.b1 * x - self.a1 * y + self.z2
        self.z2 = self.b2 * x - self.a2 * y
        return y

    # -----------------------------------------------------------------------
    # dc_gain()
    # What:  how much a constant input is multiplied by once settled:
    #        H(1) = (b0 + b1 + b2) / (1 + a1 + a2)  (z = 1 is frequency 0).
    #        A high-pass gives 0, a notch gives 1.
    # Time:  O(1).  Space: O(1).
    # Data structures: floats.  Files used: none.
    # -----------------------------------------------------------------------
    def dc_gain(self) -> float:
        """Steady-state gain for a constant input."""
        return (self.b0 + self.b1 + self.b2) / (1 + self.a1 + self.a2)

    # -----------------------------------------------------------------------
    # prime(x0)
    # What:  sets the memory as if the input had always been x0, so a recording that
    #        starts at, say, 600 mV does not make the filter "ring" at the start.
    #        In steady state y = dc_gain * x0, and the update equations give
    #            z2 = b2 x0 - a2 y,   z1 = b1 x0 - a1 y + z2.
    # Time:  O(1).  Space: O(1).
    # Data structures: floats.  Files used: none.
    # -----------------------------------------------------------------------
    def prime(self, x0: float) -> float:
        """Start in steady state for a constant input x0; returns the steady output."""
        y0 = self.dc_gain() * x0
        self.z2 = self.b2 * x0 - self.a2 * y0
        self.z1 = self.b1 * x0 - self.a1 * y0 + self.z2
        return y0

    # -----------------------------------------------------------------------
    # filter(values)
    # What:  runs step() over a whole array, carrying the memory along.
    # Time:  O(n): one step per sample (a plain loop, because each output needs
    #        the previous one, so it cannot be done "all at once" in NumPy).
    # Space: O(n): the output array.
    # Data structures: NumPy float64 array in and out.
    # Files used: none.
    # -----------------------------------------------------------------------
    def filter(self, values: np.ndarray) -> np.ndarray:
        """Filter a whole array (memory continues from the last call)."""
        output = np.empty(len(values))
        for index, x in enumerate(values):
            output[index] = self.step(float(x))
        return output


# ---------------------------------------------------------------------------
# highpass_biquad(cutoff_hz, rate_hz, q)
# What:  designs a 2nd-order high-pass from the standard "Audio EQ Cookbook"
#        (R. Bristow-Johnson) formulas, which use the bilinear transform:
#            w0 = 2*pi*f0/fs,  alpha = sin(w0) / (2Q)
#            b0 = b2 = (1 + cos w0) / 2,  b1 = -(1 + cos w0)
#            a0 = 1 + alpha,  a1 = -2 cos w0,  a2 = 1 - alpha
#        and divides everything by a0 so the filter equations need no division.
# Time:  O(1).  Space: O(1).
# Data structures: returns a Biquad dataclass.
# Files used: none.
# ---------------------------------------------------------------------------
def highpass_biquad(cutoff_hz: float, rate_hz: float, q: float) -> Biquad:
    """2nd-order high-pass section (RBJ cookbook)."""
    w0 = 2 * math.pi * cutoff_hz / rate_hz
    alpha = math.sin(w0) / (2 * q)
    cos_w0 = math.cos(w0)
    a0 = 1 + alpha
    return Biquad(b0=(1 + cos_w0) / 2 / a0, b1=-(1 + cos_w0) / a0, b2=(1 + cos_w0) / 2 / a0,
                  a1=-2 * cos_w0 / a0, a2=(1 - alpha) / a0)


# ---------------------------------------------------------------------------
# notch_biquad(centre_hz, rate_hz, bandwidth_hz)
# What:  designs a notch: zero gain exactly at centre_hz, gain ~1 elsewhere.
#        Cookbook formulas with Q = centre / bandwidth:
#            b0 = 1, b1 = -2 cos w0, b2 = 1, a0 = 1 + alpha, a1 = -2 cos w0, a2 = 1 - alpha
#        The zeros sit exactly on the unit circle at +-w0 (that is what makes the
#        gain 0 there); the poles just inside set how narrow the notch is. Using the
#        same bandwidth for every harmonic (rather than the same Q) removes the same
#        +-2 Hz around 50, 100, 150 and 200 Hz and makes all four settle equally fast
#        (time constant about 1 / (pi * bandwidth) = 80 ms).
# Time:  O(1).  Space: O(1).
# Data structures: returns a Biquad dataclass.
# Files used: none.
# ---------------------------------------------------------------------------
def notch_biquad(centre_hz: float, rate_hz: float, bandwidth_hz: float) -> Biquad:
    """2nd-order notch section (RBJ cookbook)."""
    w0 = 2 * math.pi * centre_hz / rate_hz
    alpha = math.sin(w0) / (2 * (centre_hz / bandwidth_hz))
    cos_w0 = math.cos(w0)
    a0 = 1 + alpha
    return Biquad(b0=1 / a0, b1=-2 * cos_w0 / a0, b2=1 / a0, a1=-2 * cos_w0 / a0, a2=(1 - alpha) / a0)


# ---------------------------------------------------------------------------
# EmgFilter
# What:  the full cleaning chain for raw EMG: a 4th-order Butterworth high-pass at
#        20 Hz (two biquads) followed by one notch per mains harmonic below Nyquist.
#        Works on a whole recording (filter) or one sample at a time (step).
# Data structures: a list of Biquad sections, applied in order. A list because the
#        order is fixed and we walk it front to back for every sample.
# Time:  O(S) per sample for S = 6 sections; O(n * S) for n samples.  Space: O(S).
# ---------------------------------------------------------------------------
@dataclass
class EmgFilter:
    """High-pass + mains notches for raw EMG at a given sample rate."""

    rate_hz: float
    cfg: DspConfig = DSP_CONFIG
    sections: list[Biquad] = field(init=False)

    # -----------------------------------------------------------------------
    # __post_init__()
    # What:  builds the sections once the sample rate is known.
    # Time:  O(H) for H mains harmonics.  Space: O(H).
    # Data structures: list of Biquad.  Files used: none.
    # -----------------------------------------------------------------------
    def __post_init__(self) -> None:
        """Design the high-pass and notch sections for this sample rate."""
        self.sections = [highpass_biquad(self.cfg.highpass_hz, self.rate_hz, q) for q in BUTTERWORTH_4_QS]
        for harmonic in range(1, self.cfg.mains_harmonics + 1):
            centre_hz = harmonic * self.cfg.mains_hz
            if centre_hz < self.rate_hz / 2:
                self.sections.append(notch_biquad(centre_hz, self.rate_hz, self.cfg.notch_bandwidth_hz))

    # -----------------------------------------------------------------------
    # prime(x0)
    # What:  starts every section in steady state for a constant input x0, passing
    #        each section's steady output on to the next (the high-pass outputs 0).
    # Time:  O(S) for S sections.  Space: O(1).
    # Data structures: walks the section list.  Files used: none.
    # -----------------------------------------------------------------------
    def prime(self, x0: float) -> None:
        """Start the whole chain in steady state for input x0."""
        for section in self.sections:
            x0 = section.prime(x0)

    # -----------------------------------------------------------------------
    # step(x)
    # What:  filters one live sample through every section in turn.
    # Time:  O(S) per sample (S = 6 sections at 500 Hz): tiny, fine for live use.
    # Space: O(1).
    # Data structures: walks the section list.  Files used: none.
    # -----------------------------------------------------------------------
    def step(self, x: float) -> float:
        """Filter one sample through the whole chain."""
        for section in self.sections:
            x = section.step(x)
        return x

    # -----------------------------------------------------------------------
    # filter(values, settle)
    # What:  filters a whole recording. With settle=True (the default, for offline
    #        use) two things stop the filters "ringing" at the start:
    #        1. Periodic padding: a copy of the first stretch of the signal is put in
    #           front, x[0..p-1] + x[0..], with p a whole number of mains cycles
    #           (20 ms = 10 samples at 500 Hz) close to settle_pad_s. Because the hum
    #           repeats every 20 ms, the join x[p-1] -> x[0] looks exactly like the real
    #           step x[p-1] -> x[p], so the notches settle on the hum's true phase
    #           before the real data starts. (A mirror image was tried first: it
    #           reverses the hum's phase and made the ringing worse.) The padding is
    #           cut off again at the end.
    #        2. prime(): the chain starts in steady state for the first padded value,
    #           so the DC offset causes no jump either.
    #        Live use (step) cannot look ahead, so it just needs ~0.5 s to settle.
    #        Running each section over the whole array in turn gives the same result
    #        as the sample-by-sample chain (the sections are linear and time-invariant).
    # Time:  O((n + p) * S): every sample (plus p padding samples) through every section.
    # Space: O(n + p): the padded array and one output per section pass.
    # Data structures: NumPy float64 arrays joined with np.concatenate.
    # Files used: none.
    # -----------------------------------------------------------------------
    def filter(self, values: np.ndarray, settle: bool = True) -> np.ndarray:
        """Filter a whole array; settle=True pads and primes so there is no start-up ringing."""
        data = np.asarray(values, dtype=np.float64)
        pad = 0
        if settle:
            cycle = max(1, round(self.rate_hz / self.cfg.mains_hz))          # > samples per mains cycle (10)
            pad = round(self.cfg.settle_pad_s * self.rate_hz / cycle) * cycle  # > whole cycles (250 = 25 cycles)
            pad = min(pad, (len(data) // cycle) * cycle)                       # > short inputs: what fits
        if pad > 0:
            data = np.concatenate((data[:pad], data))     # > x[0..p-1] in front of x: continues the hum
        if settle and len(data) > 0:
            self.prime(float(data[0]))
        for section in self.sections:
            data = section.filter(data)
        return data[pad:]                                  # > drop the padding again


# ---------------------------------------------------------------------------
# rectify(values)
# What:  full-wave rectification: every sample becomes its size, |x|. After the
#        high-pass the signal swings around 0, so this turns the swings into a
#        positive "how much is happening" signal (the lecture notes' "rectify").
# Time:  O(n).  Space: O(n).
# Data structures: NumPy arrays (np.abs is one element-wise operation, not an algorithm).
# Files used: none.
# ---------------------------------------------------------------------------
def rectify(values: np.ndarray) -> np.ndarray:
    """Full-wave rectification |x|."""
    return np.abs(values)


# ---------------------------------------------------------------------------
# moving_average(values, window)
# What:  trailing moving average: output i is the mean of the last `window` samples
#        up to and including i (fewer at the very start). Trailing = it only uses
#        the past, so the same result is possible live.
#        Trick for O(n): with the running total c[i] = x[0] + ... + x[i-1],
#        the sum of x[lo..i] is c[i+1] - c[lo], one subtraction per output.
# Time:  O(n): one cumulative sum, then constant work per output.
# Space: O(n): the cumulative sum and the output.
# Data structures: NumPy arrays; index arrays (lo, counts) let NumPy do every
#        output at once instead of a Python loop.
# Files used: none.
# ---------------------------------------------------------------------------
def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    """Trailing moving average over `window` samples (shorter at the start)."""
    if window < 1:
        raise ValueError("window must be at least 1 sample")
    running_total = np.concatenate(([0.0], np.cumsum(values, dtype=np.float64)))
    index = np.arange(len(values))
    lo = np.maximum(0, index - window + 1)               # > first sample inside each window
    counts = index + 1 - lo                               # > how many samples each window holds
    return (running_total[index + 1] - running_total[lo]) / counts


# ---------------------------------------------------------------------------
# moving_rms(values, window)
# What:  trailing moving RMS: sqrt of the moving average of the squares. On a
#        high-passed EMG signal this is the "RMS envelope": the muscle's activation
#        over time. Squaring already makes every value positive, so a separate
#        rectify step is not needed (|x|^2 = x^2).
# Time:  O(n), via moving_average.
# Space: O(n).
# Data structures: NumPy arrays.
# Files used: none.
# ---------------------------------------------------------------------------
def moving_rms(values: np.ndarray, window: int) -> np.ndarray:
    """Trailing moving RMS over `window` samples."""
    mean_square = moving_average(values * values, window)
    return np.sqrt(np.maximum(mean_square, 0.0))         # > max(.., 0): rounding can give -1e-12


# ---------------------------------------------------------------------------
# emg_envelope(values, rate_hz, cfg)
# What:  the whole Step 2 pipeline on one recording:
#        raw mV -> EmgFilter (20 Hz high-pass + mains notches) -> 100 ms moving RMS.
#        The result is in mV at the ESP32 pin; Step 4 divides it by the MVC to get %MVC.
# Time:  O(n * S) for the filter plus O(n) for the RMS.
# Space: O(n).
# Data structures: NumPy arrays; an EmgFilter (list of Biquad sections).
# Files used: none.
# ---------------------------------------------------------------------------
def emg_envelope(values: np.ndarray, rate_hz: float, cfg: DspConfig = DSP_CONFIG) -> np.ndarray:
    """RMS envelope (mV) of raw EMG: filter, then a trailing RMS of envelope_window_s."""
    cleaned = EmgFilter(rate_hz, cfg).filter(values)
    window = max(1, round(cfg.envelope_window_s * rate_hz))   # > 0.1 s * 500 Hz = 50 samples
    return moving_rms(cleaned, window)


# ---------------------------------------------------------------------------
# RunningRms
# What:  the live version of moving_rms: feed one sample at a time, get the RMS of
#        the last `window` samples straight away. Keeps a running sum of squares:
#        add the new square, subtract the square that falls out of the window.
#        (Over very long sessions rounding could build up; recompute() resets it.)
# Data structures: a deque with maxlen = window as a ring buffer: appending to a
#        full deque drops the oldest value in O(1), unlike list.pop(0) which is O(n).
# Time:  O(1) per sample (step); O(window) to recompute.  Space: O(window).
# ---------------------------------------------------------------------------
@dataclass
class RunningRms:
    """Trailing RMS over the last `window` samples, one sample at a time."""

    window: int
    buffer: deque = field(init=False)
    sum_of_squares: float = field(init=False, default=0.0)

    # -----------------------------------------------------------------------
    # __post_init__()
    # What:  makes the empty ring buffer.  Time: O(1).  Space: O(window).
    # Data structures: deque(maxlen=window).  Files used: none.
    # -----------------------------------------------------------------------
    def __post_init__(self) -> None:
        """Create the empty ring buffer."""
        if self.window < 1:
            raise ValueError("window must be at least 1 sample")
        self.buffer = deque(maxlen=self.window)

    # -----------------------------------------------------------------------
    # step(x)
    # What:  adds one sample and returns the RMS of the samples now in the window.
    # Time:  O(1): one subtraction, one addition, one append, one sqrt.
    # Space: O(1) extra (the buffer is already allocated).
    # Data structures: the deque ring buffer.  Files used: none.
    # -----------------------------------------------------------------------
    def step(self, x: float) -> float:
        """Add one sample; return the current RMS."""
        if len(self.buffer) == self.window:              # > full: the oldest sample is about to drop out
            self.sum_of_squares -= self.buffer[0] ** 2
        self.buffer.append(x)
        self.sum_of_squares += x * x
        return math.sqrt(max(self.sum_of_squares, 0.0) / len(self.buffer))

    # -----------------------------------------------------------------------
    # recompute()
    # What:  recalculates the sum of squares exactly from the buffer, clearing any
    #        rounding that has built up over a long live session.
    # Time:  O(window).  Space: O(1).
    # Data structures: the deque.  Files used: none.
    # -----------------------------------------------------------------------
    def recompute(self) -> None:
        """Reset the running sum from the buffer contents."""
        self.sum_of_squares = sum(value * value for value in self.buffer)
