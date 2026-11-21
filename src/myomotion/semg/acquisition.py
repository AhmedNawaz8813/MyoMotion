"""Reading the ESP32 sEMG stream live (Step 5).

The ESP32 build (firmware myomotion_emg_wifi.ino) makes its own Wi-Fi network and sends text
lines over TCP to whoever connects to 192.168.4.1:3333:

    D,<t_us>,<ch1>,<ch2>,<ch3>     one sample: ESP32 micros() and millivolts at each ADC pin
    E,<t_us>,LED_ON                the reply to a sync command (ignored here)

TCP delivers a stream of bytes, not lines: one read can end in the middle of a line, and the
next read starts with the rest of it. LineParser keeps the unfinished piece until its end
arrives. Only Python's standard library is used (socket), so no extra package is needed.

The encrypted firmware (hardware/myomotion_emg_secure) seals the same lines with AES-GCM and adds
homomorphic P lines: give EspStream the keys and it decrypts first (src/myomotion/semg/secure_link.py),
so everything after it works unchanged.

Data structures (this file): a bytes buffer for an unfinished line; EmgSample and BoardEvent dataclasses; lists of
    samples and events; a TCP socket; a SecureFrameDecoder when the link is encrypted.
Time complexity (this file): O(b) per read for b received bytes (every line split and parsed once); O(1) per sample
    for the 32-bit clock unwrap.
"""

from __future__ import annotations   # > type hints are stored as text, not run: allows forward references

import socket                         # > TCP connection to the ESP32 (standard library)
from dataclasses import dataclass, field

MICROS_WRAP = 2 ** 32                 # micros() is 32-bit: it restarts from 0 every ~71.6 minutes
SAMPLE_PREFIX = "D,"                  # sample lines start with this; everything else is skipped
EVENT_PREFIX = "E,"                   # board event lines (E,<t_us>,LED_ON): kept in LineParser.events
SYNC_COMMAND = b"S"                   # the firmware flashes the sync LED and answers E,<t_us>,LED_ON


# ---------------------------------------------------------------------------
# AcquisitionConfig  (the CONFIG for this file)
# What:  where the ESP32 is and how to read from it.
# Data structures: a frozen dataclass (named, typed, cannot change by accident).
# Time:  O(1) to build (fixed fields).
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class AcquisitionConfig:
    """Connection settings for the ESP32's Wi-Fi stream."""

    host: str = "192.168.4.1"         # the ESP32's own address on its MyoMotion-EMG network
    port: int = 3333                  # TCP port the firmware listens on
    connect_timeout_s: float = 5.0    # give up connecting after this long
    read_timeout_s: float = 0.02      # wait at most 20 ms for new bytes, so the display stays responsive
    read_bytes: int = 4096            # read up to this many bytes at a time


ACQUISITION_CONFIG = AcquisitionConfig()


# ---------------------------------------------------------------------------
# EmgSample
# What:  one sample from the ESP32: its clock and the millivolts of every channel.
# Data structures: a frozen dataclass; the channel values are a tuple because a
#        sample never changes and the number of channels is fixed per line.
# Time:  O(1) to build (an int and a tuple of 3 floats).
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class EmgSample:
    """One sample: ESP32 time (unwrapped microseconds) and millivolts per channel."""

    t_us: int
    values_mv: tuple[float, ...]


# ---------------------------------------------------------------------------
# parse_line(text)
# What:  turns one text line into an EmgSample, or None if it is not a sample line
#        (E lines, start-up messages, or a damaged line).
# Time:  O(c) for the c comma-separated fields.  Space: O(c).
# Data structures: str.split gives a list of fields; the values become a tuple.
# Files used: none.
# ---------------------------------------------------------------------------
def parse_line(text: str) -> EmgSample | None:
    """Parse 'D,t_us,ch1,ch2,ch3' (t_us not yet unwrapped); None for anything else."""
    if not text.startswith(SAMPLE_PREFIX):
        return None
    fields = text.split(",")
    if len(fields) < 3:                                  # > needs at least D, t_us and one channel
        return None
    try:
        return EmgSample(t_us=int(fields[1]), values_mv=tuple(float(value) for value in fields[2:]))
    except ValueError:                                   # > a damaged line, e.g. half of it lost
        return None


# ---------------------------------------------------------------------------
# BoardEvent / parse_event(text)
# What:  an E line from the ESP32, e.g. E,<t_us>,LED_ON (the moment the sync LED came on).
# Time:  O(1).  Space: O(1).
# Data structures: frozen dataclass.  Files used: none.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class BoardEvent:
    """One event the ESP32 reported, on its own clock."""

    t_us: int
    name: str


# Time:  O(c) for the c comma-separated fields (3).  Space: O(c).
# Data structures: str.split gives a list of fields; a BoardEvent dataclass (or None) out.
def parse_event(text: str) -> BoardEvent | None:
    """Parse 'E,t_us,NAME'; None for anything else."""
    fields = text.split(",")
    if not text.startswith(EVENT_PREFIX) or len(fields) != 3:
        return None
    try:
        return BoardEvent(t_us=int(fields[1]), name=fields[2])
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# LineParser
# What:  feed it raw bytes in any chunk sizes; it returns the complete samples.
#        It keeps the unfinished end of each chunk for the next call, counts damaged
#        sample lines, and unwraps the ESP32's 32-bit micros() clock: when the time
#        suddenly jumps back by more than half the range, the clock has wrapped, so
#        2^32 is added from then on. Board events (E lines, e.g. LED_ON) are kept in
#        `events` for whoever needs them (the synced recorder).
# Data structures: a bytes buffer for the unfinished line; ints for the counters; a list
#        of BoardEvents (the caller empties it).
# Time:  O(b) per feed() for b received bytes.  Space: O(b).
# ---------------------------------------------------------------------------
@dataclass
class LineParser:
    """Bytes in, complete EmgSamples out (with an unwrapped clock)."""

    bad_lines: int = 0
    events: list = field(default_factory=list)
    _pending: bytes = field(default=b"", repr=False)
    _wrap_offset: int = field(default=0, repr=False)
    _last_raw_us: int | None = field(default=None, repr=False)

    # -----------------------------------------------------------------------
    # feed(chunk)
    # What:  adds a chunk, splits off every complete line, parses the sample lines.
    # Time:  O(b) for b bytes (one split, one parse per line).
    # Space: O(b): the lines of this chunk.
    # Data structures: list of byte lines; list of EmgSample out.
    # Files used: none.
    # -----------------------------------------------------------------------
    def feed(self, chunk: bytes) -> list[EmgSample]:
        """Add bytes; return the samples whose lines are now complete."""
        lines = (self._pending + chunk).split(b"\n")
        self._pending = lines.pop()                      # > the last piece has no newline yet: keep it
        samples: list[EmgSample] = []
        for raw_line in lines:
            text = raw_line.decode("ascii", errors="replace").strip()   # > strip removes the \r of \r\n
            sample = parse_line(text)
            if sample is None:
                event = parse_event(text)
                if event is not None:
                    self.events.append(BoardEvent(t_us=self._unwrap(event.t_us), name=event.name))
                elif text.startswith(SAMPLE_PREFIX):
                    self.bad_lines += 1
                continue
            samples.append(EmgSample(t_us=self._unwrap(sample.t_us), values_mv=sample.values_mv))
        return samples

    # -----------------------------------------------------------------------
    # _unwrap(raw_us)
    # What:  32-bit micros() -> ever-increasing microseconds.
    # Time:  O(1).  Space: O(1).
    # Data structures: ints.  Files used: none.
    # -----------------------------------------------------------------------
    def _unwrap(self, raw_us: int) -> int:
        """Add 2^32 for every wrap of the ESP32 clock seen so far."""
        if self._last_raw_us is not None and raw_us < self._last_raw_us - MICROS_WRAP // 2:
            self._wrap_offset += MICROS_WRAP
        self._last_raw_us = raw_us
        return raw_us + self._wrap_offset


# ---------------------------------------------------------------------------
# EspStream
# What:  a TCP connection to the ESP32 that returns whatever samples have arrived,
#        waiting at most read_timeout_s, so a display loop never freezes.
#        Use it with "with": the connection is closed even after an error.
#        With keys (encrypted firmware), every chunk is verified and decrypted first;
#        the homomorphic P lines of the latest read are kept in last_he_samples.
# Data structures: a socket and a LineParser (+ a SecureFrameDecoder when encrypted).
# Files used: none (network only); src/myomotion/semg/secure_link.py when encrypted.
# Time:  O(b) per read() for b received bytes.  Space: O(b).
# ---------------------------------------------------------------------------
class EspStream:
    """Live sample source: the ESP32's Wi-Fi TCP stream (plain or encrypted)."""

    # -----------------------------------------------------------------------
    # __init__(cfg, keys)
    # What:  connects to the ESP32. Raises ConnectionError with a hint if it cannot.
    #        keys = LinkKeys from secure_link.load_keys() for the encrypted firmware.
    # Time:  O(1) (one network handshake).  Space: O(1).
    # Data structures: socket, LineParser, SecureFrameDecoder.  Files used: none.
    # -----------------------------------------------------------------------
    def __init__(self, cfg: AcquisitionConfig = ACQUISITION_CONFIG, keys: object | None = None) -> None:
        """Open the TCP connection to the ESP32."""
        self.cfg = cfg
        self.parser = LineParser()
        self.decoder = None
        self.last_he_samples: list = []
        if keys is not None:
            # > imported only here, so the plain firmware does not need the cryptography package
            from myomotion.semg.secure_link import SecureFrameDecoder
            self.decoder = SecureFrameDecoder(keys)
        try:
            self.sock = socket.create_connection((cfg.host, cfg.port), timeout=cfg.connect_timeout_s)
        except OSError as error:
            raise ConnectionError(f"Could not connect to {cfg.host}:{cfg.port} ({error}). Is the laptop on the "
                                  "MyoMotion-EMG Wi-Fi, the ESP32 powered, and no other program (e.g. "
                                  "record_session.py) connected to it?") from error
        self.sock.settimeout(cfg.read_timeout_s)

    # -----------------------------------------------------------------------
    # read()
    # What:  the samples that arrived since the last call ([] if none within the
    #        timeout). Raises ConnectionError if the ESP32 closed the connection.
    # Time:  O(b) for the b bytes received.  Space: O(b).
    # Data structures: list of EmgSample.  Files used: none.
    # -----------------------------------------------------------------------
    def read(self) -> list[EmgSample]:
        """Return the newly arrived samples (possibly none)."""
        try:
            chunk = self.sock.recv(self.cfg.read_bytes)
        except (socket.timeout, BlockingIOError):        # > nothing new yet: not an error
            return []
        if not chunk:                                     # > recv returns b"" only when the other side closed
            raise ConnectionError("the ESP32 closed the connection")
        if self.decoder is None:
            return self.parser.feed(chunk)
        from myomotion.semg.secure_link import split_he_lines
        others, self.last_he_samples = split_he_lines(self.decoder.feed(chunk))
        return self.parser.feed(others)

    # -----------------------------------------------------------------------
    # flash_sync()
    # What:  asks the ESP32 to flash the sync LED; its E,<t_us>,LED_ON reply arrives in
    #        parser.events with the next read().
    # Time:  O(1).  Space: O(1).
    # Data structures: none.  Files used: none.
    # -----------------------------------------------------------------------
    def flash_sync(self) -> None:
        """Send the sync command."""
        self.sock.sendall(SYNC_COMMAND)

    # Time:  O(1).
    # Data structures: the TCP socket.
    def close(self) -> None:
        """Close the connection."""
        self.sock.close()

    # Time:  O(1).
    # Data structures: returns itself (context manager: `with EspStream(...) as s:`).
    def __enter__(self) -> EspStream:
        return self

    # Time:  O(1).
    # Data structures: closes the socket on leaving the `with` block.
    def __exit__(self, *exc_info: object) -> None:
        self.close()
