"""Encrypted ESP32 -> laptop link: AES-128-GCM for the link, additive homomorphic encryption for the envelope.

Firmware: hardware/myomotion_emg_secure/ (uses the ESP32's hardware AES through mbedtls).

Layer 1, link encryption (who can read and change the data on the way)
    Every batch the ESP32 sends (about 20 ms of samples) is sealed with AES-128-GCM: encrypted, and signed
    with a 16-byte tag. Without the link key the Wi-Fi traffic is unreadable, and a changed, replayed or
    forged batch fails the tag check and is thrown away. Lines on the wire:
        H,<session hex>,<key id>,<version>        hello, in clear, once per connection
        X,<counter>,<base64 ciphertext>,<base64 tag>   one sealed batch
    Nonce = the 8-byte random session (new every connection) + the 4-byte batch counter, so a nonce is
    never reused with the same key. Inside a batch are the usual text lines:
        D,<t_us>,<ch1>,<ch2>,<ch3>     raw millivolts (for the dataset, exactly as before)
        E,<t_us>,LED_ON                the sync flash
        P,<n>,<t_us>,<c1>,<c2>,<c3>    homomorphic ciphertexts of sample n (16 hex digits each)

Layer 2, additive homomorphic encryption (computing on data you cannot read)
    Scheme of Castelluccia, Mykletun and Tsudik (2005), made for sensor networks:
        encrypt:  c = (p + k) mod 2^64        decrypt a sum:  sum(p) = (sum(c) - sum(k)) mod 2^64
    p = (filtered sample)^2 x 1000 as a whole number (mV^2 in thousandths): the ESP32 runs the same filter as
    src/myomotion/semg/dsp.py (20 Hz high-pass + mains notches) and squares the result. k comes from the HE
    key: the first 8 bytes of AES(he_key, session | n | channel), a fresh key for every sample and channel.
    Adding ciphertexts adds the hidden values underneath, so a party WITHOUT the key (HeAggregator) can add
    up each 100 ms window. Only the key holder (HeDecryptor) removes the keys from the window sum, divides
    by the window length and takes the square root: that is exactly the RMS envelope of
    src/myomotion/semg/dsp.py (RunningRms over 50 samples), computed by someone who never saw a sample.

Keys: two random 128-bit keys made by scripts/semg/make_link_keys.py, stored outside git (laptop:
~/.config/myomotion/keys.json; firmware: hardware/myomotion_emg_secure/secrets.h, git-ignored).
AES itself comes from the `cryptography` package on the laptop and from the ESP32's hardware: encryption
primitives must never be home-made. Everything built on top of AES here is our own code.

Data structures (this file): a LinkKeys dataclass of bytes; NumPy uint64 arrays for keystreams, ciphertexts and
    window sums (arithmetic wraps modulo 2^64 by itself); a deque ring buffer (HeAggregator); a bytes buffer and int
    counters (SecureFrameDecoder); AES-ECB and AES-GCM objects from the cryptography library (never home-made).
Time complexity (this file): O(N * C) for N samples and C channels (one AES block per sample and channel; sliding
    window sums from one cumulative sum); O(C) per sample when live; O(m) per sealed batch of m bytes.
"""

from __future__ import annotations   # > type hints are stored as text, not run: allows forward references

import base64
import hashlib
import json
import math
import os
import secrets
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from myomotion.semg.dsp import DSP_CONFIG, EmgFilter

HE_MODULUS = 2 ** 64                 # ciphertexts and sums live modulo 2^64 (one unsigned 64-bit word)
KEY_BYTES = 16                       # AES-128
SESSION_BYTES = 8
TAG_BYTES = 16
CHANNELS = 3


# ---------------------------------------------------------------------------
# SecureConfig  (the CONFIG for this file; the firmware has the same numbers)
# What:  protocol constants shared by the ESP32 and the laptop.
# Data structures: a frozen dataclass.
# Time:  O(1) to build (fixed fields).
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SecureConfig:
    """Settings of the encrypted link and the homomorphic envelope."""

    version: str = "1"                       # protocol version in the hello line
    aad: bytes = b"MyoMotion-X1"             # extra data signed with every batch (binds the protocol)
    power_scale: float = 1000.0              # p = round(y^2 * 1000): mV^2 in thousandths
    rate_hz: float = 500.0                   # sample rate the ESP32 filter is designed for
    window: int = round(DSP_CONFIG.envelope_window_s * 500.0)   # 50 samples = the 100 ms RMS window
    key_file: Path = Path("~/.config/myomotion/keys.json")


SECURE_CONFIG = SecureConfig()


# Time:  O(1) (an exception type).
# Data structures: a subclass of ValueError carrying the message.
class KeyMismatch(ValueError):
    """The ESP32 and the laptop hold different keys."""


# ---------------------------------------------------------------------------
# LinkKeys
# What:  the two secret keys and their public fingerprint (key id). The id lets the
#        ESP32 and the laptop check they hold the same keys without revealing them.
# Data structures: frozen dataclass of bytes.
# Time:  O(1) to build (two 16-byte keys).
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LinkKeys:
    """Link key (AES-GCM) and homomorphic key (AES keystream)."""

    link_key: bytes
    he_key: bytes

    # Time:  O(k) for the k = 32 key bytes (one SHA-256).  Space: O(1).
    # Data structures: bytes in; the first 4 digest bytes as a hex str out.
    @property
    def key_id(self) -> str:
        """First 4 bytes of SHA-256 over both keys, as hex (safe to show)."""
        return hashlib.sha256(b"MyoMotion keys" + self.link_key + self.he_key).digest()[:4].hex()


# ---------------------------------------------------------------------------
# make_keys() / save_keys(keys, path) / load_keys(path)
# What:  make two random keys (operating-system randomness), write them to a JSON
#        file only the user can read (permission 600), and read them back.
# Time:  O(1).  Space: O(1).
# Data structures: dict -> JSON.  Files used: reads / writes the key file.
# ---------------------------------------------------------------------------
def make_keys() -> LinkKeys:
    """Two fresh random 128-bit keys."""
    return LinkKeys(link_key=secrets.token_bytes(KEY_BYTES), he_key=secrets.token_bytes(KEY_BYTES))


# Time:  O(1) (two 16-byte keys).  Space: O(1).
# Data structures: dict -> JSON, written through an OS file descriptor opened with permission 600.
def save_keys(keys: LinkKeys, path: Path) -> Path:
    """Write the key file (permission 600)."""
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    contents = {"link_key": keys.link_key.hex(), "he_key": keys.he_key.hex(), "key_id": keys.key_id}
    # > create the file with owner-only permission from the start (no moment where others could read it)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "w") as key_file:
        json.dump(contents, key_file, indent=2)
    return path


# Time:  O(1).  Space: O(1).
# Data structures: dict from JSON; a LinkKeys dataclass of bytes out.
def load_keys(path: Path = SECURE_CONFIG.key_file) -> LinkKeys:
    """Read the key file."""
    path = Path(path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"no key file at {path}: run  python scripts/semg/make_link_keys.py  first")
    contents = json.loads(path.read_text())
    keys = LinkKeys(link_key=bytes.fromhex(contents["link_key"]), he_key=bytes.fromhex(contents["he_key"]))
    if len(keys.link_key) != KEY_BYTES or len(keys.he_key) != KEY_BYTES:
        raise ValueError(f"{path} does not hold two 16-byte keys")
    return keys


# ---------------------------------------------------------------------------
# he_power(y_mv, cfg)
# What:  the value that gets encrypted: the filtered sample squared, as a whole
#        number of thousandths of mV^2 (homomorphic addition works on integers).
# Time:  O(1).  Space: O(1).
# Data structures: int.  Files used: none.
# ---------------------------------------------------------------------------
def he_power(y_mv: float, cfg: SecureConfig = SECURE_CONFIG) -> int:
    """round(y^2 * power_scale), half away from zero like C's llround."""
    return int(math.floor(y_mv * y_mv * cfg.power_scale + 0.5))


# ---------------------------------------------------------------------------
# he_key_blocks(session, first_n, count, channels)
# What:  the AES input blocks for samples first_n .. first_n+count-1:
#        session (8 bytes) | n (4 bytes, big-endian) | channel (1 byte) | 3 zero bytes.
#        Every (session, sample, channel) gets its own block, so every key is used once.
# Time:  O(count * channels).  Space: O(count * channels * 16) bytes.
# Data structures: one bytearray (AES-ECB then encrypts all blocks in one call).
# Files used: none.
# ---------------------------------------------------------------------------
def he_key_blocks(session: bytes, first_n: int, count: int, channels: int = CHANNELS) -> bytes:
    """Concatenated 16-byte keystream inputs."""
    blocks = bytearray()
    for n in range(first_n, first_n + count):
        prefix = session + (n % 2 ** 32).to_bytes(4, "big")
        for channel in range(channels):
            blocks += prefix + bytes((channel, 0, 0, 0))
    return bytes(blocks)


# ---------------------------------------------------------------------------
# he_keys(he_key, session, first_n, count, channels)
# What:  the homomorphic keys k for a run of samples: first 8 bytes (little-endian)
#        of each AES block's output. Only the key holder can make these.
# Time:  O(count * channels) AES blocks (one library call).  Space: same.
# Data structures: NumPy uint64 array (count x channels): uint64 arithmetic wraps
#        modulo 2^64 by itself, which is exactly the scheme's modulus.
# Files used: none.
# ---------------------------------------------------------------------------
def he_keys(he_key: bytes, session: bytes, first_n: int, count: int, channels: int = CHANNELS) -> np.ndarray:
    """Keystream values k[sample, channel] as uint64."""
    encryptor = Cipher(algorithms.AES(he_key), modes.ECB()).encryptor()
    output = encryptor.update(he_key_blocks(session, first_n, count, channels)) + encryptor.finalize()
    # > each 16-byte block -> two 8-byte words; keep the first word of every block
    return np.frombuffer(output, dtype="<u8").reshape(count, channels, 2)[:, :, 0].copy()


# ---------------------------------------------------------------------------
# HeSample
# What:  one P line: sample number, ESP32 time and one ciphertext per channel.
# Data structures: frozen dataclass; ciphertexts as a tuple of ints.
# Time:  O(C) to build for C channels.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class HeSample:
    """Homomorphic ciphertexts of one sample."""

    n: int
    t_us: int
    ciphertexts: tuple[int, ...]


# ---------------------------------------------------------------------------
# parse_he_line(text) / format_he_line(n, t_us, ciphertexts)
# What:  P,<n>,<t_us>,<c1 hex>,<c2 hex>,<c3 hex>  <-> HeSample.
# Time:  O(channels).  Space: O(channels).
# Data structures: str.split list; tuple.  Files used: none.
# ---------------------------------------------------------------------------
def parse_he_line(text: str) -> HeSample | None:
    """HeSample from a P line, or None if it is not a valid P line."""
    if not text.startswith("P,"):
        return None
    fields = text.split(",")
    try:
        return HeSample(n=int(fields[1]), t_us=int(fields[2]), ciphertexts=tuple(int(f, 16) for f in fields[3:]))
    except (ValueError, IndexError):
        return None


# Time:  O(C) for C channels.  Space: O(C).
# Data structures: a tuple/list of ints in; one text line out (16 hex digits per ciphertext).
def format_he_line(n: int, t_us: int, ciphertexts: tuple[int, ...] | list[int]) -> str:
    """The P line for one sample."""
    return f"P,{n},{t_us}," + ",".join(f"{c:016x}" for c in ciphertexts) + "\n"


# ---------------------------------------------------------------------------
# HeWindow
# What:  what the aggregator hands to the key holder: an encrypted window sum per
#        channel and which samples it covers.
# Data structures: frozen dataclass.
# Time:  O(1) to build (C window sums).
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class HeWindow:
    """Encrypted sums of samples first_n .. last_n."""

    first_n: int
    last_n: int
    t_us: int                         # time of the newest sample in the window
    sums: tuple[int, ...]             # one encrypted sum per channel

    # Time:  O(1).
    # Data structures: ints.
    @property
    def count(self) -> int:
        """How many samples the sums cover."""
        return self.last_n - self.first_n + 1


# ---------------------------------------------------------------------------
# HeAggregator  (holds NO key)
# What:  keeps a running encrypted sum of the last `window` samples per channel:
#        add the newest ciphertext, subtract the one that falls out (homomorphic
#        subtraction works just like addition, modulo 2^64). A gap in the sample
#        numbers (a rejected batch) starts the window again, so the key holder always
#        knows exactly which samples are inside.
# Time:  O(channels) per sample.  Space: O(window * channels).
# Data structures: deque(maxlen=window) ring buffer of HeSamples (as RunningRms in dsp.py);
#        a list of running sums.
# ---------------------------------------------------------------------------
@dataclass
class HeAggregator:
    """Window sums over ciphertexts, without any key."""

    window: int = SECURE_CONFIG.window
    channels: int = CHANNELS
    buffer: deque = field(init=False)
    sums: list[int] = field(init=False)

    # Time:  O(C).  Space: O(window * C).
    # Data structures: deque(maxlen=window) ring buffer; a list of C running sums.
    def __post_init__(self) -> None:
        """Empty ring buffer and zero sums."""
        self.buffer = deque(maxlen=self.window)
        self.sums = [0] * self.channels

    # Time:  O(C) per sample: subtract the oldest and add the newest ciphertext (mod 2^64), no re-summing of the
    #        window.  Space: O(1) extra.
    # Data structures: the deque ring buffer; the list of running sums; a HeWindow dataclass out.
    def add(self, sample: HeSample) -> HeWindow | None:
        """Add one sample; return the window sums once the window is full."""
        if self.buffer and sample.n != self.buffer[-1].n + 1:   # > missing samples: start a new window
            self.__post_init__()
        if len(self.buffer) == self.window:
            oldest = self.buffer[0]
            for channel in range(self.channels):
                self.sums[channel] = (self.sums[channel] - oldest.ciphertexts[channel]) % HE_MODULUS
        self.buffer.append(sample)
        for channel in range(self.channels):
            self.sums[channel] = (self.sums[channel] + sample.ciphertexts[channel]) % HE_MODULUS
        if len(self.buffer) < self.window:
            return None
        return HeWindow(first_n=self.buffer[0].n, last_n=sample.n, t_us=sample.t_us, sums=tuple(self.sums))


# ---------------------------------------------------------------------------
# HeDecryptor  (the key holder)
# What:  turns an encrypted window sum into the RMS envelope:
#            sum(p) = (sum(c) - sum(k)) mod 2^64,   envelope = sqrt(sum(p) / 1000 / count)
# Time:  O(count * channels) AES blocks per window.  Space: O(count * channels).
# Data structures: NumPy uint64 keys; float result.  Files used: none.
# ---------------------------------------------------------------------------
@dataclass
class HeDecryptor:
    """Decrypts window sums with the homomorphic key."""

    he_key: bytes
    session: bytes
    channels: int = CHANNELS
    cfg: SecureConfig = SECURE_CONFIG

    # Time:  O(n * C) keystream blocks for the n samples of the window.  Space: O(n * C).
    # Data structures: NumPy uint64 key array summed with wrap-around; a list of ints out.
    def power_sums(self, window: HeWindow) -> list[int]:
        """sum(p) per channel (whole numbers of thousandths of mV^2)."""
        keys = he_keys(self.he_key, self.session, window.first_n, window.count, self.channels)
        key_sums = keys.sum(axis=0, dtype=np.uint64)            # > uint64: wraps modulo 2^64 by itself
        return [(window.sums[c] - int(key_sums[c])) % HE_MODULUS for c in range(self.channels)]

    # Time:  O(n * C) via power_sums, plus C square roots.  Space: O(C).
    # Data structures: a list of floats.
    def envelope_mv(self, window: HeWindow) -> list[float]:
        """RMS envelope (mV) of each channel over the window."""
        return [math.sqrt(total / self.cfg.power_scale / window.count) for total in self.power_sums(window)]


# ---------------------------------------------------------------------------
# aggregate_windows(ciphertexts, window)  (no key)  /  decrypt_windows(...)  (key holder)
# What:  the same two roles for a whole recording at once: sliding window sums of
#        the ciphertexts with a cumulative sum (each window = cs[i] - cs[i - window]),
#        then removing the keys the same way.
# Time:  O(N * channels).  Space: O(N * channels).
# Data structures: NumPy uint64 arrays; cumulative sums wrap modulo 2^64 by themselves.
# Files used: none.
# ---------------------------------------------------------------------------
def sliding_sums_u64(values: np.ndarray, window: int) -> np.ndarray:
    """Sums of every `window` consecutive rows, modulo 2^64 (row i = rows i-window+1 .. i)."""
    with np.errstate(over="ignore"):
        cumulative = np.cumsum(values.astype(np.uint64), axis=0, dtype=np.uint64)
        shifted = np.vstack([np.zeros((1, values.shape[1]), dtype=np.uint64), cumulative[:-window]])
        return cumulative[window - 1:] - shifted[:len(cumulative) - window + 1]


# Time:  O(N * C) (sliding sums from a cumulative sum).  Space: O(N * C).
# Data structures: NumPy uint64 arrays (sums wrap modulo 2^64 by themselves).
def aggregate_windows(ciphertexts: np.ndarray, window: int = SECURE_CONFIG.window) -> np.ndarray:
    """Encrypted window sums (N - window + 1 rows): no key needed."""
    return sliding_sums_u64(ciphertexts, window)


# Time:  O(N * C): one keystream block per sample and channel, then sliding sums.  Space: O(N * C).
# Data structures: NumPy uint64 and float64 arrays.
def decrypt_windows(encrypted_sums: np.ndarray, he_key: bytes, session: bytes, first_n: int, total: int,
                    window: int = SECURE_CONFIG.window, cfg: SecureConfig = SECURE_CONFIG) -> np.ndarray:
    """RMS envelope (mV) for every full window of samples first_n .. first_n+total-1."""
    keys = he_keys(he_key, session, first_n, total, encrypted_sums.shape[1])
    with np.errstate(over="ignore"):
        power = encrypted_sums - sliding_sums_u64(keys, window)   # > (sum c - sum k) mod 2^64 = sum p
    return np.sqrt(power.astype(np.float64) / cfg.power_scale / window)


# ---------------------------------------------------------------------------
# plain_envelope(values_mv, cfg)
# What:  the reference the homomorphic result must match: the same causal chain the
#        ESP32 runs (filter primed on the first sample, then sample by sample) and the
#        100 ms trailing RMS of src/myomotion/semg/dsp.py.
# Time:  O(N * S) for S = 6 filter sections.  Space: O(N).
# Data structures: EmgFilter (list of biquads), NumPy arrays.
# Files used: src/myomotion/semg/dsp.py (EmgFilter).
# ---------------------------------------------------------------------------
def plain_envelope(values_mv: np.ndarray, cfg: SecureConfig = SECURE_CONFIG) -> np.ndarray:
    """RMS envelope of every full window, computed in the clear."""
    chain = EmgFilter(cfg.rate_hz)
    chain.prime(float(values_mv[0]))
    filtered = np.array([chain.step(float(v)) for v in values_mv])
    squares = filtered * filtered
    cumulative = np.concatenate(([0.0], np.cumsum(squares)))
    return np.sqrt((cumulative[cfg.window:] - cumulative[:-cfg.window]) / cfg.window)


# ---------------------------------------------------------------------------
# SecureFrameDecoder
# What:  layer 1 on the laptop: bytes from the ESP32 in, the decrypted inner text out.
#        Reads the hello (session and key id), then opens each sealed batch:
#        checks that the counter only goes up (a replayed batch is rejected), checks
#        the GCM tag (a changed or forged batch is rejected) and decrypts. Clear-text
#        sample lines are ignored: nobody can inject data by sending plain D lines.
# Time:  O(b) for b bytes.  Space: O(b).
# Data structures: bytes buffer for an unfinished line; counters (ints).
# ---------------------------------------------------------------------------
@dataclass
class SecureFrameDecoder:
    """Bytes from the secure firmware -> verified, decrypted text lines."""

    keys: LinkKeys
    cfg: SecureConfig = SECURE_CONFIG
    session: bytes | None = None
    frames_ok: int = 0
    frames_rejected: int = 0
    plain_lines_ignored: int = 0
    _last_counter: int = field(default=-1, repr=False)
    _pending: bytes = field(default=b"", repr=False)

    # Time:  O(1).
    # Data structures: an AESGCM object holding the link key.
    def __post_init__(self) -> None:
        """AES-GCM with the link key."""
        self._aead = AESGCM(self.keys.link_key)

    # Time:  O(b) for b received bytes (split once, every line opened once).  Space: O(b).
    # Data structures: a bytes buffer for the unfinished line; a list of byte lines; a list of decrypted batches.
    def feed(self, chunk: bytes) -> bytes:
        """Add received bytes; return the decrypted text of every complete, valid batch."""
        lines = (self._pending + chunk).split(b"\n")
        self._pending = lines.pop()                      # > the last piece has no newline yet: keep it
        opened: list[bytes] = []
        for raw in lines:
            line = raw.strip()
            if line.startswith(b"H,"):
                self._hello(line.decode("ascii", errors="replace"))
            elif line.startswith(b"X,"):
                plain = self._open(line)
                if plain is not None:
                    opened.append(plain)
            elif line:
                self.plain_lines_ignored += 1
        return b"".join(opened)

    # Time:  O(1) (4 fields).  Space: O(1).
    # Data structures: str.split list; the session id as bytes.
    def _hello(self, text: str) -> None:
        """New session: remember its nonce prefix and check the key id."""
        fields = text.split(",")
        if len(fields) < 4:
            return
        if fields[2] != self.keys.key_id:
            raise KeyMismatch(f"the ESP32 has keys {fields[2]} but this laptop's key file has {self.keys.key_id}: "
                              "flash the firmware with the secrets.h made together with this key file")
        session = bytes.fromhex(fields[1])
        if session != self.session:                     # > a repeated hello of the same session changes nothing
            self.session = session
            self._last_counter = -1

    # Time:  O(m) for an m-byte batch (base64 decode + AES-GCM).  Space: O(m).
    # Data structures: bytes; an int counter for the replay check.
    def _open(self, line: bytes) -> bytes | None:
        """Verify and decrypt one X line; None (and counted) if it is not genuine."""
        try:
            _, counter_text, cipher_text, tag_text = line.decode("ascii").split(",")
            counter = int(counter_text)
            sealed = base64.b64decode(cipher_text, validate=True) + base64.b64decode(tag_text, validate=True)
        except (ValueError, UnicodeDecodeError):
            self.frames_rejected += 1
            return None
        if self.session is None or counter <= self._last_counter:   # > no hello yet, or a replayed batch
            self.frames_rejected += 1
            return None
        try:
            plain = self._aead.decrypt(self.session + counter.to_bytes(4, "big"), sealed, self.cfg.aad)
        except InvalidTag:                                           # > changed, forged or wrong key
            self.frames_rejected += 1
            return None
        self._last_counter = counter
        self.frames_ok += 1
        return plain


# ---------------------------------------------------------------------------
# split_he_lines(text)
# What:  separates the P lines (homomorphic ciphertexts) from the rest of a decrypted
#        batch, so the D and E lines can go to the normal LineParser unchanged.
# Time:  O(b).  Space: O(b).
# Data structures: lists of lines.  Files used: none.
# ---------------------------------------------------------------------------
def split_he_lines(text: bytes) -> tuple[bytes, list[HeSample]]:
    """(the other lines as bytes, the HeSamples)."""
    others: list[bytes] = []
    samples: list[HeSample] = []
    for line in text.split(b"\n"):
        if line.startswith(b"P,"):
            sample = parse_he_line(line.decode("ascii", errors="replace").strip())
            if sample is not None:
                samples.append(sample)
        elif line:
            others.append(line + b"\n")
    return b"".join(others), samples


# ---------------------------------------------------------------------------
# ReferenceEncoder
# What:  the ESP32 firmware's sending side in Python (same protocol, same filter,
#        same keys): used by the tests and by a fake secure ESP32. The C++ version is
#        hardware/myomotion_emg_secure/secure_core.h; tests/test_secure_link.py checks
#        both against this file's decoder.
# Time:  O(S + channels) per sample (filters + one AES block per channel).
# Space: O(batch).
# Data structures: one EmgFilter per channel; AES-ECB and AES-GCM objects; counters.
# ---------------------------------------------------------------------------
@dataclass
class ReferenceEncoder:
    """Python twin of the secure firmware's encoder."""

    keys: LinkKeys
    session: bytes = field(default_factory=lambda: secrets.token_bytes(SESSION_BYTES))
    cfg: SecureConfig = SECURE_CONFIG
    sample_number: int = 0
    frame_counter: int = 0

    # Time:  O(C * S) to build C filter chains of S sections.  Space: O(C * S).
    # Data structures: a list of EmgFilter; an AESGCM object.
    def __post_init__(self) -> None:
        """Filters (primed on the first sample) and the two AES objects."""
        self.filters = [EmgFilter(self.cfg.rate_hz) for _ in range(CHANNELS)]
        self.primed = False
        self._aead = AESGCM(self.keys.link_key)

    # Time:  O(1).
    # Data structures: a str encoded to bytes.
    def hello(self) -> bytes:
        """The clear-text hello line."""
        return f"H,{self.session.hex()},{self.keys.key_id},{self.cfg.version}\n".encode()

    # Time:  O(C * S) per sample: filter, square and encrypt each of the C channels.  Space: O(C).
    # Data structures: a list of ciphertext ints; two text lines out.
    def sample_lines(self, t_us: int, values_mv: tuple[float, ...]) -> str:
        """D line + P line for one sample (filter -> square -> encrypt)."""
        if not self.primed:
            for chain, value in zip(self.filters, values_mv):
                chain.prime(float(value))
            self.primed = True
        keys = he_keys(self.keys.he_key, self.session, self.sample_number, 1)[0]
        ciphertexts = []
        for channel, (chain, value) in enumerate(zip(self.filters, values_mv)):
            power = he_power(chain.step(float(value)), self.cfg)
            ciphertexts.append((power + int(keys[channel])) % HE_MODULUS)
        d_line = f"D,{t_us % 2 ** 32}," + ",".join(str(int(v)) for v in values_mv) + "\n"
        p_line = format_he_line(self.sample_number, t_us % 2 ** 32, ciphertexts)
        self.sample_number += 1
        return d_line + p_line

    # Time:  O(m) for an m-byte text (AES-GCM + base64).  Space: O(m).
    # Data structures: bytes; one X line out.
    def seal(self, text: str) -> bytes:
        """One X line: AES-GCM over the text, counter in the nonce."""
        nonce = self.session + self.frame_counter.to_bytes(4, "big")
        sealed = self._aead.encrypt(nonce, text.encode(), self.cfg.aad)
        cipher, tag = sealed[:-TAG_BYTES], sealed[-TAG_BYTES:]
        line = (f"X,{self.frame_counter},{base64.b64encode(cipher).decode()},"
                f"{base64.b64encode(tag).decode()}\n").encode()
        self.frame_counter += 1
        return line
