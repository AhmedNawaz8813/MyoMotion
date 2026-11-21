"""Tests for the encrypted link and the homomorphic envelope (src/myomotion/semg/secure_link.py), the firmware's
C++ core (hardware/myomotion_emg_secure/secure_core.h) and the scripts make_link_keys.py / check_he.py.

Run from the repo root:  python -m pytest tests/test_secure_link.py -q
No hardware: the ESP32 is played by ReferenceEncoder (the firmware's Python twin) and, when g++ and OpenSSL are
installed, by the firmware's own C++ code compiled for this computer.
"""

from __future__ import annotations

import csv
import json
import shutil
import socket
import stat
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np
import pytest

from myomotion.semg.acquisition import AcquisitionConfig, EspStream, LineParser
from myomotion.semg.secure_link import (HE_MODULUS, HeAggregator, HeDecryptor, HeSample, KeyMismatch, LinkKeys,
                                        ReferenceEncoder, SecureFrameDecoder, aggregate_windows, decrypt_windows,
                                        load_keys, make_keys, plain_envelope, save_keys, split_he_lines)

REPO = Path(__file__).resolve().parents[1]
RATE_HZ = 500.0
SESSION = bytes.fromhex("0123456789abcdef")


def emg_like(seconds: float, seed: int = 0) -> np.ndarray:
    """Raw-EMG-like millivolts: offset, 50 Hz hum, rest noise and one strong burst in the middle."""
    rng = np.random.default_rng(seed)
    t_s = np.arange(round(seconds * RATE_HZ)) / RATE_HZ
    size = np.where((t_s > seconds / 3) & (t_s < 2 * seconds / 3), 150.0, 5.0)
    return np.clip(np.round(650 + 70 * np.sin(2 * np.pi * 50 * t_s) + size * rng.standard_normal(len(t_s))),
                   142, 3100)


def encode(keys: LinkKeys, values: np.ndarray, batch: int = 10, session: bytes = SESSION) -> bytes:
    """What the secure ESP32 would send for these samples (hello + sealed batches of 10)."""
    encoder = ReferenceEncoder(keys, session=session)
    wire = encoder.hello()
    for start in range(0, len(values), batch):
        text = "".join(encoder.sample_lines(900_000_000 + 2000 * k, (float(values[k]), 150.0, 150.0))
                       for k in range(start, min(len(values), start + batch)))
        wire += encoder.seal(text)
    return wire


def test_keys_save_load_and_stay_private(tmp_path: Path) -> None:
    keys = make_keys()
    path = save_keys(keys, tmp_path / "keys.json")
    assert stat.S_IMODE(path.stat().st_mode) == 0o600                  # > only the owner can read the keys
    assert load_keys(path) == keys and len(keys.key_id) == 8
    assert make_keys().key_id != keys.key_id


def test_link_decrypts_everything_and_rejects_tampered_replayed_and_plain_data() -> None:
    keys = make_keys()
    values = emg_like(2.0)
    wire = encode(keys, values)
    decoder = SecureFrameDecoder(keys)
    others, he = split_he_lines(decoder.feed(wire[:777]) + decoder.feed(wire[777:]))   # > split anywhere
    samples = LineParser().feed(others)
    assert [s.values_mv[0] for s in samples] == list(values) and len(he) == len(values)
    assert decoder.frames_ok == len(values) // 10 and decoder.frames_rejected == 0

    lines = wire.split(b"\n")
    tampered = bytearray(lines[3])
    tampered[12] = ord("A") if tampered[12] != ord("A") else ord("B")       # > one changed character
    attacker = SecureFrameDecoder(keys)
    attacker.feed(lines[0] + b"\n" + bytes(tampered) + b"\n" + lines[2] + b"\n" + lines[2] + b"\n"
                  + b"D,1,2,3,4\n")
    assert attacker.frames_ok == 1                                           # > only the genuine batch
    assert attacker.frames_rejected == 2                                     # > the changed one and the replay
    assert attacker.plain_lines_ignored == 1                                 # > injected clear-text sample


def test_wrong_keys_are_refused_with_an_explanation() -> None:
    wire = encode(make_keys(), emg_like(0.2))
    with pytest.raises(KeyMismatch, match="flash the firmware"):
        SecureFrameDecoder(make_keys()).feed(wire)


def test_aggregator_needs_no_key_and_the_key_holder_gets_the_plain_envelope() -> None:
    keys = make_keys()
    values = emg_like(3.0, seed=1)
    _, he = split_he_lines(SecureFrameDecoder(keys).feed(encode(keys, values)))
    aggregator = HeAggregator()                                   # > constructed without any key
    windows = [w for w in (aggregator.add(sample) for sample in he) if w is not None]
    decryptor = HeDecryptor(keys.he_key, SESSION)
    live = np.array([decryptor.envelope_mv(w)[0] for w in windows[::25]])
    plain = plain_envelope(values)
    assert len(windows) == len(plain)
    assert np.max(np.abs(live - plain[::25])) < 1e-3                         # > mV
    bulk = decrypt_windows(aggregate_windows(np.array([s.ciphertexts for s in he], dtype=np.uint64)),
                           keys.he_key, SESSION, he[0].n, len(he))
    assert np.max(np.abs(bulk[:, 0] - plain)) < 1e-3
    assert plain.max() > 100 > 10 > plain.min()                              # > the burst is really there


def test_equal_values_give_unrelated_ciphertexts() -> None:
    """A constant input (zero after the filter) still gives a fresh-looking ciphertext every sample."""
    keys = make_keys()
    _, he = split_he_lines(SecureFrameDecoder(keys).feed(encode(keys, np.full(200, 700.0))))
    ciphertexts = [s.ciphertexts[0] for s in he]
    assert len(set(ciphertexts)) == len(ciphertexts)
    assert all(0 <= c < HE_MODULUS for c in ciphertexts)


def test_aggregator_starts_again_after_a_missing_batch() -> None:
    aggregator = HeAggregator(window=3, channels=1)
    results = [aggregator.add(HeSample(n, 0, (n,))) for n in (0, 1, 2, 3, 7, 8, 9)]
    assert results[2].first_n == 0 and results[3].first_n == 1
    assert results[4] is None and results[5] is None                         # > gap: window refills first
    assert results[6].first_n == 7 and results[6].sums == (24,)


def test_firmware_core_in_cpp_matches_python_exactly(tmp_path: Path) -> None:
    """Compiles hardware/myomotion_emg_secure/secure_core.h for this PC (OpenSSL instead of the ESP32's AES)."""
    if shutil.which("g++") is None or not Path("/usr/include/openssl/evp.h").exists():
        pytest.skip("needs g++ and the OpenSSL headers")
    harness = tmp_path / "harness"
    subprocess.run(["g++", "-std=c++17", "-O2", "-I", str(REPO / "hardware" / "myomotion_emg_secure"),
                    str(REPO / "tests" / "firmware" / "secure_core_harness.cpp"), "-lcrypto", "-o", str(harness)],
                   check=True, capture_output=True)
    keys = make_keys()
    values = emg_like(2.0, seed=2)
    rows = "".join(f"{900_000_000 + 2000 * k},{int(values[k])},150,150\n" + ("E,900002000\n" if k == 1 else "")
                   for k in range(len(values)))
    wire = subprocess.run([str(harness), keys.link_key.hex(), keys.he_key.hex(), keys.key_id, SESSION.hex()],
                          input=rows.encode(), capture_output=True, check=True).stdout
    decoder = SecureFrameDecoder(keys)
    others, he = split_he_lines(decoder.feed(wire))
    assert decoder.frames_rejected == 0 and others.count(b"E,") == 1
    assert [s.values_mv[0] for s in LineParser().feed(others)] == list(values)
    _, reference = split_he_lines(SecureFrameDecoder(keys).feed(encode(keys, values)))
    assert [s.ciphertexts for s in he] == [s.ciphertexts for s in reference]   # > bit for bit


def test_esp_stream_decrypts_a_fake_secure_esp32() -> None:
    keys = make_keys()
    wire = encode(keys, emg_like(0.4))
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(1)

    def serve() -> None:
        connection, _ = server.accept()
        connection.sendall(wire)
        connection.close()

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    samples, he = [], []
    with EspStream(AcquisitionConfig(host="127.0.0.1", port=server.getsockname()[1], read_timeout_s=0.5),
                   keys=keys) as stream:
        with pytest.raises(ConnectionError):
            for _ in range(50):
                samples += stream.read()
                he += stream.last_he_samples
    thread.join(timeout=2)
    server.close()
    assert len(samples) == len(he) == 200 and stream.decoder.frames_rejected == 0


def test_make_link_keys_writes_both_copies_and_will_not_overwrite(tmp_path: Path) -> None:
    script = [sys.executable, str(REPO / "scripts" / "semg" / "make_link_keys.py"),
              "--key-file", str(tmp_path / "keys.json"), "--header", str(tmp_path / "secrets.h")]
    first = subprocess.run(script, capture_output=True, text=True)
    assert first.returncode == 0, first.stdout + first.stderr
    keys = load_keys(tmp_path / "keys.json")
    header = (tmp_path / "secrets.h").read_text()
    assert f'KEY_ID = "{keys.key_id}"' in header and keys.link_key.hex()[:2] in header.replace("0x", "")
    assert subprocess.run(script, capture_output=True, text=True).returncode == 1          # > refuses
    assert subprocess.run(script + ["--force"], capture_output=True, text=True).returncode == 0
    assert load_keys(tmp_path / "keys.json") != keys


def test_check_he_script_passes_on_a_recording(tmp_path: Path) -> None:
    """A folder as record_session.py --secure writes it, checked by scripts/semg/check_he.py."""
    keys = make_keys()
    save_keys(keys, tmp_path / "keys.json")
    values = emg_like(3.0, seed=3)
    decoder = SecureFrameDecoder(keys)
    others, he = split_he_lines(decoder.feed(encode(keys, values)))
    folder = tmp_path / "S01_forearm_fist"
    folder.mkdir()
    with (folder / "emg.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["pc_time_s", "t_us", "ch1", "ch2", "ch3"])
        writer.writerows([0, s.t_us, *[int(v) for v in s.values_mv]] for s in LineParser().feed(others))
    with (folder / "he.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["pc_time_s", "n", "t_us", "c1", "c2", "c3"])
        writer.writerows([0, s.n, s.t_us, *[f"{c:016x}" for c in s.ciphertexts]] for s in he)
    (folder / "link.json").write_text(json.dumps({"session": SESSION.hex(), "key_id": keys.key_id,
                                                  "batches_verified": decoder.frames_ok, "batches_rejected": 0}))
    run = subprocess.run([sys.executable, str(REPO / "scripts" / "semg" / "check_he.py"), str(folder),
                          "--keys", str(tmp_path / "keys.json")], capture_output=True, text=True)
    assert run.returncode == 0, run.stdout + run.stderr
    assert "PASS" in run.stdout and (folder / "he.png").exists()
    assert json.loads((folder / "he.json").read_text())["max_difference_mv"] < 1e-3
