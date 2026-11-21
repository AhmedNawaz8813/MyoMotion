"""Make the two secret keys for the encrypted ESP32 link (once, before flashing the secure firmware).

Usage, from the repo root:
    python scripts/semg/make_link_keys.py
    python scripts/semg/make_link_keys.py --force        # replace existing keys (then re-flash the ESP32!)

Writes the same two random 128-bit keys to:
    ~/.config/myomotion/keys.json                     the laptop's copy (permission 600, outside the repo)
    hardware/myomotion_emg_secure/secrets.h           the firmware's copy (git-ignored: never commit it)
The key id printed at the end is a fingerprint, safe to show: the ESP32 sends it in its hello line and the
laptop refuses to talk to a board whose id differs. Thin wrapper: the logic is in
src/myomotion/semg/secure_link.py (make_keys, save_keys, LinkKeys.key_id).

Data structures (this file): a LinkKeys dataclass of bytes; a dict for keys.json; text for secrets.h.
Time complexity (this file): O(1): two 16-byte random keys and one SHA-256.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

# > This file is scripts/semg/make_link_keys.py, so parents[2] is the repo root; adding <repo>/src to
# > sys.path lets `import myomotion` work without `pip install -e .` (as in Ahmad's scripts).
_REPO = Path(__file__).resolve().parents[2]
_SRC = str(_REPO / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from myomotion.semg.secure_link import SECURE_CONFIG, LinkKeys, make_keys, save_keys  # noqa: E402

SECRETS_HEADER = _REPO / "hardware" / "myomotion_emg_secure" / "secrets.h"
DEFAULT_WIFI_PASSWORD = "myomotion2026"      # the Wi-Fi password the laptop already knows (WPA2, 8+ characters)


# ---------------------------------------------------------------------------
# c_array(data)
# What:  bytes -> "0x12, 0xab, ..." for a C array.
# Time:  O(n).  Space: O(n).
# Data structures: str join.  Files used: none.
# ---------------------------------------------------------------------------
def c_array(data: bytes) -> str:
    """C initialiser for a byte array."""
    return ", ".join(f"0x{byte:02x}" for byte in data)


# ---------------------------------------------------------------------------
# secrets_header(keys, wifi_password)
# What:  the text of secrets.h for the firmware.
# Time:  O(1).  Space: O(1).
# Data structures: str.  Files used: none.
# ---------------------------------------------------------------------------
def secrets_header(keys: LinkKeys, wifi_password: str) -> str:
    """Contents of hardware/myomotion_emg_secure/secrets.h."""
    return (
        "// secrets.h: made by scripts/semg/make_link_keys.py on "
        f"{datetime.now().isoformat(timespec='seconds')}. NEVER commit this file (it is git-ignored).\n"
        "// The laptop's copy of the same keys is ~/.config/myomotion/keys.json.\n"
        "#pragma once\n"
        "#include <stdint.h>\n\n"
        f'static const char* const WIFI_PASSWORD_SECRET = "{wifi_password}";\n'
        f"static const uint8_t LINK_KEY[16] = {{{c_array(keys.link_key)}}};\n"
        f"static const uint8_t HE_KEY[16] = {{{c_array(keys.he_key)}}};\n"
        f'static const char* const KEY_ID = "{keys.key_id}";\n'
    )


# ---------------------------------------------------------------------------
# main()
# What:  refuses to replace existing keys without --force (the board and the laptop
#        would no longer match), then writes both copies.
# Time / Space: O(1).
# Data structures: argparse.Namespace.
# Files used: writes the key file and secrets.h (see the docstring).
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Make the keys for the encrypted ESP32 link")
    parser.add_argument("--key-file", type=Path, default=SECURE_CONFIG.key_file, help="laptop key file")
    parser.add_argument("--header", type=Path, default=SECRETS_HEADER, help="firmware secrets.h")
    parser.add_argument("--wifi-password", default=DEFAULT_WIFI_PASSWORD, help="ESP32 Wi-Fi password (8+ chars)")
    parser.add_argument("--force", action="store_true", help="replace existing keys")
    args = parser.parse_args(argv)

    key_file, header = args.key_file.expanduser(), args.header.expanduser()
    if len(args.wifi_password) < 8:
        print("The Wi-Fi password needs at least 8 characters (WPA2).")
        return 1
    existing = [str(p) for p in (key_file, header) if p.exists()]
    if existing and not args.force:
        print("Keys already exist: " + ", ".join(existing) + "\nUse --force to replace them, then re-flash the "
              "ESP32 (the old firmware would no longer match).")
        return 1

    keys = make_keys()
    save_keys(keys, key_file)
    header.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(header, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "w") as header_file:
        header_file.write(secrets_header(keys, args.wifi_password))
    print(f"Key id {keys.key_id}\n  laptop:   {key_file}\n  firmware: {header}\n"
          "Now flash hardware/myomotion_emg_secure/myomotion_emg_secure.ino to the ESP32.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
