// secure_core.h: the encrypted link and the homomorphic envelope, in plain C++ (no Arduino code here).
//
// Used by myomotion_emg_secure.ino on the ESP32, and by tests/firmware/secure_core_harness.cpp on a laptop,
// so the exact code that runs on the board is also tested on a PC (tests/test_secure_link.py).
// The laptop side is src/myomotion/semg/secure_link.py; both follow the same protocol:
//
//   H,<session hex>,<key id>,<version>              hello, in clear, once per connection
//   X,<counter>,<base64 ciphertext>,<base64 tag>     one batch sealed with AES-128-GCM
//     nonce = 8-byte random session + 4-byte counter (big-endian); extra signed data = "MyoMotion-X1"
//   inside a batch:  D,<t_us>,<ch1>,<ch2>,<ch3>      raw millivolts (as before)
//                    E,<t_us>,LED_ON                 sync flash
//                    P,<n>,<t_us>,<c1>,<c2>,<c3>     homomorphic ciphertexts, 16 hex digits each
//
// Homomorphic scheme (Castelluccia, Mykletun, Tsudik 2005): c = (p + k) mod 2^64 with
//   p = (filtered sample)^2 * 1000 as a whole number, filter = 20 Hz high-pass + 50/100/150/200 Hz notches
//       (identical to src/myomotion/semg/dsp.py, double precision)
//   k = first 8 bytes (little-endian) of AES(HE key, session | n (4 bytes, big-endian) | channel | 0 0 0)
// Adding ciphertexts adds the hidden values, so the laptop can sum a 100 ms window without the key.
//
// The two AES operations are given from outside (BlockCipher, AeadCipher): the ESP32 uses mbedtls with
// its hardware AES, the PC test uses OpenSSL. Encryption primitives are never home-made.
//
// Data structures (this file): Biquad structs (5 coefficients + 2 state values); a fixed array of Biquads
//   per channel (EmgFilterChain; no heap on the microcontroller); fixed char / byte arrays for lines, nonces
//   and tags; two abstract interfaces (BlockCipher, AeadCipher) so the same encoder runs on mbedtls (ESP32)
//   and OpenSSL (PC test).
// Time complexity (this file): per sample O(C * (S + 1)): S = 6 filter steps and one AES block for each of
//   the C = 3 channels; per batch O(m) for AES-GCM and base64 of the m-byte batch text; the rest is O(1).

#pragma once

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

namespace myomotion {

// ---------------------------------------------------------------------------
// Protocol constants (the same numbers as SecureConfig in secure_link.py)
// ---------------------------------------------------------------------------
const char* const PROTOCOL_VERSION = "1";
const char* const LINK_AAD = "MyoMotion-X1";
const double POWER_SCALE = 1000.0;          // p = y^2 in thousandths of mV^2
const double RATE_HZ = 500.0;               // the filter is designed for 500 samples per second
const int EMG_CHANNELS = 3;
const int LINK_SESSION_BYTES = 8;
const int LINK_TAG_BYTES = 16;
const int LINK_NONCE_BYTES = 12;

// Filter design (the same as DspConfig in dsp.py)
const double HIGHPASS_HZ = 20.0;
const double MAINS_HZ = 50.0;
const int MAINS_HARMONICS = 4;
const double NOTCH_BANDWIDTH_HZ = 4.0;
const int MAX_SECTIONS = 2 + MAINS_HARMONICS;
const double PI_VALUE = 3.14159265358979323846;

// ---------------------------------------------------------------------------
// Biquad
// What:  one second-order filter section, Direct Form II Transposed, sample by sample.
//        Same equations as Biquad in dsp.py (step, dc_gain, prime).
// Time:  O(1) per sample.  Space: O(1): five coefficients and two memory values.
// Data structures: a plain struct of seven doubles.
// ---------------------------------------------------------------------------
struct Biquad {
  double b0 = 1, b1 = 0, b2 = 0, a1 = 0, a2 = 0;
  double z1 = 0, z2 = 0;

  // Time O(1): three multiply-adds per output.
  double step(double x) {
    double y = b0 * x + z1;
    z1 = b1 * x - a1 * y + z2;
    z2 = b2 * x - a2 * y;
    return y;
  }

  // Time O(1): the gain for a constant input.
  double dcGain() const { return (b0 + b1 + b2) / (1 + a1 + a2); }

  // Start as if the input had always been x0; returns the steady output.
  // Time O(1).
  double prime(double x0) {
    double y0 = dcGain() * x0;
    z2 = b2 * x0 - a2 * y0;
    z1 = b1 * x0 - a1 * y0 + z2;
    return y0;
  }
};

// RBJ cookbook high-pass section (as highpass_biquad in dsp.py).
// Time O(1) (one sin, one cos).  Data structures: returns a Biquad struct by value.
inline Biquad highpassBiquad(double cutoffHz, double rateHz, double q) {
  double w0 = 2 * PI_VALUE * cutoffHz / rateHz;
  double alpha = sin(w0) / (2 * q);
  double cosW0 = cos(w0);
  double a0 = 1 + alpha;
  Biquad s;
  s.b0 = (1 + cosW0) / 2 / a0;
  s.b1 = -(1 + cosW0) / a0;
  s.b2 = (1 + cosW0) / 2 / a0;
  s.a1 = -2 * cosW0 / a0;
  s.a2 = (1 - alpha) / a0;
  return s;
}

// RBJ cookbook notch section (as notch_biquad in dsp.py).
// Time O(1) (one sin, one cos).  Data structures: returns a Biquad struct by value.
inline Biquad notchBiquad(double centreHz, double rateHz, double bandwidthHz) {
  double w0 = 2 * PI_VALUE * centreHz / rateHz;
  double alpha = sin(w0) / (2 * (centreHz / bandwidthHz));
  double cosW0 = cos(w0);
  double a0 = 1 + alpha;
  Biquad s;
  s.b0 = 1 / a0;
  s.b1 = -2 * cosW0 / a0;
  s.b2 = 1 / a0;
  s.a1 = -2 * cosW0 / a0;
  s.a2 = (1 - alpha) / a0;
  return s;
}

// ---------------------------------------------------------------------------
// EmgFilterChain
// What:  4th-order Butterworth high-pass (two sections) + one notch per mains harmonic below
//        Nyquist: the same chain as EmgFilter in dsp.py.
// Time:  O(S) per sample for S = 6 sections.  Space: O(S).
// Data structures: fixed array of Biquads (no heap on the microcontroller).
// ---------------------------------------------------------------------------
struct EmgFilterChain {
  Biquad sections[MAX_SECTIONS];
  int count = 0;

  // Time O(S): one section per high-pass stage and per mains harmonic below Nyquist.
  void design(double rateHz) {
    count = 0;
    const double qs[2] = {1 / (2 * cos(PI_VALUE / 8)), 1 / (2 * cos(3 * PI_VALUE / 8))};
    for (int i = 0; i < 2; i++) {
      sections[count++] = highpassBiquad(HIGHPASS_HZ, rateHz, qs[i]);
    }
    for (int harmonic = 1; harmonic <= MAINS_HARMONICS; harmonic++) {
      double centreHz = harmonic * MAINS_HZ;
      if (centreHz < rateHz / 2) {
        sections[count++] = notchBiquad(centreHz, rateHz, NOTCH_BANDWIDTH_HZ);
      }
    }
  }

  // Time O(S): each section is primed with the steady output of the one before it.
  void prime(double x0) {
    for (int i = 0; i < count; i++) {
      x0 = sections[i].prime(x0);
    }
  }

  // Time O(S): the sample passes through the sections in order.
  double step(double x) {
    for (int i = 0; i < count; i++) {
      x = sections[i].step(x);
    }
    return x;
  }
};

// The value that gets encrypted: round(y^2 * 1000), the same rounding as he_power in secure_link.py.
// Time O(1).  Data structures: a double in, a uint64_t out.
inline uint64_t hePower(double yMv) {
  return (uint64_t)floor(yMv * yMv * POWER_SCALE + 0.5);
}

// ---------------------------------------------------------------------------
// Text helpers (our own, so they work the same on every printf implementation)
// ---------------------------------------------------------------------------

// 16 lowercase hex digits of a 64-bit number. Time O(16).
// Data structures: a 16-character lookup string; a 17-byte char array out (16 digits + NUL).
inline void hex64(uint64_t value, char out[17]) {
  const char* digits = "0123456789abcdef";
  for (int i = 15; i >= 0; i--) {
    out[i] = digits[value & 0xF];
    value >>= 4;
  }
  out[16] = '\0';
}

// Base64 (RFC 4648, with = padding). Returns the length written, or 0 if it does not fit.
// Time O(n). Every 3 bytes become 4 characters from a 64-character alphabet.
// Data structures: a 64-character lookup table; each 3-byte group held as a 24-bit number in a uint32_t.
inline size_t base64Encode(const uint8_t* data, size_t length, char* out, size_t capacity) {
  const char* alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  size_t needed = 4 * ((length + 2) / 3);
  if (needed + 1 > capacity) {
    return 0;
  }
  size_t o = 0;
  for (size_t i = 0; i < length; i += 3) {
    uint32_t chunk = (uint32_t)data[i] << 16;
    if (i + 1 < length) chunk |= (uint32_t)data[i + 1] << 8;
    if (i + 2 < length) chunk |= (uint32_t)data[i + 2];
    out[o++] = alphabet[(chunk >> 18) & 0x3F];
    out[o++] = alphabet[(chunk >> 12) & 0x3F];
    out[o++] = (i + 1 < length) ? alphabet[(chunk >> 6) & 0x3F] : '=';
    out[o++] = (i + 2 < length) ? alphabet[chunk & 0x3F] : '=';
  }
  out[o] = '\0';
  return o;
}

// ---------------------------------------------------------------------------
// Crypto interfaces, implemented by the caller (mbedtls on the ESP32, OpenSSL on a PC)
// ---------------------------------------------------------------------------
// BlockCipher: interface (one pure virtual function).  Time O(1) per 16-byte block.
class BlockCipher {                      // AES-128 on one 16-byte block, key already loaded
 public:
  virtual ~BlockCipher() {}
  virtual void encryptBlock(const uint8_t in[16], uint8_t out[16]) = 0;
};

// AeadCipher: interface (one pure virtual function).  Time O(m) to seal m bytes.
class AeadCipher {                       // AES-128-GCM, key already loaded
 public:
  virtual ~AeadCipher() {}
  virtual bool seal(const uint8_t nonce[LINK_NONCE_BYTES], const uint8_t* aad, size_t aadLength,
                    const uint8_t* plain, size_t length, uint8_t* cipher, uint8_t tag[LINK_TAG_BYTES]) = 0;
};

// ---------------------------------------------------------------------------
// SecureEncoder
// What:  everything the ESP32 sends, built here: the hello line, the D + P lines of each sample
//        (filter -> square -> encrypt with a fresh homomorphic key) and the sealed X lines.
// Time:  per sample O(S * EMG_CHANNELS) filter steps + EMG_CHANNELS AES blocks; per batch one AES-GCM pass.
// Space: O(1) apart from the caller's buffers.
// Data structures: one EmgFilterChain per channel; counters; the session bytes.
// ---------------------------------------------------------------------------
class SecureEncoder {
 public:
  // Time O(1).  Data structures: references to the two ciphers; the key id string (not copied).
  SecureEncoder(BlockCipher& heCipher, AeadCipher& linkCipher, const char* keyId)
      : heCipher_(heCipher), linkCipher_(linkCipher), keyId_(keyId) {}

  // A new connection: new random session, counters and filters start again.
  // Time O(S * C) to design the C filter chains.  Data structures: the 8-byte session array; the chains.
  void begin(const uint8_t session[LINK_SESSION_BYTES]) {
    memcpy(session_, session, LINK_SESSION_BYTES);
    sampleNumber_ = 0;
    frameCounter_ = 0;
    primed_ = false;
    for (int c = 0; c < EMG_CHANNELS; c++) {
      filters_[c].design(RATE_HZ);
    }
  }

  // H,<session hex>,<key id>,<version>\n  Returns the length, 0 if it does not fit.
  // Time O(1) (8 session bytes -> 16 hex digits).  Data structures: a 17-byte char array on the stack.
  size_t helloLine(char* out, size_t capacity) const {
    char sessionHex[2 * LINK_SESSION_BYTES + 1];
    for (int i = 0; i < LINK_SESSION_BYTES; i++) {
      snprintf(sessionHex + 2 * i, 3, "%02x", session_[i]);
    }
    int length = snprintf(out, capacity, "H,%s,%s,%s\n", sessionHex, keyId_, PROTOCOL_VERSION);
    return (length > 0 && (size_t)length < capacity) ? (size_t)length : 0;
  }

  // The homomorphic key of sample n, channel c (first 8 bytes of the AES output, little-endian).
  // Time O(1): one AES block (fixed 10 rounds).  Data structures: two 16-byte arrays (counter block, output).
  uint64_t heKey(uint32_t n, int channel) {
    uint8_t block[16] = {0};
    uint8_t out[16];
    memcpy(block, session_, LINK_SESSION_BYTES);
    block[8] = (uint8_t)(n >> 24);
    block[9] = (uint8_t)(n >> 16);
    block[10] = (uint8_t)(n >> 8);
    block[11] = (uint8_t)n;
    block[12] = (uint8_t)channel;
    heCipher_.encryptBlock(block, out);
    uint64_t key = 0;
    for (int i = 7; i >= 0; i--) {
      key = (key << 8) | out[i];
    }
    return key;
  }

  // Appends the D and P lines of one sample to the batch text. False (nothing added) if it does not fit.
  // Time O(C * (S + 1)) = O(1) for fixed C and S: per channel S filter steps, one AES block, 16 hex digits.
  // Data structures: a 160-byte line array on the stack; the caller's batch text (appended in place).
  bool appendSample(char* text, size_t& used, size_t capacity, uint32_t timeUs, const uint16_t mv[EMG_CHANNELS]) {
    char line[160];
    int length = snprintf(line, sizeof(line), "D,%lu,%u,%u,%u\nP,%lu,%lu", (unsigned long)timeUs,
                          (unsigned)mv[0], (unsigned)mv[1], (unsigned)mv[2], (unsigned long)sampleNumber_,
                          (unsigned long)timeUs);
    if (length <= 0) {
      return false;
    }
    size_t total = (size_t)length + EMG_CHANNELS * 17 + 1;
    if (used + total >= capacity) {
      return false;
    }
    if (!primed_) {                                  // first sample of the session: no start-up jump
      for (int c = 0; c < EMG_CHANNELS; c++) {
        filters_[c].prime((double)mv[c]);
      }
      primed_ = true;
    }
    memcpy(text + used, line, (size_t)length);
    used += (size_t)length;
    for (int c = 0; c < EMG_CHANNELS; c++) {
      uint64_t cipher = hePower(filters_[c].step((double)mv[c])) + heKey(sampleNumber_, c);   // wraps mod 2^64
      char hex[17];
      hex64(cipher, hex);
      text[used++] = ',';
      memcpy(text + used, hex, 16);
      used += 16;
    }
    text[used++] = '\n';
    text[used] = '\0';
    sampleNumber_++;
    return true;
  }

  // Appends any ready-made line (e.g. "E,123,LED_ON\n"). False if it does not fit.
  // Time O(L) for an L-character line.  Data structures: the caller's char array (appended in place).
  static bool appendLine(char* text, size_t& used, size_t capacity, const char* line) {
    size_t length = strlen(line);
    if (used + length >= capacity) {
      return false;
    }
    memcpy(text + used, line, length);
    used += length;
    text[used] = '\0';
    return true;
  }

  // Seals the batch text into one X line. Returns its length, 0 on failure.
  // cipherScratch must hold `length` bytes.
  // Time O(m) for m bytes: one AES-GCM pass + base64 (4 characters per 3 bytes).
  // Data structures: a 12-byte nonce and a 16-byte tag array on the stack; the caller's arrays.
  size_t seal(const char* text, size_t length, uint8_t* cipherScratch, char* out, size_t capacity) {
    uint8_t nonce[LINK_NONCE_BYTES];
    uint8_t tag[LINK_TAG_BYTES];
    memcpy(nonce, session_, LINK_SESSION_BYTES);
    nonce[8] = (uint8_t)(frameCounter_ >> 24);
    nonce[9] = (uint8_t)(frameCounter_ >> 16);
    nonce[10] = (uint8_t)(frameCounter_ >> 8);
    nonce[11] = (uint8_t)frameCounter_;
    if (!linkCipher_.seal(nonce, (const uint8_t*)LINK_AAD, strlen(LINK_AAD), (const uint8_t*)text, length,
                          cipherScratch, tag)) {
      return 0;
    }
    int head = snprintf(out, capacity, "X,%lu,", (unsigned long)frameCounter_);
    if (head <= 0 || (size_t)head >= capacity) {
      return 0;
    }
    size_t used = (size_t)head;
    size_t written = base64Encode(cipherScratch, length, out + used, capacity - used);
    if (written == 0 && length > 0) {
      return 0;
    }
    used += written;
    if (used + 1 >= capacity) {
      return 0;
    }
    out[used++] = ',';
    written = base64Encode(tag, LINK_TAG_BYTES, out + used, capacity - used);
    if (written == 0 || used + written + 2 > capacity) {
      return 0;
    }
    used += written;
    out[used++] = '\n';
    out[used] = '\0';
    frameCounter_++;
    return used;
  }

  // Time O(1): read-only counters (for tests and the status line).
  uint32_t framesSealed() const { return frameCounter_; }
  uint32_t samplesEncoded() const { return sampleNumber_; }

 private:
  BlockCipher& heCipher_;
  AeadCipher& linkCipher_;
  const char* keyId_;
  uint8_t session_[LINK_SESSION_BYTES] = {0};
  uint32_t sampleNumber_ = 0;
  uint32_t frameCounter_ = 0;
  bool primed_ = false;
  EmgFilterChain filters_[EMG_CHANNELS];
};

}  // namespace myomotion
