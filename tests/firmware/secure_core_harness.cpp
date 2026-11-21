// Runs the ESP32's secure encoder (hardware/myomotion_emg_secure/secure_core.h) on a PC, with OpenSSL in
// place of the ESP32's hardware AES, so tests/test_secure_link.py can check the C++ output against the
// Python decoder (src/myomotion/semg/secure_link.py).
//
// Build:  g++ -std=c++17 -O2 -I hardware/myomotion_emg_secure tests/firmware/secure_core_harness.cpp -lcrypto
// Run:    harness <link key hex> <he key hex> <key id> <session hex> < samples.csv > wire.txt
//         samples.csv lines: t_us,mv1,mv2,mv3    (one batch per 10 samples, like the firmware)
//         a line "E,<t_us>" adds a sync-flash line to the current batch.

#include <openssl/evp.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "secure_core.h"

using myomotion::LINK_NONCE_BYTES;
using myomotion::LINK_TAG_BYTES;

static void hexToBytes(const char* hex, uint8_t* out, size_t count) {
  for (size_t i = 0; i < count; i++) {
    unsigned value = 0;
    sscanf(hex + 2 * i, "%2x", &value);
    out[i] = (uint8_t)value;
  }
}

class OpenSslBlock : public myomotion::BlockCipher {
 public:
  explicit OpenSslBlock(const uint8_t key[16]) {
    ctx_ = EVP_CIPHER_CTX_new();
    EVP_EncryptInit_ex(ctx_, EVP_aes_128_ecb(), nullptr, key, nullptr);
    EVP_CIPHER_CTX_set_padding(ctx_, 0);
  }
  ~OpenSslBlock() override { EVP_CIPHER_CTX_free(ctx_); }
  void encryptBlock(const uint8_t in[16], uint8_t out[16]) override {
    int length = 0;
    EVP_EncryptUpdate(ctx_, out, &length, in, 16);
  }

 private:
  EVP_CIPHER_CTX* ctx_;
};

class OpenSslGcm : public myomotion::AeadCipher {
 public:
  explicit OpenSslGcm(const uint8_t key[16]) { memcpy(key_, key, 16); }
  bool seal(const uint8_t nonce[LINK_NONCE_BYTES], const uint8_t* aad, size_t aadLength, const uint8_t* plain,
            size_t length, uint8_t* cipher, uint8_t tag[LINK_TAG_BYTES]) override {
    EVP_CIPHER_CTX* ctx = EVP_CIPHER_CTX_new();
    int out = 0;
    bool ok = EVP_EncryptInit_ex(ctx, EVP_aes_128_gcm(), nullptr, nullptr, nullptr) == 1 &&
              EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_GCM_SET_IVLEN, LINK_NONCE_BYTES, nullptr) == 1 &&
              EVP_EncryptInit_ex(ctx, nullptr, nullptr, key_, nonce) == 1 &&
              EVP_EncryptUpdate(ctx, nullptr, &out, aad, (int)aadLength) == 1 &&
              EVP_EncryptUpdate(ctx, cipher, &out, plain, (int)length) == 1 &&
              EVP_EncryptFinal_ex(ctx, cipher + out, &out) == 1 &&
              EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_GCM_GET_TAG, LINK_TAG_BYTES, tag) == 1;
    EVP_CIPHER_CTX_free(ctx);
    return ok;
  }

 private:
  uint8_t key_[16];
};

static char text[4096];
static uint8_t scratch[4096];
static char frame[6144];

int main(int argc, char** argv) {
  if (argc != 5) {
    fprintf(stderr, "usage: harness <link key hex> <he key hex> <key id> <session hex>\n");
    return 2;
  }
  uint8_t linkKey[16], heKey[16], session[8];
  hexToBytes(argv[1], linkKey, 16);
  hexToBytes(argv[2], heKey, 16);
  hexToBytes(argv[4], session, 8);
  OpenSslBlock block(heKey);
  OpenSslGcm gcm(linkKey);
  myomotion::SecureEncoder encoder(block, gcm, argv[3]);
  encoder.begin(session);

  char hello[64];
  fwrite(hello, 1, encoder.helloLine(hello, sizeof(hello)), stdout);

  size_t used = 0;
  int inBatch = 0;
  char line[128];
  while (fgets(line, sizeof(line), stdin)) {
    if (line[0] == 'E') {
      char event[48];
      snprintf(event, sizeof(event), "E,%lu,LED_ON\n", strtoul(line + 2, nullptr, 10));
      myomotion::SecureEncoder::appendLine(text, used, sizeof(text), event);
      continue;
    }
    unsigned long timeUs = 0;
    unsigned mv[3] = {0, 0, 0};
    if (sscanf(line, "%lu,%u,%u,%u", &timeUs, &mv[0], &mv[1], &mv[2]) != 4) {
      continue;
    }
    const uint16_t values[3] = {(uint16_t)mv[0], (uint16_t)mv[1], (uint16_t)mv[2]};
    encoder.appendSample(text, used, sizeof(text), (uint32_t)timeUs, values);
    if (++inBatch == 10) {
      fwrite(frame, 1, encoder.seal(text, used, scratch, frame, sizeof(frame)), stdout);
      used = 0;
      inBatch = 0;
    }
  }
  if (used > 0) {
    fwrite(frame, 1, encoder.seal(text, used, scratch, frame, sizeof(frame)), stdout);
  }
  return 0;
}
