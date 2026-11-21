// MyoMotion EMG over Wi-Fi, ENCRYPTED: 3 channels, 500 samples per second, ESP32 -> laptop.
//
// The same sampling and Wi-Fi as myomotion_emg_wifi.ino (the proven plain version, kept unchanged as a
// fallback), plus two layers of encryption (protocol in secure_core.h and src/myomotion/semg/secure_link.py):
//   1. Link: every batch (~20 ms of samples) is sealed with AES-128-GCM using the ESP32's hardware AES.
//      Without the link key the traffic is unreadable; a changed or replayed batch is rejected.
//   2. Homomorphic: every sample is filtered (20 Hz high-pass + mains notches), squared and encrypted as
//      c = (p + k) mod 2^64, so the laptop can add up 100 ms windows without the key (RMS envelope).
//
// Before flashing: run  python scripts/semg/make_link_keys.py  in the repo. It writes secrets.h next to
// this file (the keys and the Wi-Fi password; git-ignored) and the laptop's copy of the same keys.
// Laptop side: record_session.py --board esp32 --secure, or scripts/semg/live_mvc.py --secure.
//
// Wiring (unchanged): divider outputs -> VP (ch1), VN (ch2), D34 (ch3); EMG GND rail -> ESP32 GND;
// sync LED on D25 (+ 220-330 ohm); bias resistor R4 from the divider output to D32 (driven HIGH).
// Board: "ESP32 Dev Module".
// USB Serial Monitor (115200 baud): network details while waiting; while connected, a health line every
// 5 s, including how long the encryption takes per batch.
//
// Data structures (this file): a lock-free single-producer / single-consumer ring buffer (a fixed array of
//   2048 Sample structs + two std::atomic indices) between the sampling task and loop(); fixed char / byte
//   arrays for the batch text, its ciphertext and the X line (global and reused, no heap); a SecureEncoder
//   (secure_core.h) holding one filter chain per channel.
// Time complexity (this file): O(C) per sample in the sampling task (C = 3 channels); per batch of b samples
//   O(b * C * (S + 1)) to filter and encrypt them (S = 6 filter sections, one AES block per channel and
//   sample) plus O(m) to seal and base64 the m-byte batch text; everything else in loop() is O(1).

#include <WiFi.h>
#include <atomic>
#include <esp_random.h>
#include <mbedtls/aes.h>
#include <mbedtls/gcm.h>

#if __has_include("secrets.h")
#include "secrets.h"
#else
#error "secrets.h is missing: run  python scripts/semg/make_link_keys.py  in the repo first"
#endif
#include "secure_core.h"

// Wi-Fi network made by the ESP32 (password from secrets.h)
const char* WIFI_NAME = "MyoMotion-EMG";
const uint16_t TCP_PORT = 3333;
const unsigned long USB_BAUD = 115200;

// Sampling (unchanged)
const int CHANNEL_COUNT = myomotion::EMG_CHANNELS;
const int CHANNEL_PINS[CHANNEL_COUNT] = {36, 39, 34};      // VP, VN, D34 (ADC1: fine with Wi-Fi)
const TickType_t SAMPLE_PERIOD_TICKS = pdMS_TO_TICKS(2);   // 2 ms = 500 samples per second
const int SAMPLING_TASK_PRIORITY = 3;
const int SAMPLING_TASK_CORE = 1;
const uint32_t SAMPLING_TASK_STACK_BYTES = 4096;

// Buffer between the sampling task and loop()
const size_t BUFFER_CAPACITY = 2048;                       // about 4 seconds of samples
const size_t SAMPLES_PER_BATCH = 10;                       // send once 10 are waiting (20 ms)
const size_t MAX_SAMPLES_PER_BATCH = 30;                   // catch up after a pause
const size_t TEXT_BYTES = 3072;                            // 30 samples x (D line + P line) < 3072
const size_t FRAME_BYTES = 4400;                           // base64 of 3072 bytes + tag + header
const size_t LINE_BYTES = 48;

// Sync LED
const int SYNC_LED_PIN = 25;

// Bias for the raw-EMG divider (rebuild of 20 Nov 2026): R4 (10 kOhm) runs from the divider output to this
// pin. Driven HIGH (3.3 V) it lifts the resting signal to ~1.32 V, the middle of the ADC range, so the lower
// half of the EMG wave is no longer cut off at the ~0.14 V floor. A GPIO is used because the wide ESP32
// board covers the breadboard row of its 3V3 pin. Signal at the pin = module volts / 5 + 1.32 V.
const int BIAS_PIN = 32;
const unsigned long LED_ON_MS = 200;
const char SYNC_COMMAND = 'S';

const unsigned long STATUS_REPORT_MS = 5000;

// ---------------------------------------------------------------------------
// The two AES operations for secure_core.h, on the ESP32's hardware AES (through mbedtls).
// Each key is loaded once in setup() (not in a global constructor, which runs before the system and its
// hardware drivers are fully started); the contexts are reused for every block / batch.
// ---------------------------------------------------------------------------
// MbedAesBlock: AES-128 on one 16-byte block (makes the homomorphic keys).
// Time: begin O(1) (key expansion, once); encryptBlock O(1) (fixed 10 rounds in hardware).  Space: O(1).
// Data structures: an mbedtls_aes_context (the expanded round keys), loaded once and reused.
class MbedAesBlock : public myomotion::BlockCipher {
 public:
  bool begin(const uint8_t key[16]) {
    mbedtls_aes_init(&ctx_);
    return mbedtls_aes_setkey_enc(&ctx_, key, 128) == 0;
  }
  void encryptBlock(const uint8_t in[16], uint8_t out[16]) override {
    mbedtls_aes_crypt_ecb(&ctx_, MBEDTLS_AES_ENCRYPT, in, out);
  }

 private:
  mbedtls_aes_context ctx_;
};

// MbedGcm: AES-128-GCM, seals one batch (encrypts it and computes its 16-byte tag).
// Time: begin O(1); seal O(m) for m bytes (one AES block per 16 bytes + the GHASH).  Space: O(1) extra.
// Data structures: an mbedtls_gcm_context (key + GHASH table), loaded once and reused.
class MbedGcm : public myomotion::AeadCipher {
 public:
  bool begin(const uint8_t key[16]) {
    mbedtls_gcm_init(&ctx_);
    return mbedtls_gcm_setkey(&ctx_, MBEDTLS_CIPHER_ID_AES, key, 128) == 0;
  }
  bool seal(const uint8_t nonce[myomotion::LINK_NONCE_BYTES], const uint8_t* aad, size_t aadLength,
            const uint8_t* plain, size_t length, uint8_t* cipher, uint8_t tag[myomotion::LINK_TAG_BYTES]) override {
    return mbedtls_gcm_crypt_and_tag(&ctx_, MBEDTLS_GCM_ENCRYPT, length, nonce, myomotion::LINK_NONCE_BYTES, aad,
                                     aadLength, plain, cipher, myomotion::LINK_TAG_BYTES, tag) == 0;
  }

 private:
  mbedtls_gcm_context ctx_;
};

// Sample: one reading of all channels.  Time O(1) to fill.  Space: 10 bytes (12 with padding).
// Data structures: a plain struct, a 32-bit time and a fixed array of three 16-bit millivolt values.
struct Sample {
  uint32_t timeUs;
  uint16_t millivolts[CHANNEL_COUNT];
};

WiFiServer server(TCP_PORT);
WiFiClient laptop;
bool wifiStarted = false;

MbedAesBlock heCipher;
MbedGcm linkCipher;
myomotion::SecureEncoder encoder(heCipher, linkCipher, KEY_ID);

// Ring buffer: only the sampling task moves writeIndex, only loop() moves readIndex.
// Data structures: a fixed array + two atomic indices (one producer, one consumer, so no lock is needed);
// one slot always stays empty to tell full from empty.  Time: O(1) push and O(1) pop.
// Space: 2048 x 12 bytes = about 24 KB of the ESP32's RAM.
Sample sampleBuffer[BUFFER_CAPACITY];
std::atomic<size_t> writeIndex(0);
std::atomic<size_t> readIndex(0);
std::atomic<uint32_t> droppedSamples(0);

// Batch buffers: global (not on loop()'s stack), reused for every batch.
char batchText[TEXT_BYTES];
uint8_t cipherScratch[TEXT_BYTES];
char frameOut[FRAME_BYTES];

bool laptopConnected = false;
unsigned long nextStatusReportMs = 0;
unsigned long samplesSent = 0;
unsigned long failedWrites = 0;
unsigned long sealFailures = 0;
unsigned long cryptoMicros = 0;        // time spent filtering + encrypting since the last status line
unsigned long cryptoBatches = 0;
bool ledIsOn = false;
unsigned long ledOffAtMs = 0;

// Time O(1): the modular difference of the two indices.  Data structures: the ring buffer's atomic indices.
size_t samplesWaiting() {
  size_t write = writeIndex.load(std::memory_order_acquire);
  size_t read = readIndex.load(std::memory_order_relaxed);
  return (write + BUFFER_CAPACITY - read) % BUFFER_CAPACITY;
}

// Reads the three channels every 2 ms into the buffer (unchanged from the plain firmware).
// Time: O(C) per 2 ms tick (C = 3 analog reads) + O(1) push.  Space: O(1) (one Sample on the stack).
// Data structures: a Sample struct; the ring buffer (the newest sample is dropped and counted when full).
void samplingTask(void* unused) {
  (void)unused;
  TickType_t lastWake = xTaskGetTickCount();
  while (true) {
    vTaskDelayUntil(&lastWake, SAMPLE_PERIOD_TICKS);
    Sample sample;
    sample.timeUs = (uint32_t)micros();
    for (int channel = 0; channel < CHANNEL_COUNT; channel++) {
      sample.millivolts[channel] = (uint16_t)analogReadMilliVolts(CHANNEL_PINS[channel]);
    }
    size_t write = writeIndex.load(std::memory_order_relaxed);
    size_t next = (write + 1) % BUFFER_CAPACITY;
    if (next == readIndex.load(std::memory_order_acquire)) {
      droppedSamples.fetch_add(1, std::memory_order_relaxed);
      continue;
    }
    sampleBuffer[write] = sample;
    writeIndex.store(next, std::memory_order_release);
  }
}

// Time O(1): the read index jumps to the write index (samples are skipped, not erased).
// Data structures: the ring buffer's atomic indices.
void clearBuffer() {
  readIndex.store(writeIndex.load(std::memory_order_acquire), std::memory_order_release);
}

// Time O(n) for n bytes handed to the TCP stack.  Data structures: a char array (not copied).
void sendToLaptop(const char* text, size_t length) {
  if (length == 0) {
    return;
  }
  size_t written = laptop.write((const uint8_t*)text, length);
  if (written != length) {
    failedWrites++;
  }
}

// Seals the batch text and sends it as one X line.
// Time O(m) for m bytes of batch text (AES-GCM + base64).
// Data structures: the global batch, ciphertext and frame arrays.
void sealAndSend(size_t textLength) {
  size_t length = encoder.seal(batchText, textLength, cipherScratch, frameOut, sizeof(frameOut));
  if (length == 0) {
    sealFailures++;
    return;
  }
  sendToLaptop(frameOut, length);
}

// Turns the sync LED on and sends the moment, sealed like everything else.
// Time O(1): one short E line sealed and sent.  Data structures: a 48-byte char array on the stack.
void flashSyncLed() {
  digitalWrite(SYNC_LED_PIN, HIGH);
  unsigned long flashUs = micros();
  ledIsOn = true;
  ledOffAtMs = millis() + LED_ON_MS;
  char line[LINE_BYTES];
  snprintf(line, sizeof(line), "E,%lu,LED_ON\n", flashUs);
  size_t used = 0;
  if (myomotion::SecureEncoder::appendLine(batchText, used, sizeof(batchText), line)) {
    sealAndSend(used);
  }
  Serial.println("Sync flash sent");
}

// Time O(1).  Data structures: none (a wrap-safe comparison of millis() with the off time).
void updateSyncLed() {
  if (ledIsOn && (long)(millis() - ledOffAtMs) >= 0) {
    digitalWrite(SYNC_LED_PIN, LOW);
    ledIsOn = false;
  }
}

// Time O(k) for k command bytes waiting.  Data structures: none (one char at a time from the TCP socket).
void handleLaptopCommands() {
  while (laptop.available() > 0) {
    char command = (char)laptop.read();
    if (command == SYNC_COMMAND) {
      flashSyncLed();
    }
  }
}

// New connection: new random session (hardware random number generator), hello line in clear.
// Time O(1) per call; on a new connection O(S * C) to design the filters again (S = 6 sections, C = 3).
// Data structures: an 8-byte random session array; a 64-byte hello line; the WiFiClient.
void checkConnection() {
  if (server.hasClient()) {
    WiFiClient incoming = server.accept();
    if (laptop) {
      laptop.stop();
    }
    laptop = incoming;
    laptop.setNoDelay(true);
    laptopConnected = true;
    clearBuffer();
    uint8_t session[myomotion::LINK_SESSION_BYTES];
    esp_fill_random(session, sizeof(session));
    encoder.begin(session);
    char hello[64];
    sendToLaptop(hello, encoder.helloLine(hello, sizeof(hello)));
    samplesSent = 0;
    failedWrites = 0;
    sealFailures = 0;
    cryptoMicros = 0;
    cryptoBatches = 0;
    droppedSamples.store(0, std::memory_order_relaxed);
    nextStatusReportMs = millis() + STATUS_REPORT_MS;
    Serial.print("Laptop connected from ");
    Serial.print(laptop.remoteIP());
    Serial.print(": sending 500 encrypted samples per second, key id ");
    Serial.println(KEY_ID);
  }
  if (laptopConnected && !laptop.connected()) {
    laptopConnected = false;
    laptop.stop();
    Serial.print("Laptop disconnected after ");
    Serial.print(samplesSent);
    Serial.println(" samples. Waiting for it to connect again...");
  }
}

// Filters, squares and encrypts the waiting samples (D + P lines), seals them and sends one X line.
// Time: O(b * C * (S + 1)) for b <= 30 samples (S filter steps and one AES block per channel and sample),
//   plus O(m) to seal and base64 the m-byte batch.  Space: O(1) extra (the global arrays are reused).
// Data structures: the ring buffer (pop); the global batch text, ciphertext and frame arrays.
void sendWaitingSamples() {
  if (samplesWaiting() < SAMPLES_PER_BATCH) {
    return;
  }
  unsigned long startUs = micros();
  size_t used = 0;
  size_t count = 0;
  while (count < MAX_SAMPLES_PER_BATCH && samplesWaiting() > 0) {
    size_t read = readIndex.load(std::memory_order_relaxed);
    const Sample& sample = sampleBuffer[read];
    if (!encoder.appendSample(batchText, used, sizeof(batchText), sample.timeUs, sample.millivolts)) {
      break;                                   // batch full: the rest goes next time
    }
    readIndex.store((read + 1) % BUFFER_CAPACITY, std::memory_order_release);
    count++;
  }
  size_t length = encoder.seal(batchText, used, cipherScratch, frameOut, sizeof(frameOut));
  cryptoMicros += micros() - startUs;
  cryptoBatches++;
  if (length == 0) {
    sealFailures++;
    return;
  }
  sendToLaptop(frameOut, length);
  samplesSent += count;
}

// Time O(1): at most one status line every 5 s.  Data structures: counters (unsigned long).
void reportStatus() {
  if ((long)(millis() - nextStatusReportMs) < 0) {
    return;
  }
  nextStatusReportMs = millis() + STATUS_REPORT_MS;
  if (!laptopConnected) {
    Serial.print("Waiting for the laptop. Wi-Fi: ");
    Serial.print(WIFI_NAME);
    Serial.print("  address: ");
    Serial.print(WiFi.softAPIP());
    Serial.print(":");
    Serial.print(TCP_PORT);
    Serial.print("  key id: ");
    Serial.println(KEY_ID);
    return;
  }
  Serial.print("sent ");
  Serial.print(samplesSent);
  Serial.print(" | dropped (buffer full) ");
  Serial.print(droppedSamples.load(std::memory_order_relaxed));
  Serial.print(" | failed writes ");
  Serial.print(failedWrites);
  Serial.print(" | seal failures ");
  Serial.print(sealFailures);
  Serial.print(" | encryption per batch ");
  Serial.print(cryptoBatches ? cryptoMicros / cryptoBatches : 0);
  Serial.print(" us | waiting in buffer ");
  Serial.println((unsigned long)samplesWaiting());
  cryptoMicros = 0;
  cryptoBatches = 0;
}

// Time O(1): two AES key expansions, the Wi-Fi access point, one task started.
// Data structures: the two cipher objects; the sampling task (FreeRTOS, pinned to core 1).
void setup() {
  Serial.begin(USB_BAUD);
  pinMode(SYNC_LED_PIN, OUTPUT);
  digitalWrite(SYNC_LED_PIN, LOW);
  pinMode(BIAS_PIN, OUTPUT);
  digitalWrite(BIAS_PIN, HIGH);                // 3.3 V for the bias resistor R4
  if (!heCipher.begin(HE_KEY) || !linkCipher.begin(LINK_KEY)) {
    Serial.println("Encryption keys could not be loaded. Re-run make_link_keys.py and flash again.");
    return;
  }

  WiFi.mode(WIFI_AP);
  if (!WiFi.softAP(WIFI_NAME, WIFI_PASSWORD_SECRET)) {
    Serial.println("Wi-Fi did not start. Press EN to restart the board.");
    return;
  }
  WiFi.setSleep(false);
  server.begin();
  server.setNoDelay(true);
  wifiStarted = true;
  xTaskCreatePinnedToCore(samplingTask, "emgSampling", SAMPLING_TASK_STACK_BYTES, nullptr,
                          SAMPLING_TASK_PRIORITY, nullptr, SAMPLING_TASK_CORE);
  Serial.println();
  Serial.print("MyoMotion EMG over Wi-Fi (encrypted) is ready. Key id ");
  Serial.println(KEY_ID);
}

// Time per pass: O(1) + at most one batch (see sendWaitingSamples); 1 ms pause so the Wi-Fi stack can run.
// Data structures: everything above; nothing allocated here.
void loop() {
  if (!wifiStarted) {
    delay(100);
    return;
  }
  checkConnection();
  updateSyncLed();
  reportStatus();
  if (!laptopConnected) {
    clearBuffer();
    delay(1);
    return;
  }
  handleLaptopCommands();
  sendWaitingSamples();
  delay(1);
}
