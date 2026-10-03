#include "AudioAnalyzer.hpp"

#include <algorithm>
#include <cmath>

namespace {

constexpr float kPi = 3.14159265358979323846f;

} // namespace

AudioAnalyzer::AudioAnalyzer() {
    rebuildBuffers();
}

int AudioAnalyzer::clampFftSize(int size) {
    // Snap to the supported power-of-two set {512, 1024, 2048, 4096}.
    if (size <= 512) return 512;
    if (size <= 1024) return 1024;
    if (size <= 2048) return 2048;
    return 4096;
}

void AudioAnalyzer::setFftSize(int size) {
    const int snapped = clampFftSize(size);
    if (snapped == fftSize_) return;
    fftSize_ = snapped;
    rebuildBuffers();
}

void AudioAnalyzer::setWindowType(AudioWindowType type) {
    if (type == windowType_) return;
    windowType_ = type;
    rebuildBuffers();
}

void AudioAnalyzer::setBeatSensitivity(float threshold) {
    beatThreshold_ = std::clamp(threshold, 1.05f, 3.0f);
}

void AudioAnalyzer::setBandEdges(float bassUpperHz, float midUpperHz) {
    bassUpperHz_ = std::max(40.0f, bassUpperHz);
    midUpperHz_ = std::max(bassUpperHz_ + 100.0f, midUpperHz);
}

void AudioAnalyzer::rebuildBuffers() {
    const int n = fftSize_;

    // Analysis window.
    window_.assign(static_cast<size_t>(n), 1.0f);
    switch (windowType_) {
        case AudioWindowType::Hann:
            for (int i = 0; i < n; ++i)
                window_[static_cast<size_t>(i)] =
                    0.5f * (1.0f - std::cos(2.0f * kPi * static_cast<float>(i) / static_cast<float>(n - 1)));
            break;
        case AudioWindowType::Hamming:
            for (int i = 0; i < n; ++i)
                window_[static_cast<size_t>(i)] =
                    0.54f - 0.46f * std::cos(2.0f * kPi * static_cast<float>(i) / static_cast<float>(n - 1));
            break;
        case AudioWindowType::Blackman:
            for (int i = 0; i < n; ++i) {
                const float x = 2.0f * kPi * static_cast<float>(i) / static_cast<float>(n - 1);
                window_[static_cast<size_t>(i)] = 0.42f - 0.5f * std::cos(x) + 0.08f * std::cos(2.0f * x);
            }
            break;
        case AudioWindowType::Rectangular:
            break; // stays 1.0
    }

    fftIn_.assign(static_cast<size_t>(n), 0.0f);
    re_.assign(static_cast<size_t>(n), 0.0f);
    im_.assign(static_cast<size_t>(n), 0.0f);

    // Bit-reversal permutation for the iterative FFT.
    bitRev_.assign(static_cast<size_t>(n), 0);
    int bits = 0;
    while ((1 << bits) < n) ++bits;
    for (int i = 0; i < n; ++i) {
        int r = 0;
        for (int b = 0; b < bits; ++b)
            if (i & (1 << b)) r |= 1 << (bits - 1 - b);
        bitRev_[static_cast<size_t>(i)] = r;
    }

    smoothedMag_.assign(static_cast<size_t>(n / 2 + 1), 0.0f);

    if (static_cast<int>(waveform_.size()) != kWaveformCapacity)
        waveform_.assign(static_cast<size_t>(kWaveformCapacity), 0.0f);
    resetHistory();
}

void AudioAnalyzer::resetHistory() {
    std::fill(smoothedMag_.begin(), smoothedMag_.end(), 0.0f);
    std::fill(waveform_.begin(), waveform_.end(), 0.0f);
    waveformPos_ = 0;
    waveformFill_ = 0;
    slowEnergy_ = 0.0f;
    features_ = AudioFeatures{};
}

void AudioAnalyzer::computeFft() {
    // Copy through the bit-reversal permutation into the scratch buffers.
    const int n = fftSize_;
    for (int i = 0; i < n; ++i) {
        re_[static_cast<size_t>(i)] = fftIn_[static_cast<size_t>(bitRev_[static_cast<size_t>(i)])];
        im_[static_cast<size_t>(i)] = 0.0f;
    }
    // Iterative radix-2 decimation-in-time butterflies.
    for (int len = 2; len <= n; len <<= 1) {
        const float ang = -2.0f * kPi / static_cast<float>(len);
        const float wRe = std::cos(ang);
        const float wIm = std::sin(ang);
        for (int i = 0; i < n; i += len) {
            float cRe = 1.0f, cIm = 0.0f;
            for (int j = 0; j < len / 2; ++j) {
                const size_t a = static_cast<size_t>(i + j);
                const size_t b = static_cast<size_t>(i + j + len / 2);
                const float tRe = cRe * re_[b] - cIm * im_[b];
                const float tIm = cRe * im_[b] + cIm * re_[b];
                re_[b] = re_[a] - tRe;
                im_[b] = im_[a] - tIm;
                re_[a] += tRe;
                im_[a] += tIm;
                const float nRe = cRe * wRe - cIm * wIm;
                cIm = cRe * wIm + cIm * wRe;
                cRe = nRe;
            }
        }
    }
}

void AudioAnalyzer::analyzeWindow(const float* mono, uint32_t sampleRate) {
    if (!mono || sampleRate == 0) return;
    const int n = fftSize_;

    // Time-domain peak (instant amplitude) + windowed copy. Peak of f32 PCM
    // is already 0..1, so no calibration constant is needed.
    float peak = 0.0f;
    for (int i = 0; i < n; ++i) {
        const float s = mono[i];
        const float a = s < 0.0f ? -s : s;
        if (a > peak) peak = a;
        fftIn_[static_cast<size_t>(i)] = s * window_[static_cast<size_t>(i)];
    }

    computeFft();

    // Linear magnitudes, normalized so a full-scale sine reads ~0.5 (Hann)
    // at its peak bin: mag = 2|X|/N (DC/Nyquist single-sided: |X|/N).
    const float invN = 1.0f / static_cast<float>(n);
    const int bins = n / 2 + 1;
    constexpr float kAttack = 0.65f;                    // fast attack ...
    const float release = (1.0f - spectrumSmoothing_) * 0.5f + 0.02f; // ... slow release
    for (int k = 0; k < bins; ++k) {
        const size_t uk = static_cast<size_t>(k);
        float mag = std::sqrt(re_[uk] * re_[uk] + im_[uk] * im_[uk]) * invN;
        if (k > 0 && k < n / 2) mag *= 2.0f;
        float& s = smoothedMag_[uk];
        s += (mag - s) * (mag >= s ? kAttack : release);
    }

    updateFeaturesFromSpectrum(sampleRate, clamp01(peak));

    // Waveform history: append the raw (unwindowed) window.
    for (int i = 0; i < n; ++i) {
        waveform_[static_cast<size_t>(waveformPos_)] = mono[i];
        waveformPos_ = (waveformPos_ + 1) % kWaveformCapacity;
    }
    waveformFill_ = std::min(waveformFill_ + n, kWaveformCapacity);

    features_.sampleRate = sampleRate;
    features_.valid = true;
}

void AudioAnalyzer::updateFeaturesFromSpectrum(uint32_t sampleRate, float peak) {
    const int n = fftSize_;
    const int bins = n / 2 + 1;
    const float nyquist = static_cast<float>(sampleRate) * 0.5f;
    const float binHz = static_cast<float>(sampleRate) / static_cast<float>(n);

    // Spectral power shares per band (self-normalizing: immune to overall
    // level, no calibration constant). Bands follow the spec defaults:
    // bass 20-250 Hz, mid 250 Hz-4 kHz, high 4-20 kHz (clamped to Nyquist).
    const float hiCap = std::min(kHighUpperHz, nyquist);
    double pBass = 0.0, pMid = 0.0, pHigh = 0.0, pTotal = 0.0;
    for (int k = 0; k < bins; ++k) {
        const float f = static_cast<float>(k) * binHz;
        if (f < kBassLowerHz || f > hiCap) continue;
        const double p = static_cast<double>(smoothedMag_[static_cast<size_t>(k)]);
        const double p2 = p * p;
        pTotal += p2;
        if (f < bassUpperHz_) pBass += p2;
        else if (f < midUpperHz_) pMid += p2;
        else pHigh += p2;
    }

    float tBass = 0.0f, tMid = 0.0f, tHigh = 0.0f;
    if (pTotal > 1e-12) {
        tBass = clamp01(static_cast<float>(pBass / pTotal));
        tMid = clamp01(static_cast<float>(pMid / pTotal));
        tHigh = clamp01(static_cast<float>(pHigh / pTotal));
    }

    // One-pole smoothing toward the band targets; (1 - smoothing) keeps the
    // slider semantics identical to the amplitude follower below.
    const float bAlpha = 1.0f - spectrumSmoothing_ * 0.9f;
    features_.bassEnergy += (tBass - features_.bassEnergy) * bAlpha;
    features_.midEnergy += (tMid - features_.midEnergy) * bAlpha;
    features_.highEnergy += (tHigh - features_.highEnergy) * bAlpha;

    // Overall amplitude: peak follower with smoothing.
    features_.instantAmplitude = peak;
    const float aAlpha = 1.0f - amplitudeSmoothing_ * 0.95f;
    features_.amplitude += (peak - features_.amplitude) * aAlpha;

    // Lightweight transient/beat detector: current peak vs a slow follower.
    // onset fires when the instant level jumps well above recent history.
    slowEnergy_ += (peak - slowEnergy_) * 0.03f;
    const float denom = slowEnergy_ * beatThreshold_;
    if (denom > 1e-5f && peak > denom) {
        const float strength = clamp01((peak - denom) / (beatThreshold_ * 0.5f + 1e-5f));
        features_.beatIntensity = std::max(features_.beatIntensity, clamp01(0.35f + 0.65f * strength));
    } else {
        features_.beatIntensity *= 0.90f; // smooth decay between updates
        if (features_.beatIntensity < 0.003f) features_.beatIntensity = 0.0f;
    }
}

void AudioAnalyzer::updateIdle(float dt) {
    if (dt <= 0.0f) return;
    // Time-based decay so paused/stopped audio settles to silence instead of
    // freezing the last spectrum on screen.
    const float k = std::exp(-dt * 6.0f);
    for (float& s : smoothedMag_) s *= k;
    features_.amplitude *= k;
    features_.instantAmplitude *= k;
    features_.bassEnergy *= k;
    features_.midEnergy *= k;
    features_.highEnergy *= k;
    features_.beatIntensity *= std::exp(-dt * 4.0f);
    slowEnergy_ *= k;
    if (features_.amplitude < 0.003f) features_.amplitude = 0.0f;
    if (features_.beatIntensity < 0.003f) features_.beatIntensity = 0.0f;
}

const std::vector<float>& AudioAnalyzer::displayBars(int count, bool logScale, uint32_t sampleRate) {
    if (count < 8) count = 8;
    if (count > 256) count = 256;
    if (static_cast<int>(displayBars_.size()) != count)
        displayBars_.assign(static_cast<size_t>(count), 0.0f);

    const int n = fftSize_;
    const int bins = n / 2 + 1;
    const float rate = sampleRate > 0 ? static_cast<float>(sampleRate)
                                      : static_cast<float>(features_.sampleRate);
    const float nyquist = rate > 0.0f ? rate * 0.5f : 20000.0f;
    const float fLo = std::min(20.0f, nyquist * 0.5f);
    const float binHz = rate > 0.0f ? rate / static_cast<float>(n) : 1.0f;

    for (int b = 0; b < count; ++b) {
        float f0, f1;
        if (logScale && fLo > 0.0f && nyquist > fLo) {
            // Logarithmic mapping: each bar spans an equal ratio, which
            // spreads the musically dense low end across the display.
            const float t0 = static_cast<float>(b) / static_cast<float>(count);
            const float t1 = static_cast<float>(b + 1) / static_cast<float>(count);
            f0 = fLo * std::pow(nyquist / fLo, t0);
            f1 = fLo * std::pow(nyquist / fLo, t1);
        } else {
            f0 = nyquist * static_cast<float>(b) / static_cast<float>(count);
            f1 = nyquist * static_cast<float>(b + 1) / static_cast<float>(count);
        }
        int k0 = static_cast<int>(f0 / binHz);
        int k1 = static_cast<int>(std::ceil(f1 / binHz));
        k0 = std::clamp(k0, 0, bins - 1);
        k1 = std::clamp(k1, k0 + 1, bins);
        float m = 0.0f;
        for (int k = k0; k < k1; ++k)
            m = std::max(m, smoothedMag_[static_cast<size_t>(k)]);
        // Gentle compression for readability (peak ~0.5 -> ~1.0).
        displayBars_[static_cast<size_t>(b)] = std::pow(clamp01(m * 2.0f), 0.6f);
    }
    return displayBars_;
}
