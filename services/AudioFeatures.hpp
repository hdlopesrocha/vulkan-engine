#pragma once

#include <cstdint>

// Compact per-frame audio features consumed by the music widget UI and,
// through the water GPU params, by the MUSIC_REACTIVE_WATER brush.
//
// All values are normalized to 0..1 unless noted. Produced by AudioAnalyzer
// (services/AudioAnalyzer.*), which owns the FFT and the smoothing state.
struct AudioFeatures {
    float instantAmplitude = 0.0f;   // peak of the current analysis window
    float amplitude = 0.0f;          // smoothed overall loudness (audioAmplitude)
    float bassEnergy = 0.0f;         // spectral power share: ~20-250 Hz
    float midEnergy = 0.0f;          // spectral power share: ~250 Hz-4 kHz
    float highEnergy = 0.0f;         // spectral power share: ~4-20 kHz
    float beatIntensity = 0.0f;      // transient onset strength, smooth decay
    uint32_t sampleRate = 0;         // sample rate of the analysed PCM
    bool valid = false;              // true once at least one window ran
};
