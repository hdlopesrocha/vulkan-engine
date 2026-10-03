#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

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

// Window function applied before the FFT (reduces spectral leakage).
enum class AudioWindowType {
    Hann = 0,       // default: good leakage suppression, music-friendly
    Hamming,
    Blackman,
    Rectangular     // no window (useful for comparison / debugging)
};

// Fast real-time FFT analyser.
//
// Design notes:
//  * Self-contained: no FFT library, no audio-library, no UI dependency,
//    so it can be unit-tested standalone and reused by any widget.
//  * Zero steady-state allocation: every buffer is (re)allocated only in
//    setFftSize()/reset(); analyzeWindow()/updateIdle() never allocate.
//  * Radix-2 Cooley-Tukey FFT, iterative in-place, N = 512..4096. At these
//    sizes the cost is negligible next to a frame (< 100k flops/window).
//  * The widget feeds one mono window per call (decoded PCM around the
//    playback cursor); hop/throttling policy lives in the caller.
class AudioAnalyzer {
public:
    static constexpr int kMinFftSize = 512;
    static constexpr int kMaxFftSize = 4096;
    static constexpr int kDefaultFftSize = 2048;
    static constexpr int kWaveformCapacity = 4096;

    // Frequency-band defaults (Hz). Kept configurable via setBandEdges()
    // instead of hard-coded at the call sites.
    static constexpr float kDefaultBassUpperHz = 250.0f;
    static constexpr float kDefaultMidUpperHz = 4000.0f;
    static constexpr float kBassLowerHz = 20.0f;
    static constexpr float kHighUpperHz = 20000.0f;

    AudioAnalyzer();
    ~AudioAnalyzer() = default;

    AudioAnalyzer(const AudioAnalyzer&) = delete;
    AudioAnalyzer& operator=(const AudioAnalyzer&) = delete;

    // Rebuild reusable buffers (window, FFT scratch, spectra). Only call on
    // user action (FFT-size combo) or track load — never per frame.
    void setFftSize(int size);
    int fftSize() const { return fftSize_; }

    void setWindowType(AudioWindowType type);
    AudioWindowType windowType() const { return windowType_; }

    // 0.0 = fully responsive .. 0.95 = very smooth.
    void setSpectrumSmoothing(float s) { spectrumSmoothing_ = clamp01(s); }
    void setAmplitudeSmoothing(float s) { amplitudeSmoothing_ = clamp01(s); }
    float spectrumSmoothing() const { return spectrumSmoothing_; }
    float amplitudeSmoothing() const { return amplitudeSmoothing_; }

    // Onset threshold as a multiple of the slow energy follower
    // (typical range 1.1 .. 2.5, default 1.4).
    void setBeatSensitivity(float threshold);
    float beatSensitivity() const { return beatThreshold_; }

    void setBandEdges(float bassUpperHz, float midUpperHz);

    // Analyse one mono window of exactly fftSize() samples at sampleRate Hz.
    // Updates features, smoothed spectrum and the waveform ring.
    void analyzeWindow(const float* mono, uint32_t sampleRate);

    // Decay beat/spectrum/amplitude toward silence (call when paused,
    // stopped, or starved of data). dt in seconds.
    void updateIdle(float dt);

    // Clear spectra, followers and waveform history; keeps settings.
    void resetHistory();

    const AudioFeatures& features() const { return features_; }

    // Smoothed linear magnitude spectrum, bins [0, fftSize()/2].
    // Normalized so a full-scale sine reads ~0.5 at its peak bin.
    const float* spectrumData() const { return smoothedMag_.data(); }
    int spectrumBins() const { return fftSize_ / 2 + 1; }

    // Display-ready bars (count entries, each 0..1 with gentle compression
    // for readability). Log mapping spreads the bass region; linear is
    // uniform in Hz. Backed by a reusable buffer (resized only when count
    // changes). sampleRate: rate the last window was analysed at.
    const std::vector<float>& displayBars(int count, bool logScale, uint32_t sampleRate);

    // Recent mono history for the waveform graph (oldest -> newest when
    // read with waveformOffset() wrap, matching ImGui::PlotLines offset).
    const float* waveformData() const { return waveform_.data(); }
    int waveformSize() const { return waveformFill_; }
    int waveformCapacity() const { return static_cast<int>(waveform_.size()); }
    int waveformOffset() const { return waveformPos_; }

private:
    static float clamp01(float v) { return v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v); }

    void rebuildBuffers();
    void computeFft();      // windowed FFT of fftIn_ -> re_/im_
    void updateFeaturesFromSpectrum(uint32_t sampleRate, float peak);
    static int clampFftSize(int size);

    int fftSize_ = kDefaultFftSize;
    AudioWindowType windowType_ = AudioWindowType::Hann;
    float spectrumSmoothing_ = 0.80f;
    float amplitudeSmoothing_ = 0.70f;
    float beatThreshold_ = 1.4f;
    float bassUpperHz_ = kDefaultBassUpperHz;
    float midUpperHz_ = kDefaultMidUpperHz;

    AudioFeatures features_{};
    float slowEnergy_ = 0.0f;   // slow follower for onset detection

    std::vector<float> window_;       // analysis window, size N
    std::vector<float> fftIn_;        // windowed input, size N
    std::vector<float> re_;           // FFT scratch, size N
    std::vector<float> im_;           // FFT scratch, size N
    std::vector<int> bitRev_;         // bit-reversal permutation, size N
    std::vector<float> smoothedMag_;  // smoothed magnitudes, size N/2+1
    std::vector<float> displayBars_;  // displayBars() scratch

    std::vector<float> waveform_;     // ring, capacity kWaveformCapacity
    int waveformPos_ = 0;             // next write index (oldest sample)
    int waveformFill_ = 0;            // valid samples (<= capacity)
};
