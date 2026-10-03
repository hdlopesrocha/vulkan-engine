#pragma once

#include "Widget.hpp"
#include "components/FilePicker.hpp"
#include "../services/AudioAnalyzer.hpp"
#include "../third_party/miniaudio/miniaudio.h"
#include <atomic>
#include <cstdint>
#include <filesystem>
#include <future>
#include <mutex>
#include <string>
#include <vector>

// Music player widget with integrated real-time audio analysis.
//
// Responsibilities (kept in this widget by design):
//  * loading music files + transport (play/pause/stop/seek/repeat)
//  * decoding the file to mono PCM (background thread, never on the UI path)
//  * real-time FFT/waveform analysis while the track plays
//  * visualising waveform + spectrum + bands inside this same widget
//  * exposing compact audio features for the MUSIC_REACTIVE_WATER brush
//    (MyApp forwards them into the water GPU params each frame).
//
// The water shader only reads the uploaded parameters; no FFT or decoding
// logic lives outside this widget + AudioAnalyzer.
class MusicWidget : public Widget {
public:
    MusicWidget();
    ~MusicWidget() override;

    void render() override;

    // Per-frame maintenance that must run even when the widget window is
    // hidden (decode completion + analysis). Same UI thread as render();
    // cheap and idempotent, safe to call every frame.
    void tick();

    // Compact features for the renderer (valid flag tells whether real
    // analysis data is present yet).
    const AudioFeatures& audioFeatures() const { return analyzer_.features(); }
    // True when the widget currently produces live music-reactive output.
    bool musicReactiveActive() const;
    bool isPlaying() const { return playbackState == PlaybackState::Playing; }

private:
    enum class PlaybackState {
        Stopped,
        Playing,
        Paused
    };

    // ── Existing transport state (unchanged behaviour) ──
    std::string selectedFile;
    FilePicker filePicker_;
    std::string pickerError;

    PlaybackState playbackState;
    bool audioInitialized;
    bool soundLoaded;
    bool repeatEnabled;
    float trackDurationSec;
    ma_engine engine;
    ma_sound sound;
    bool engineReady;
    bool soundReady;

    bool initAudio();
    void unloadSound();
    bool loadSelectedTrack();
    float currentPlaybackPositionSeconds() const;
    void seekTo(float seconds);
    void applyRepeatSetting();

    void play();
    void pause();
    void stop();
    void pollPlayerState();

    // ── Decoded PCM (mono f32, native file rate) ──
    struct DecodedTrack {
        std::vector<float> mono;
        uint32_t sampleRate = 0;
    };
    DecodedTrack track_;
    std::mutex trackMutex_;          // guards track_ + staging below
    bool trackReady_ = false;        // UI thread: decoded PCM available
    std::string decodedPath_;        // file track_ was decoded from
    std::string decodeError_;        // last background-decode failure

    // Background decode handoff (worker fills staging under lock; the UI
    // thread consumes it once the future is ready — future readiness
    // implies staging visibility, so no flag race is possible).
    DecodedTrack staging_;
    std::string stagingError_;
    std::string stagingPath_;
    std::string pendingPath_;        // newest decode request (UI thread)
    std::atomic<bool> decodeInFlight_{false};
    std::future<void> decodeFuture_;

    void startDecodeAsync(const std::string& path);
    void launchDecodeWorker(const std::string& path);
    void pollDecode();
    static void decodeFileWorker(const std::string& path, DecodedTrack& out, std::string& error);

    // ── Analysis ──
    AudioAnalyzer analyzer_;
    std::vector<float> windowScratch_; // reusable gather buffer (size = FFT)
    ma_uint64 lastAnalysisCursor_ = 0;
    bool analysisInit_ = false;
    double lastTickTime_ = 0.0;

    // UI settings (wired to AudioAnalyzer).
    int fftSizeIdx_ = 2;             // {"512","1024","2048","4096"} -> 2048
    int windowIdx_ = 0;              // {"Hann","Hamming","Blackman","Rectangular"}
    float spectrumSmoothing_ = 0.80f;
    float amplitudeSmoothing_ = 0.70f;
    float beatSensitivity_ = 1.4f;
    bool showWaveform_ = true;
    bool showFFT_ = true;
    bool logFrequency_ = true;
    bool reactiveEnabled_ = true;    // forward features to the water params

    void updateAnalysis(double now);
    void renderAnalysisSection();
    void renderWaveform();
    void renderFftSpectrum();
    void renderAnalysisControls();
};
