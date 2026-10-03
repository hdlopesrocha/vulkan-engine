#include "MusicWidget.hpp"
#include "components/ImGuiHelpers.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <cstdlib>
#include <imgui.h>
#include <utility>
#include <vector>

namespace {

// FFT-size combo entries; index 2 (2048) is the default.
constexpr int kFftSizes[] = {512, 1024, 2048, 4096};
constexpr const char* kFftLabels = "512\0"
                                   "1024\0"
                                   "2048\0"
                                   "4096\0";
constexpr const char* kWindowLabels = "Hann\0"
                                      "Hamming\0"
                                      "Blackman\0"
                                      "Rectangular\0";

// Guard against pathological files: 32M mono frames (~12 min @ 44.1 kHz,
// 128 MB f32) is plenty for a game-ambient track and bounds memory.
constexpr ma_uint64 kMaxDecodeFrames = 32ULL * 1024ULL * 1024ULL;

// Map a frequency to a 0..1 canvas position (log or linear).
float freqToT(float freq, float nyquist, bool logScale) {
    if (nyquist <= 0.0f) return 0.0f;
    if (logScale) {
        const float lo = std::min(20.0f, nyquist * 0.5f);
        if (freq <= lo) return 0.0f;
        if (freq >= nyquist) return 1.0f;
        return std::log(freq / lo) / std::log(nyquist / lo);
    }
    return std::clamp(freq / nyquist, 0.0f, 1.0f);
}

} // namespace

MusicWidget::MusicWidget()
    : Widget("Music Player", u8"\uf001"),
      selectedFile(),
      filePicker_("Select MP3 File", ".mp3"),
      pickerError(),
      playbackState(PlaybackState::Stopped),
      audioInitialized(false),
      soundLoaded(false),
      repeatEnabled(false),
      trackDurationSec(0.0f),
      engine(),
      sound(),
      engineReady(false),
      soundReady(false) {
    analyzer_.setFftSize(kFftSizes[fftSizeIdx_]);
    analyzer_.setWindowType(AudioWindowType::Hann);
    analyzer_.setSpectrumSmoothing(spectrumSmoothing_);
    analyzer_.setAmplitudeSmoothing(amplitudeSmoothing_);
    analyzer_.setBeatSensitivity(beatSensitivity_);
}

MusicWidget::~MusicWidget() {
    // A decode may still be running: wait for it (it only touches the
    // mutex-guarded staging area, so joining is safe). get() reaps the
    // worker thread; the future destructor would block anyway.
    if (decodeFuture_.valid()) {
        decodeFuture_.get();
    }
    unloadSound();
    if (engineReady) {
        ma_engine_uninit(&engine);
        engineReady = false;
    }
}

bool MusicWidget::musicReactiveActive() const {
    return reactiveEnabled_ && trackReady_ && analyzer_.features().valid &&
           playbackState == PlaybackState::Playing;
}

bool MusicWidget::initAudio() {
    if (audioInitialized) {
        return true;
    }
    ma_result result = ma_engine_init(nullptr, &engine);
    if (result != MA_SUCCESS) {
        pickerError = "Failed to initialize audio engine.";
        return false;
    }
    engineReady = true;
    audioInitialized = true;
    return true;
}

void MusicWidget::unloadSound() {
    if (soundReady) {
        ma_sound_uninit(&sound);
        soundReady = false;
    }
    soundLoaded = false;
    playbackState = PlaybackState::Stopped;
    trackDurationSec = 0.0f;
}

bool MusicWidget::loadSelectedTrack() {
    if (selectedFile.empty()) {
        return false;
    }
    if (!initAudio()) {
        return false;
    }

    unloadSound();

    ma_result result = ma_sound_init_from_file(&engine, selectedFile.c_str(), 0, nullptr, nullptr, &sound);
    if (result != MA_SUCCESS) {
        pickerError = "Failed to load audio file with miniaudio.";
        return false;
    }

    soundReady = true;
    soundLoaded = true;
    applyRepeatSetting();

    ma_uint64 lengthFrames = 0;
    if (ma_sound_get_length_in_pcm_frames(&sound, &lengthFrames) == MA_SUCCESS) {
        ma_uint32 sampleRate = 0;
        ma_sound_get_data_format(&sound, nullptr, nullptr, &sampleRate, nullptr, 0);
        if (sampleRate > 0) {
            trackDurationSec = static_cast<float>(static_cast<double>(lengthFrames) / static_cast<double>(sampleRate));
        }
    }

    // Drop the previous analysis state and decode the new file in the
    // background so the UI thread never blocks on I/O + decode.
    {
        std::lock_guard<std::mutex> lock(trackMutex_);
        track_ = DecodedTrack{};
        trackReady_ = false;
        decodedPath_.clear();
        decodeError_.clear();
    }
    analyzer_.resetHistory();
    analysisInit_ = false;
    startDecodeAsync(selectedFile);

    return true;
}

float MusicWidget::currentPlaybackPositionSeconds() const {
    if (!soundReady) {
        return 0.0f;
    }

    ma_uint64 cursorFrames = 0;
    ma_uint32 sampleRate = 0;
    if (ma_sound_get_cursor_in_pcm_frames(&sound, &cursorFrames) != MA_SUCCESS) {
        return 0.0f;
    }
    ma_sound_get_data_format(&sound, nullptr, nullptr, &sampleRate, nullptr, 0);
    if (sampleRate == 0) {
        return 0.0f;
    }
    return static_cast<float>(static_cast<double>(cursorFrames) / static_cast<double>(sampleRate));
}

void MusicWidget::seekTo(float seconds) {
    if (!soundReady) {
        return;
    }
    if (seconds < 0.0f) {
        seconds = 0.0f;
    }
    if (trackDurationSec > 0.0f && seconds > trackDurationSec) {
        seconds = trackDurationSec;
    }

    ma_uint32 sampleRate = 0;
    ma_sound_get_data_format(&sound, nullptr, nullptr, &sampleRate, nullptr, 0);
    if (sampleRate == 0) {
        return;
    }
    ma_uint64 targetFrame = static_cast<ma_uint64>(seconds * static_cast<float>(sampleRate));
    ma_sound_seek_to_pcm_frame(&sound, targetFrame);
    // Force the analyser to re-seed at the new cursor instead of smearing
    // the old spectrum across the seek discontinuity.
    analysisInit_ = false;
}

void MusicWidget::applyRepeatSetting() {
    if (soundReady) {
        ma_sound_set_looping(&sound, repeatEnabled ? MA_TRUE : MA_FALSE);
    }
}

void MusicWidget::play() {
    pollPlayerState();
    if (selectedFile.empty()) {
        pickerError = "Select an MP3 file first.";
        return;
    }

    if (!soundLoaded && !loadSelectedTrack()) {
        return;
    }

    if (ma_sound_start(&sound) == MA_SUCCESS) {
        playbackState = PlaybackState::Playing;
        pickerError.clear();
    } else {
        pickerError = "Failed to start playback.";
    }
}

void MusicWidget::pause() {
    pollPlayerState();
    if (playbackState == PlaybackState::Playing && soundReady) {
        if (ma_sound_stop(&sound) == MA_SUCCESS) {
            playbackState = PlaybackState::Paused;
            pickerError.clear();
        }
    }
}

void MusicWidget::stop() {
    pollPlayerState();
    if (soundReady) {
        ma_sound_stop(&sound);
        ma_sound_seek_to_pcm_frame(&sound, 0);
    }
    playbackState = PlaybackState::Stopped;
    analysisInit_ = false;
}

void MusicWidget::pollPlayerState() {
    if (!soundReady || playbackState != PlaybackState::Playing) {
        return;
    }

    ma_uint64 cursor = 0;
    ma_uint64 length = 0;
    if (ma_sound_get_cursor_in_pcm_frames(&sound, &cursor) == MA_SUCCESS &&
        ma_sound_get_length_in_pcm_frames(&sound, &length) == MA_SUCCESS &&
        length > 0 && cursor >= length) {
        if (repeatEnabled) {
            ma_sound_seek_to_pcm_frame(&sound, 0);
            ma_sound_start(&sound);
        } else {
            playbackState = PlaybackState::Stopped;
            ma_sound_seek_to_pcm_frame(&sound, 0);
        }
        analysisInit_ = false;
    }
}

// ── Background decode ────────────────────────────────────────────────
// Decodes any miniaudio-supported file (same decoder family as playback, so
// format support is identical by construction) to mono f32 at the file's
// native rate. Runs on a worker via std::async; only the mutex-guarded
// staging area is shared with the UI thread.
void MusicWidget::decodeFileWorker(const std::string& path, DecodedTrack& out, std::string& error) {
    DecodedTrack local;
    std::string localError;

    ma_decoder_config config = ma_decoder_config_init_default();
    config.format = ma_format_f32; // f32 mono, native sample rate
    config.channels = 1;

    ma_decoder decoder;
    if (ma_decoder_init_file(path.c_str(), &config, &decoder) != MA_SUCCESS) {
        localError = "Failed to decode audio file for analysis.";
    } else {
        ma_uint32 rate = 0;
        ma_decoder_get_data_format(&decoder, nullptr, nullptr, &rate, nullptr, 0);
        local.sampleRate = rate;

        ma_uint64 totalGuess = 0;
        if (ma_decoder_get_length_in_pcm_frames(&decoder, &totalGuess) == MA_SUCCESS && totalGuess > 0)
            local.mono.reserve(static_cast<size_t>(std::min<ma_uint64>(totalGuess, kMaxDecodeFrames)));

        std::vector<float> chunk(8192);
        for (;;) {
            ma_uint64 framesRead = 0;
            if (ma_decoder_read_pcm_frames(&decoder, chunk.data(), chunk.size(), &framesRead) != MA_SUCCESS)
                break;
            if (framesRead == 0) break;
            // resize + memcpy (instead of range insert) to keep GCC's
            // -Warray-bounds quiet about the chunked append.
            const size_t oldSize = local.mono.size();
            const size_t roomLeft = static_cast<size_t>(kMaxDecodeFrames) > oldSize
                                        ? static_cast<size_t>(kMaxDecodeFrames) - oldSize
                                        : 0;
            const size_t take = std::min<size_t>(static_cast<size_t>(framesRead), roomLeft);
            local.mono.resize(oldSize + take);
            std::memcpy(local.mono.data() + oldSize, chunk.data(), take * sizeof(float));
            if (take < static_cast<size_t>(framesRead) ||
                local.mono.size() >= static_cast<size_t>(kMaxDecodeFrames))
                break; // cap reached: truncated but usable
        }
        ma_decoder_uninit(&decoder);
        if (local.mono.empty() && localError.empty())
            localError = "Decoded audio is empty.";
    }

    // Publish under the caller's lock (see launchDecodeWorker).
    out = std::move(local);
    error = std::move(localError);
}

void MusicWidget::launchDecodeWorker(const std::string& path) {
    decodeInFlight_.store(true, std::memory_order_release);
    decodeFuture_ = std::async(std::launch::async, [this, path]() {
        DecodedTrack local;
        std::string error;
        decodeFileWorker(path, local, error);
        std::lock_guard<std::mutex> lock(trackMutex_);
        staging_ = std::move(local);
        stagingError_ = std::move(error);
        stagingPath_ = path;
    });
}

void MusicWidget::startDecodeAsync(const std::string& path) {
    // Newest request wins; a running worker is never interrupted (it only
    // touches the lock-guarded staging area) — the fresher path is picked
    // up by pollDecode() as soon as the worker finishes.
    pendingPath_ = path;
    if (decodeInFlight_.load(std::memory_order_acquire))
        return;
    if (decodeFuture_.valid())
        decodeFuture_.get(); // reap the finished thread
    launchDecodeWorker(path);
}

void MusicWidget::pollDecode() {
    // Reap a finished worker and publish its staging area.
    if (decodeInFlight_.load(std::memory_order_acquire) && decodeFuture_.valid() &&
        decodeFuture_.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
        decodeFuture_.get();
        decodeInFlight_.store(false, std::memory_order_release);
        std::lock_guard<std::mutex> lock(trackMutex_);
        if (!staging_.mono.empty() && stagingPath_ == pendingPath_) {
            track_ = std::move(staging_);
            staging_ = DecodedTrack{};
            decodedPath_ = stagingPath_;
            trackReady_ = true;
            decodeError_ = stagingError_;
            analyzer_.resetHistory();
            analysisInit_ = false;
        } else if (!stagingError_.empty() && stagingPath_ == pendingPath_) {
            decodeError_ = stagingError_;
        }
        // else: stale result for a superseded file — discard.
    }
    // Kick off the pending request (covers the superseded-file case above).
    if (!decodeInFlight_.load(std::memory_order_acquire)) {
        std::string want;
        {
            std::lock_guard<std::mutex> lock(trackMutex_);
            if (trackReady_ && decodedPath_ == pendingPath_)
                return;
            want = pendingPath_;
        }
        if (!want.empty()) {
            if (decodeFuture_.valid())
                decodeFuture_.get();
            launchDecodeWorker(want);
        }
    }
}

// ── Per-frame analysis ───────────────────────────────────────────────
// Runs on the UI thread (called from render() and, for the hidden-widget
// case, from MyApp after the widget pass). Feeds the analyser one FFT
// window per playback hop of N/2 frames; idle-decays otherwise. Never
// allocates: windowScratch_ is sized once per FFT-size setting.
void MusicWidget::tick() {
    pollDecode();

    const double now = ImGui::GetTime();
    float dt = lastTickTime_ > 0.0 ? static_cast<float>(now - lastTickTime_) : 0.016f;
    dt = std::clamp(dt, 0.0f, 0.25f);
    lastTickTime_ = now;
    updateAnalysis(static_cast<double>(dt));
}

void MusicWidget::updateAnalysis(double dtDouble) {
    const float dt = static_cast<float>(dtDouble);
    pollPlayerState();

    bool haveTrack = false;
    {
        std::lock_guard<std::mutex> lock(trackMutex_);
        haveTrack = trackReady_ && !track_.mono.empty();
    }
    if (!haveTrack || playbackState != PlaybackState::Playing || !soundReady) {
        analyzer_.updateIdle(dt);
        return;
    }

    ma_uint64 cursor = 0;
    ma_uint32 soundRate = 0;
    if (ma_sound_get_cursor_in_pcm_frames(&sound, &cursor) != MA_SUCCESS) {
        analyzer_.updateIdle(dt);
        return;
    }
    ma_sound_get_data_format(&sound, nullptr, nullptr, &soundRate, nullptr, 0);

    // The engine may resample: map the sound cursor into decoded-PCM frames.
    uint32_t trackRate = 0;
    size_t pcmSize = 0;
    {
        std::lock_guard<std::mutex> lock(trackMutex_);
        trackRate = track_.sampleRate;
        pcmSize = track_.mono.size();
    }
    const double rateScale = (soundRate > 0 && trackRate > 0)
                                 ? static_cast<double>(trackRate) / static_cast<double>(soundRate)
                                 : 1.0;
    const ma_int64 pcmPos = static_cast<ma_int64>(static_cast<double>(cursor) * rateScale);

    const int n = analyzer_.fftSize();
    if (static_cast<int>(windowScratch_.size()) != n)
        windowScratch_.assign(static_cast<size_t>(n), 0.0f);

    // Seek / loop / track-change discontinuity: re-seed instead of smearing.
    if (!analysisInit_ || std::llabs(static_cast<ma_int64>(pcmPos) - static_cast<ma_int64>(lastAnalysisCursor_)) >
                                         static_cast<ma_int64>(n * 4)) {
        analyzer_.resetHistory();
        lastAnalysisCursor_ = static_cast<ma_uint64>(pcmPos > 0 ? pcmPos : 0);
        analysisInit_ = true;
    }

    // One FFT per hop of N/2 decoded frames (~43 updates/s at 48 kHz/N=2048:
    // at most one window per UI frame, trivially cheap).
    if (pcmPos >= static_cast<ma_int64>(lastAnalysisCursor_) + n / 2 ||
        pcmPos < static_cast<ma_int64>(lastAnalysisCursor_)) {
        std::lock_guard<std::mutex> lock(trackMutex_);
        if (!track_.mono.empty()) {
            const ma_int64 start = pcmPos - n / 2; // centre the window on the cursor
            const auto& mono = track_.mono;
            for (int i = 0; i < n; ++i) {
                const ma_int64 idx = start + i;
                windowScratch_[static_cast<size_t>(i)] =
                    (idx >= 0 && static_cast<size_t>(idx) < mono.size()) ? mono[static_cast<size_t>(idx)] : 0.0f;
            }
            analyzer_.analyzeWindow(windowScratch_.data(), track_.sampleRate);
            lastAnalysisCursor_ = static_cast<ma_uint64>(pcmPos > 0 ? pcmPos : 0);
            (void)pcmSize;
            return;
        }
    }
    // Cursor barely moved (or lock missed): gentle decay, no freeze.
    analyzer_.updateIdle(dt);
}

// ── Widget rendering ─────────────────────────────────────────────────

void MusicWidget::render() {
    tick();
    pollPlayerState();

    ImGuiHelpers::WindowGuard wg(displayTitle().c_str(), &isOpen);
    if (!wg.visible()) {
        return;
    }

    // ── MUSIC (existing interface, unchanged) ──
    const char* stateText = "Stopped";
    if (playbackState == PlaybackState::Playing) {
        stateText = "Playing";
    } else if (playbackState == PlaybackState::Paused) {
        stateText = "Paused";
    }

    ImGui::Text("State: %s", stateText);
    ImGui::TextWrapped("File: %s", selectedFile.empty() ? "(none selected)" : selectedFile.c_str());

    if (trackDurationSec > 0.0f) {
        float pos = currentPlaybackPositionSeconds();
        if (ImGui::SliderFloat("Seek", &pos, 0.0f, trackDurationSec, "%.1f s")) {
            seekTo(pos);
        }
        ImGui::Text("%.1f / %.1f s", currentPlaybackPositionSeconds(), trackDurationSec);
    } else {
        ImGui::BeginDisabled();
        float disabledSeek = 0.0f;
        ImGui::SliderFloat("Seek", &disabledSeek, 0.0f, 1.0f, "Duration unknown");
        ImGui::EndDisabled();
    }

        if (ImGui::Button(repeatEnabled
            ? reinterpret_cast<const char*>(u8"\uf01e##repeat_on")
            : reinterpret_cast<const char*>(u8"\uf01e##repeat_off"))) {
        repeatEnabled = !repeatEnabled;
        applyRepeatSetting();
    }
    ImGuiHelpers::SetTooltipIfHovered(repeatEnabled ? "Repeat: ON" : "Repeat: OFF");

    ImGui::SameLine();
    if (ImGui::Button(reinterpret_cast<const char*>(u8"\uf07c##open_mp3"))) {
        filePicker_.open(selectedFile.empty() ? std::filesystem::path("music.mp3") : std::filesystem::path(selectedFile));
    }
    ImGuiHelpers::SetTooltipIfHovered("Open MP3 file");

    ImGui::SameLine();
    ImGui::BeginDisabled(selectedFile.empty());
        if (ImGui::Button(playbackState == PlaybackState::Playing
            ? reinterpret_cast<const char*>(u8"\uf04c##toggle_play_pause")
            : reinterpret_cast<const char*>(u8"\uf04b##toggle_play_pause"))) {
        if (playbackState == PlaybackState::Playing) {
            pause();
        } else {
            play();
        }
    }
    ImGui::EndDisabled();
    ImGuiHelpers::SetTooltipIfHovered(playbackState == PlaybackState::Playing ? "Pause" : "Play");

    ImGui::SameLine();
    ImGui::BeginDisabled(playbackState == PlaybackState::Stopped);
    if (ImGui::Button(reinterpret_cast<const char*>(u8"\uf04d##stop"))) {
        stop();
    }
    ImGui::EndDisabled();
    ImGuiHelpers::SetTooltipIfHovered("Stop");

    if (!pickerError.empty()) {
        ImGui::Spacing();
        ImGui::TextColored(ImVec4(1.0f, 0.35f, 0.35f, 1.0f), "%s", pickerError.c_str());
    }

    // ── ANALYSIS (new, same widget — no separate window) ──
    renderAnalysisSection();

    std::filesystem::path chosenFile;
    if (filePicker_.render(chosenFile)) {
        selectedFile = chosenFile.string();
        pickerError.clear();
        loadSelectedTrack();
    }
}

void MusicWidget::renderAnalysisSection() {
    ImGui::Separator();
    ImGui::Text("ANALYSIS");

    // Status line: decode progress / analysis source.
    {
        std::lock_guard<std::mutex> lock(trackMutex_);
        if (decodeInFlight_.load(std::memory_order_acquire) && !trackReady_) {
            ImGui::TextDisabled("Decoding audio for analysis...");
        } else if (trackReady_) {
            const double secs = track_.sampleRate > 0
                                    ? static_cast<double>(track_.mono.size()) / track_.sampleRate
                                    : 0.0;
            ImGui::TextDisabled("Analysing %u Hz mono (%.1f s decoded)", track_.sampleRate, secs);
        } else {
            ImGui::TextDisabled("No audio decoded yet.");
        }
        if (!decodeError_.empty()) {
            ImGui::TextColored(ImVec4(1.0f, 0.35f, 0.35f, 1.0f), "%s", decodeError_.c_str());
        }
    }

    // Live feature readout for the MUSIC_REACTIVE_WATER consumer.
    const AudioFeatures& f = analyzer_.features();
    if (f.valid) {
        ImGui::Text("Amplitude: %.2f  Bass: %.2f  Mid: %.2f  High: %.2f  Beat: %.2f",
                    f.amplitude, f.bassEnergy, f.midEnergy, f.highEnergy, f.beatIntensity);
    } else {
        ImGui::TextDisabled("Amplitude: --  Bass: --  Mid: --  High: --  Beat: --");
    }

    if (showWaveform_) renderWaveform();
    if (showFFT_) renderFftSpectrum();
    renderAnalysisControls();
}

void MusicWidget::renderWaveform() {
    ImGui::Spacing();
    ImGui::Text("WAVEFORM");
    const int size = analyzer_.waveformSize();
    if (size > 8) {
        // Single batched polyline; fixed -1..1 range makes quiet/loud
        // sections and transients immediately obvious.
        ImGui::PlotLines("##music_waveform", analyzer_.waveformData(), size,
                         analyzer_.waveformOffset(), nullptr, -1.0f, 1.0f, ImVec2(-1.0f, 80.0f));
    } else {
        ImGui::BeginDisabled();
        static float empty[2] = {0.0f, 0.0f};
        ImGui::PlotLines("##music_waveform_empty", empty, 2, 0, "Play a track to see the waveform",
                         -1.0f, 1.0f, ImVec2(-1.0f, 80.0f));
        ImGui::EndDisabled();
    }
}

void MusicWidget::renderFftSpectrum() {
    ImGui::Spacing();
    const AudioFeatures& f = analyzer_.features();
    if (f.beatIntensity > 0.45f)
        ImGui::Text("FFT SPECTRUM  (beat)");
    else
        ImGui::Text("FFT SPECTRUM");

    constexpr int kBars = 64;
    uint32_t rate = 0;
    {
        std::lock_guard<std::mutex> lock(trackMutex_);
        rate = trackReady_ ? track_.sampleRate : f.sampleRate;
    }
    if (rate == 0) rate = 48000;
    const float nyquist = static_cast<float>(rate) * 0.5f;

    const std::vector<float>& bars = analyzer_.displayBars(kBars, logFrequency_, rate);

    const float canvasH = 140.0f;
    ImGui::BeginChild("##music_fft_canvas", ImVec2(-1.0f, canvasH), true,
                      ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);
    const ImVec2 min = ImGui::GetCursorScreenPos();
    const ImVec2 avail = ImGui::GetContentRegionAvail();
    if (avail.x <= 4.0f || avail.y <= 4.0f) {
        ImGui::EndChild();
        return;
    }
    constexpr float kLabelH = 14.0f; // bottom strip for frequency labels
    const ImVec2 p0(min.x, min.y);
    const ImVec2 p1(min.x + avail.x, min.y + avail.y - kLabelH);
    const float plotH = p1.y - p0.y;
    ImDrawList* dl = ImGui::GetWindowDrawList();

    // Subtle band backgrounds: bass / mid / high.
    const float bandEdges[4] = {20.0f, 250.0f, 4000.0f, 20000.0f};
    const ImU32 bandCols[3] = {
        IM_COL32(255, 150, 60, 22),  // bass: warm
        IM_COL32(120, 220, 130, 18), // mid: green
        IM_COL32(110, 170, 255, 22), // high: blue
    };
    const char* bandNames[3] = {"BASS", "MID", "HIGH"};
    for (int b = 0; b < 3; ++b) {
        const float x0 = p0.x + freqToT(bandEdges[b], nyquist, logFrequency_) * (p1.x - p0.x);
        const float x1 = p0.x + freqToT(bandEdges[b + 1], nyquist, logFrequency_) * (p1.x - p0.x);
        if (x1 <= x0 + 1.0f) continue;
        dl->AddRectFilled(ImVec2(x0, p0.y), ImVec2(x1, p1.y), bandCols[b]);
        if (x1 - x0 > 34.0f)
            dl->AddText(ImVec2(x0 + 3.0f, p0.y + 2.0f), IM_COL32(255, 255, 255, 110), bandNames[b]);
    }

    // Bars: one rect each (64 draw commands in a single draw list — cheap).
    // Colour follows the band the bar centre falls in.
    const float barW = (p1.x - p0.x) / static_cast<float>(kBars);
    for (int i = 0; i < kBars; ++i) {
        const float v = std::clamp(bars[static_cast<size_t>(i)], 0.0f, 1.0f);
        if (v <= 0.001f) continue;
        // Band of this bar's centre frequency for tinting.
        float fc;
        if (logFrequency_) {
            const float lo = std::min(20.0f, nyquist * 0.5f);
            fc = lo * std::pow(nyquist / lo, (static_cast<float>(i) + 0.5f) / static_cast<float>(kBars));
        } else {
            fc = nyquist * (static_cast<float>(i) + 0.5f) / static_cast<float>(kBars);
        }
        ImU32 col = fc < 250.0f ? IM_COL32(255, 170, 80, 235)
                    : fc < 4000.0f ? IM_COL32(140, 230, 150, 235)
                                   : IM_COL32(130, 185, 255, 235);
        const float x0 = p0.x + static_cast<float>(i) * barW;
        const float x1 = x0 + barW - 1.0f;
        const float y1 = p1.y;
        const float y0 = y1 - v * plotH;
        dl->AddRectFilled(ImVec2(x0, y0), ImVec2(x1, y1), col);
    }

    // Frequency labels (a useful subset, never every bin).
    if (logFrequency_) {
        constexpr float kTicks[] = {20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000};
        char buf[16];
        for (float tick : kTicks) {
            if (tick >= nyquist) continue;
            const float x = p0.x + freqToT(tick, nyquist, true) * (p1.x - p0.x);
            dl->AddLine(ImVec2(x, p1.y), ImVec2(x, p1.y + 3.0f), IM_COL32(255, 255, 255, 90));
            if (tick >= 1000.0f) std::snprintf(buf, sizeof(buf), "%.0fk", tick / 1000.0f);
            else if (tick >= 100.0f) std::snprintf(buf, sizeof(buf), "%.0f", tick);
            else std::snprintf(buf, sizeof(buf), "%.0f", tick);
            dl->AddText(ImVec2(x - 8.0f, p1.y + 3.0f), IM_COL32(255, 255, 255, 130), buf);
        }
    } else {
        char buf[16];
        for (int i = 0; i <= 4; ++i) {
            const float tick = nyquist * static_cast<float>(i) / 4.0f;
            const float x = p0.x + (p1.x - p0.x) * static_cast<float>(i) / 4.0f;
            dl->AddLine(ImVec2(x, p1.y), ImVec2(x, p1.y + 3.0f), IM_COL32(255, 255, 255, 90));
            std::snprintf(buf, sizeof(buf), "%.1fk", tick / 1000.0f);
            dl->AddText(ImVec2(x - 10.0f, p1.y + 3.0f), IM_COL32(255, 255, 255, 130), buf);
        }
    }
    ImGui::EndChild();
}

void MusicWidget::renderAnalysisControls() {
    ImGui::Spacing();
    if (ImGui::CollapsingHeader("Analysis Settings", ImGuiTreeNodeFlags_DefaultOpen)) {
        bool fftChanged = false;
        if (fftSizeIdx_ < 0 || fftSizeIdx_ > 3) fftSizeIdx_ = 2;
        if (ImGui::Combo("FFT Size", &fftSizeIdx_, kFftLabels)) {
            analyzer_.setFftSize(kFftSizes[fftSizeIdx_]);
            windowScratch_.clear(); // force gather-buffer resize
            analysisInit_ = false;  // re-seed at the cursor
            fftChanged = true;
        }
        ImGuiHelpers::SetTooltipIfHovered("FFT window length. Larger = finer frequency detail, slower response.");
        (void)fftChanged;

        if (windowIdx_ < 0 || windowIdx_ > 3) windowIdx_ = 0;
        if (ImGui::Combo("Window", &windowIdx_, kWindowLabels)) {
            analyzer_.setWindowType(static_cast<AudioWindowType>(windowIdx_));
        }
        ImGuiHelpers::SetTooltipIfHovered("Window applied before the FFT (Hann recommended for music).");

        if (ImGui::SliderFloat("Spectrum Smoothing", &spectrumSmoothing_, 0.0f, 0.95f, "%.2f")) {
            analyzer_.setSpectrumSmoothing(spectrumSmoothing_);
        }
        if (ImGui::SliderFloat("Amplitude Smoothing", &amplitudeSmoothing_, 0.0f, 0.95f, "%.2f")) {
            analyzer_.setAmplitudeSmoothing(amplitudeSmoothing_);
        }
        if (ImGui::SliderFloat("Beat Sensitivity", &beatSensitivity_, 1.05f, 3.0f, "%.2f")) {
            analyzer_.setBeatSensitivity(beatSensitivity_);
        }
        ImGuiHelpers::SetTooltipIfHovered("Onset threshold vs recent loudness. Lower = easier to trigger.");

        ImGui::Checkbox("Show Waveform", &showWaveform_);
        ImGui::SameLine();
        ImGui::Checkbox("Show FFT", &showFFT_);
        ImGui::SameLine();
        ImGui::Checkbox("Log Frequency", &logFrequency_);
        ImGui::Checkbox("Music-reactive water output", &reactiveEnabled_);
        ImGuiHelpers::SetTooltipIfHovered("Forward amplitude / bass / mid / high / beat to the water GPU params.");
    }
}
