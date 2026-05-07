#include "MusicWidget.hpp"
#include "components/ImGuiHelpers.hpp"

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <cstdlib>
#include <imgui.h>
#include <vector>

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
      soundReady(false) {}

MusicWidget::~MusicWidget() {
    unloadSound();
    if (engineReady) {
        ma_engine_uninit(&engine);
        engineReady = false;
    }
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
    }
}

void MusicWidget::render() {
    pollPlayerState();

    ImGuiHelpers::WindowGuard wg(displayTitle().c_str(), &isOpen);
    if (!wg.visible()) {
        return;
    }

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

    std::filesystem::path chosenFile;
    if (filePicker_.render(chosenFile)) {
        selectedFile = chosenFile.string();
        pickerError.clear();
        loadSelectedTrack();
    }
}
