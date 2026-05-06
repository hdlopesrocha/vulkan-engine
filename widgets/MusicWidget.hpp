#pragma once

#include "Widget.hpp"
#include "../third_party/miniaudio/miniaudio.h"
#include <filesystem>
#include <string>

class MusicWidget : public Widget {
public:
    MusicWidget();
    ~MusicWidget() override;

    void render() override;

private:
    enum class PlaybackState {
        Stopped,
        Playing,
        Paused
    };

    std::string selectedFile;
    std::filesystem::path pickerDir;
    char pickerNameBuf[256];
    bool pickerOpenPending;
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

    void openFilePicker();
    void renderFilePickerPopup();
    void executePickerSelection();

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
};
