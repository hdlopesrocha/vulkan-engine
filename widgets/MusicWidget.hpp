#pragma once

#include "Widget.hpp"
#include "components/FilePicker.hpp"
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
};
