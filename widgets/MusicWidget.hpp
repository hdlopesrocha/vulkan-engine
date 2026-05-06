#pragma once

#include "Widget.hpp"
#include <chrono>
#include <filesystem>
#include <string>
#include <sys/types.h>

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

    pid_t playerPid;
    PlaybackState playbackState;
    bool repeatEnabled;
    float trackDurationSec;
    float playbackOffsetSec;
    std::chrono::steady_clock::time_point playbackStartTime;

    void openFilePicker();
    void renderFilePickerPopup();
    void executePickerSelection();
    float queryDurationSeconds(const std::string& filePath) const;
    float currentPlaybackPositionSeconds() const;
    void seekTo(float seconds);
    void terminatePlayerProcess();

    bool launchPlayerProcess(const std::string& filePath, float startSeconds = 0.0f);
    void play();
    void pause();
    void stop();
    void pollPlayerState();
};
