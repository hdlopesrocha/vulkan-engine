#include "MusicWidget.hpp"
#include "components/ImGuiHelpers.hpp"

#include <algorithm>
#include <cerrno>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <cstdlib>
#include <imgui.h>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

namespace {
std::string shellEscapeSingleQuoted(const std::string& value) {
    std::string escaped;
    escaped.reserve(value.size() + 8);
    for (char c : value) {
        if (c == '\'') {
            escaped += "'\\''";
        } else {
            escaped += c;
        }
    }
    return escaped;
}
}

MusicWidget::MusicWidget()
    : Widget("Music Player", u8"\uf001"),
      selectedFile(),
      pickerDir("."),
      pickerNameBuf(),
      pickerOpenPending(false),
      pickerError(),
      playerPid(-1),
    playbackState(PlaybackState::Stopped),
    repeatEnabled(false),
    trackDurationSec(0.0f),
    playbackOffsetSec(0.0f),
    playbackStartTime(std::chrono::steady_clock::now()) {
    const char* defaultName = "music.mp3";
    std::memcpy(pickerNameBuf, defaultName, std::strlen(defaultName));
    pickerNameBuf[std::strlen(defaultName)] = '\0';
}

MusicWidget::~MusicWidget() {
    stop();
    pollPlayerState();
}

void MusicWidget::openFilePicker() {
    pickerOpenPending = true;
    pickerError.clear();

    std::filesystem::path current = selectedFile.empty() ? std::filesystem::path(".") : std::filesystem::path(selectedFile);
    std::error_code ec;
    if (std::filesystem::is_directory(current, ec)) {
        pickerDir = current;
    } else if (current.has_parent_path()) {
        pickerDir = current.parent_path();
    } else {
        pickerDir = std::filesystem::current_path(ec);
    }

    std::filesystem::path absoluteDir = std::filesystem::absolute(pickerDir, ec);
    if (!ec) {
        pickerDir = absoluteDir;
    }

    std::string base = current.filename().string();
    if (base.empty()) {
        base = "music.mp3";
    }
    if (base.size() >= sizeof(pickerNameBuf)) {
        base.resize(sizeof(pickerNameBuf) - 1);
    }
    std::memcpy(pickerNameBuf, base.c_str(), base.size());
    pickerNameBuf[base.size()] = '\0';
}

void MusicWidget::executePickerSelection() {
    std::filesystem::path selected = pickerDir / pickerNameBuf;
    if (selected.extension() != ".mp3") {
        selected += ".mp3";
    }

    std::error_code ec;
    if (!std::filesystem::exists(selected, ec) || !std::filesystem::is_regular_file(selected, ec)) {
        pickerError = "Selected file does not exist.";
        return;
    }

    selectedFile = selected.string();
    trackDurationSec = queryDurationSeconds(selectedFile);
    playbackOffsetSec = 0.0f;
    pickerError.clear();
    ImGui::CloseCurrentPopup();
}

float MusicWidget::queryDurationSeconds(const std::string& filePath) const {
    std::string cmd = "ffprobe -v error -show_entries format=duration "
                      "-of default=noprint_wrappers=1:nokey=1 '" +
                      shellEscapeSingleQuoted(filePath) + "' 2>/dev/null";

    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) {
        return 0.0f;
    }

    char buffer[128] = {0};
    std::string output;
    while (fgets(buffer, sizeof(buffer), pipe)) {
        output += buffer;
    }
    pclose(pipe);

    try {
        const double duration = std::stod(output);
        return duration > 0.0 ? static_cast<float>(duration) : 0.0f;
    } catch (...) {
        return 0.0f;
    }
}

float MusicWidget::currentPlaybackPositionSeconds() const {
    float pos = playbackOffsetSec;
    if (playbackState == PlaybackState::Playing && playerPid > 0) {
        const auto now = std::chrono::steady_clock::now();
        const auto elapsed = std::chrono::duration<float>(now - playbackStartTime).count();
        pos += elapsed;
    }
    if (trackDurationSec > 0.0f && pos > trackDurationSec) {
        pos = trackDurationSec;
    }
    if (pos < 0.0f) {
        pos = 0.0f;
    }
    return pos;
}

void MusicWidget::terminatePlayerProcess() {
    if (playerPid > 0) {
        kill(playerPid, SIGTERM);
        waitpid(playerPid, nullptr, WNOHANG);
    }
    playerPid = -1;
}

void MusicWidget::seekTo(float seconds) {
    if (seconds < 0.0f) {
        seconds = 0.0f;
    }
    if (trackDurationSec > 0.0f && seconds > trackDurationSec) {
        seconds = trackDurationSec;
    }

    playbackOffsetSec = seconds;
    if (playbackState == PlaybackState::Playing || playbackState == PlaybackState::Paused) {
        terminatePlayerProcess();
        playbackState = PlaybackState::Stopped;
        if (!launchPlayerProcess(selectedFile, playbackOffsetSec)) {
            pickerError = "Failed to seek with current audio backend.";
        }
    }
}

bool MusicWidget::launchPlayerProcess(const std::string& filePath, float startSeconds) {
    pid_t pid = fork();
    if (pid < 0) {
        return false;
    }

    if (pid == 0) {
        char secbuf[32] = {0};
        std::snprintf(secbuf, sizeof(secbuf), "%.3f", startSeconds);

        if (startSeconds > 0.0f) {
            execlp("ffplay", "ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", "-ss", secbuf, filePath.c_str(), static_cast<char*>(nullptr));
            execlp("mpv", "mpv", "--no-video", "--really-quiet", "--start", secbuf, filePath.c_str(), static_cast<char*>(nullptr));
        } else {
            execlp("ffplay", "ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", filePath.c_str(), static_cast<char*>(nullptr));
            execlp("mpv", "mpv", "--no-video", "--really-quiet", filePath.c_str(), static_cast<char*>(nullptr));
            execlp("mpg123", "mpg123", "-q", filePath.c_str(), static_cast<char*>(nullptr));
        }
        _exit(127);
    }

    playerPid = pid;
    playbackState = PlaybackState::Playing;
    playbackStartTime = std::chrono::steady_clock::now();
    return true;
}

void MusicWidget::play() {
    pollPlayerState();
    if (selectedFile.empty()) {
        pickerError = "Select an MP3 file first.";
        return;
    }

    if (playbackState == PlaybackState::Paused && playerPid > 0) {
        if (kill(playerPid, SIGCONT) == 0) {
            playbackState = PlaybackState::Playing;
            playbackStartTime = std::chrono::steady_clock::now();
            pickerError.clear();
            return;
        }
        stop();
    }

    if (playbackState == PlaybackState::Playing) {
        return;
    }

    if (!launchPlayerProcess(selectedFile, playbackOffsetSec)) {
        pickerError = "Failed to launch audio player (ffplay/mpg123/mpv).";
    } else {
        pickerError.clear();
    }
}

void MusicWidget::pause() {
    pollPlayerState();
    if (playbackState == PlaybackState::Playing && playerPid > 0) {
        if (kill(playerPid, SIGSTOP) == 0) {
            playbackOffsetSec = currentPlaybackPositionSeconds();
            playbackState = PlaybackState::Paused;
            pickerError.clear();
        }
    }
}

void MusicWidget::stop() {
    pollPlayerState();
    terminatePlayerProcess();
    playbackState = PlaybackState::Stopped;
    playbackOffsetSec = 0.0f;
}

void MusicWidget::pollPlayerState() {
    if (playerPid <= 0) {
        return;
    }

    int status = 0;
    pid_t r = waitpid(playerPid, &status, WNOHANG);
    if (r == playerPid) {
        playerPid = -1;
        if (repeatEnabled && !selectedFile.empty()) {
            playbackOffsetSec = 0.0f;
            if (!launchPlayerProcess(selectedFile, 0.0f)) {
                playbackState = PlaybackState::Stopped;
                pickerError = "Failed to restart track in repeat mode.";
            }
        } else {
            playbackState = PlaybackState::Stopped;
            playbackOffsetSec = 0.0f;
        }
    }
}

void MusicWidget::renderFilePickerPopup() {
    if (pickerOpenPending) {
        ImGui::OpenPopup("Select MP3 File");
        pickerOpenPending = false;
    }

    ImGui::SetNextWindowSize(ImVec2(700.0f, 420.0f), ImGuiCond_Appearing);
    if (!ImGui::BeginPopupModal("Select MP3 File", nullptr, ImGuiWindowFlags_NoCollapse)) {
        return;
    }

    std::error_code ec;
    if (!std::filesystem::exists(pickerDir, ec) || !std::filesystem::is_directory(pickerDir, ec)) {
        pickerDir = std::filesystem::current_path(ec);
    }

    ImGui::TextWrapped("Current directory: %s", pickerDir.string().c_str());
    if (ImGui::Button("Up") && pickerDir.has_parent_path()) {
        pickerDir = pickerDir.parent_path();
    }
    ImGui::SameLine();
    if (ImGui::Button("Root##mp3")) {
        pickerDir = std::filesystem::path("/");
    }
    ImGui::SameLine();
    if (ImGui::Button("Home##mp3")) {
        const char* home = std::getenv("HOME");
        if (home && *home) {
            pickerDir = std::filesystem::path(home);
        }
    }

    std::vector<std::filesystem::directory_entry> dirs;
    std::vector<std::filesystem::directory_entry> files;
    for (const auto& entry : std::filesystem::directory_iterator(pickerDir, ec)) {
        if (ec) {
            break;
        }
        if (entry.is_directory(ec)) {
            dirs.push_back(entry);
        } else if (entry.is_regular_file(ec)) {
            files.push_back(entry);
        }
    }

    auto byName = [](const std::filesystem::directory_entry& a, const std::filesystem::directory_entry& b) {
        return a.path().filename().string() < b.path().filename().string();
    };
    std::sort(dirs.begin(), dirs.end(), byName);
    std::sort(files.begin(), files.end(), byName);

    ImGui::BeginChild("##mp3_picker_entries", ImVec2(0.0f, 260.0f), true);
    for (const auto& d : dirs) {
        std::string label = "[DIR] " + d.path().filename().string();
        if (ImGui::Selectable(label.c_str(), false)) {
            pickerDir = d.path();
        }
    }
    for (const auto& f : files) {
        if (f.path().extension() != ".mp3") {
            continue;
        }
        std::string name = f.path().filename().string();
        if (ImGui::Selectable(name.c_str(), false)) {
            if (name.size() >= sizeof(pickerNameBuf)) {
                name.resize(sizeof(pickerNameBuf) - 1);
            }
            std::memcpy(pickerNameBuf, name.c_str(), name.size());
            pickerNameBuf[name.size()] = '\0';
        }
        if (ImGui::IsItemHovered() && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
            executePickerSelection();
        }
    }
    ImGui::EndChild();

    ImGui::Text("File name:");
    ImGui::InputText("##mp3_picker_name", pickerNameBuf, sizeof(pickerNameBuf));

    if (ImGui::Button("Select")) {
        executePickerSelection();
    }
    ImGui::SameLine();
    if (ImGui::Button("Cancel##mp3_picker")) {
        pickerError.clear();
        ImGui::CloseCurrentPopup();
    }

    if (!pickerError.empty()) {
        ImGui::TextColored(ImVec4(1.0f, 0.35f, 0.35f, 1.0f), "%s", pickerError.c_str());
    }

    ImGui::EndPopup();
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

    ImGui::Checkbox("Repeat", &repeatEnabled);

    if (ImGui::Button("Open MP3...")) {
        openFilePicker();
    }

    ImGui::BeginDisabled(selectedFile.empty());
    if (ImGui::Button(playbackState == PlaybackState::Paused ? "Resume" : "Play")) {
        play();
    }
    ImGui::EndDisabled();

    ImGui::SameLine();
    ImGui::BeginDisabled(playbackState != PlaybackState::Playing);
    if (ImGui::Button("Pause")) {
        pause();
    }
    ImGui::EndDisabled();

    ImGui::SameLine();
    ImGui::BeginDisabled(playbackState == PlaybackState::Stopped);
    if (ImGui::Button("Stop")) {
        stop();
    }
    ImGui::EndDisabled();

    if (!pickerError.empty()) {
        ImGui::Spacing();
        ImGui::TextColored(ImVec4(1.0f, 0.35f, 0.35f, 1.0f), "%s", pickerError.c_str());
    }

    renderFilePickerPopup();
}
