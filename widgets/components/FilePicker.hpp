#pragma once

#include <filesystem>
#include <string>
#include <vector>

// Reusable ImGui file-picker popup component.
//
// Usage:
//   FilePicker picker("Select MP3 File", ".mp3");
//   picker.addBookmark(reinterpret_cast<const char*>(u8"\uf07c##bm"), "Scenes", "scenes");
//
//   // To open (e.g. from a button/menu click):
//   picker.open(currentPath);
//
//   // Every frame in your ImGui render (any scope):
//   std::filesystem::path chosen;
//   if (picker.render(chosen)) {
//       // use chosen
//   }
class FilePicker {
public:
    struct Bookmark {
        std::string iconLabel; // full ImGui button label, e.g. "\uf07c##id"
        std::string tooltip;
        std::filesystem::path path;
    };

    // popupId:   unique ImGui popup title / ID string, e.g. "Select MP3 File"
    // extension: file-extension filter including dot, e.g. ".mp3"
    FilePicker(std::string popupId, std::string extension);

    // Add an optional bookmark navigation button shown after the Home button.
    void addBookmark(std::string iconLabel, std::string tooltip, std::filesystem::path bookmarkPath);

    // Open the picker popup.  initialPath seeds the initial directory and filename.
    // saveMode: skip existence check on confirm and show all regular files in the listing.
    // header:   optional one-line text rendered inside the popup (e.g. "Mode: Save").
    void open(std::filesystem::path initialPath = ".", bool saveMode = false, std::string header = "");

    bool isSaveMode() const { return saveMode_; }

    // Call every frame inside any ImGui scope.
    // Returns true and fills outPath when the user confirms a selection.
    bool render(std::filesystem::path& outPath);

private:
    std::string popupId_;
    std::string extension_;
    std::filesystem::path pickerDir_;
    char nameBuf_[256];
    bool openPending_ = false;
    bool saveMode_ = false;
    std::string header_;
    std::string error_;
    std::vector<Bookmark> bookmarks_;

    bool confirmSelection(std::filesystem::path& outPath);
};
