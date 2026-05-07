#include "FilePicker.hpp"
#include "ImGuiHelpers.hpp"

#include <imgui.h>
#include <algorithm>
#include <cstring>
#include <cstdlib>
#include <vector>

FilePicker::FilePicker(std::string popupId, std::string extension)
    : popupId_(std::move(popupId))
    , extension_(std::move(extension))
    , pickerDir_(".")
    , nameBuf_()
{}

void FilePicker::addBookmark(std::string iconLabel, std::string tooltip, std::filesystem::path bookmarkPath) {
    bookmarks_.push_back({std::move(iconLabel), std::move(tooltip), std::move(bookmarkPath)});
}

void FilePicker::open(std::filesystem::path initialPath, bool saveMode, std::string header) {
    saveMode_ = saveMode;
    header_ = std::move(header);
    openPending_ = true;
    error_.clear();

    if (initialPath.empty()) {
        initialPath = ".";
    }

    std::error_code ec;
    std::filesystem::path baseDir;
    if (std::filesystem::is_directory(initialPath, ec)) {
        baseDir = initialPath;
    } else if (initialPath.has_parent_path()) {
        baseDir = initialPath.parent_path();
    } else {
        baseDir = std::filesystem::current_path(ec);
    }
    if (baseDir.empty()) {
        baseDir = std::filesystem::current_path(ec);
    }
    pickerDir_ = std::filesystem::absolute(baseDir, ec);
    if (ec) {
        pickerDir_ = baseDir;
    }

    std::string filename = initialPath.filename().string();
    if (filename.empty() || std::filesystem::is_directory(initialPath, ec)) {
        filename = "file" + extension_;
    }
    if (filename.size() >= sizeof(nameBuf_)) {
        filename.resize(sizeof(nameBuf_) - 1);
    }
    std::memcpy(nameBuf_, filename.c_str(), filename.size());
    nameBuf_[filename.size()] = '\0';
}

bool FilePicker::confirmSelection(std::filesystem::path& outPath) {
    std::filesystem::path selected = pickerDir_ / nameBuf_;
    if (selected.extension() != extension_) {
        selected += extension_;
    }

    std::error_code ec;
    if (!saveMode_ && (!std::filesystem::exists(selected, ec) || !std::filesystem::is_regular_file(selected, ec))) {
        error_ = "Selected file does not exist.";
        return false;
    }

    outPath = selected;
    error_.clear();
    ImGui::CloseCurrentPopup();
    return true;
}

bool FilePicker::render(std::filesystem::path& outPath) {
    if (openPending_) {
        ImGui::OpenPopup(popupId_.c_str());
        openPending_ = false;
    }

    ImVec2 center = ImGui::GetMainViewport()->GetCenter();
    ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
    ImGui::SetNextWindowSize(ImVec2(700.0f, 420.0f), ImGuiCond_Appearing);
    if (!ImGui::BeginPopupModal(popupId_.c_str(), nullptr, ImGuiWindowFlags_NoCollapse)) {
        return false;
    }

    bool confirmed = false;
    std::error_code ec;

    if (!std::filesystem::exists(pickerDir_, ec) || !std::filesystem::is_directory(pickerDir_, ec)) {
        pickerDir_ = std::filesystem::current_path(ec);
    }

    if (!header_.empty()) {
        ImGui::TextUnformatted(header_.c_str());
    }

    ImGui::TextWrapped("Current directory: %s", pickerDir_.string().c_str());

    if (ImGui::Button(reinterpret_cast<const char*>(u8"\uf062##fp_up")) && pickerDir_.has_parent_path()) {
        pickerDir_ = pickerDir_.parent_path();
    }
    ImGuiHelpers::SetTooltipIfHovered("Up one folder");
    ImGui::SameLine();

    if (ImGui::Button(reinterpret_cast<const char*>(u8"\uf0ac##fp_root"))) {
        pickerDir_ = std::filesystem::path("/");
    }
    ImGuiHelpers::SetTooltipIfHovered("Go to root (/)");
    ImGui::SameLine();

    if (ImGui::Button(reinterpret_cast<const char*>(u8"\uf015##fp_home"))) {
        const char* home = std::getenv("HOME");
        if (home && *home) {
            pickerDir_ = std::filesystem::path(home);
        }
    }
    ImGuiHelpers::SetTooltipIfHovered("Go to home folder");

    for (const auto& bm : bookmarks_) {
        ImGui::SameLine();
        if (ImGui::Button(bm.iconLabel.c_str())) {
            pickerDir_ = bm.path;
        }
        ImGuiHelpers::SetTooltipIfHovered("%s", bm.tooltip.c_str());
    }

    // Directory listing
    std::vector<std::filesystem::directory_entry> dirs;
    std::vector<std::filesystem::directory_entry> files;
    for (const auto& entry : std::filesystem::directory_iterator(pickerDir_, ec)) {
        if (ec) break;
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

    std::string childId = "##fp_entries_" + popupId_;
    ImGui::BeginChild(childId.c_str(), ImVec2(0.0f, 260.0f), true);
    for (const auto& d : dirs) {
        std::string label = "[DIR] " + d.path().filename().string();
        if (ImGui::Selectable(label.c_str(), false)) {
            pickerDir_ = d.path();
        }
    }
    for (const auto& f : files) {
        const bool matchesExt = f.path().extension() == extension_;
        if (!saveMode_ && !matchesExt) {
            continue;
        }
        const std::string name = f.path().filename().string();
        if (ImGui::Selectable(name.c_str(), false)) {
            std::string clipped = name;
            if (clipped.size() >= sizeof(nameBuf_)) {
                clipped.resize(sizeof(nameBuf_) - 1);
            }
            std::memcpy(nameBuf_, clipped.c_str(), clipped.size());
            nameBuf_[clipped.size()] = '\0';
        }
        if (ImGui::IsItemHovered() && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
            if (confirmSelection(outPath)) {
                confirmed = true;
            }
        }
    }
    ImGui::EndChild();

    ImGui::Text("File name:");
    std::string inputId = "##fp_name_" + popupId_;
    ImGui::InputText(inputId.c_str(), nameBuf_, sizeof(nameBuf_));

    if (ImGui::Button(reinterpret_cast<const char*>(u8"\uf00c##fp_select"))) {
        if (confirmSelection(outPath)) {
            confirmed = true;
        }
    }
    ImGuiHelpers::SetTooltipIfHovered("Select file");
    ImGui::SameLine();

    if (ImGui::Button(reinterpret_cast<const char*>(u8"\uf00d##fp_cancel"))) {
        error_.clear();
        ImGui::CloseCurrentPopup();
    }
    ImGuiHelpers::SetTooltipIfHovered("Cancel");

    if (!error_.empty()) {
        ImGui::TextColored(ImVec4(1.0f, 0.35f, 0.35f, 1.0f), "%s", error_.c_str());
    }

    ImGui::EndPopup();
    return confirmed;
}
