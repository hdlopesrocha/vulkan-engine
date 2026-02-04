#include "ScrollablePicker.hpp"
#include <imgui.h>

namespace ImGuiComponents {

bool ScrollableTexturePicker(const char* id, size_t count, size_t &currentIndex, std::function<ImTextureID(size_t)> getTexture, float thumb, int rows, bool centerOnSelection, bool showTooltip) {
    if (count == 0) return false;
    bool changed = false;

    // compute child height based on number of rows
    const float vpad = 8.0f;
    float childH = thumb * std::max(1, rows) + vpad;
    // No horizontal scrollbar by default; items will wrap into rows
    ImGui::BeginChild(id, ImVec2(0, childH), false, ImGuiWindowFlags_None);

    // compute how many items fit per row based on available width (use item spacing)
    float availX = ImGui::GetContentRegionAvail().x;
    float itemSpacing = ImGui::GetStyle().ItemSpacing.x;
    int itemsPerRow = std::max(1, static_cast<int>(availX / (thumb + itemSpacing)));

    for (size_t i = 0; i < count; ++i) {
        ImGui::PushID(static_cast<int>(i));
        ImTextureID img = getTexture(i);
        if (!img) {
            ImGui::Dummy(ImVec2(thumb, thumb));
        } else {
            if (i == currentIndex) {
                if (ImGui::ImageButton(img, ImVec2(thumb, thumb))) { currentIndex = i; changed = true; }
                // framed border
                ImVec2 mn = ImGui::GetItemRectMin();
                ImVec2 mx = ImGui::GetItemRectMax();
                ImU32 borderCol = ImGui::GetColorU32(ImGuiCol_Border);
                ImDrawList* dl = ImGui::GetWindowDrawList();
                dl->AddRect(mn, mx, borderCol, 4.0f, 0, 3.0f);
                if (centerOnSelection) ImGui::SetScrollHereX(0.5f);
            } else {
                if (ImGui::ImageButton(img, ImVec2(thumb, thumb))) { currentIndex = i; changed = true; if (centerOnSelection) ImGui::SetScrollHereX(0.5f); }
            }
            if (showTooltip && ImGui::IsItemHovered()) {
                ImGui::SetTooltip("%zu", i);
            }
        }
        ImGui::PopID();

        // layout: insert SameLine when not end of row
        if ((static_cast<int>(i) + 1) % itemsPerRow != 0) ImGui::SameLine();
    }

    ImGui::EndChild();
    return changed;
}

} // namespace ImGuiComponents
