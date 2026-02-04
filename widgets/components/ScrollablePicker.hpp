#pragma once

#include <cstddef>
#include <functional>
#include <imgui.h>

namespace ImGuiComponents {

// Render a horizontally-scrollable thumbnail picker.
// - id: unique ImGui id for the child region
// - count: number of items
// - currentIndex: reference to the currently selected index (updated if selection changes)
// - getTexture: callback that returns an ImTextureID for a given index (may return nullptr)
// - thumb: thumbnail size in pixels (square)
// - centerOnSelection: if true, the picker will try to keep the selected item centered (may override user scroll)
// - showTooltip: if true, show a tooltip with the index when hovering
// Returns true if the selection changed.
bool ScrollableTexturePicker(const char* id, size_t count, size_t &currentIndex, std::function<ImTextureID(size_t)> getTexture, float thumb = 48.0f, int rows = 1, bool centerOnSelection = false, bool showTooltip = true);

} // namespace ImGuiComponents
