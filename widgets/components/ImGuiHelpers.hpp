// Lightweight ImGui helper utilities for widgets
#pragma once

#include <imgui.h>
#include <cstdarg>

#include "../../events/ControllerContext.hpp"

namespace ImGuiHelpers {

// RAII guard for ImGui windows. Usage:
// ImGuiHelpers::WindowGuard g("Title", &isOpen); if (!g.visible()) return;
class WindowGuard {
public:
    WindowGuard(const char* title, bool* p_open = nullptr, ImGuiWindowFlags flags = 0)
        : m_popen(p_open), m_visible(ImGui::Begin(title, p_open, flags)) {}

    ~WindowGuard() { ImGui::End(); }

    bool visible() const { return m_visible; }

    bool* openPtr() const { return m_popen; }

private:
    bool* m_popen;
    bool m_visible;
};

// Show a tooltip when the previous item is hovered. Accepts printf-style format.
inline void SetTooltipIfHovered(const char* fmt, ...) {
    if (ImGui::IsItemHovered()) {
        va_list args;
        va_start(args, fmt);
        ImGui::BeginTooltip();
        ImGui::TextV(fmt, args);
        ImGui::EndTooltip();
        va_end(args);
    }
}

// Display an image or a placeholder text when texture is not available.
inline void ImageOrUnavailable(ImTextureID tex, const ImVec2& size, const char* unavailableText = "Preview unavailable") {
    if (tex) ImGui::Image(tex, size);
    else ImGui::TextUnformatted(unavailableText);
}

// Draw the active-page category badge (colored dot + CAM/BRU/LIT label) shared
// by the gamepad and controller-parameter widgets.
inline void CategoryBadge(PageCategory category) {
    const char* label = "?";
    ImVec4 col = ImVec4(0.6f, 0.6f, 0.6f, 1.0f);
    switch (category) {
        case PageCategory::CAMERA: label = "CAM"; col = ImVec4(0.2f, 0.5f, 0.9f, 1.0f); break;
        case PageCategory::BRUSH:  label = "BRU"; col = ImVec4(0.2f, 0.9f, 0.3f, 1.0f); break;
        case PageCategory::LIGHT:  label = "LIT"; col = ImVec4(0.9f, 0.9f, 0.2f, 1.0f); break;
    }

    ImDrawList* dl = ImGui::GetWindowDrawList();
    ImVec2 pos = ImGui::GetCursorScreenPos();
    ImGui::Dummy(ImVec2(36, 0));
    ImVec2 center = ImVec2(pos.x + 10.0f, pos.y + 10.0f);
    dl->AddCircleFilled(center, 8.0f, ImGui::ColorConvertFloat4ToU32(col));
    dl->AddText(ImVec2(pos.x + 22.0f, pos.y), ImGui::ColorConvertFloat4ToU32(ImVec4(1,1,1,1)), label);
}

} // namespace ImGuiHelpers
