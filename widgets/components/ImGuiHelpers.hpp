// Lightweight ImGui helper utilities for widgets
#pragma once

#include <imgui.h>
#include <cstdarg>

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

} // namespace ImGuiHelpers
