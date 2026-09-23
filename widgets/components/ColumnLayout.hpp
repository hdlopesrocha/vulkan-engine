#pragma once

#include <imgui.h>
#include <algorithm>
#include <functional>
#include <vector>

#include "ImGuiHelpers.hpp"

namespace ImGuiComponents {

constexpr float kColumnWidth = 256.0f;

struct ColumnSection {
    std::function<void()> draw;
    float width = kColumnWidth;
};

inline void ColSeparator(float width = kColumnWidth) {
    const ImVec2 sp = ImGui::GetCursorScreenPos();
    ImGui::GetWindowDrawList()->AddLine(
        ImVec2(sp.x, sp.y), ImVec2(sp.x + width, sp.y),
        ImGui::GetColorU32(ImGuiCol_Separator));
    ImGui::Dummy(ImVec2(width, 1.0f));
}

inline void TooltipOnHover(const char* desc) {
    if (!desc || !*desc) return;
    ImGuiHelpers::SetTooltipIfHovered("%s", desc);
}

inline void FieldLabel(const char* label, const char* desc = nullptr) {
    ImGui::TextUnformatted(label);
    TooltipOnHover(desc);
}

inline bool CheckboxField(const char* label, bool* value, const char* desc = nullptr) {
    const bool changed = ImGui::Checkbox(label, value);
    TooltipOnHover(desc);
    return changed;
}

inline bool SliderFloatField(const char* label, float* value, float min, float max,
                             const char* format = "%.3f", const char* desc = nullptr,
                             ImGuiSliderFlags flags = 0) {
    FieldLabel(label, desc);
    ImGui::SetNextItemWidth(kColumnWidth);
    ImGui::PushID(label);
    const bool changed = ImGui::SliderFloat("##v", value, min, max, format, flags);
    ImGui::PopID();
    TooltipOnHover(desc);
    return changed;
}

inline bool SliderIntField(const char* label, int* value, int min, int max,
                           const char* format = "%d", const char* desc = nullptr) {
    FieldLabel(label, desc);
    ImGui::SetNextItemWidth(kColumnWidth);
    ImGui::PushID(label);
    const bool changed = ImGui::SliderInt("##v", value, min, max, format);
    ImGui::PopID();
    TooltipOnHover(desc);
    return changed;
}

inline bool SliderFloat3Field(const char* label, float* value, float min, float max,
                              const char* format = "%.3f", const char* desc = nullptr) {
    FieldLabel(label, desc);
    ImGui::SetNextItemWidth(kColumnWidth);
    ImGui::PushID(label);
    const bool changed = ImGui::SliderFloat3("##v", value, min, max, format);
    ImGui::PopID();
    TooltipOnHover(desc);
    return changed;
}

inline bool ColorEdit3Field(const char* label, float* color, const char* desc = nullptr) {
    FieldLabel(label, desc);
    ImGui::SetNextItemWidth(kColumnWidth);
    ImGui::PushID(label);
    const bool changed = ImGui::ColorEdit3("##v", color);
    ImGui::PopID();
    TooltipOnHover(desc);
    return changed;
}

inline float EstimateSectionHeight(int controlRows) {
    return 26.0f + static_cast<float>(controlRows) * 42.0f;
}

inline void LayoutSections(const std::vector<ColumnSection>& sections,
                           std::vector<float>& cachedH,
                           const std::vector<float>& estimates) {
    const int n = static_cast<int>(sections.size());
    if (n == 0) return;
    if (static_cast<int>(cachedH.size()) != n) {
        cachedH.assign(n, 170.0f);
        for (int i = 0; i < n && i < static_cast<int>(estimates.size()); ++i) {
            if (estimates[i] > 1.0f) cachedH[i] = estimates[i];
        }
    }

    float availH = ImGui::GetContentRegionAvail().y;
    if (availH < 50.0f) availH = 500.0f;
    const float flowH = availH - 4.0f;

    constexpr float kSectionGap = 8.0f;
    std::vector<int> colOf(n, 0);
    std::vector<float> colH(1, 0.0f);
    std::vector<float> colW(1, 0.0f);
    for (int i = 0; i < n; ++i) {
        const float h = cachedH[i] > 1.0f ? cachedH[i] : 170.0f;
        int best = -1;
        float bestRem = 1e30f;
        for (int c = 0; c < static_cast<int>(colH.size()); ++c) {
            const float rem = flowH - colH[c];
            if ((colH[c] <= 0.0f || rem >= h) && rem - h < bestRem) {
                best = c;
                bestRem = rem - h;
            }
        }
        if (best < 0) {
            best = static_cast<int>(colH.size());
            colH.emplace_back(0.0f);
            colW.emplace_back(0.0f);
        }
        colOf[i] = best;
        colH[best] += h + kSectionGap;
        colW[best] = std::max(colW[best], sections[i].width);
    }

    const float spacingX = ImGui::GetStyle().ItemSpacing.x;
    for (int c = 0; c < static_cast<int>(colH.size()); ++c) {
        if (c > 0) ImGui::SameLine(0.0f, spacingX);
        ImGui::BeginGroup();
        ImGui::Dummy(ImVec2(colW[c], 0.0f));
        ImGui::PushTextWrapPos(ImGui::GetCursorPos().x + colW[c]);
        bool first = true;
        for (int i = 0; i < n; ++i) {
            if (colOf[i] != c) continue;
            if (!first) {
                ImGui::Spacing();
                ColSeparator(colW[c]);
                ImGui::Spacing();
            }
            const float y0 = ImGui::GetCursorScreenPos().y;
            sections[i].draw();
            const float y1 = ImGui::GetCursorScreenPos().y;
            const float measured = y1 - y0;
            if (measured > 1.0f) cachedH[i] = measured;
            first = false;
        }
        ImGui::PopTextWrapPos();
        ImGui::EndGroup();
    }
}

} // namespace ImGuiComponents
