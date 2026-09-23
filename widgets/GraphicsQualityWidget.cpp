#include "GraphicsQualityWidget.hpp"

#include "../events/EventManager.hpp"
#include "../events/SetGraphicsQualityEvent.hpp"
#include "components/ImGuiHelpers.hpp"

#include <imgui.h>
#include <memory>

GraphicsQualityWidget::GraphicsQualityWidget(EventManager* eventManager_)
    : Widget("Graphics Quality"), eventManager(eventManager_) {
    // Part of the main UI: visible on startup, still togglable from the
    // Windows menu like every other widget.
    isOpen = true;
}

void GraphicsQualityWidget::render() {
    if (!eventManager) return;

    // Decoration-free overlay anchored to the extreme right edge (bottom-right
    // with pivot (1,1)), matching the Stats/Gamepad overlay style. Pivot +
    // ImGuiCond_Always keeps it pinned there even if the ini file moves it.
    const ImGuiWindowFlags flags = ImGuiWindowFlags_NoDecoration |
                                   ImGuiWindowFlags_AlwaysAutoResize |
                                   ImGuiWindowFlags_NoFocusOnAppearing |
                                   ImGuiWindowFlags_NoNav;
    const float padding = 10.0f;
    const ImVec2 display = ImGui::GetIO().DisplaySize;
    ImGui::SetNextWindowPos(ImVec2(display.x - padding, display.y - padding),
                            ImGuiCond_Always, ImVec2(1.0f, 1.0f));
    ImGui::SetNextWindowBgAlpha(0.35f);

    ImGui::Begin(displayTitle().c_str(), nullptr, flags);

    ImGui::TextUnformatted("Graphics");

    if (ImGui::Button("Maximum", ImVec2(120.0f, 0.0f))) {
        eventManager->queue(std::make_shared<SetGraphicsQualityEvent>(GraphicsQuality::Maximum));
    }
    ImGuiHelpers::SetTooltipIfHovered(
        "RT reflections/refraction/thickness/depth, water blur, tessellation and shadows on. "
        "Water volumetric scattering, caustics, glitter and foam restored.");

    if (ImGui::Button("Minimal", ImVec2(120.0f, 0.0f))) {
        eventManager->queue(std::make_shared<SetGraphicsQualityEvent>(GraphicsQuality::Minimal));
    }
    ImGuiHelpers::SetTooltipIfHovered(
        "No RT refraction/reflection/depth/thickness, no water blur, no tessellation, no shadows. "
        "Water volumetric scattering, caustics, glitter and foam disabled.");

    ImGui::End();
}
