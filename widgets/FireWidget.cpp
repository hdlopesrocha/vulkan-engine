#include "FireWidget.hpp"

#include <imgui.h>
#include "components/ImGuiHelpers.hpp"

FireWidget::FireWidget(VegetationRenderer* vegetationRenderer_)
    : Widget("Fire"), vegetationRenderer(vegetationRenderer_) {
    isOpen = false;
}

void FireWidget::render() {
    if (!vegetationRenderer) return;

    VegetationRenderer::FireSettings& fire = vegetationRenderer->getFireSettings();

    ImGuiHelpers::WindowGuard wg(displayTitle().c_str(), &isOpen);
    if (!wg.visible()) return;

    ImGui::TextWrapped("Animated fire billboards. Fire instances are created from lava terrain (brush index 4) "
                       "and render procedurally on the crossed-plane mesh instead of the atlas texture.");
    ImGui::Separator();

    if (ImGui::CollapsingHeader("Fire", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Checkbox("Enable Fire", &fire.enabled);
        ImGuiHelpers::SetTooltipIfHovered("Master switch. Disabled fire instances collapse (no atlas fallback).");

        ImGui::SliderFloat("Size", &fire.size, 0.5f, 30.0f, "%.2f");
        ImGuiHelpers::SetTooltipIfHovered("Fire quad width in world units. Distant impostor snapshots are captured at this size — "
                                           "re-capture via the Impostor Viewer after changing it.");

        ImGui::SliderFloat("Height Scale", &fire.heightScale, 0.2f, 4.0f, "%.2f");
        ImGuiHelpers::SetTooltipIfHovered("Vertical stretch of the flame quad.");

        ImGui::SliderFloat("Density", &fire.density, 0.0f, 1.0f, "%.3f");
        ImGuiHelpers::SetTooltipIfHovered("Fraction of lava-terrain (brush 4) instances that become fire. "
                                           "Applies to newly generated chunks (reload or regenerate the scene to respawn).");

        ImGui::SliderFloat("Alpha", &fire.alpha, 0.0f, 1.0f, "%.2f");
        ImGuiHelpers::SetTooltipIfHovered("Global flame opacity multiplier.");

        // Snapshots are static: re-bake the far-fire impostor views after
        // tuning size/height/noise/colors so distant flames keep matching.
        if (ImGui::Button("Recapture Fire Impostors")) {
            if (onRecaptureFire) onRecaptureFire();
        }
        ImGuiHelpers::SetTooltipIfHovered("Re-bakes only the fire snapshot layers (60-79) from the current settings.");

        if (ImGui::Button("Reset Fire Defaults")) {
            fire = VegetationRenderer::FireSettings{};
        }
    }

    if (ImGui::CollapsingHeader("Animation", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::SliderFloat("Speed", &fire.speed, 0.0f, 8.0f, "%.2f");
        ImGuiHelpers::SetTooltipIfHovered("Animation speed multiplier.");

        ImGui::SliderFloat("Rise Speed", &fire.riseSpeed, 0.0f, 5.0f, "%.2f");
        ImGuiHelpers::SetTooltipIfHovered("How fast the flames scroll upward.");

        ImGui::SliderFloat("Flicker", &fire.flicker, 0.0f, 1.0f, "%.2f");
        ImGuiHelpers::SetTooltipIfHovered("Temporal flicker amount (breathing + shimmer).");

        ImGui::SliderFloat("Noise Scale", &fire.noiseScale, 0.5f, 8.0f, "%.2f");
        ImGuiHelpers::SetTooltipIfHovered("Flame noise frequency (higher = smaller tongues).");

        ImGui::SliderFloat("Turbulence", &fire.turbulence, 0.0f, 2.0f, "%.2f");
        ImGuiHelpers::SetTooltipIfHovered("Sideways flame distortion and sway.");

        ImGui::SliderFloat("Wind Influence", &fire.windInfluence, 0.0f, 1.0f, "%.2f");
        ImGuiHelpers::SetTooltipIfHovered("How much wind bends the flames.");
    }

    if (ImGui::CollapsingHeader("Look", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::SliderFloat("Intensity", &fire.intensity, 0.0f, 4.0f, "%.2f");
        ImGuiHelpers::SetTooltipIfHovered("Overall brightness multiplier.");

        ImGui::SliderFloat("Emissive", &fire.emissive, 0.0f, 8.0f, "%.2f");
        ImGuiHelpers::SetTooltipIfHovered("Emissive boost — fire glows instead of receiving light.");

        ImGui::SliderFloat("Smoke", &fire.smoke, 0.0f, 1.0f, "%.2f");
        ImGuiHelpers::SetTooltipIfHovered("Smoke veil above the flame.");

        ImGui::ColorEdit3("Inner Color", &fire.innerColor.x);
        ImGuiHelpers::SetTooltipIfHovered("Hot core color (near-white yellow).");

        ImGui::ColorEdit3("Mid Color", &fire.midColor.x);
        ImGuiHelpers::SetTooltipIfHovered("Mid flame color (orange).");

        ImGui::ColorEdit3("Outer Color", &fire.outerColor.x);
        ImGuiHelpers::SetTooltipIfHovered("Flame edge color (dark red).");
    }
}
