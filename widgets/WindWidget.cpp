#include "WindWidget.hpp"

#include <imgui.h>
#include "components/ImGuiHelpers.hpp"

WindWidget::WindWidget(VegetationRenderer* vegetationRenderer)
    : Widget("Vegetation Wind", u8"\uf72e"), vegetationRenderer(vegetationRenderer) {
    isOpen = false;
}

void WindWidget::render() {
    if (!vegetationRenderer) return;

    VegetationRenderer::WindSettings& wind = vegetationRenderer->getWindSettings();
    VegetationRenderer::DistanceDensitySettings& density = vegetationRenderer->getDistanceDensitySettings();

    ImGuiHelpers::WindowGuard wg(displayTitle().c_str(), &isOpen);
    if (!wg.visible()) return;

    if (ImGui::CollapsingHeader("Wind", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Checkbox("Enable Wind", &wind.enabled);

        ImGui::SliderFloat2("Direction (X/Z)", &wind.direction.x, -1.0f, 1.0f, "%.2f");
        ImGuiHelpers::SetTooltipIfHovered("Wind direction in world XZ plane.");

        ImGui::SliderFloat("Strength", &wind.strength, 0.0f, 24.0f, "%.2f");
        ImGuiHelpers::SetTooltipIfHovered("Overall horizontal bend amount.");

        ImGui::SliderFloat("Base Frequency", &wind.baseFrequency, 0.0001f, 0.05f, "%.5f");
        ImGuiHelpers::SetTooltipIfHovered("Scale of large wind waves.");

        ImGui::SliderFloat("Speed", &wind.speed, 0.0f, 10.0f, "%.2f");
        ImGuiHelpers::SetTooltipIfHovered("How fast wind noise advects over time.");

        ImGui::SliderFloat("Gust Frequency", &wind.gustFrequency, 0.0001f, 0.12f, "%.5f");
        ImGuiHelpers::SetTooltipIfHovered("Scale of smaller gust patterns.");

        ImGui::SliderFloat("Gust Strength", &wind.gustStrength, 0.0f, 2.0f, "%.2f");
        ImGuiHelpers::SetTooltipIfHovered("How strongly gust noise amplifies sway.");

        ImGui::SliderFloat("Skew Amount", &wind.skewAmount, 0.0f, 10.0f, "%.2f");
        ImGuiHelpers::SetTooltipIfHovered("Side skew of billboard tops to simulate blade lean.");

        ImGui::SliderFloat("Trunk Stiffness", &wind.trunkStiffness, 0.0f, 1.0f, "%.2f");
        ImGuiHelpers::SetTooltipIfHovered("0 = very stiff base, 1 = bends from base to tip.");

        ImGui::SliderFloat("Noise Scale", &wind.noiseScale, 0.1f, 6.0f, "%.2f");
        ImGuiHelpers::SetTooltipIfHovered("Global scale multiplier for both Perlin fields.");

        ImGui::SliderFloat("Vertical Flutter", &wind.verticalFlutter, 0.0f, 2.0f, "%.2f");
        ImGuiHelpers::SetTooltipIfHovered("Adds subtle upward flicker at blade tips.");

        ImGui::SliderFloat("Turbulence", &wind.turbulence, 0.0f, 2.0f, "%.2f");
        ImGuiHelpers::SetTooltipIfHovered("Cross-wind variation so chunks do not sway uniformly.");

        if (ImGui::Button("Reset Wind Defaults")) {
            wind = VegetationRenderer::WindSettings{};
        }
    }

    if (ImGui::CollapsingHeader("Distance Density", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Checkbox("Enable Distance Density", &density.enabled);
        ImGuiHelpers::SetTooltipIfHovered("Reduce vegetation instance counts for distant chunks by rewriting indirect draw counts.");

        ImGui::SliderFloat("Full Density Distance", &density.fullDensityDistance, 1.0f, 4096.0f, "%.0f");
        ImGuiHelpers::SetTooltipIfHovered("Chunks at or nearer than this distance keep their full vegetation count.");

        float minDistance = density.minDensityDistance;
        if (ImGui::SliderFloat("Minimum Density Distance", &minDistance, 2.0f, 16384.0f, "%.0f")) {
            density.minDensityDistance = minDistance;
        }
        if (density.minDensityDistance <= density.fullDensityDistance) {
            density.minDensityDistance = density.fullDensityDistance + 1.0f;
        }
        ImGuiHelpers::SetTooltipIfHovered("Beyond this distance, chunks clamp to the minimum density factor.");

        ImGui::SliderFloat("Minimum Density Factor", &density.minDensityFactor, 0.01f, 1.0f, "%.2f");
        ImGuiHelpers::SetTooltipIfHovered("Fraction of original instances preserved for the farthest chunks.");

        if (ImGui::Button("Reset Distance Defaults")) {
            density = VegetationRenderer::DistanceDensitySettings{};
        }
    }
}
