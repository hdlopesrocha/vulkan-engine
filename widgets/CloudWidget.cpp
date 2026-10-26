#include "CloudWidget.hpp"
#include "components/ColumnLayout.hpp"
#include <vector>

CloudWidget::CloudWidget(CloudSettings& settingsRef)
    : Widget("Clouds", u8"\uf0c2"), settings(settingsRef) {}

namespace {
// Full-width controls line up with the shared column packer.
constexpr float kSettingsColWidth = ImGuiComponents::kColumnWidth;
} // namespace

void CloudWidget::render() {
    // Same window setup as Settings: resizable, horizontal scrollbar only for
    // column overflow, mouse wheel never scrolls vertically.
    auto wg = ImGuiComponents::BeginColumnWindow(displayTitle().c_str(), &isOpen);
    if (!wg.visible()) return;

    // ---- Sections: each must be self-contained (header + controls) so it can
    // ---- be moved as a whole into another column when the current one fills up.
    std::vector<ImGuiComponents::ColumnSection> sections;
    sections.reserve(6);

    // 0: Master
    sections.emplace_back([this]() {
        ImGui::Text("Master");
        ImGuiComponents::ColSeparator();
        if (ImGui::Checkbox("Clouds Enabled", &settings.enabled)) {
        }
        ImGuiComponents::TooltipOnHover("Master toggle for volumetric clouds (sky + shadows + reflections).\nAlso gated by Settings 'Clouds' toggle; Minimal preset disables.");
        ImGuiComponents::FieldLabel("Density Scale", "Global extinction multiplier for all tiers.");
        ImGui::SetNextItemWidth(kSettingsColWidth);
        ImGui::SliderFloat("##Density Scale", &settings.densityScale, 0.0f, 3.0f, "%.2f");
        ImGuiComponents::FieldLabel("Shadow Strength", "How strongly clouds darken terrain/water/vegetation.\n0 = clouds visible but cast no shadow.");
        ImGui::SetNextItemWidth(kSettingsColWidth);
        ImGui::SliderFloat("##Shadow Strength", &settings.shadowStrength, 0.0f, 1.0f, "%.2f");
        ImGuiComponents::TooltipOnHover("How strongly clouds darken terrain/water/vegetation.\n0 = clouds visible but cast no shadow.");
        ImGuiComponents::FieldLabel("Exposure", "Cloud brightness multiplier.");
        ImGui::SetNextItemWidth(kSettingsColWidth);
        ImGui::SliderFloat("##Exposure", &settings.exposure, 0.1f, 3.0f, "%.2f");
        ImGuiComponents::FieldLabel("Raymarch Steps", "Steps per slab in the fullscreen sky pass (equirect uses half).\nMore = denser, slower.");
        ImGui::SetNextItemWidth(kSettingsColWidth);
        ImGui::SliderInt("##Raymarch Steps", &settings.raymarchSteps, 4, 24);
        ImGuiComponents::TooltipOnHover("Steps per slab in the fullscreen sky pass (equirect uses half).\nMore = denser, slower.");
        ImGuiComponents::FieldLabel("Light Steps", "Steps toward the sun per sample (self-shadowing inside clouds).");
        ImGui::SetNextItemWidth(kSettingsColWidth);
        ImGui::SliderInt("##Light Steps", &settings.lightSteps, 1, 6);
        ImGuiComponents::TooltipOnHover("Steps toward the sun per sample (self-shadowing inside clouds).");
    });

    // 1: Low clouds (~2 km)
    sections.emplace_back([this]() {
        ImGui::Text("Low Clouds (~2 km)");
        ImGuiComponents::ColSeparator();
        ImGui::TextWrapped("Stratus, stratocumulus, cumulus. Fog = low cloud at ground.");
        if (ImGui::Checkbox("Low Enabled", &settings.lowEnabled)) {
        }
        ImGuiComponents::FieldLabel("Low Coverage", "0 = clear, 1 = overcast.");
        ImGui::SetNextItemWidth(kSettingsColWidth);
        ImGui::SliderFloat("##Low Coverage", &settings.lowCoverage, 0.0f, 1.0f, "%.2f");
        ImGuiComponents::FieldLabel("Low Density", "Extinction multiplier for this tier.");
        ImGui::SetNextItemWidth(kSettingsColWidth);
        ImGui::SliderFloat("##Low Density", &settings.lowDensity, 0.0f, 3.0f, "%.2f");
        ImGuiComponents::FieldLabel("Low Scale", "Horizontal feature scale (1/m). Smaller = larger cloud masses.");
        ImGui::SetNextItemWidth(kSettingsColWidth);
        ImGui::SliderFloat("##Low Scale", &settings.lowScale, 0.00005f, 0.002f, "%.6f");
        ImGuiComponents::TooltipOnHover("Horizontal feature scale (1/m). Smaller = larger cloud masses.");
        ImGuiComponents::FieldLabel("Low Base (m)", "Slab base height above origin.");
        ImGui::SetNextItemWidth(kSettingsColWidth);
        ImGui::SliderFloat("##Low Base (m)", &settings.lowBaseHeight, 0.0f, 4000.0f, "%.0f");
        ImGuiComponents::FieldLabel("Low Thick (m)", "Slab thickness.");
        ImGui::SetNextItemWidth(kSettingsColWidth);
        ImGui::SliderFloat("##Low Thick (m)", &settings.lowThickness, 50.0f, 1500.0f, "%.0f");
        ImGuiComponents::FieldLabel("Low Wind Mul", "Multiplier on global wind for this tier.");
        ImGui::SetNextItemWidth(kSettingsColWidth);
        ImGui::SliderFloat("##Low Wind Mul", &settings.lowWindSpeedMul, 0.0f, 4.0f, "%.2f");
    });

    // 2: Mid clouds (2-7 km)
    sections.emplace_back([this]() {
        ImGui::Text("Mid Clouds (2-7 km)");
        ImGuiComponents::ColSeparator();
        ImGui::TextWrapped("Altocumulus, altostratus (water/ice mix).");
        if (ImGui::Checkbox("Mid Enabled", &settings.midEnabled)) {
        }
        ImGuiComponents::FieldLabel("Mid Coverage", "0 = clear, 1 = overcast.");
        ImGui::SetNextItemWidth(kSettingsColWidth);
        ImGui::SliderFloat("##Mid Coverage", &settings.midCoverage, 0.0f, 1.0f, "%.2f");
        ImGuiComponents::FieldLabel("Mid Density", "Extinction multiplier for this tier.");
        ImGui::SetNextItemWidth(kSettingsColWidth);
        ImGui::SliderFloat("##Mid Density", &settings.midDensity, 0.0f, 3.0f, "%.2f");
        ImGuiComponents::FieldLabel("Mid Scale", "Horizontal feature scale (1/m).");
        ImGui::SetNextItemWidth(kSettingsColWidth);
        ImGui::SliderFloat("##Mid Scale", &settings.midScale, 0.00005f, 0.002f, "%.6f");
        ImGuiComponents::FieldLabel("Mid Base (m)", "Slab base height above origin.");
        ImGui::SetNextItemWidth(kSettingsColWidth);
        ImGui::SliderFloat("##Mid Base (m)", &settings.midBaseHeight, 2000.0f, 7000.0f, "%.0f");
        ImGuiComponents::FieldLabel("Mid Thick (m)", "Slab thickness.");
        ImGui::SetNextItemWidth(kSettingsColWidth);
        ImGui::SliderFloat("##Mid Thick (m)", &settings.midThickness, 100.0f, 2000.0f, "%.0f");
        ImGuiComponents::FieldLabel("Mid Wind Mul", "Multiplier on global wind for this tier.");
        ImGui::SetNextItemWidth(kSettingsColWidth);
        ImGui::SliderFloat("##Mid Wind Mul", &settings.midWindSpeedMul, 0.0f, 5.0f, "%.2f");
    });

    // 3: High clouds (5-13 km)
    sections.emplace_back([this]() {
        ImGui::Text("High Clouds (5-13 km)");
        ImGuiComponents::ColSeparator();
        ImGui::TextWrapped("Cirrus, cirrocumulus, cirrostratus (ice crystals, streaky).");
        if (ImGui::Checkbox("High Enabled", &settings.highEnabled)) {
        }
        ImGuiComponents::FieldLabel("High Coverage", "0 = clear, 1 = overcast.");
        ImGui::SetNextItemWidth(kSettingsColWidth);
        ImGui::SliderFloat("##High Coverage", &settings.highCoverage, 0.0f, 1.0f, "%.2f");
        ImGuiComponents::FieldLabel("High Density", "Extinction multiplier for this tier.");
        ImGui::SetNextItemWidth(kSettingsColWidth);
        ImGui::SliderFloat("##High Density", &settings.highDensity, 0.0f, 3.0f, "%.2f");
        ImGuiComponents::FieldLabel("High Scale", "Horizontal feature scale (1/m).");
        ImGui::SetNextItemWidth(kSettingsColWidth);
        ImGui::SliderFloat("##High Scale", &settings.highScale, 0.00005f, 0.002f, "%.6f");
        ImGuiComponents::FieldLabel("High Base (m)", "Slab base height above origin.");
        ImGui::SetNextItemWidth(kSettingsColWidth);
        ImGui::SliderFloat("##High Base (m)", &settings.highBaseHeight, 5000.0f, 13000.0f, "%.0f");
        ImGuiComponents::FieldLabel("High Thick (m)", "Slab thickness.");
        ImGui::SetNextItemWidth(kSettingsColWidth);
        ImGui::SliderFloat("##High Thick (m)", &settings.highThickness, 100.0f, 2500.0f, "%.0f");
        ImGuiComponents::FieldLabel("High Wind Mul", "Multiplier on global wind for this tier.");
        ImGui::SetNextItemWidth(kSettingsColWidth);
        ImGui::SliderFloat("##High Wind Mul", &settings.highWindSpeedMul, 0.0f, 6.0f, "%.2f");
    });

    // 4: Wind / Animation
    sections.emplace_back([this]() {
        ImGui::Text("Wind / Animation");
        ImGuiComponents::ColSeparator();
        ImGuiComponents::FieldLabel("Wind Response", "Cloud gain on the shared wind field (Wind widget: ambient direction/strength, gusts, tornadoes). 0 = clouds ignore wind.");
        ImGui::SetNextItemWidth(kSettingsColWidth);
        ImGui::SliderFloat("##Wind Response", &settings.windResponse, 0.0f, 10.0f, "%.2f");
        ImGuiComponents::TooltipOnHover("Cloud gain on the shared wind field (Wind widget: ambient direction/strength, gusts, tornadoes). 0 = clouds ignore wind.");
        ImGuiComponents::FieldLabel("Time Scale", "Animation speed multiplier.");
        ImGui::SetNextItemWidth(kSettingsColWidth);
        ImGui::SliderFloat("##Time Scale", &settings.timeScale, 0.0f, 4.0f, "%.2f");
        ImGuiComponents::FieldLabel("Detail", "Edge erosion detail (second noise octave). 0 = smooth puffs.");
        ImGui::SetNextItemWidth(kSettingsColWidth);
        ImGui::SliderFloat("##Detail", &settings.detailStrength, 0.0f, 1.0f, "%.2f");
        ImGuiComponents::TooltipOnHover("Edge erosion detail (second noise octave). 0 = smooth puffs.");
    });

    // 5: Lighting
    sections.emplace_back([this]() {
        ImGui::Text("Lighting");
        ImGuiComponents::ColSeparator();
        ImGuiComponents::FieldLabel("Silver Lining", "Bright rim when looking near the sun through thin cloud.");
        ImGui::SetNextItemWidth(kSettingsColWidth);
        ImGui::SliderFloat("##Silver Lining", &settings.silverLining, 0.0f, 2.0f, "%.2f");
        ImGuiComponents::TooltipOnHover("Bright rim when looking near the sun through thin cloud.");
        ImGuiComponents::FieldLabel("Ambient Boost", "Ambient lift inside clouds.");
        ImGui::SetNextItemWidth(kSettingsColWidth);
        ImGui::SliderFloat("##Ambient Boost", &settings.ambientBoost, 0.0f, 1.0f, "%.2f");
        ImGuiComponents::FieldLabel("Forward g", "Henyey-Greenstein asymmetry: 0 = isotropic, >0 = forward scatter.");
        ImGui::SetNextItemWidth(kSettingsColWidth);
        ImGui::SliderFloat("##Forward g", &settings.sunForwardG, -0.9f, 0.9f, "%.2f");
        ImGuiComponents::TooltipOnHover("Henyey-Greenstein asymmetry: 0 = isotropic, >0 = forward scatter.");
    });

    // ---- Packing: best-fit over the last frame's measured section heights.
    static std::vector<float> cachedH;
    static const std::vector<float> estimates = {260.0f, 340.0f, 340.0f,
                                                 340.0f, 220.0f, 200.0f};
    ImGuiComponents::LayoutSections(sections, cachedH, estimates);
}
