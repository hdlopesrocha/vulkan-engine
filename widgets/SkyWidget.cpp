#include "SkyWidget.hpp"

SkyWidget::SkyWidget(SkySettings &settingsRef) : Widget("Sky"), settings(settingsRef) {}

void SkyWidget::render() {
    if (!isOpen) return;
    if (ImGui::Begin(title.c_str(), &isOpen)) {
        // Sky mode selection
        const char* skyModeItems[] = { "Gradient", "Grid" };
        int currentMode = static_cast<int>(settings.mode);
        if (ImGui::Combo("Sky Mode", &currentMode, skyModeItems, IM_ARRAYSIZE(skyModeItems))) {
            settings.mode = static_cast<SkySettings::Mode>(currentMode);
        }
        ImGui::Separator();

        float hc[3] = { settings.horizonColor.r, settings.horizonColor.g, settings.horizonColor.b };
        if (ImGui::ColorEdit3("Horizon Color", hc)) {
            settings.horizonColor = glm::vec3(hc[0], hc[1], hc[2]);
        }
        float zc[3] = { settings.zenithColor.r, settings.zenithColor.g, settings.zenithColor.b };
        if (ImGui::ColorEdit3("Zenith Color", zc)) {
            settings.zenithColor = glm::vec3(zc[0], zc[1], zc[2]);
        }
        ImGui::Separator();
        ImGui::Text("Night settings:");
        float nh[3] = { settings.nightHorizon.r, settings.nightHorizon.g, settings.nightHorizon.b };
        if (ImGui::ColorEdit3("Night Horizon", nh)) settings.nightHorizon = glm::vec3(nh[0], nh[1], nh[2]);
        float nz[3] = { settings.nightZenith.r, settings.nightZenith.g, settings.nightZenith.b };
        if (ImGui::ColorEdit3("Night Zenith", nz)) settings.nightZenith = glm::vec3(nz[0], nz[1], nz[2]);
        ImGui::SliderFloat("Night Intensity", &settings.nightIntensity, 0.0f, 1.0f);
        ImGui::SliderFloat("Star Intensity", &settings.starIntensity, 0.0f, 2.0f);
        ImGui::Separator();
        ImGui::SliderFloat("Warmth", &settings.warmth, 0.0f, 1.0f, "%.2f");
        ImGui::SliderFloat("Exponent", &settings.exponent, 0.1f, 5.0f, "%.2f");
        ImGui::SliderFloat("Sun Flare", &settings.sunFlare, 0.0f, 2.0f, "%.2f");
        ImGui::Text("Note: Light direction is controlled by Light widget");
    }
    ImGui::End();
}

// Accessors are now inlined in the header (forward to settings)
