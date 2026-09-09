#pragma once

// Hybrid RT debug + performance widget (§19).
// Raster owns primary, CSM owns macro shadows, RT owns secondary visibility.
// Controls: RT reflections/refractions/thickness/local-shadows, water pipeline
// vs inline, ray distances, roughness gate, IOR/absorption, RT debug views,
// CSM vs RT shadow comparison modes. Timings come from the stats overlay.

#include "Widget.hpp"
#include "../utils/Settings.hpp"
#include <imgui.h>

class HybridRTWidget : public Widget {
public:
    explicit HybridRTWidget(Settings& s)
        : Widget("Hybrid RT"), settings(s) {}

    void render() override {
        ImGui::TextWrapped("Raster=primary, CSM=macro shadows, RT=secondary (reflections, refraction, thickness, contact).");
        ImGui::Separator();
        ImGui::Checkbox("RT reflections", &settings.rtReflections);
        ImGui::Checkbox("RT refractions", &settings.rtRefractions);
        ImGui::Checkbox("RT water thickness", &settings.rtThickness);
        ImGui::Checkbox("RT local/contact shadows (augment CSM)", &settings.rtLocalShadows);
        ImGui::Checkbox("Water via RT pipeline (off = inline queries)", &settings.rtWaterPipeline);
        ImGui::Separator();
        ImGui::SliderFloat("Max reflect dist", &settings.rtMaxReflectDist, 10.0f, 2000.0f, "%.0f");
        ImGui::SliderFloat("Max refract dist", &settings.rtMaxRefractDist, 10.0f, 1000.0f, "%.0f");
        ImGui::SliderFloat("Max contact dist", &settings.rtMaxShadowDist, 1.0f, 60.0f, "%.1f");
        ImGui::SliderFloat("Roughness threshold", &settings.rtRoughnessThreshold, 0.0f, 1.0f, "%.2f");
        ImGui::SliderFloat("Self-skip dist", &settings.rtSelfSkipDist, 0.0f, 15.0f, "%.2f");
        ImGui::Separator();
        ImGui::TextWrapped("Water look (IOR, absorption, depth cap, shore fade) lives in Water Settings, per water layer.");
        ImGui::Separator();
        // RT debug views drive rt.debug.x (RT pipeline + raster RT branches).
        const char* rtViews[] = {"Off", "50 Reflect-only", "51 Refract-only", "52 Thickness",
                                 "53 Fresnel", "54 Absorption", "55 CSM-only", "56 RT-local-only",
                                 "57 CSM+RT combined"};
        int rtIdx = 0;
        const int rtVals[] = {0, 50, 51, 52, 53, 54, 55, 56, 57};
        for (int i = 0; i < 9; ++i) if (settings.rtDebugView == rtVals[i]) rtIdx = i;
        if (ImGui::Combo("RT debug view", &rtIdx, rtViews, 9)) {
            settings.rtDebugView = rtVals[rtIdx];
            // Mirror into the raster debugMode so solid AND water show it.
            if (settings.rtDebugView != 0) settings.debugMode = settings.rtDebugView;
        }
        ImGui::TextWrapped("Raster debugMode also selects these views (50-57) for solid+water.");
        ImGui::Separator();
        ImGui::TextWrapped("CSM stays authoritative: keep RT local shadows OFF unless inspecting contact detail. Proxy BLAS is coarse by design — never use RT for macro terrain shadows.");
    }

private:
    Settings& settings;
};
