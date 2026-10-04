#include "SdfWidget.hpp"

#include <imgui.h>
#include "components/ImGuiHelpers.hpp"

SdfWidget::SdfWidget(SdfRenderer* sdf)
    : Widget("SDF Volumes"), sdfRenderer(sdf) {
    isOpen = false;
    if (sdfRenderer) {
        const float d = sdfRenderer->lavaDensity();
        lavaAreaPerFlame = (d > 0.0f) ? (1.0f / d) : 10000.0f;
        lavaScale = sdfRenderer->lavaScale();
        lavaSpikiness = sdfRenderer->lavaSpikiness();
        lavaTipRadius = sdfRenderer->lavaTipRadius();
        lavaBaseRadius = sdfRenderer->lavaBaseRadius();
        lavaHeight = sdfRenderer->lavaHeight();
        lavaSpikeFreq = sdfRenderer->lavaSpikeFreq();
    }
}

void SdfWidget::render() {
    ImGuiHelpers::WindowGuard wg(displayTitle().c_str(), &isOpen);
    if (!wg.visible()) return;

    ImGui::TextWrapped("Generic GPU SDF renderer (surface/volume/emissive/transparent). "
                       "Fire is the first volumetric consumer: flames anchor to "
                       "brush-4 lava terrain as chunks stream in. Traversal, grid "
                       "and raymarcher are shared by all future SDF effects.");
    ImGui::Separator();

    if (ImGui::CollapsingHeader("Fire (SDF volume)", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::SliderFloat("Time scale", &timeScale, 0.0f, 4.0f, "%.2f");
        if (sdfRenderer) {
            if (ImGui::SliderFloat("m² per flame", &lavaAreaPerFlame, 100.0f, 100000.0f, "%.0f",
                                   ImGuiSliderFlags_Logarithmic)) {
                sdfRenderer->setLavaDensity(1.0f / std::max(lavaAreaPerFlame, 1.0f));
            }
            ImGuiHelpers::SetTooltipIfHovered("Lava surface area per flame (default 1 per 10000 m²). Applies to newly streamed chunks.");
            if (ImGui::SliderFloat("Flame scale", &lavaScale, 0.25f, 1024.0f, "%.1f",
                                  ImGuiSliderFlags_Logarithmic)) {
                sdfRenderer->setLavaScale(lavaScale);
            }
            ImGuiHelpers::SetTooltipIfHovered("Flame size multiplier (log scale, up to 1024). Applies to newly streamed chunks.");
            if (ImGui::SliderFloat("Spikiness", &lavaSpikiness, 0.0f, 1.5f, "%.2f")) {
                sdfRenderer->setLavaSpikiness(lavaSpikiness);
            }
            ImGuiHelpers::SetTooltipIfHovered("Flame tongue amplitude (0 = smooth rounded capsule). Applies on next scene rebuild.");
            if (ImGui::SliderFloat("Tip radius", &lavaTipRadius, 0.01f, 32.0f, "%.2f",
                                   ImGuiSliderFlags_Logarithmic)) {
                sdfRenderer->setLavaTipRadius(lavaTipRadius);
            }
            ImGuiHelpers::SetTooltipIfHovered("Flame tip roundness, local units x flame scale (= meters at scale 32). 0.01 ~ sharp cone tip. Applies on next scene rebuild.");
            if (ImGui::SliderFloat("Base radius", &lavaBaseRadius, 0.05f, 32.0f, "%.2f",
                                   ImGuiSliderFlags_Logarithmic)) {
                sdfRenderer->setLavaBaseRadius(lavaBaseRadius);
            }
            ImGuiHelpers::SetTooltipIfHovered("Flame base width, local units x flame scale (= meters at scale 32, up to 1024 m). Applies on next scene rebuild.");
            if (ImGui::SliderFloat("Height", &lavaHeight, 0.5f, 10.0f, "%.2f")) {
                sdfRenderer->setLavaHeight(lavaHeight);
            }
            ImGuiHelpers::SetTooltipIfHovered("Flame base-to-tip height in local units (x instance scale). Applies on next scene rebuild.");
            if (ImGui::SliderFloat("Spike freq", &lavaSpikeFreq, 0.5f, 6.0f, "%.2f")) {
                sdfRenderer->setLavaSpikeFreq(lavaSpikeFreq);
            }
            ImGuiHelpers::SetTooltipIfHovered("Tongue count around the flame axis. Applies on next scene rebuild.");
        }
        if (sdfRenderer) {
            auto st = sdfRenderer->getStats();
            ImGui::Text("flames=%u chunks=%u containers=%u cells=%u draws=%u",
                st.lavaAnchors, st.lavaChunks,
                st.containerCount, st.gridCellCount, st.lastDrawInstances);
        }
    }

    if (sdfRenderer && ImGui::CollapsingHeader("Renderer", ImGuiTreeNodeFlags_DefaultOpen)) {
        static int mode = 1;
        ImGui::RadioButton("Surface", &mode, 0);
        ImGui::SameLine(); ImGui::RadioButton("Volume", &mode, 1);
        ImGui::SameLine(); ImGui::RadioButton("Emissive", &mode, 2);
        ImGui::SameLine(); ImGui::RadioButton("Transparent", &mode, 3);
        sdfRenderer->setRenderMode(static_cast<SdfRenderer::RenderMode>(mode));
        // Debug view: bit 0 = on, bits 1-2 = Steps / Candidates / Field.
        // Candidates BLACK = no flame SDF ever evaluated along the ray.
        // Field WHITE = ray passed close to a flame surface.
        static int dbg = 0;
        ImGui::RadioButton("Debug off", &dbg, 0);
        ImGui::SameLine(); ImGui::RadioButton("Steps", &dbg, 1);
        ImGui::SameLine(); ImGui::RadioButton("Candidates", &dbg, 2);
        ImGui::SameLine(); ImGui::RadioButton("Field", &dbg, 3);
        uint32_t flags = 0u;
        if (dbg == 1) flags = 1u;
        else if (dbg == 2) flags = 1u | (1u << 1u);
        else if (dbg == 3) flags = 1u | (2u << 1u);
        sdfRenderer->setDebugFlags(flags);
    }
}
