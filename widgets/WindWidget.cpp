#include "WindWidget.hpp"

#include <cstdio>
#include <imgui.h>
#include <string>
#include "components/ImGuiHelpers.hpp"

WindWidget::WindWidget(VegetationRenderer* vegetationRenderer_)
    : Widget("Vegetation Wind", u8"\uf72e"), vegetationRenderer(vegetationRenderer_) {
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

    if (ImGui::CollapsingHeader("Tornadoes")) {
        WindFieldSettings& field = vegetationRenderer->getWindFieldSettings();

        int activeCount = 0;
        for (const TornadoSettings& t : field.tornadoes) {
            if (t.active) ++activeCount;
        }

        // Full-width add: activates the first inactive slot (defaults come
        // from TornadoSettings value-init, so pre-configured slots survive).
        // Disabled when all 4 slots are active.
        ImGui::BeginDisabled(activeCount >= static_cast<int>(kWindFieldMaxTornadoes));
        if (ImGui::Button("Add Tornado##tornado_add", ImVec2(-1, 0))) {
            for (size_t i = 0; i < field.tornadoes.size(); ++i) {
                if (!field.tornadoes[i].active) {
                    field.tornadoes[i].active = true;
                    selectedTornado = static_cast<int>(i);
                    break;
                }
            }
        }
        ImGui::EndDisabled();
        ImGuiHelpers::SetTooltipIfHovered("Activate the first free tornado slot.");

        // Slot list: name + active state. Selection drives the sliders below.
        if (selectedTornado < 0 || selectedTornado >= static_cast<int>(kWindFieldMaxTornadoes)) {
            selectedTornado = 0;
        }
        for (int i = 0; i < static_cast<int>(kWindFieldMaxTornadoes); ++i) {
            const TornadoSettings& t = field.tornadoes[i];
            char selLabel[64];
            std::snprintf(selLabel, sizeof(selLabel), "Tornado %d (%s)##tornado%d_sel",
                          i, t.active ? "active" : "inactive", i);
            if (ImGui::Selectable(selLabel, selectedTornado == i)) {
                selectedTornado = i;
            }
        }

        // Per-selected-tornado sliders. Only the selected slot is drawn, and
        // every label carries a ##tornado{i}_field suffix so no two controls
        // in this window ever share an ImGui ID.
        TornadoSettings& sel = field.tornadoes[static_cast<size_t>(selectedTornado)];
        const int si = selectedTornado;
        auto tornadoLabel = [&](const char* visible, const char* key) {
            char buf[96];
            std::snprintf(buf, sizeof(buf), "%s##tornado%d_%s", visible, si, key);
            return std::string(buf);
        };

        const std::string lBase = tornadoLabel("Base XZ", "base");
        ImGui::SliderFloat2(lBase.c_str(), &sel.baseXZ.x, -2000.0f, 2000.0f, "%.0f");
        ImGuiHelpers::SetTooltipIfHovered("Funnel origin in world XZ (metres).");

        const std::string lGround = tornadoLabel("Ground Y", "ground");
        ImGui::SliderFloat(lGround.c_str(), &sel.groundY, -100.0f, 500.0f, "%.0f");
        ImGuiHelpers::SetTooltipIfHovered("World Y of the funnel base (sea level = 0).");

        const std::string lRadius = tornadoLabel("Radius", "radius");
        ImGui::SliderFloat(lRadius.c_str(), &sel.radius, 5.0f, 200.0f, "%.0f");
        ImGuiHelpers::SetTooltipIfHovered("Rankine core radius in metres.");

        const std::string lHeight = tornadoLabel("Height", "height");
        ImGui::SliderFloat(lHeight.c_str(), &sel.height, 20.0f, 600.0f, "%.0f");
        ImGuiHelpers::SetTooltipIfHovered("Funnel height; updraft decays to zero at the top.");

        const std::string lStrength = tornadoLabel("Strength", "strength");
        ImGui::SliderFloat(lStrength.c_str(), &sel.strength, 0.0f, 100.0f, "%.1f");
        ImGuiHelpers::SetTooltipIfHovered("Max tangential speed at the core edge (m/s).");

        const std::string lDirection = tornadoLabel("Direction", "direction");
        ImGui::SliderFloat(lDirection.c_str(), &sel.direction, -1.0f, 1.0f, "%.2f");
        ImGuiHelpers::SetTooltipIfHovered("-1 = clockwise, +1 = counter-clockwise seen from above, 0 = pure updraft.");

        const std::string lPhase = tornadoLabel("Phase", "phase");
        ImGui::SliderFloat(lPhase.c_str(), &sel.phase, 0.0f, 6.28f, "%.2f");
        ImGuiHelpers::SetTooltipIfHovered("Staggers the wander orbit and direction swing (seconds).");

        const std::string lDelta = tornadoLabel("Delta Time", "deltaTime");
        ImGui::SliderFloat(lDelta.c_str(), &sel.deltaTime, 0.0f, 4.0f, "%.2f");
        ImGuiHelpers::SetTooltipIfHovered("Per-tornado time scale; 1 = real time, 0 = frozen.");

        const std::string lDrift = tornadoLabel("Drift XZ", "drift");
        ImGui::SliderFloat2(lDrift.c_str(), &sel.driftVelocity.x, -20.0f, 20.0f, "%.1f");
        ImGuiHelpers::SetTooltipIfHovered("Constant base translation velocity (m/s).");

        const std::string lWanderR = tornadoLabel("Wander Radius", "wanderRadius");
        ImGui::SliderFloat(lWanderR.c_str(), &sel.wanderRadius, 0.0f, 500.0f, "%.0f");
        ImGuiHelpers::SetTooltipIfHovered("Centre orbit radius around the drifted base (metres).");

        const std::string lWanderS = tornadoLabel("Wander Speed", "wanderSpeed");
        ImGui::SliderFloat(lWanderS.c_str(), &sel.wanderSpeed, 0.0f, 2.0f, "%.2f");
        ImGuiHelpers::SetTooltipIfHovered("Centre orbit speed (radians/second).");

        const std::string lSwingA = tornadoLabel("Swing Amplitude", "swingAmp");
        ImGui::SliderFloat(lSwingA.c_str(), &sel.swingAmplitude, 0.0f, 1.0f, "%.2f");
        ImGuiHelpers::SetTooltipIfHovered("0 = static direction; blends the sign toward oscillation.");

        const std::string lSwingF = tornadoLabel("Swing Frequency", "swingFreq");
        ImGui::SliderFloat(lSwingF.c_str(), &sel.swingFrequency, 0.0f, 2.0f, "%.2f");
        ImGuiHelpers::SetTooltipIfHovered("Direction swing speed (radians/second).");

        // Delete: deactivates the selected slot (red styling precedent:
        // VegetationAtlasEditor "Delete Tile"). Disabled on inactive slots.
        ImGui::BeginDisabled(!sel.active);
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.8f, 0.2f, 0.2f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(1.0f, 0.3f, 0.3f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.6f, 0.1f, 0.1f, 1.0f));
        if (ImGui::Button("Delete Tornado##tornado_delete")) {
            sel.active = false;
        }
        ImGui::PopStyleColor(3);
        ImGui::EndDisabled();
        ImGuiHelpers::SetTooltipIfHovered("Deactivate the selected tornado slot.");

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
