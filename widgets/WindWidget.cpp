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

        ImGui::TextWrapped(
            "The wind velocity IS the 3D fBm field below: there is no authored "
            "direction, the local fBm vector sets it. All consumers (vegetation, "
            "clouds, fire, wind trails) sample that one field.");

        if (ImGui::Button("Reset Wind Defaults")) {
            wind = VegetationRenderer::WindSettings{};
        }
    }

    // ── Vector fBm wind field ───────────────────────────────────────────────
    // Every axis is its own octave stack over the shared Perlin field, so the
    // wind velocity is a true 3D vector: its local direction comes from the
    // noise, nothing is authored.
    if (ImGui::CollapsingHeader("FBM Field", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::TextWrapped(
            "The 3D vector fBm IS the wind velocity, shared by vegetation, clouds, "
            "fire and the wind trails.");

        ImGui::SliderFloat("Frequency", &wind.fbmFrequency, 0.0002f, 0.05f, "%.5f");
        ImGuiHelpers::SetTooltipIfHovered(
            "Spatial frequency of the first octave (1/m). Smaller = larger gust cells\n"
            "(0.002 ~= 500 m features).");

        ImGui::SliderInt("Octaves", &wind.fbmOctaves, 1,
                         static_cast<int>(kWindFbmMaxOctaves));
        ImGuiHelpers::SetTooltipIfHovered(
            "Number of noise layers. Each octave adds detail at lacunarity x frequency\n"
            "and gain x amplitude. Cost scales linearly: 3 noise samples per octave\n"
            "per wind evaluation, so lower this first if the field is too slow.");

        ImGui::SliderFloat("Lacunarity", &wind.fbmLacunarity, 1.0f, 4.0f, "%.2f");
        ImGuiHelpers::SetTooltipIfHovered("Frequency multiplier between octaves (classic fBm: 2.0).");

        ImGui::SliderFloat("Gain", &wind.fbmGain, 0.0f, 0.95f, "%.2f");
        ImGuiHelpers::SetTooltipIfHovered(
            "Amplitude multiplier between octaves (a.k.a. persistence).\n"
            "0 = only the first octave, 0.5 = classic fBm detail.");

        ImGui::SliderFloat("Amplitude", &wind.fbmAmplitude, 0.0f, 50.0f, "%.2f m/s");
        ImGuiHelpers::SetTooltipIfHovered(
            "Peak wind speed of the fBm field, in m/s. 0 disables the wind\n"
            "(there is no separate mean flow).");

        ImGui::SliderFloat("Vertical Weight", &wind.fbmVertical, 0.0f, 1.0f, "%.2f");
        ImGuiHelpers::SetTooltipIfHovered(
            "Authority of the Y (vertical) component: 1 = all three axes equal\n"
            "(a true 3D field), 0 = horizontal gusts only.");

        ImGui::SliderFloat("Speed", &wind.speed, 0.0f, 10.0f, "%.2f");
        ImGuiHelpers::SetTooltipIfHovered(
            "Temporal evolution speed of the fBm field (0 = frozen pattern).\n"
            "Also scales the per-blade sway scroll.");

        ImGui::DragFloat("Seed", &wind.fbmSeed, 0.25f, -1000.0f, 1000.0f, "%.2f");
        ImGuiHelpers::SetTooltipIfHovered("Domain offset; different seeds give a different wind pattern.");

        if (ImGui::Button("Reset FBM Defaults")) {
            const VegetationRenderer::WindSettings defaults{};
            wind.fbmFrequency = defaults.fbmFrequency;
            wind.fbmOctaves = defaults.fbmOctaves;
            wind.fbmLacunarity = defaults.fbmLacunarity;
            wind.fbmGain = defaults.fbmGain;
            wind.fbmAmplitude = defaults.fbmAmplitude;
            wind.fbmVertical = defaults.fbmVertical;
            wind.speed = defaults.speed;
            wind.fbmSeed = defaults.fbmSeed;
        }
    }

    // ── Per-blade sway detail (vegetation only) ─────────────────────────────
    if (ImGui::CollapsingHeader("Vegetation Sway")) {
        ImGui::SliderFloat("Sway Strength", &wind.strength, 0.0f, 24.0f, "%.2f");
        ImGuiHelpers::SetTooltipIfHovered(
            "Blade bend amount (metres) at full lean. The sway DIRECTION is extracted\n"
            "from the shared fBm wind at each blade — there is no authored direction.");

        ImGui::SliderFloat("Lean Velocity", &wind.leanVelocity, 0.5f, 30.0f, "%.1f m/s");
        ImGuiHelpers::SetTooltipIfHovered(
            "Wind speed at which the blade lean reaches ~63% of Sway Strength\n"
            "(saturating response: light air barely moves the grass, strong wind\n"
            "lays it over). The lean direction is the local wind direction.");

        ImGui::SliderFloat("Swirl Velocity", &wind.swirlVelocity, 0.0f, 60.0f, "%.1f m/s");
        ImGuiHelpers::SetTooltipIfHovered(
            "Below this wind speed blades only lean downwind. Above it they also\n"
            "swirl: a cross-wind rotation that grows with the EXCESS speed and is\n"
            "scaled by Turbulence (neighbouring gust cells counter-rotate).");

        ImGui::SliderFloat("Base Frequency", &wind.baseFrequency, 0.0001f, 0.05f, "%.5f");
        ImGuiHelpers::SetTooltipIfHovered("Scale of the large sway waves.");

        ImGui::SliderFloat("Gust Frequency", &wind.gustFrequency, 0.0001f, 0.12f, "%.5f");
        ImGuiHelpers::SetTooltipIfHovered("Scale of the finer sway patterns.");

        ImGui::SliderFloat("Gust Strength", &wind.gustStrength, 0.0f, 2.0f, "%.2f");
        ImGuiHelpers::SetTooltipIfHovered("How strongly the sway gust noise amplifies blade motion.");

        ImGui::SliderFloat("Skew Amount", &wind.skewAmount, 0.0f, 10.0f, "%.2f");
        ImGuiHelpers::SetTooltipIfHovered("Side skew of blade tips to simulate lean.");

        ImGui::SliderFloat("Trunk Stiffness", &wind.trunkStiffness, 0.0f, 1.0f, "%.2f");
        ImGuiHelpers::SetTooltipIfHovered("0 = very stiff base, 1 = bends from base to tip.");

        ImGui::SliderFloat("Noise Scale", &wind.noiseScale, 0.1f, 6.0f, "%.2f");
        ImGuiHelpers::SetTooltipIfHovered("Global scale multiplier for both sway Perlin fields.");

        ImGui::SliderFloat("Vertical Flutter", &wind.verticalFlutter, 0.0f, 2.0f, "%.2f");
        ImGuiHelpers::SetTooltipIfHovered("Adds subtle upward flicker at blade tips.");

        ImGui::SliderFloat("Turbulence", &wind.turbulence, 0.0f, 2.0f, "%.2f");
        ImGuiHelpers::SetTooltipIfHovered("Cross-wind variation so chunks do not sway uniformly.");
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

        // Delete: deactivates the selected slot. Disabled on inactive slots.
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
