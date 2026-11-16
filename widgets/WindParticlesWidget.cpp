#include "WindParticlesWidget.hpp"

#include <algorithm>
#include <imgui.h>
#include "components/ImGuiHelpers.hpp"
#include "../types/Settings.hpp"
#include "../vulkan/renderer/vegetation/VegetationRenderer.hpp"
#include "../vulkan/renderer/wind/WindRenderer.hpp"

WindParticlesWidget::WindParticlesWidget(Settings* settings_, VegetationRenderer* vegetationRenderer_)
    : Widget("Wind Trails", u8"\uf72e"), settings(settings_), vegetationRenderer(vegetationRenderer_) {
    isOpen = false;
}

void WindParticlesWidget::render() {
    if (!settings) return;

    ImGuiHelpers::WindowGuard wg(displayTitle().c_str(), &isOpen);
    if (!wg.visible()) return;

    // Master gate (also exposed in the Settings widget).
    ImGui::Checkbox("Enable Wind Trails", &settings->windParticlesEnabled);
    ImGuiHelpers::SetTooltipIfHovered(
        "GPU-advected particles that follow the shared wind field and draw\n"
        "fading world-space trails. Toggling it off also stops the simulation.");

    WindParticlesSettings& trail = settings->windParticles;

    if (ImGui::CollapsingHeader("Trails", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::SliderInt("Particles", &trail.particleCount, 0,
                         static_cast<int>(WindRenderer::kMaxParticles));
        ImGuiHelpers::SetTooltipIfHovered(
            "Simulated particles (capacity %u). Each one leaves its own trail.",
            WindRenderer::kMaxParticles);

        ImGui::SliderFloat("Trail Duration", &trail.trailDuration, 0.1f, 6.0f, "%.2f s");
        ImGuiHelpers::SetTooltipIfHovered(
            "Lifetime of a trail in seconds. Samples are stored at a fixed rate\n"
            "(%.0f Hz), so the length is frame-rate independent. Capacity (%u\n"
            "samples) caps the maximum at ~%.1f s.",
            1.0f / WindRenderer::kTrailSampleInterval,
            WindRenderer::kTrailCapacity,
            static_cast<float>(WindRenderer::kTrailCapacity) * WindRenderer::kTrailSampleInterval);

        ImGui::SliderFloat("Line Width", &trail.lineWidth, 0.5f, 12.0f, "%.1f px");
        ImGuiHelpers::SetTooltipIfHovered(
            "Trail width in screen pixels (expanded to camera-facing quads, so it\n"
            "does not need the wideLines device feature).");

        ImGui::SliderFloat("Opacity", &trail.opacity, 0.0f, 1.0f, "%.2f");
        ImGuiHelpers::SetTooltipIfHovered("Peak alpha of the newest trail segment; it fades to 0 at the tail.");

        ImGui::SliderFloat("Speed Range", &trail.colorSpeedMax, 0.5f, 60.0f, "%.1f m/s");
        ImGuiHelpers::SetTooltipIfHovered(
            "Trails are colored by the local wind speed: blue (0) -> green ->\n"
            "yellow -> red at this speed. Lower it to highlight slower gusts;\n"
            "tornado wind (>30 m/s) is always red.");
    }

    if (ImGui::CollapsingHeader("Spawn Volume", ImGuiTreeNodeFlags_DefaultOpen)) {
        // The particle far plane is a hard bound: [0, camera far]. The camera
        // near/far live in Settings > Camera; both sliders are clamped every
        // frame here and again when the simulation packs its UBO.
        const float cameraFar = std::max(settings->farPlane, 1.0f);
        trail.spawnFarDistance = std::clamp(trail.spawnFarDistance, 0.0f, cameraFar);

        ImGui::SliderFloat("Near Plane", &trail.spawnNearDistance, 0.0f,
                           std::max(trail.spawnFarDistance, 1.0f), "%.0f m");
        ImGuiHelpers::SetTooltipIfHovered(
            "Closest distance from the camera at which a respawned particle can appear.\n"
            "Bounded by the Far Plane below.");

        ImGui::SliderFloat("Far Plane", &trail.spawnFarDistance, 0.0f, cameraFar, "%.0f m");
        ImGuiHelpers::SetTooltipIfHovered(
            "Farthest distance wind particles may live: respawns happen inside\n"
            "[Near Plane, Far Plane] and a particle that drifts past this distance is\n"
            "respawned. Clamped to the camera far plane (Settings > Camera).");

        // Authoritative clamp after both sliders (also covers camera-far edits).
        trail.spawnFarDistance = std::clamp(trail.spawnFarDistance, 0.0f, cameraFar);
        trail.spawnNearDistance = std::clamp(trail.spawnNearDistance, 0.0f, trail.spawnFarDistance);

        ImGui::TextDisabled("Camera far plane: %.0f m (Settings > Camera)", cameraFar);
        ImGui::TextWrapped(
            "Particles are seeded (and re-seeded when they leave the view frustum or\n"
            "cross the far plane) at random points INSIDE the view, between these\n"
            "distances.");
    }

    // ── Wind ────────────────────────────────────────────────────────────────
    // These sliders edit the shared wind field (VegetationRenderer::WindSettings
    // -> WindField UBO, binding 27), so this renderer, the grass and the clouds
    // all follow the same wind. Tornadoes live in the Wind widget.
    if (!vegetationRenderer) {
        ImGui::TextDisabled("No vegetation renderer: wind parameters unavailable.");
        return;
    }
    VegetationRenderer::WindSettings& wind = vegetationRenderer->getWindSettings();
    if (ImGui::CollapsingHeader("Wind (shared field)", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Checkbox("Enable Wind", &wind.enabled);
        ImGuiHelpers::SetTooltipIfHovered("Master gate for the shared wind field: vegetation sway and these trails.\nClouds/fire keep sampling the field itself.");

        ImGui::TextWrapped(
            "The wind velocity is the 3D fBm field itself: there is no authored\n"
            "direction, the local fBm vector sets it.");
    }

    // Full parameter set lives in the Wind widget; the same controls are
    // mirrored here because the trails are the most direct visualization of it.
    if (ImGui::CollapsingHeader("FBM Field (wind velocity)")) {
        ImGui::SliderFloat("Frequency", &wind.fbmFrequency, 0.0002f, 0.05f, "%.5f");
        ImGuiHelpers::SetTooltipIfHovered("Spatial frequency of the first octave (1/m).");
        ImGui::SliderInt("Octaves", &wind.fbmOctaves, 1, static_cast<int>(kWindFbmMaxOctaves));
        ImGuiHelpers::SetTooltipIfHovered("Noise layers; each octave costs three field samples.");
        ImGui::SliderFloat("Lacunarity", &wind.fbmLacunarity, 1.0f, 4.0f, "%.2f");
        ImGuiHelpers::SetTooltipIfHovered("Frequency multiplier between octaves.");
        ImGui::SliderFloat("Gain", &wind.fbmGain, 0.0f, 0.95f, "%.2f");
        ImGuiHelpers::SetTooltipIfHovered("Amplitude multiplier between octaves (persistence).");
        ImGui::SliderFloat("Amplitude", &wind.fbmAmplitude, 0.0f, 50.0f, "%.2f m/s");
        ImGuiHelpers::SetTooltipIfHovered("Peak wind speed (m/s); 0 disables the field.");
        ImGui::SliderFloat("Vertical Weight", &wind.fbmVertical, 0.0f, 1.0f, "%.2f");
        ImGuiHelpers::SetTooltipIfHovered("Authority of the vertical (Y) component.");
        ImGui::SliderFloat("Speed", &wind.speed, 0.0f, 10.0f, "%.2f");
        ImGuiHelpers::SetTooltipIfHovered("Temporal evolution speed of the fBm field.");
        ImGui::DragFloat("Seed", &wind.fbmSeed, 0.25f, -1000.0f, 1000.0f, "%.2f");
        ImGuiHelpers::SetTooltipIfHovered("Domain offset; changes the wind pattern.");
    }

    if (ImGui::Button("Reset Trail Defaults")) {
        trail = WindParticlesSettings{};
    }
    ImGuiHelpers::SetTooltipIfHovered("Restore only the trail/spawn settings (the shared wind field is untouched).");
}
