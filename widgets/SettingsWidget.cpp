#include "SettingsWidget.hpp"
#include "components/ColumnLayout.hpp"
#include "vulkan/includes/debug/DebugModes.hpp"
#include <vector>

SettingsWidget::SettingsWidget(Settings& settingsRef, ShadowParams* shadowParams_) : Widget("Settings", u8"\uf013"), settings(settingsRef), shadowParams(shadowParams_) {
    isOpen = true;
}

void SettingsWidget::resetToDefaults() {
    settings.resetToDefaults();
}

namespace {
// Full-width controls line up with the shared column packer.
constexpr float kSettingsColWidth = ImGuiComponents::kColumnWidth;
} // namespace

void SettingsWidget::render() {
    // Resizable. Only the horizontal scrollbar is kept: vertical overflow is
    // handled by flowing sections into new columns, and the mouse wheel never
    // scrolls vertically (NoScrollWithMouse).
    auto wg = ImGuiComponents::BeginColumnWindow(displayTitle().c_str(), &isOpen);
    if (!wg.visible()) return;

    // ---- Sections: each must be self-contained (header + controls) so it can
    // ---- be moved as a whole into another column when the current one fills up.
    // ---- Order is deliberate: everyday render controls first, advanced/debug
    // ---- last. Tiny related groups are merged (Display & Performance, Input)
    // ---- so every block stays similarly sized and columns pack without big
    // ---- empty gaps.
    std::vector<ImGuiComponents::ColumnSection> sections;
    sections.reserve(12);

    // 0: Rendering (core toggles — most used)
    sections.emplace_back([this]() {
        ImGui::Text("Rendering");
        ImGuiComponents::ColSeparator();
        ImGui::Checkbox("Render Solid", &settings.renderSolid);
        ImGuiComponents::TooltipOnHover("Toggle rendering of the main solid scene (terrain/meshes)");
        // Grass representation selector: bladed tufts + impostors, or none.
        {
            const char* kGrassModes[] = { "Bladed", "None" };
            int grassMode = static_cast<int>(settings.grassMode);
            ImGuiComponents::FieldLabel("Grass", "Grass representation selector");
            ImGui::SetNextItemWidth(kSettingsColWidth);
            if (ImGui::Combo("##GrassMode", &grassMode, kGrassModes, 2)) {
                settings.grassMode = static_cast<Settings::GrassMode>(grassMode);
            }
            ImGuiComponents::TooltipOnHover("Grass representation:\n"
                           "  Bladed = triangle blade tufts + captured impostors\n"
                           "  None   = no grass");
        }
        ImGuiComponents::SliderIntField("Raycast Pixel Size", &settings.raycastPixelSize, 1, 8, "%d px",
                "SDF and sky-cloud raymarchers cast one ray per NxN screen-pixel\n"
                "block (block center): 1 = one ray per pixel, 2 = 2x2 pixelation\n"
                "(default), up to 8x8. Higher = more pixelated and cheaper.");
        // SDF march tiers (perf report 25 M12): step budget + smoke resolve
        // samples. Maximum keeps 64/12; Minimal sets 32/6 via the preset.
        // Streams through SdfParamsUBO (no idle, no rebuild), unlike the
        // texture-array tier below it.
        if (ImGuiComponents::SliderIntField("SDF Max Steps", &settings.sdfMaxSteps, 16, 256, "%d steps",
                "SDF march step budget (shader hard-caps at 256). 64 = reference\n"
                "(Maximum), 32 = Minimal tier. Lower = cheaper and coarser.")) {
            if (settings.sdfMaxSteps < 16) settings.sdfMaxSteps = 16;
            if (settings.sdfMaxSteps > 256) settings.sdfMaxSteps = 256;
        }
        if (ImGuiComponents::SliderIntField("SDF Smoke Samples", &settings.sdfSmokeSamples, 4, 12, "%d",
                "Phase-B smoke resolve samples across the measured thickness.\n"
                "12 = reference (Maximum), 6 = Minimal tier. Thin smoke costs\n"
                "thin samples; the shader loop keeps trip count 12 with a break.")) {
            if (settings.sdfSmokeSamples < 4) settings.sdfSmokeSamples = 4;
            if (settings.sdfSmokeSamples > 12) settings.sdfSmokeSamples = 12;
        }
        ImGui::Checkbox("Ray Marching", &settings.rayMarchingEnabled);
        ImGuiComponents::TooltipOnHover("Master gate for the generic SDF raymarcher\n"
                       "(fire, smoke) and its grass shadows.\n"
                       "Off = cleared SDF targets (Minimal preset disables it).\n"
                       "The sky cloud raymarch has its own Volumetric Clouds toggle.");
        ImGui::Checkbox("Volumetric Clouds", &settings.cloudsEnabled);
        ImGuiComponents::TooltipOnHover("Master toggle for volumetric clouds (sky raymarch + ground shadows + reflections).\nMinimal preset disables; per-tier controls live in the Clouds widget.");
        // Wind-trail particles (WindRenderer): GPU particles advected by the
        // shared wind field, drawing fading 2 s trails. Trail duration/color/
        // width and the wind parameters live in the Wind Trails widget.
        // toggled; the simulation stops/resumes on the next frame
        ImGuiComponents::CheckboxField("Wind Trails",
                &settings.windParticlesEnabled,
                "GPU particles advected by the shared wind field, each leaving a\n"
                "fading world-space trail. A particle that leaves the view frustum\n"
                "respawns at a random point inside it. Configure count/duration/\n"
                "color/width and the wind parameters in the Wind Trails widget.");
        // Raymarched wind-velocity isosurface (WindRaymarchRenderer).
        // toggled; the overlay appears on the next frame
        ImGuiComponents::CheckboxField("Wind Raymarch",
                &settings.windRaymarchEnabled,
                "Raymarches the shared wind field and renders the shape where the\n"
                "wind speed equals a target velocity. The target and march quality\n"
                "live in the Wind widget (Raymarch Surface).");
        ImGuiComponents::FieldLabel("Vegetation render scale", "Offscreen vegetation resolution");
        ImGui::SetNextItemWidth(kSettingsColWidth);
        ImGui::SliderFloat("##Vegetation render scale", &settings.vegetationRenderScale, 0.25f, 1.0f, "%.2f");
        ImGuiComponents::TooltipOnHover("Vegetation offscreen targets render at this fraction of the swapchain size\n"
                       "(1.0 = full resolution). Leaf detail is leaf-scale, so 0.5 on dense scenes\n"
                       "cuts the shaded vegetation pixels 4x; the composite upsamples the color and\n"
                       "takes the closest of the 2x2 depth taps for the occlusion test.\n"
                       "Applied immediately (one device idle on the frame it changes).");
        ImGuiComponents::FieldLabel("Texture array size", "Material texture-array resolution (px per layer)");
        ImGui::SetNextItemWidth(kSettingsColWidth);
        {
            const char* kTextureSizeItems[] = { "512", "1024", "2048" };
            int sizeIdx = (settings.textureArraySize >= 2048) ? 2
                        : (settings.textureArraySize >= 1024 ? 1 : 0);
            if (ImGui::Combo("##Texture array size", &sizeIdx, kTextureSizeItems, 3)) {
                settings.textureArraySize = (sizeIdx == 2) ? 2048 : (sizeIdx == 1 ? 1024 : 512);
            }
        }
        ImGuiComponents::TooltipOnHover("Resolution of the five material texture arrays (albedo, normal, bump,\n"
                       "roughness, ao). 1024 is ~750 MB of device memory with the content-sized\n"
                       "layer count; 512 quarters it (sources are box-filtered down).\n"
                       "Changing it rebuilds the arrays: one device idle, then a full re-upload\n"
                       "and mixer refresh, so expect a brief hitch on that frame.");
        ImGui::Checkbox("Enable Normal Mapping", &settings.normalMappingEnabled);
        ImGuiComponents::TooltipOnHover("Globally enable/disable normal mapping (normal maps still needed in textures)");
        ImGui::Checkbox("Enable Roughness", &settings.roughnessEnabled);
        ImGuiComponents::TooltipOnHover("Globally enable/disable roughness map influence on specular");
        ImGui::Checkbox("Enable Ambient Occlusion", &settings.aoEnabled);
        ImGuiComponents::TooltipOnHover("Globally enable/disable ambient occlusion mapping");
    });

    // 4: Display & Performance (merged: presentation controls in one place)
    sections.emplace_back([this]() {
        ImGui::Text("Display & Performance");
        ImGuiComponents::ColSeparator();
        if (ImGui::Button("Reset to Defaults")) {
            resetToDefaults();
        }
    });

    // 5: Shadow Effects
    sections.emplace_back([this]() {
        ImGui::Text("Shadow Effects");
        ImGuiComponents::ColSeparator();
        // Applied by MyApp on the next frame (start/stop the server there).
        ImGuiComponents::CheckboxField("Serve This Scene", &settings.serveScene,
            "Start a SceneServer bound to this application's live scene: the bundled web\n"
            "viewer and engine clients in --remote mode receive the chunk ladder plus every\n"
            "live octree edit (same protocol as bin/server).");
        if (settings.serveScene) {
            ImGui::SetNextItemWidth(120.0f);
            ImGui::InputInt("Server Port", &settings.serveScenePort);
            settings.serveScenePort = std::clamp(settings.serveScenePort, 1024, 65535);
        }

        ImGui::Checkbox("Enable Shadows", &settings.enableShadows);
        ImGuiComponents::TooltipOnHover("Globally enable or disable all shadowing");
        if (shadowParams) {
            for (int i = 0; i < SHADOW_CASCADE_COUNT; i++) {
                ImGui::Text("  Cascade %d", i);
            }
        }
    });

    // 6: Camera
    sections.emplace_back([this]() {
        ImGui::Text("Camera");
        ImGuiComponents::ColSeparator();
        ImGuiComponents::FieldLabel("Near Plane", "Near clip plane distance (affects depth precision)");
        ImGui::SetNextItemWidth(kSettingsColWidth);
        ImGui::DragFloat("##Near Plane", &settings.nearPlane, 0.01f, 0.001f, 100.0f, "%.3f");
        ImGuiComponents::TooltipOnHover("Near clip plane distance (affects depth precision)");
        ImGuiComponents::FieldLabel("Far Plane", "Far clip plane distance (view distance)");
        ImGui::SetNextItemWidth(kSettingsColWidth);
        ImGui::DragFloat("##Far Plane", &settings.farPlane, 10.0f, 100.0f, 100000.0f, "%.1f");
        ImGuiComponents::TooltipOnHover("Far clip plane distance (view distance)");
    });

    // 7: Vegetation Impostors
    sections.emplace_back([this]() {
        ImGui::Text("Vegetation Impostors");
        ImGuiComponents::ColSeparator();
        ImGuiComponents::FieldLabel("Impostor Distance", "Beyond this distance vegetation (and SDF grass in Grass Raycast mode) is replaced by pre-captured impostors.\nSet to 0 to disable impostor rendering.");
        ImGui::SetNextItemWidth(kSettingsColWidth);
        ImGui::DragFloat("##Impostor Distance", &settings.impostorDistance, 5.0f, 0.0f, 5000.0f, "%.0f m");
        ImGuiComponents::TooltipOnHover("Beyond this distance vegetation (and SDF grass in Grass Raycast mode) is replaced by pre-captured impostors.\nSet to 0 to disable impostor rendering.");
        if (settings.impostorDistance < 0.0f) settings.impostorDistance = 0.0f;
    });

    // 8: Level of Detail
    sections.emplace_back([this]() {
        ImGui::Text("Level of Detail");
        ImGuiComponents::ColSeparator();
        ImGuiComponents::FieldLabel("LoD Distance Bias",
            "Scales the distance at which each coarser LoD level takes over "
            "(transition at distance = level * chunkSize * bias). Larger = full "
            "detail farther away (more triangles); smaller = coarser meshes "
            "closer (fewer triangles). 0 = always coarsest, 64+ = full detail "
            "everywhere.");
        ImGui::SetNextItemWidth(kSettingsColWidth);
        // live: the per-frame GPU band test reads settings.lodBias directly
        ImGui::SliderFloat("##LoD Distance Bias", &settings.lodBias, 0.0f, 64.0f, "%.1f");
        ImGuiComponents::TooltipOnHover(
            "Scales the distance at which each coarser LoD level takes over "
            "(transition at distance = level * chunkSize * bias). Larger = full "
            "detail farther away (more triangles); smaller = coarser meshes "
            "closer (fewer triangles). 0 = always coarsest, 64+ = full detail "
            "everywhere.");

        ImGuiComponents::FieldLabel("Max Target LoD",
            "Caps the coarsest LoD level the renderer may select for a chunk. "
            "Lower = only finer (more detailed) chunk levels are drawn; 16+ = "
            "unlimited (chunk ladders rarely exceed ~5 levels).");
        ImGui::SetNextItemWidth(kSettingsColWidth);
        // live: the per-frame GPU band test reads settings.maxTargetLod directly
        ImGui::SliderInt("##Max Target LoD", &settings.maxTargetLod, 0, 24, "%d");
        ImGuiComponents::TooltipOnHover(
            "Caps the coarsest LoD level the renderer may select for a chunk. "
            "Lower = only finer (more detailed) chunk levels are drawn; 16+ = "
            "unlimited (chunk ladders rarely exceed ~5 levels).");
    });

    // 9: Tessellation
    sections.emplace_back([this]() {
        ImGui::Text("Tessellation");
        ImGuiComponents::ColSeparator();
        ImGui::Checkbox("Enable Tessellation", &settings.tessellationEnabled);
        ImGuiComponents::TooltipOnHover("Global toggle: when disabled, tessellation and displacement are skipped");
        ImGui::Checkbox("Enable Shadow Tessellation", &settings.shadowTessellationEnabled);
        ImGuiComponents::TooltipOnHover("Global toggle: when disabled, tessellation and displacement are skipped");
        ImGuiComponents::FieldLabel("Tessellation Factor", "Multiplies per-material min/max tess levels globally");
        ImGui::SetNextItemWidth(kSettingsColWidth);
        ImGui::SliderFloat("##Tessellation Factor", &settings.tessellationFactor, 0.0f, 8.0f, "%.2f");
        ImGuiComponents::TooltipOnHover("Multiplies per-material min/max tess levels globally");
        ImGuiComponents::FieldLabel("Tess Min Distance", nullptr);
        ImGui::SetNextItemWidth(kSettingsColWidth);
        ImGui::SliderFloat("##Tess Min Distance", &settings.tessMinDistance, 1.0f, 2048.0f, "%.1f");
        ImGuiComponents::FieldLabel("Tess Max Distance", nullptr);
        ImGui::SetNextItemWidth(kSettingsColWidth);
        ImGui::SliderFloat("##Tess Max Distance", &settings.tessMaxDistance, 1.0f, 2048.0f, "%.1f");
    });

    // 10: Triplanar Mapping
    sections.emplace_back([this]() {
        ImGui::Text("Triplanar Mapping");
        ImGuiComponents::ColSeparator();
        ImGuiComponents::FieldLabel("Triplanar Threshold", "? (dead-zone before blending)");
        ImGui::SetNextItemWidth(kSettingsColWidth);
        ImGui::SliderFloat("##Triplanar Threshold", &settings.triplanarThreshold, 0.0f, 0.5f, "%.3f");
        ImGuiComponents::TooltipOnHover("? (dead-zone before blending)");
        ImGuiComponents::FieldLabel("Triplanar Exponent", "? (>1 = steeper)");
        ImGui::SetNextItemWidth(kSettingsColWidth);
        ImGui::SliderFloat("##Triplanar Exponent", &settings.triplanarExponent, 1.0f, 12.0f, "%.2f");
        ImGuiComponents::TooltipOnHover("? (>1 = steeper)");
    });

    // 11: Debug Visualisation (advanced — last)
    sections.emplace_back([this]() {
        ImGui::Text("Debug Visualisation");
        ImGuiComponents::ColSeparator();
        // Canonical IDs/names live in vulkan/includes/debug/DebugModes.hpp; the
        // shaders mirror them in includes/DebugModes.glsl. Both surfaces
        // dispatch on the same IDs, so a view only ever affects the surface
        // it applies to and the rest keeps rendering normally.
        constexpr int modeCount = static_cast<int>(DebugMode::Count);
        // Clamp stale/out-of-range IDs instead of indexing out of bounds.
        int current = static_cast<int>(debugModeFromInt(settings.debugMode));
        // Index shown in the header so no SameLine widget is needed (narrow-column safe).
        ImGui::Text("Debug Mode (%d)", current);
        ImGui::SetNextItemWidth(kSettingsColWidth);
        if (ImGui::Combo("##Debug Mode", &current, kDebugModeNames, modeCount)) {
            settings.debugMode = current;
        }
        ImGuiComponents::TooltipOnHover(
            "Shader diagnostic views. The Octree SDF Raymarch view is produced "
            "by the octree raymarch pass and auto-enables that path even when "
            "'Octree SDF Raymarch' is unchecked: each enabled Solid layer gets "
            "a stable hue (see the Layers widget), leaf cells show their "
            "lattice/refinement depth, magenta marks node-cap fallback cells "
            "and blue marks cells with missing corners.");
        if (current == static_cast<int>(DebugMode::OctreeSdfRaymarch)) {
            ImGui::TextDisabled("Uses the octree SDF raymarch pass (auto-enabled).");
        }

        // toggled overlay of per-mesh bounding boxes
        ImGui::Checkbox("Show Mesh Bounding Boxes", &settings.showBoundingBoxes);
        ImGuiComponents::TooltipOnHover("Render bounding boxes for meshes currently uploaded to the GPU");

        // toggled SDF leaf cube overlay
        ImGui::Checkbox("Show SDF Cubes (indirect cull)", &settings.showSDFDebug);
        ImGuiComponents::TooltipOnHover("Render leaf-node cube faces colored by SDF sign; frustum-culled on the GPU via IndirectRenderer.comp (only visible cubes are drawn)");
    });

    // ---- Packing: best-fit over the last frame's measured section heights.
    static std::vector<float> cachedH;
    static const std::vector<float> estimates = {
        230.0f, 230.0f, 300.0f, 240.0f, 210.0f, 190.0f,
        0.0f,   0.0f,   180.0f, 260.0f, 0.0f,   230.0f,
    };
    ImGuiComponents::LayoutSections(sections, cachedH, estimates);
}
