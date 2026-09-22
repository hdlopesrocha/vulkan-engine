#include "SettingsWidget.hpp"
#include "components/ImGuiHelpers.hpp"
#include "vulkan/includes/DebugModes.hpp"
#include <vector>
#include <functional>

SettingsWidget::SettingsWidget(Settings& settingsRef, ShadowParams* shadowParams_) : Widget("Settings", u8"\uf013"), settings(settingsRef), shadowParams(shadowParams_) {
    isOpen = true;
}

void SettingsWidget::resetToDefaults() {
    settings.resetToDefaults();
}

namespace {
// Every column is exactly this wide; separators match it 1:1.
constexpr float kSettingsColWidth = 256.0f;

// Fixed-width separator: ImGui::Separator() spans the whole window content,
// which overflows/underflows a 256px column (especially with h-scroll).
// This draws exactly one column width so all section dividers align.
inline void ColSeparator() {
    const ImVec2 sp = ImGui::GetCursorScreenPos();
    ImGui::GetWindowDrawList()->AddLine(
        ImVec2(sp.x, sp.y), ImVec2(sp.x + kSettingsColWidth, sp.y),
        ImGui::GetColorU32(ImGuiCol_Separator));
    ImGui::Dummy(ImVec2(kSettingsColWidth, 1.0f));
}

// Label rendered above its control; description appears as a tooltip when the
// label (or the control below) is hovered.
inline void TooltipOnHover(const char* desc) {
    if (!desc || !*desc) return;
    ImGuiHelpers::SetTooltipIfHovered("%s", desc);
}

inline void LabelOnTop(const char* label, const char* desc = nullptr) {
    ImGui::TextUnformatted(label);
    TooltipOnHover(desc);
}

// Full-width slider/drag/combo with its visible label rendered above.
inline void FieldLabel(const char* label, const char* desc = nullptr) {
    LabelOnTop(label, desc);
}
} // namespace

void SettingsWidget::render() {
    // Resizable. Only the horizontal scrollbar is kept: vertical overflow is
    // handled by flowing sections into new columns, and the mouse wheel never
    // scrolls vertically (NoScrollWithMouse).
    ImGui::SetNextWindowPos(ImVec2(0, 24), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(1280, 720), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSizeConstraints(ImVec2(320, 300), ImVec2(FLT_MAX, FLT_MAX));
    ImGuiHelpers::WindowGuard wg(displayTitle().c_str(), &isOpen,
        ImGuiWindowFlags_HorizontalScrollbar | ImGuiWindowFlags_NoScrollWithMouse);
    if (!wg.visible()) return;

    // ---- Sections: each must be self-contained (header + controls) so it can
    // ---- be moved as a whole into another column when the current one fills up.
    // ---- Order is deliberate: everyday render controls first, advanced/debug
    // ---- last. Tiny related groups are merged (Display & Performance, Input)
    // ---- and the oversized Hybrid RT block is split, so every block is
    // ---- similarly sized and columns pack without big empty gaps.
    std::vector<std::function<void()>> sections;
    sections.reserve(13);

    // 0: Rendering (core toggles — most used)
    sections.emplace_back([this]() {
        ImGui::Text("Rendering");
        ColSeparator();
        if (ImGui::Checkbox("Render Solid", &settings.renderSolid)) {
            // toggled
        }
        TooltipOnHover("Toggle rendering of the main solid scene (terrain/meshes)");
        if (ImGui::Checkbox("Render Water", &settings.waterEnabled)) {
            // toggled
        }
        TooltipOnHover("When off, water passes are skipped and only the solid scene is composited");
        if (ImGui::Checkbox("Water Blur (global)", &settings.blurEnabled)) {
            // toggled
        }
        TooltipOnHover("Global gate for the per-material refraction/tint blur.\n"
                       "The blur runs only where this AND the layer's 'Enable Blur' are on.\n"
                       "Reflections and surface highlights are never blurred.");
        if (ImGui::Checkbox("Render Vegetation", &settings.vegetationEnabled)) {
            // toggled
        }
        TooltipOnHover("Toggle billboarding vegetation draws");
        if (ImGui::Checkbox("Enable Normal Mapping", &settings.normalMappingEnabled)) {
            // toggled
        }
        TooltipOnHover("Globally enable/disable normal mapping (normal maps still needed in textures)");
        if (ImGui::Checkbox("Enable Roughness", &settings.roughnessEnabled)) {
            // toggled
        }
        TooltipOnHover("Globally enable/disable roughness map influence on specular");
        if (ImGui::Checkbox("Enable Ambient Occlusion", &settings.aoEnabled)) {
            // toggled
        }
        TooltipOnHover("Globally enable/disable ambient occlusion mapping");
    });

    // 1: Hybrid RT toggles (raster owns primary, CSM macro shadows,
    // RT secondary visibility)
    sections.emplace_back([this]() {
        ImGui::Text("Hybrid RT");
        ColSeparator();
        ImGui::TextWrapped("Raster=primary, CSM=macro shadows, RT=secondary (reflections, refraction, thickness, contact).");
        if (ImGui::Checkbox("RT solid reflections", &settings.rtReflections)) {
        }
        TooltipOnHover("Solid mirror/SSR reflection rays (sky on miss/off).");
        if (ImGui::Checkbox("RT water reflections", &settings.rtWaterReflections)) {
        }
        TooltipOnHover("Water surface reflection rays (inline and RT-pipeline).\n"
                       "Per-material 'Enable Reflection' still gates each water layer.");
        if (ImGui::Checkbox("RT refractions", &settings.rtRefractions)) {
        }
        TooltipOnHover("Water refraction via Snell IOR (the same ray also carries RT thickness).\n"
                       "Off = NO water refraction at all: the sky-fallback bent ray is disabled too.");
        if (ImGui::Checkbox("RT water thickness", &settings.rtThickness)) {
        }
        TooltipOnHover("Use the refraction ray's path length as water thickness + Beer-Lambert absorption.\n"
                       "Off = thickness comes only from the raster back face.");
        if (ImGui::Checkbox("RT water depth (regions)", &settings.rtWaterDepth)) {
        }
        TooltipOnHover("Shore-wave region depth from a ray-traced solid bottom (world-space drop)\n"
                       "instead of the raster solid + water back-face depth. Requires RT;\n"
                       "falls back to the raster path where the ray misses.");
        if (ImGui::Checkbox("RT local/contact shadows (augment CSM)", &settings.rtLocalShadows)) {
        }
        TooltipOnHover("Selective RT contact shadows augmenting CSM (off = CSM-only, recommended)");
        if (ImGui::Checkbox("Water via RT pipeline (off = inline queries)", &settings.rtWaterPipeline)) {
        }
        TooltipOnHover("Water via async RT pipeline outputs (off = inline ray queries)");
        FieldLabel("Reflection bounces", "Extra mirror rays on reflective hits");
        ImGui::SetNextItemWidth(kSettingsColWidth);
        ImGui::SliderInt("##Reflection bounces", &settings.rtReflectionBounces, 0, 3);
        TooltipOnHover("Extra mirror rays when a water/solid reflection hits another\n"
                       "reflective surface (0 = single reflection, 1 = reflection inside\n"
                       "the reflection, up to 3). More bounces cost more ray work.");
        if (ImGui::Checkbox("Water in main pass (Phase-1, blend into solid)", &settings.waterInMainPass)) {
        }
        TooltipOnHover("Draw water with the alpha-blended main pipeline into the solid color/depth targets "
                       "(no separate water pass/composite water). Smoother work-in-progress: in-trace "
                       "screen lookups use the previous frame's solid/vegetation targets.");
    });

    // 2: RT Distances (split out so the RT block packs into columns)
    sections.emplace_back([this]() {
        ImGui::Text("RT Distances");
        ColSeparator();
        FieldLabel("Max reflect dist", "Reflection ray Tmax (world units)");
        ImGui::SetNextItemWidth(kSettingsColWidth);
        ImGui::SliderFloat("##Max reflect dist", &settings.rtMaxReflectDist, 10.0f, 2000.0f, "%.0f");
        TooltipOnHover("Reflection ray Tmax (world units)");
        FieldLabel("Max refract dist", "Refraction ray Tmax (also deep-water thickness)");
        ImGui::SetNextItemWidth(kSettingsColWidth);
        ImGui::SliderFloat("##Max refract dist", &settings.rtMaxRefractDist, 10.0f, 1000.0f, "%.0f");
        TooltipOnHover("Refraction ray Tmax (also deep-water thickness)");
        FieldLabel("Max contact dist", "Local shadow ray Tmax (contact range only)");
        ImGui::SetNextItemWidth(kSettingsColWidth);
        ImGui::SliderFloat("##Max contact dist", &settings.rtMaxShadowDist, 1.0f, 60.0f, "%.1f");
        TooltipOnHover("Local shadow ray Tmax (contact range only)");
        FieldLabel("Roughness threshold", "Roughness above this skips RT reflections (env approx)");
        ImGui::SetNextItemWidth(kSettingsColWidth);
        ImGui::SliderFloat("##Roughness threshold", &settings.rtRoughnessThreshold, 0.0f, 1.0f, "%.2f");
        TooltipOnHover("Roughness above this skips RT reflections (env approx)");
        FieldLabel("Self-skip dist", "Ignore proxy hits closer than this (own-box guard)");
        ImGui::SetNextItemWidth(kSettingsColWidth);
        ImGui::SliderFloat("##Self-skip dist", &settings.rtSelfSkipDist, 0.0f, 15.0f, "%.2f");
        TooltipOnHover("Ignore proxy hits closer than this (own-box guard)");
    });

    // 3: RT Ray Budget
    sections.emplace_back([this]() {
        ImGui::Text("RT Ray Budget");
        ColSeparator();
        ImGui::TextWrapped("Ray budget (2-4x fewer inline rays, no visible change):");
        FieldLabel("Ray scale", "0 = full-rate inline rays (reference), 1 = checkerboard half-rate");
        const char* rayScales[] = {"Full-rate (reference)", "Checkerboard half-rate"};
        int rayIdx = (settings.rtRayScale == 1) ? 1 : 0;
        ImGui::SetNextItemWidth(kSettingsColWidth);
        if (ImGui::Combo("##Ray scale", &rayIdx, rayScales, 2)) {
            settings.rtRayScale = (rayIdx == 1) ? 1 : 0;
        }
        TooltipOnHover("0 = full-rate inline rays (reference), 1 = checkerboard half-rate");
        FieldLabel("Ray contrib min", "Skip the inline ray when the lobe contribution is below this");
        ImGui::SetNextItemWidth(kSettingsColWidth);
        ImGui::SliderFloat("##Ray contrib min", &settings.rtRayContribMin, 0.0f, 0.2f, "%.3f");
        TooltipOnHover("Skip the inline ray when the lobe contribution is below this");
        if (ImGui::Checkbox("Water single-ray (Fresnel xor, off = dual reference)", &settings.rtSingleRay)) {
        }
        TooltipOnHover("Water traces reflection XOR refraction stochastically (probability = Fresnel mix).\n"
                       "Only the REFRACTION ray is cut (it recovers from the raster bottom/sky);\n"
                       "reflection always traces full-rate. Checkerboard applies to solid reflections.");

        ImGui::TextWrapped("Water look (IOR, absorption, depth cap, shore fade) lives in Water Settings, per water layer.");
        ImGui::TextWrapped("CSM stays authoritative: keep RT local shadows OFF unless inspecting contact detail. Proxy BLAS is coarse by design — never use RT for macro terrain shadows.");
    });

    // 4: Display & Performance (merged: presentation controls in one place)
    sections.emplace_back([this]() {
        ImGui::Text("Display & Performance");
        ColSeparator();
        if (ImGui::Checkbox("V-Sync (MAILBOX/FIFO)", &settings.vsyncEnabled)) {
            // Will be read by VulkanApp to recreate swapchain with different present mode
        }
        TooltipOnHover("When disabled, uses IMMEDIATE mode for uncapped FPS (may cause tearing)");
        if (ImGui::Checkbox("Wireframe Mode", &settings.wireframeMode)) {
            // toggle wireframe rendering
        }
        TooltipOnHover("Render meshes in wireframe (requires GPU support)");
        if (ImGui::Checkbox("Water Wireframe", &settings.waterWireframeMode)) {
            // toggle water wireframe only
        }
        TooltipOnHover("Render water surface in white wireframe");
        if (ImGui::Button("Reset to Defaults")) {
            resetToDefaults();
        }
    });

    // 5: Shadow Effects
    sections.emplace_back([this]() {
        ImGui::Text("Shadow Effects");
        ColSeparator();
        if (ImGui::Checkbox("Enable Shadows", &settings.enableShadows)) {

        }
        TooltipOnHover("Globally enable or disable all shadowing");
        if (ImGui::Button("Dump Shadow Depth")) {
            if (onDumpShadowDepth) onDumpShadowDepth();
        }
        TooltipOnHover("Write shadow depth PGM for debugging");
        if (shadowParams) {
            FieldLabel("Base Ortho Size", "Shadow camera orthographic size for the base cascade");
            ImGui::SetNextItemWidth(kSettingsColWidth);
            ImGui::SliderFloat("##Base Ortho Size", &shadowParams->orthoSize, 10.0f, 2048.0f, "%.0f");
            TooltipOnHover("Shadow camera orthographic size for the base cascade");
            for (int i = 0; i < SHADOW_CASCADE_COUNT; i++) {
                ImGui::Text("  Cascade %d", i);
            }
        }
    });

    // 6: Camera
    sections.emplace_back([this]() {
        ImGui::Text("Camera");
        ColSeparator();
        FieldLabel("Near Plane", "Near clip plane distance (affects depth precision)");
        ImGui::SetNextItemWidth(kSettingsColWidth);
        ImGui::DragFloat("##Near Plane", &settings.nearPlane, 0.01f, 0.001f, 100.0f, "%.3f");
        TooltipOnHover("Near clip plane distance (affects depth precision)");
        FieldLabel("Far Plane", "Far clip plane distance (view distance)");
        ImGui::SetNextItemWidth(kSettingsColWidth);
        ImGui::DragFloat("##Far Plane", &settings.farPlane, 10.0f, 100.0f, 100000.0f, "%.1f");
        TooltipOnHover("Far clip plane distance (view distance)");
    });

    // 7: Vegetation Impostors
    sections.emplace_back([this]() {
        ImGui::Text("Vegetation Impostors");
        ColSeparator();
        FieldLabel("Impostor Distance", "Beyond this distance vegetation is replaced by pre-captured impostors.\nSet to 0 to disable impostor rendering.");
        ImGui::SetNextItemWidth(kSettingsColWidth);
        ImGui::DragFloat("##Impostor Distance", &settings.impostorDistance, 5.0f, 0.0f, 5000.0f, "%.0f m");
        TooltipOnHover("Beyond this distance vegetation is replaced by pre-captured impostors.\nSet to 0 to disable impostor rendering.");
        if (settings.impostorDistance < 0.0f) settings.impostorDistance = 0.0f;
    });

    // 8: Level of Detail
    sections.emplace_back([this]() {
        ImGui::Text("Level of Detail");
        ColSeparator();
        FieldLabel("LoD Distance Bias",
            "Scales the distance at which each coarser LoD level takes over "
            "(transition at distance = level * chunkSize * bias). Larger = full "
            "detail farther away (more triangles); smaller = coarser meshes "
            "closer (fewer triangles). 0 = always coarsest, 64+ = full detail "
            "everywhere.");
        ImGui::SetNextItemWidth(kSettingsColWidth);
        if (ImGui::SliderFloat("##LoD Distance Bias", &settings.lodBias, 0.0f, 64.0f, "%.1f")) {
            // live: the per-frame GPU band test reads settings.lodBias directly
        }
        TooltipOnHover(
            "Scales the distance at which each coarser LoD level takes over "
            "(transition at distance = level * chunkSize * bias). Larger = full "
            "detail farther away (more triangles); smaller = coarser meshes "
            "closer (fewer triangles). 0 = always coarsest, 64+ = full detail "
            "everywhere.");

        FieldLabel("Max Target LoD",
            "Caps the coarsest LoD level the renderer may select for a chunk. "
            "Lower = only finer (more detailed) chunk levels are drawn; 16+ = "
            "unlimited (chunk ladders rarely exceed ~5 levels).");
        ImGui::SetNextItemWidth(kSettingsColWidth);
        if (ImGui::SliderInt("##Max Target LoD", &settings.maxTargetLod, 0, 24, "%d")) {
            // live: the per-frame GPU band test reads settings.maxTargetLod directly
        }
        TooltipOnHover(
            "Caps the coarsest LoD level the renderer may select for a chunk. "
            "Lower = only finer (more detailed) chunk levels are drawn; 16+ = "
            "unlimited (chunk ladders rarely exceed ~5 levels).");
    });

    // 9: Tessellation
    sections.emplace_back([this]() {
        ImGui::Text("Tessellation");
        ColSeparator();
        if (ImGui::Checkbox("Enable Tessellation", &settings.tessellationEnabled)) {
            // toggled globally
        }
        TooltipOnHover("Global toggle: when disabled, tessellation and displacement are skipped");
        if (ImGui::Checkbox("Enable Shadow Tessellation", &settings.shadowTessellationEnabled)) {
            // toggled globally
        }
        TooltipOnHover("Global toggle: when disabled, tessellation and displacement are skipped");
        if (ImGui::Checkbox("Adaptive Tessellation", &settings.adaptiveTessellation)) {
        }
        TooltipOnHover("Enable camera-distance driven tessellation level");
        FieldLabel("Tessellation Factor", "Multiplies per-material min/max tess levels globally");
        ImGui::SetNextItemWidth(kSettingsColWidth);
        ImGui::SliderFloat("##Tessellation Factor", &settings.tessellationFactor, 0.0f, 8.0f, "%.2f");
        TooltipOnHover("Multiplies per-material min/max tess levels globally");
        FieldLabel("Tess Min Distance", nullptr);
        ImGui::SetNextItemWidth(kSettingsColWidth);
        ImGui::SliderFloat("##Tess Min Distance", &settings.tessMinDistance, 1.0f, 2048.0f, "%.1f");
        FieldLabel("Tess Max Distance", nullptr);
        ImGui::SetNextItemWidth(kSettingsColWidth);
        ImGui::SliderFloat("##Tess Max Distance", &settings.tessMaxDistance, 1.0f, 2048.0f, "%.1f");
    });

    // 10: Triplanar Mapping
    sections.emplace_back([this]() {
        ImGui::Text("Triplanar Mapping");
        ColSeparator();
        FieldLabel("Triplanar Threshold", "? (dead-zone before blending)");
        ImGui::SetNextItemWidth(kSettingsColWidth);
        ImGui::SliderFloat("##Triplanar Threshold", &settings.triplanarThreshold, 0.0f, 0.5f, "%.3f");
        TooltipOnHover("? (dead-zone before blending)");
        FieldLabel("Triplanar Exponent", "? (>1 = steeper)");
        ImGui::SetNextItemWidth(kSettingsColWidth);
        ImGui::SliderFloat("##Triplanar Exponent", &settings.triplanarExponent, 1.0f, 12.0f, "%.2f");
        TooltipOnHover("? (>1 = steeper)");
    });

    // 11: Input (merged: axes + sensitivity)
    sections.emplace_back([this]() {
        ImGui::Text("Input");
        ColSeparator();
        if (ImGui::Checkbox("Flip keyboard rotation axes", &settings.flipKeyboardRotation)) {
            // toggled
        }
        TooltipOnHover("Invert yaw/pitch directions for keyboard rotation controls");
        if (ImGui::Checkbox("Flip gamepad rotation axes", &settings.flipGamepadRotation)) {
            // toggled
        }
        TooltipOnHover("Invert yaw/pitch directions for gamepad right-stick");
        FieldLabel("Move Speed", "Movement speed in units/second used by keyboard and gamepad");
        ImGui::SetNextItemWidth(kSettingsColWidth);
        ImGui::SliderFloat("##Move Speed", &settings.moveSpeed, 0.1f, 20.0f, "%.2f");
        TooltipOnHover("Movement speed in units/second used by keyboard and gamepad");
        FieldLabel("Angular Speed (deg/s)", "Angular rotation speed in degrees/second used by keyboard and gamepad");
        ImGui::SetNextItemWidth(kSettingsColWidth);
        ImGui::SliderFloat("##Angular Speed (deg/s)", &settings.angularSpeedDeg, 1.0f, 360.0f, "%.0f");
        TooltipOnHover("Angular rotation speed in degrees/second used by keyboard and gamepad");
    });

    // 12: Debug Visualisation (advanced — last)
    sections.emplace_back([this]() {
        ImGui::Text("Debug Visualisation");
        ColSeparator();
        // Canonical IDs/names live in vulkan/includes/DebugModes.hpp; the
        // shaders mirror them in includes/debug_modes.glsl. Both surfaces
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

        if (ImGui::Checkbox("Show Mesh Bounding Boxes", &settings.showBoundingBoxes)) {
            // toggled overlay of per-mesh bounding boxes
        }
        TooltipOnHover("Render bounding boxes for meshes currently uploaded to the GPU");

        if (ImGui::Checkbox("Show SDF Cubes (indirect cull)", &settings.showSDFDebug)) {
            // toggled SDF leaf cube overlay
        }
        TooltipOnHover("Render leaf-node cube faces colored by SDF sign; frustum-culled on the GPU via indirect.comp (only visible cubes are drawn)");
    });

    // ---- Packing: best-fit over the last frame's measured section heights.
    // ---- Sections are visited in importance order; each goes into the fullest
    // ---- column that still fits it, so leftover gaps get filled instead of
    // ---- stranding empty column space. New columns open only when nothing
    // ---- fits; overflow scrolls horizontally (fixed 256px columns).
    const int n = static_cast<int>(sections.size());
    static std::vector<float> cachedH;
    if (static_cast<int>(cachedH.size()) != n) {
        // First frame estimates; refined by measurement from frame 2 on.
        cachedH.assign(n, 170.0f);
        cachedH[0] = 190.0f;   // Rendering
        cachedH[1] = 230.0f;   // Hybrid RT toggles
        cachedH[2] = 300.0f;   // RT Distances
        cachedH[3] = 240.0f;   // RT Ray Budget
        cachedH[4] = 210.0f;   // Display & Performance
        cachedH[5] = 190.0f;   // Shadow Effects
        cachedH[8] = 180.0f;   // Level of Detail
        cachedH[9] = 260.0f;   // Tessellation
        cachedH[11] = 280.0f;  // Input
        cachedH[12] = 230.0f;  // Debug Visualisation
    }

    float availW = ImGui::GetContentRegionAvail().x;
    float availH = ImGui::GetContentRegionAvail().y;
    if (availW < 50.0f) availW = 300.0f;
    if (availH < 50.0f) availH = 500.0f;
    // Safety margin so rounding/trailing spacing can never push content 1px
    // past the bottom edge (which would summon a vertical scrollbar).
    // Columns are sized against flowH, guaranteeing content height < availH.
    const float flowH = availH - 4.0f;

    // No width cap: as many columns as the height requires. Horizontal
    // scrolling handles overflow instead of squeezing columns.
    constexpr float kSectionGap = 8.0f;
    std::vector<int> colOf(n, 0);
    std::vector<float> colH(1, 0.0f);
    for (int i = 0; i < n; ++i) {
        const float h = cachedH[i] > 1.0f ? cachedH[i] : 170.0f;
        // Tightest column that fits wins; empty columns always accept so a
        // section taller than the window still lands somewhere sane.
        int best = -1;
        float bestRem = 1e30f;
        for (int c = 0; c < static_cast<int>(colH.size()); ++c) {
            const float rem = flowH - colH[c];
            if ((colH[c] <= 0.0f || rem >= h) && rem - h < bestRem) {
                best = c;
                bestRem = rem - h;
            }
        }
        if (best < 0) {
            best = static_cast<int>(colH.size());
            colH.emplace_back(0.0f);
        }
        colOf[i] = best;
        colH[best] += h + kSectionGap;
    }
    const int nCols = static_cast<int>(colH.size());

    auto renderSectionMeasured = [&](int idx, bool firstInColumn) {
        if (!firstInColumn) {
            ImGui::Spacing();
            ColSeparator();
            ImGui::Spacing();
        }
        const float y0 = ImGui::GetCursorScreenPos().y;
        sections[idx]();
        const float y1 = ImGui::GetCursorScreenPos().y;
        const float measured = y1 - y0;
        if (measured > 1.0f) cachedH[idx] = measured;
    };

    // Fixed column width: every column is exactly kSettingsColWidth; overflow scrolls.
    constexpr float colWidth = kSettingsColWidth;
    const float spacingX = ImGui::GetStyle().ItemSpacing.x;

    for (int c = 0; c < nCols; ++c) {
        if (c > 0) ImGui::SameLine(0.0f, spacingX);
        ImGui::BeginGroup();
        // Enforce the column width so TextWrapped + full-width widgets wrap
        // consistently and the group extends past the window edge when needed.
        ImGui::Dummy(ImVec2(colWidth, 0.0f));
        ImGui::PushTextWrapPos(ImGui::GetCursorPos().x + colWidth);
        bool first = true;
        for (int i = 0; i < n; ++i) {
            if (colOf[i] != c) continue;
            renderSectionMeasured(i, first);
            first = false;
        }
        ImGui::PopTextWrapPos();
        ImGui::EndGroup();
    }
}
