#include "SettingsWidget.hpp"
#include "components/ImGuiHelpers.hpp"
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
    ImGui::SetNextWindowPos(ImVec2(32, 32), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(420, 600), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSizeConstraints(ImVec2(320, 300), ImVec2(FLT_MAX, FLT_MAX));
    ImGuiHelpers::WindowGuard wg(displayTitle().c_str(), &isOpen,
        ImGuiWindowFlags_HorizontalScrollbar | ImGuiWindowFlags_NoScrollWithMouse);
    if (!wg.visible()) return;

    // ---- Sections: each must be self-contained (header + controls) so it can
    // ---- be moved as a whole into another column when the current one fills up.
    std::vector<std::function<void()>> sections;
    sections.reserve(12);

    // 0: Shadow Effects
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

    // 1: Performance
    sections.emplace_back([this]() {
        ImGui::Text("Performance");
        ColSeparator();
        if (ImGui::Checkbox("V-Sync (MAILBOX/FIFO)", &settings.vsyncEnabled)) {
            // Will be read by VulkanApp to recreate swapchain with different present mode
        }
        TooltipOnHover("When disabled, uses IMMEDIATE mode for uncapped FPS (may cause tearing)");
    });

    // 2: Camera
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

    // 3: Vegetation Impostors
    sections.emplace_back([this]() {
        ImGui::Text("Vegetation Impostors");
        ColSeparator();
        FieldLabel("Impostor Distance", "Beyond this distance vegetation is replaced by pre-captured impostors.\nSet to 0 to disable impostor rendering.");
        ImGui::SetNextItemWidth(kSettingsColWidth);
        ImGui::DragFloat("##Impostor Distance", &settings.impostorDistance, 5.0f, 0.0f, 5000.0f, "%.0f m");
        TooltipOnHover("Beyond this distance vegetation is replaced by pre-captured impostors.\nSet to 0 to disable impostor rendering.");
        if (settings.impostorDistance < 0.0f) settings.impostorDistance = 0.0f;
    });

    // 4: Rendering
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

    // 5: Input
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
    });

    // 6: Input Sensitivity
    sections.emplace_back([this]() {
        ImGui::Text("Input Sensitivity");
        ColSeparator();
        FieldLabel("Move Speed", "Movement speed in units/second used by keyboard and gamepad");
        ImGui::SetNextItemWidth(kSettingsColWidth);
        ImGui::SliderFloat("##Move Speed", &settings.moveSpeed, 0.1f, 20.0f, "%.2f");
        TooltipOnHover("Movement speed in units/second used by keyboard and gamepad");
        FieldLabel("Angular Speed (deg/s)", "Angular rotation speed in degrees/second used by keyboard and gamepad");
        ImGui::SetNextItemWidth(kSettingsColWidth);
        ImGui::SliderFloat("##Angular Speed (deg/s)", &settings.angularSpeedDeg, 1.0f, 360.0f, "%.0f");
        TooltipOnHover("Angular rotation speed in degrees/second used by keyboard and gamepad");
    });

    // 7: Tessellation
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

    // 8: Triplanar Mapping
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

    // 9: Level of Detail
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

    // 10: Display Mode
    sections.emplace_back([this]() {
        ImGui::Text("Display Mode");
        ColSeparator();
        if (ImGui::Button("Reset to Defaults")) {
            resetToDefaults();
        }
        if (ImGui::Checkbox("Wireframe Mode", &settings.wireframeMode)) {
            // toggle wireframe rendering
        }
        TooltipOnHover("Render meshes in wireframe (requires GPU support)");
        if (ImGui::Checkbox("Water Wireframe", &settings.waterWireframeMode)) {
            // toggle water wireframe only
        }
        TooltipOnHover("Render water surface in white wireframe");
    });

    // 11: Debug Visualisation
    sections.emplace_back([this]() {
        ImGui::Text("Debug Visualisation");
        ColSeparator();
        const char* debugItems[] = {
            // 0: default
            "Default Render",
            // Solid: normals & geometry (1-7)
            "Fragment Normal",
            "World Normal",
            "Normal from Derivatives",
            "TES Face Normal",
            "Triplanar Normal",
            "Per-Projection Triplanar Normals (RGB)",
            "Triplanar Weights",
            // Solid: material & textures (8-27)
            "UV Coordinates",
            "Albedo Texture",
            "Albedo Samples (R/G/B)",
            "Triplanar Albedo",
            "Normal Texture",
            "Bump Texture",
            "Triplanar Bump (Height)",
            "Per-Projection Triplanar Bump (RGB)",
            "Height Pre-Projection",
            "Per-Projection Triplanar Heights (RGB)",
            "UV vs Triplanar Height Diff",
            "UV vs Triplanar Bump Diff",
            "Tex Indices (RGB)",
            "Tex Weights (RGB)",
            "Triplanar UV X (first)",
            "Triplanar UV Y (first)",
            "Triplanar UV Z (first)",
            "Per-Projection Triplanar Normal X (first)",
            "Per-Projection Triplanar Normal Y (first)",
            "Per-Projection Triplanar Normal Z (first)",
            // Solid: roughness, AO, light (28-34)
            "Roughness (map)",
            "Material Roughness (map x factor)",
            "Ambient Occlusion (map)",
            "N·L (grayscale)",
            "Light Vector (RGB)",
            "Shadow Diagnostics",
            "Reflection Factor (env map)",
            // Water (35-49)
            "Water Screen UV",
            "Water Noise",
            "Water Displaced Normal",
            "Water Displacement",
            "Sky Reflection (equirect)",
            "Reflection Vector (visualize)",
            "Water Reflection Color",
            "Caustics: Area Contraction (front/back/blend)",
            "Caustics: Anisotropy (front/back/blend)",
            "Caustics: Cloud/Line Components (cloud,line,raw)",
            "Caustics: Final Caustic Mask",
            "Back-face Depth (raw)",
            "Front-face Depth (linearized)",
            "Back-face Depth (linearized)",
            "Water Thickness (normalized)",
            // Hybrid RT & misc (50-58, shared with the RT widget)
            "RT Reflection Only",
            "RT Refraction Only",
            "RT Thickness",
            "Fresnel",
            "Absorption (Beer-Lambert)",
            "CSM Shadows Only",
            "RT Local Shadows Only",
            "CSM + RT Combined Shadow",
            "Tessellation Level Heat"
        };
        int current = settings.debugMode;
        // Index shown in the header so no SameLine widget is needed (narrow-column safe).
        ImGui::Text("Debug Mode (%d)", settings.debugMode);
        ImGui::SetNextItemWidth(kSettingsColWidth);
        if (ImGui::Combo("##Debug Mode", &current, debugItems, IM_ARRAYSIZE(debugItems))) {
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

    // ---- Multi-column flow: fill each column top-to-bottom using the last
    // ---- frame's measured section heights; when the next section would
    // ---- overflow the visible height, start a new column. Order is preserved.
    // ---- Columns use a fixed minimum width; if they exceed the window width
    // ---- the window scrolls horizontally (HorizontalScrollbar flag above).
    const int n = static_cast<int>(sections.size());
    static std::vector<float> cachedH;
    if (static_cast<int>(cachedH.size()) != n) {
        // First frame estimates; tall sections get larger guesses so the
        // initial column split is close to the measured one.
        cachedH.assign(n, 160.0f);
        if (n > 0) cachedH[0] = 190.0f;
        if (n > 4) cachedH[4] = 230.0f;
        if (n > 7) cachedH[7] = 240.0f;
        if (n > 11) cachedH[11] = 210.0f;
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
    int nCols = 1;
    float curH = 0.0f;
    for (int i = 0; i < n; ++i) {
        float h = cachedH[i] > 1.0f ? cachedH[i] : 160.0f;
        // Fill current column as much as possible; overflow -> new column.
        if (curH > 0.0f && curH + h > flowH) {
            ++nCols;
            curH = 0.0f;
        }
        colOf[i] = nCols - 1;
        curH += h + kSectionGap;
    }
    if (nCols > n) nCols = n;

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
