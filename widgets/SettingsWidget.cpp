#include "SettingsWidget.hpp"

SettingsWidget::SettingsWidget(Settings& settingsRef) : Widget("Settings", u8"\uf013"), settings(settingsRef) {
    
}

void SettingsWidget::resetToDefaults() {
    settings.resetToDefaults();
}

void SettingsWidget::render() {
    if (!isOpen) return;

    if (ImGui::Begin(displayTitle().c_str(), &isOpen)) {
        ImGui::Text("Shadow Effects");
        ImGui::Separator();
        if (ImGui::Checkbox("Enable Shadows", &settings.enableShadows)) {

        }
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Globally enable or disable all shadowing");
        if (ImGui::Button("Dump Shadow Depth")) {
            if (onDumpShadowDepth) onDumpShadowDepth();
        }
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Write shadow depth PGM for debugging");


        ImGui::Separator();
        ImGui::Text("Performance");
        ImGui::Separator();
        if (ImGui::Checkbox("V-Sync (MAILBOX/FIFO)", &settings.vsyncEnabled)) {
            // Will be read by VulkanApp to recreate swapchain with different present mode
        }
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("When disabled, uses IMMEDIATE mode for uncapped FPS (may cause tearing)");

        ImGui::Separator();

        ImGui::Text("Camera");
        ImGui::Separator();
        ImGui::DragFloat("Near Plane", &settings.nearPlane, 0.01f, 0.001f, 100.0f, "%.3f");
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Near clip plane distance (affects depth precision)");
        ImGui::DragFloat("Far Plane", &settings.farPlane, 10.0f, 100.0f, 100000.0f, "%.1f");
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Far clip plane distance (view distance)");

        ImGui::Separator();

        ImGui::Text("Input");
        ImGui::Separator();
            ImGui::Text("Rendering");
            ImGui::Separator();
            if (ImGui::Checkbox("Render Water", &settings.waterEnabled)) {
                // toggled
            }
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("When off, water passes are skipped and only the solid scene is composited");
            if (ImGui::Checkbox("Render Vegetation", &settings.vegetationEnabled)) {
                // toggled
            }
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("Toggle billboarding vegetation draws");
            if (ImGui::Checkbox("Enable Normal Mapping", &settings.normalMappingEnabled)) {
                // toggled
            }
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("Globally enable/disable normal mapping (normal maps still needed in textures)");
        if (ImGui::Checkbox("Flip keyboard rotation axes", &settings.flipKeyboardRotation)) {
            // toggled
        }
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Invert yaw/pitch directions for keyboard rotation controls");
        if (ImGui::Checkbox("Flip gamepad rotation axes", &settings.flipGamepadRotation)) {
            // toggled
        }
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Invert yaw/pitch directions for gamepad right-stick");

        ImGui::Separator();

        ImGui::Text("Input Sensitivity");
        ImGui::Separator();
        ImGui::SliderFloat("Move Speed", &settings.moveSpeed, 0.1f, 20.0f, "%.2f");
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Movement speed in units/second used by keyboard and gamepad");
        ImGui::SliderFloat("Angular Speed (deg/s)", &settings.angularSpeedDeg, 1.0f, 360.0f, "%.0f");
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Angular rotation speed in degrees/second used by keyboard and gamepad");

        ImGui::Separator();

        ImGui::Text("Tessellation");
        ImGui::Separator();
        if (ImGui::Checkbox("Enable Tessellation", &settings.tessellationEnabled)) {
            // toggled globally
        }
        if (ImGui::Checkbox("Enable Shadow Tessellation", &settings.shadowTessellationEnabled)) {
            // toggled globally
        }
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Global toggle: when disabled, tessellation and displacement are skipped");
        ImGui::Checkbox("Adaptive Tessellation", &settings.adaptiveTessellation);
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Enable camera-distance driven tessellation level");
        ImGui::SliderFloat("Tess Min Level", &settings.tessMinLevel, 1.0f, 64.0f, "%.1f");
        ImGui::SliderFloat("Tess Max Level", &settings.tessMaxLevel, 1.0f, 64.0f, "%.1f");
        ImGui::SliderFloat("Tess Min Distance", &settings.tessMinDistance, 1.0f, 200.0f, "%.1f");
        ImGui::SliderFloat("Tess Max Distance", &settings.tessMaxDistance, 1.0f, 200.0f, "%.1f");

        ImGui::Separator();
        ImGui::Text("Triplanar Mapping");
        ImGui::Separator();
        if (ImGui::SliderFloat("Triplanar Threshold", &settings.triplanarThreshold, 0.0f, 0.5f, "%.3f")) {
            ImGui::SameLine(); ImGui::TextDisabled("? (dead-zone before blending)");
        }
        if (ImGui::SliderFloat("Triplanar Exponent", &settings.triplanarExponent, 1.0f, 12.0f, "%.2f")) {
            ImGui::SameLine(); ImGui::TextDisabled("? (>1 = steeper)");
        }

        if (ImGui::Button("Reset to Defaults")) {
            resetToDefaults();
        }
        ImGui::Separator();
        if (ImGui::Checkbox("Wireframe Mode", &settings.wireframeMode)) {
            // toggle wireframe rendering
        }
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Render meshes in wireframe (requires GPU support)");

        ImGui::Separator();
        ImGui::Text("Debug Visualisation");
        ImGui::Separator();
        const char* debugItems[] = { 
            "Default Render", 
            "Fragment Normal", 
            "World Normal", 
            "UV Coordinates", 
            "Normal (TBN)", 
            "Albedo Texture", 
            "Normal Texture", 
            "Bump Texture", 
            "Height Pre-Projection", 
            "Normal from Derivatives", 
            "Light Vector (RGB)", 
            "N·L (grayscale)", 
            "Shadow Diagnostics", 
            "Triplanar Weights", 
            "Tex Indices (RGB)", 
            "Tex Weights (RGB)", 
            "Albedo Samples (R/G/B)", 
            "Triplanar Albedo", 
            "Per-Projection Triplanar Heights (RGB)", 
            "UV vs Triplanar Height Diff", 
            "Triplanar Normal", 
            "Per-Projection Triplanar Normals (RGB)", 
            "Triplanar Bump (Height)", 
            "Per-Projection Triplanar Bump (RGB)", 
            "UV vs Triplanar Bump Diff",
            "Triplanar UV X (first)",
            "Triplanar UV Y (first)",
            "Triplanar UV Z (first)",
            "Per-Projection Triplanar Normal X (first)",
            "Per-Projection Triplanar Normal Y (first)",
            "Per-Projection Triplanar Normal Z (first)",
            "TES Face Normal (sharp - purple for up)",
            "Water Displacement (tess debug)",
            "Water Scene Color (raw sample)",
            "Water Screen UV",
            "Water Noise",
            "Water Displaced Normal",
            // Water reflection sampling debug modes (37..38)
            "Reflection Vector (visualize)",
            "Water Reflection UV",
            "Caustics: Area Contraction (front/back/blend)",
            "Caustics: Anisotropy (front/back/blend)",
            "Caustics: Cloud/Line Components (cloud,line,raw)",
            "Caustics: Final Caustic Mask"
            , "Back-face Depth (raw)",
            "Front-face Depth (linearized)",
            "Back-face Depth (linearized)",
            "Scene Depth (early UV, linearized)",
            "Effective Back Depth (min, linearized)",
            "Water Thickness (normalized)"
        };
        int current = settings.debugMode;
        if (ImGui::Combo("Debug Mode", &current, debugItems, IM_ARRAYSIZE(debugItems))) {
            settings.debugMode = current;
        }
        ImGui::SameLine(); ImGui::TextDisabled("(%d)", settings.debugMode);

        if (ImGui::Checkbox("Show Mesh Bounding Boxes", &settings.showBoundingBoxes)) {
            // toggled overlay of per-mesh bounding boxes
        }
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Render bounding boxes for meshes currently uploaded to the GPU");

        if (ImGui::Checkbox("Show Debug Cubes", &settings.showDebugCubes)) {
            // toggled overlay of debug octree cubes and node instance cubes
        }
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Render debug octree/node cubes produced by the engine and explorer");
    }
    ImGui::End();
}
