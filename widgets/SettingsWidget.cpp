#include "SettingsWidget.hpp"

SettingsWidget::SettingsWidget() : Widget("Settings") {
    isOpen = true;
}

void SettingsWidget::resetToDefaults() {
    enableShadows = true;

    flipKeyboardRotation = false;
    flipGamepadRotation = false;
    moveSpeed = 2.5f;
    angularSpeedDeg = 45.0f;
    debugMode = 0;
    adaptiveTessellation = true;
    tessMinLevel = 1.0f;
    tessMaxLevel = 32.0f;
    tessMaxDistance = 30.0f;
    tessMinDistance = 10.0f;
    // Triplanar defaults
    triplanarThreshold = 0.12f;
    triplanarExponent = 1.0f;
}

void SettingsWidget::render() {
    if (!isOpen) return;

    if (ImGui::Begin(title.c_str(), &isOpen)) {
        ImGui::Text("Shadow Effects");
        ImGui::Separator();
        if (ImGui::Checkbox("Enable Shadows", &enableShadows)) {

        }
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Globally enable or disable all shadowing");
        if (ImGui::Button("Dump Shadow Depth")) {
            if (onDumpShadowDepth) onDumpShadowDepth();
        }
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Write shadow depth PGM for debugging");


        ImGui::Separator();
        ImGui::Text("Performance");
        ImGui::Separator();
        if (ImGui::Checkbox("V-Sync (MAILBOX/FIFO)", &vsyncEnabled)) {
            // Will be read by VulkanApp to recreate swapchain with different present mode
        }
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("When disabled, uses IMMEDIATE mode for uncapped FPS (may cause tearing)");

        ImGui::Separator();

        ImGui::Text("Input");
        ImGui::Separator();
            ImGui::Text("Rendering");
            ImGui::Separator();
            if (ImGui::Checkbox("Enable Normal Mapping", &normalMappingEnabled)) {
                // toggled
            }
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("Globally enable/disable normal mapping (normal maps still needed in textures)");
        if (ImGui::Checkbox("Flip keyboard rotation axes", &flipKeyboardRotation)) {
            // toggled
        }
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Invert yaw/pitch directions for keyboard rotation controls");
        if (ImGui::Checkbox("Flip gamepad rotation axes", &flipGamepadRotation)) {
            // toggled
        }
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Invert yaw/pitch directions for gamepad right-stick");

        ImGui::Separator();

        ImGui::Text("Input Sensitivity");
        ImGui::Separator();
        ImGui::SliderFloat("Move Speed", &moveSpeed, 0.1f, 20.0f, "%.2f");
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Movement speed in units/second used by keyboard and gamepad");
        ImGui::SliderFloat("Angular Speed (deg/s)", &angularSpeedDeg, 1.0f, 360.0f, "%.0f");
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Angular rotation speed in degrees/second used by keyboard and gamepad");

        ImGui::Separator();

        ImGui::Text("Tessellation");
        ImGui::Separator();
        if (ImGui::Checkbox("Enable Tessellation", &tessellationEnabled)) {
            // toggled globally
        }
        if (ImGui::Checkbox("Enable Shadow Tessellation", &shadowTessellationEnabled)) {
            // toggled globally
        }
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Global toggle: when disabled, tessellation and displacement are skipped");
        ImGui::Checkbox("Adaptive Tessellation", &adaptiveTessellation);
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Enable camera-distance driven tessellation level");
        ImGui::SliderFloat("Tess Min Level", &tessMinLevel, 1.0f, 64.0f, "%.1f");
        ImGui::SliderFloat("Tess Max Level", &tessMaxLevel, 1.0f, 64.0f, "%.1f");
        ImGui::SliderFloat("Tess Min Distance", &tessMinDistance, 1.0f, 200.0f, "%.1f");
        ImGui::SliderFloat("Tess Max Distance", &tessMaxDistance, 1.0f, 200.0f, "%.1f");

        ImGui::Separator();
        ImGui::Text("Triplanar Mapping");
        ImGui::Separator();
        if (ImGui::SliderFloat("Triplanar Threshold", &triplanarThreshold, 0.0f, 0.5f, "%.3f")) {
            ImGui::SameLine(); ImGui::TextDisabled("? (dead-zone before blending)");
        }
        if (ImGui::SliderFloat("Triplanar Exponent", &triplanarExponent, 1.0f, 12.0f, "%.2f")) {
            ImGui::SameLine(); ImGui::TextDisabled("? (>1 = steeper)");
        }

        if (ImGui::Button("Reset to Defaults")) {
            resetToDefaults();
        }
        ImGui::Separator();
        if (ImGui::Checkbox("Wireframe Mode", &wireframeMode)) {
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
            "TES Face Normal (sharp - purple for up)" 
        };
        int current = debugMode;
        if (ImGui::Combo("Debug Mode", &current, debugItems, IM_ARRAYSIZE(debugItems))) {
            debugMode = current;
        }
        ImGui::SameLine(); ImGui::TextDisabled("(%d)", debugMode);
    }
    ImGui::End();
}
