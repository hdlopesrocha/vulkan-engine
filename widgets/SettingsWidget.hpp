#pragma once

#include "Widget.hpp"
#include <imgui.h>
#include <functional>

class SettingsWidget : public Widget {
public:
    SettingsWidget() : Widget("Settings") {
        isOpen = true;
    }
    
    void render() override {
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
            const char* debugItems[] = { "Default Render", "Fragment Normal", "World Normal", "UV Coordinates", "Tangent (TBN)", "Bitangent (TBN)", "Normal (TBN)", "Albedo Texture", "Normal Texture", "Bump Texture", "Height Pre-Projection", "Normal from Derivatives", "Light Vector (RGB)", "N·L (grayscale)", "Shadow Diagnostics", "Triplanar Weights" };
            int current = debugMode;
            if (ImGui::Combo("Debug Mode", &current, debugItems, IM_ARRAYSIZE(debugItems))) {
                debugMode = current;
            }
        }
        ImGui::End();
    }
    
    // Getters for the settings
    bool getShadowsEnabled() const { return enableShadows; }
    bool getFlipKeyboardRotation() const { return flipKeyboardRotation; }
    bool getFlipGamepadRotation() const { return flipGamepadRotation; }
    float getMoveSpeed() const { return moveSpeed; }
    float getAngularSpeedDeg() const { return angularSpeedDeg; }
    bool getWireframeEnabled() const { return wireframeMode; }
        int getDebugMode() const { return debugMode; }
    bool getNormalMappingEnabled() const { return normalMappingEnabled; }

    // Callback setter for debug actions
    void setDumpShadowDepthCallback(std::function<void()> cb) { onDumpShadowDepth = cb; }
    
private:
    bool enableShadows = true;                 // Global toggle for shadow mapping
    
    bool flipKeyboardRotation = false;         // Flip keyboard rotation axes
    bool flipGamepadRotation = false;          // Flip gamepad rotation axes
    float moveSpeed = 2.5f;                    // movement units/sec
    float angularSpeedDeg = 45.0f;             // degrees/sec for rotation
    bool wireframeMode = false;                // render wireframe when true
        int debugMode = 0;                         // 0=Default,1=Fragment Normal,2=World Normal,3=UV,4=Tangent,5=Bitangent,6=Normal (TBN),7=Albedo,8=Normal Tex,9=Bump,10=Pre-Projection,11=Normal from Derivatives,12=Light Vector,13=N·L,14=Shadow Diagnostics,15=Triplanar Weights
    bool normalMappingEnabled = true;          // Global toggle for normal mapping
    std::function<void()> onDumpShadowDepth;
    
    void resetToDefaults() {
        enableShadows = true;
        
        flipKeyboardRotation = false;
        flipGamepadRotation = false;
        moveSpeed = 2.5f;
        angularSpeedDeg = 45.0f;
        debugMode = 0;
    }
};
