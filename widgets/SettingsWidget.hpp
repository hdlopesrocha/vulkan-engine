#pragma once

#include "Widget.hpp"
#include <imgui.h>

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
            
            if (ImGui::Checkbox("Enable Parallax Self-Shadowing", &enableSelfShadowing)) {
                // Self-shadowing: bumps casting shadows on themselves
            }
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Enables shadows within the parallax surface\n(bumps casting shadows into carved areas)");
            }
            
            if (ImGui::Checkbox("Enable Parallax Shadow Displacement", &enableShadowDisplacement)) {
                // Shadow displacement: external shadows following height map
            }
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Makes external shadows follow the height displacement\n(cube shadows conform to bumps on the plane)");
            }
            
            if (ImGui::Checkbox("Enable Parallax in Shadow Pass", &enableParallaxInShadowPass)) {
                // Parallax in shadow pass: cast shadows match visual appearance
            }
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Applies parallax displacement when casting shadows\n(makes cast shadows match the visual surface detail)");
            }
            
            ImGui::Separator();
            ImGui::Text("Performance");
            ImGui::Separator();
            
            ImGui::Text("Self-Shadow Quality:");
            ImGui::SliderFloat("##SelfShadowQuality", &selfShadowQuality, 0.1f, 2.0f, "%.1f");
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Multiplier for self-shadow ray samples\nLower = faster, Higher = better quality");
            }
            
            ImGui::Separator();

            ImGui::Text("Input");
            ImGui::Separator();
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

            ImGui::Text("Parallax LOD");
            ImGui::Separator();
            ImGui::SliderFloat("Parallax Near", &parallaxNear, 0.1f, 10.0f, "%.2f");
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("Distance at which full parallax detail is used (units)");
            ImGui::SliderFloat("Parallax Far", &parallaxFar, 0.2f, 100.0f, "%.1f");
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("Distance at which parallax detail is reduced to the reduction factor (units)");
            ImGui::SliderFloat("Parallax Reduction", &parallaxReduction, 0.05f, 1.0f, "%.2f");
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("Fraction of original parallax detail to use at 'Parallax Far' (0.05..1.0)");

            if (ImGui::Button("Reset to Defaults")) {
                resetToDefaults();
            }
            ImGui::Separator();
            if (ImGui::Checkbox("Wireframe Mode", &wireframeMode)) {
                // toggle wireframe rendering
            }
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("Render meshes in wireframe (requires GPU support)");
        }
        ImGui::End();
    }
    
    // Getters for the settings
    bool getSelfShadowingEnabled() const { return enableSelfShadowing; }
    bool getShadowDisplacementEnabled() const { return enableShadowDisplacement; }
    bool getParallaxInShadowPassEnabled() const { return enableParallaxInShadowPass; }
    float getSelfShadowQuality() const { return selfShadowQuality; }
    bool getFlipKeyboardRotation() const { return flipKeyboardRotation; }
    bool getFlipGamepadRotation() const { return flipGamepadRotation; }
    float getMoveSpeed() const { return moveSpeed; }
    float getAngularSpeedDeg() const { return angularSpeedDeg; }
    float getParallaxNear() const { return parallaxNear; }
    float getParallaxFar() const { return parallaxFar; }
    float getParallaxReduction() const { return parallaxReduction; }
    bool getWireframeEnabled() const { return wireframeMode; }
    
private:
    bool enableSelfShadowing = true;           // Parallax self-shadowing (bumps on themselves)
    bool enableShadowDisplacement = true;      // External shadows follow height
    bool enableParallaxInShadowPass = true;    // Cast shadows match appearance
    float selfShadowQuality = 0.5f;            // Quality multiplier for self-shadow rays
    bool flipKeyboardRotation = false;         // Flip keyboard rotation axes
    bool flipGamepadRotation = false;          // Flip gamepad rotation axes
    float moveSpeed = 2.5f;                    // movement units/sec
    float angularSpeedDeg = 45.0f;             // degrees/sec for rotation
    float parallaxNear = 1.0f;                 // near distance for full parallax detail
    float parallaxFar = 25.0f;                 // far distance where reduction applies
    float parallaxReduction = 0.3f;            // reduction factor at far distance (0..1)
    bool wireframeMode = false;                // render wireframe when true
    
    void resetToDefaults() {
        enableSelfShadowing = true;
        enableShadowDisplacement = true;
        enableParallaxInShadowPass = true;
        selfShadowQuality = 0.5f;
        flipKeyboardRotation = false;
        flipGamepadRotation = false;
        moveSpeed = 2.5f;
        angularSpeedDeg = 45.0f;
        parallaxNear = 1.0f;
        parallaxFar = 25.0f;
        parallaxReduction = 0.3f;
    }
};
