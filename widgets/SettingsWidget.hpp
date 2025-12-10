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
            
            if (ImGui::Button("Reset to Defaults")) {
                resetToDefaults();
            }
        }
        ImGui::End();
    }
    
    // Getters for the settings
    bool getSelfShadowingEnabled() const { return enableSelfShadowing; }
    bool getShadowDisplacementEnabled() const { return enableShadowDisplacement; }
    bool getParallaxInShadowPassEnabled() const { return enableParallaxInShadowPass; }
    float getSelfShadowQuality() const { return selfShadowQuality; }
    
private:
    bool enableSelfShadowing = true;           // Parallax self-shadowing (bumps on themselves)
    bool enableShadowDisplacement = true;      // External shadows follow height
    bool enableParallaxInShadowPass = true;    // Cast shadows match appearance
    float selfShadowQuality = 0.5f;            // Quality multiplier for self-shadow rays
    
    void resetToDefaults() {
        enableSelfShadowing = true;
        enableShadowDisplacement = true;
        enableParallaxInShadowPass = true;
        selfShadowQuality = 0.5f;
    }
};
