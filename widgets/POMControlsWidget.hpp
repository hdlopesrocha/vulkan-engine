#pragma once

#include "Widget.hpp"
#include <imgui.h>

class POMControlsWidget : public Widget {
public:
    POMControlsWidget(float* pomHeightScale, float* pomMinLayers, float* pomMaxLayers,
                      bool* pomEnabled, bool* flipNormalY, bool* flipTangentHandedness,
                      bool* flipParallaxDirection, float* ambientFactor,
                      float* specularStrength, float* shininess)
        : Widget("POM Controls (Legacy)"),
          pomHeightScale(pomHeightScale), pomMinLayers(pomMinLayers),
          pomMaxLayers(pomMaxLayers), pomEnabled(pomEnabled),
          flipNormalY(flipNormalY), flipTangentHandedness(flipTangentHandedness),
          flipParallaxDirection(flipParallaxDirection), ambientFactor(ambientFactor),
          specularStrength(specularStrength), shininess(shininess) {
        // Start hidden since per-texture controls are preferred
        isOpen = false;
    }
    
    void render() override {
        if (!ImGui::Begin(title.c_str(), &isOpen)) {
            ImGui::End();
            return;
        }
        
        ImGui::TextColored(ImVec4(1.0f, 0.7f, 0.0f, 1.0f), "Note: These are legacy global controls.");
        ImGui::TextColored(ImVec4(1.0f, 0.7f, 0.0f, 1.0f), "Use per-texture materials in Textures window.");
        ImGui::Separator();
        
        ImGui::Checkbox("Enable POM", pomEnabled);
        ImGui::SliderFloat("Height Scale", pomHeightScale, 0.0f, 0.2f, "%.3f");
        ImGui::SliderFloat("Min Layers", pomMinLayers, 1.0f, 64.0f, "%.0f");
        ImGui::SliderFloat("Max Layers", pomMaxLayers, 1.0f, 128.0f, "%.0f");
        ImGui::Checkbox("Flip normal Y", flipNormalY);
        ImGui::Checkbox("Flip parallax direction", flipParallaxDirection);
        ImGui::Checkbox("Flip tangent handedness", flipTangentHandedness);
        ImGui::Separator();
        ImGui::Text("Lighting");
        ImGui::SliderFloat("Ambient", ambientFactor, 0.0f, 1.0f, "%.2f");
        ImGui::SliderFloat("Specular Strength", specularStrength, 0.0f, 2.0f, "%.2f");
        ImGui::SliderFloat("Shininess", shininess, 1.0f, 256.0f, "%.0f");
        
        ImGui::End();
    }
    
private:
    float* pomHeightScale;
    float* pomMinLayers;
    float* pomMaxLayers;
    bool* pomEnabled;
    bool* flipNormalY;
    bool* flipTangentHandedness;
    bool* flipParallaxDirection;
    float* ambientFactor;
    float* specularStrength;
    float* shininess;
};
