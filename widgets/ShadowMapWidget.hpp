#pragma once

#include "Widget.hpp"
#include "../vulkan/ShadowMapper.hpp"
#include <imgui.h>
#include <glm/glm.hpp>

class ShadowMapWidget : public Widget {
public:
    ShadowMapWidget(ShadowMapper* shadowMapper) 
        : Widget("Shadow Map"), shadowMapper(shadowMapper) {}
    
    void render() override {
        if (!ImGui::Begin(title.c_str(), &isOpen)) {
            ImGui::End();
            return;
        }
        
        ImGui::Text("Shadow Map Size: %dx%d", shadowMapper->getShadowMapSize(), shadowMapper->getShadowMapSize());
        
        // Debug: show light direction and position
        glm::vec3 lightDirVec = glm::normalize(glm::vec3(1.0f, 1.0f, 1.0f));
        glm::vec3 sceneCenter = glm::vec3(0.0f, 0.0f, 0.0f);
        glm::vec3 lightPosVec = sceneCenter + lightDirVec * 20.0f;
        ImGui::Text("Light Dir: %.2f, %.2f, %.2f", lightDirVec.x, lightDirVec.y, lightDirVec.z);
        ImGui::Text("Light Pos: %.2f, %.2f, %.2f", lightPosVec.x, lightPosVec.y, lightPosVec.z);
        ImGui::Text("Scene Center: %.2f, %.2f, %.2f", sceneCenter.x, sceneCenter.y, sceneCenter.z);
        
        // Size slider for shadow map display
        ImGui::SliderFloat("Display Size", &displaySize, 256.0f, 2048.0f, "%.0f");
        
        // Display shadow map as a texture at adjustable size
        if (shadowMapper->getImGuiDescriptorSet() != VK_NULL_HANDLE) {
            ImGui::Image((ImTextureID)shadowMapper->getImGuiDescriptorSet(), ImVec2(displaySize, displaySize));
        } else {
            ImGui::Text("Shadow map not available");
        }
        
        ImGui::End();
    }
    
private:
    ShadowMapper* shadowMapper;
    float displaySize = 512.0f;
};
