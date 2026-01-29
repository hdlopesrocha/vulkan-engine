#include "ShadowMapWidget.hpp"

ShadowMapWidget::ShadowMapWidget(ShadowRenderer* shadowMapper, ShadowParams* shadowParams)
    : Widget("Shadow Map"), shadowMapper(shadowMapper), shadowParams(shadowParams) {
}

void ShadowMapWidget::render() {
    if (!ImGui::Begin(title.c_str(), &isOpen)) {
        ImGui::End();
        return;
    }

    ImGui::Text("Shadow Map Size: %dx%d", shadowMapper->getShadowMapSize(), shadowMapper->getShadowMapSize());

    // Display shadow parameters from ShadowParams
    if (shadowParams) {
        ImGui::Separator();
        ImGui::Text("Shadow Parameters:");
        ImGui::Text("Ortho Size: %.1f", shadowParams->orthoSize);
        
        // Allow editing ortho size
        ImGui::Separator();
        ImGui::SliderFloat("Ortho Size", &shadowParams->orthoSize, 10.0f, 2048.0f, "%.0f");
    }

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
