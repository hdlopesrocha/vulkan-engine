#include "ShadowMapWidget.hpp"
#include "../vulkan/ubo/UniformObject.hpp"
#include "components/ImGuiHelpers.hpp"

ShadowMapWidget::ShadowMapWidget(ShadowRenderer* shadowMapper, ShadowParams* shadowParams)
    : Widget("Shadow Map", u8"\uf0c1"), shadowMapper(shadowMapper), shadowParams(shadowParams) {
}

void ShadowMapWidget::render() {
    ImGuiHelpers::WindowGuard wg(displayTitle().c_str(), &isOpen);
    if (!wg.visible()) return;

    ImGui::Text("Shadow Map Size: %dx%d", shadowMapper->getShadowMapSize(), shadowMapper->getShadowMapSize());

    // Display shadow parameters from ShadowParams
    if (shadowParams) {
        ImGui::Separator();
        ImGui::Text("Shadow Parameters:");
        ImGui::Text("Base Ortho Size: %.1f", shadowParams->orthoSize);
        for (int i = 0; i < SHADOW_CASCADE_COUNT; i++) {
            ImGui::Text("  Cascade %d: ortho = %.1f (x%.0f)", i,
                shadowParams->orthoSize * shadowParams->cascadeMultipliers[i],
                shadowParams->cascadeMultipliers[i]);
        }
        
        // Allow editing ortho size
        ImGui::Separator();
        ImGui::SliderFloat("Base Ortho Size", &shadowParams->orthoSize, 10.0f, 2048.0f, "%.0f");
    }

    // Size slider for shadow map display
    ImGui::SliderFloat("Display Size", &displaySize, 128.0f, 1024.0f, "%.0f");

    // Display all cascade shadow maps
    for (int i = 0; i < SHADOW_CASCADE_COUNT; i++) {
        ImGui::Separator();
        float ortho = shadowParams ? shadowParams->orthoSize * shadowParams->cascadeMultipliers[i] : 0.0f;
        ImGui::Text("Cascade %d (ortho=%.0f)", i, ortho);
        VkDescriptorSet ds = shadowMapper->getImGuiDescriptorSet(i);
        ImGuiHelpers::ImageOrUnavailable((ImTextureID)ds, ImVec2(displaySize, displaySize), "Shadow map not available");
    }
}
