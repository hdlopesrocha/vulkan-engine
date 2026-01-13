#include "RenderPassDebugWidget.hpp"
#include "../vulkan/VulkanApp.hpp"
#include "../vulkan/WaterRenderer.hpp"
#include <imgui.h>
#include <backends/imgui_impl_vulkan.h>

RenderPassDebugWidget::RenderPassDebugWidget(VulkanApp* app, WaterRenderer* waterRenderer)
    : Widget("Render Pass Debug"), app(app), waterRenderer(waterRenderer) {
}

RenderPassDebugWidget::~RenderPassDebugWidget() {
    cleanup();
}

void RenderPassDebugWidget::cleanup() {
    if (sceneColorDescriptor != VK_NULL_HANDLE) {
        ImGui_ImplVulkan_RemoveTexture(sceneColorDescriptor);
        sceneColorDescriptor = VK_NULL_HANDLE;
    }
    if (sceneDepthDescriptor != VK_NULL_HANDLE) {
        ImGui_ImplVulkan_RemoveTexture(sceneDepthDescriptor);
        sceneDepthDescriptor = VK_NULL_HANDLE;
    }
}

void RenderPassDebugWidget::updateDescriptors(uint32_t frameIndex) {
    currentFrame = frameIndex;
    
    // Clean up old descriptors
    cleanup();
    
    // Create ImGui descriptor sets for the textures
    VkImageView sceneColorView = waterRenderer->getSceneColorImageView(frameIndex);
    VkImageView sceneDepthView = waterRenderer->getSceneDepthImageView(frameIndex);
    
    if (sceneColorView != VK_NULL_HANDLE) {
        sceneColorDescriptor = ImGui_ImplVulkan_AddTexture(
            waterRenderer->getLinearSampler(),
            sceneColorView,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
        );
    }
    
    if (sceneDepthView != VK_NULL_HANDLE) {
        sceneDepthDescriptor = ImGui_ImplVulkan_AddTexture(
            waterRenderer->getLinearSampler(),
            sceneDepthView,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
        );
    }
}

void RenderPassDebugWidget::render() {
    ImGui::Begin("Render Pass Debug", nullptr, ImGuiWindowFlags_AlwaysAutoResize);
    
    ImGui::Text("Frame: %d", currentFrame);
    ImGui::SliderFloat("Preview Scale", &previewScale, 0.1f, 1.0f);
    ImGui::Separator();
    
    ImVec2 previewSize(app->getWidth() * previewScale, app->getHeight() * previewScale);
    
    ImGui::Checkbox("Show Scene Color", &showSceneColor);
    if (showSceneColor && sceneColorDescriptor != VK_NULL_HANDLE) {
        ImGui::Text("Scene Color (Offscreen):");
        ImGui::Image((ImTextureID)sceneColorDescriptor, previewSize);
    }
    
    ImGui::Separator();
    
    ImGui::Checkbox("Show Scene Depth", &showSceneDepth);
    if (showSceneDepth && sceneDepthDescriptor != VK_NULL_HANDLE) {
        ImGui::Text("Scene Depth (Offscreen):");
        ImGui::Image((ImTextureID)sceneDepthDescriptor, previewSize);
    }
    
    ImGui::End();
}
