#include "RenderPassDebugWidget.hpp"
#include "../vulkan/VulkanApp.hpp"
#include "../vulkan/WaterRenderer.hpp"
#include <imgui.h>
#include <backends/imgui_impl_vulkan.h>

RenderPassDebugWidget::RenderPassDebugWidget(VulkanApp* app, WaterRenderer* waterRenderer, SolidRenderer* solidRenderer)
    : Widget("Render Pass Debug"), app(app), waterRenderer(waterRenderer), solidRenderer(solidRenderer) {
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
    if (waterDepthDescriptor != VK_NULL_HANDLE) {
        ImGui_ImplVulkan_RemoveTexture(waterDepthDescriptor);
        waterDepthDescriptor = VK_NULL_HANDLE;
    }
    if (waterNormalDescriptor != VK_NULL_HANDLE) {
        ImGui_ImplVulkan_RemoveTexture(waterNormalDescriptor);
        waterNormalDescriptor = VK_NULL_HANDLE;
    }
    if (waterMaskDescriptor != VK_NULL_HANDLE) {
        ImGui_ImplVulkan_RemoveTexture(waterMaskDescriptor);
        waterMaskDescriptor = VK_NULL_HANDLE;
    }
}

void RenderPassDebugWidget::updateDescriptors(uint32_t frameIndex) {
    if (!waterRenderer || !solidRenderer) return;
    currentFrame = frameIndex;
    
    // Clean up old descriptors
    cleanup();
    
    // Create ImGui descriptor sets for the textures
    VkImageView sceneColorView = solidRenderer->getColorView(frameIndex);
    VkImageView sceneDepthView = solidRenderer->getDepthView(frameIndex);
    VkImageView waterDepthView = waterRenderer->getWaterDepthView();
    VkImageView waterNormalView = waterRenderer->getWaterNormalView();
    VkImageView waterMaskView = waterRenderer->getWaterMaskView();
    
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

    if (waterDepthView != VK_NULL_HANDLE) {
        waterDepthDescriptor = ImGui_ImplVulkan_AddTexture(
            waterRenderer->getLinearSampler(),
            waterDepthView,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
        );
    }

    if (waterNormalView != VK_NULL_HANDLE) {
        waterNormalDescriptor = ImGui_ImplVulkan_AddTexture(
            waterRenderer->getLinearSampler(),
            waterNormalView,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
        );
    }

    if (waterMaskView != VK_NULL_HANDLE) {
        waterMaskDescriptor = ImGui_ImplVulkan_AddTexture(
            waterRenderer->getLinearSampler(),
            waterMaskView,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
        );
    }
}

void RenderPassDebugWidget::render() {
    ImGui::Begin("Render Pass Debug", nullptr, ImGuiWindowFlags_AlwaysAutoResize);
    if (!waterRenderer) {
        ImGui::TextUnformatted("Water renderer not set.");
        ImGui::End();
        return;
    }

    // Refresh descriptors for the active frame each draw
    updateDescriptors(app->getCurrentFrame());
    
    ImGui::Text("Frame: %d", currentFrame);
    ImGui::SliderFloat("Preview Scale", &previewScale, 0.1f, 1.0f);
    ImGui::Separator();
    
    ImVec2 previewSize(app->getWidth() * previewScale, app->getHeight() * previewScale);
    
    ImGui::TextUnformatted("Solid Pass");
    ImGui::Checkbox("Show Solid Color", &showSceneColor);
    if (showSceneColor && sceneColorDescriptor != VK_NULL_HANDLE) {
        ImGui::Text("Solid Color (Offscreen):");
        ImGui::Image((ImTextureID)sceneColorDescriptor, previewSize);
    }
    
    ImGui::Separator();
    
    ImGui::Checkbox("Show Solid Depth", &showSceneDepth);
    if (showSceneDepth && sceneDepthDescriptor != VK_NULL_HANDLE) {
        ImGui::Text("Solid Depth (Offscreen):");
        ImGui::Image((ImTextureID)sceneDepthDescriptor, previewSize);
    }

    ImGui::Separator();

    ImGui::TextUnformatted("Liquid Pass");
    ImGui::Checkbox("Show Water Depth", &showWaterDepth);
    if (showWaterDepth && waterDepthDescriptor != VK_NULL_HANDLE) {
        ImGui::Text("Water Depth:");
        ImGui::Image((ImTextureID)waterDepthDescriptor, previewSize);
    }

    ImGui::Checkbox("Show Water Normal", &showWaterNormal);
    if (showWaterNormal && waterNormalDescriptor != VK_NULL_HANDLE) {
        ImGui::Text("Water Normal:");
        ImGui::Image((ImTextureID)waterNormalDescriptor, previewSize);
    }

    ImGui::Checkbox("Show Water Mask", &showWaterMask);
    if (showWaterMask && waterMaskDescriptor != VK_NULL_HANDLE) {
        ImGui::Text("Water Mask:");
        ImGui::Image((ImTextureID)waterMaskDescriptor, previewSize);
    }
    
    ImGui::End();
}
