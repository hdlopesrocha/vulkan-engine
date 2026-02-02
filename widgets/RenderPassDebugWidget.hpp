#pragma once
#include "Widget.hpp"
#include <vulkan/vulkan.h>
#include "../vulkan/SolidRenderer.hpp"

class VulkanApp;
class WaterRenderer;

class RenderPassDebugWidget : public Widget {
private:
    VulkanApp* app;
    WaterRenderer* waterRenderer;
    SolidRenderer* solidRenderer;
    
    // ImGui texture IDs for displaying render targets
    VkDescriptorSet sceneColorDescriptor = VK_NULL_HANDLE;
    VkDescriptorSet sceneDepthDescriptor = VK_NULL_HANDLE;
    VkDescriptorSet waterDepthDescriptor = VK_NULL_HANDLE;
    VkDescriptorSet waterNormalDescriptor = VK_NULL_HANDLE;
    VkDescriptorSet waterMaskDescriptor = VK_NULL_HANDLE;
    
    bool showSceneColor = true;
    bool showSceneDepth = true;
    bool showWaterDepth = false;
    bool showWaterNormal = false;
    bool showWaterMask = false;
    float previewScale = 0.3f;
    int currentFrame = 0;

public:
    RenderPassDebugWidget(VulkanApp* app, WaterRenderer* waterRenderer, SolidRenderer* solidRenderer);
    ~RenderPassDebugWidget();
    
    void render() override;
    void updateDescriptors(uint32_t frameIndex);
    void cleanup();
    void setWaterRenderer(WaterRenderer* renderer) { waterRenderer = renderer; }
    void setSolidRenderer(SolidRenderer* renderer) { solidRenderer = renderer; }
};
