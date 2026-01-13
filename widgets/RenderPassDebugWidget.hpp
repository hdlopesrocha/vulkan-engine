#pragma once
#include "Widget.hpp"
#include <vulkan/vulkan.h>

class VulkanApp;
class WaterRenderer;

class RenderPassDebugWidget : public Widget {
private:
    VulkanApp* app;
    WaterRenderer* waterRenderer;
    
    // ImGui texture IDs for displaying render targets
    VkDescriptorSet sceneColorDescriptor = VK_NULL_HANDLE;
    VkDescriptorSet sceneDepthDescriptor = VK_NULL_HANDLE;
    
    bool showSceneColor = true;
    bool showSceneDepth = true;
    float previewScale = 0.3f;
    int currentFrame = 0;

public:
    RenderPassDebugWidget(VulkanApp* app, WaterRenderer* waterRenderer);
    ~RenderPassDebugWidget();
    
    void render() override;
    void updateDescriptors(uint32_t frameIndex);
    void cleanup();
};
