#pragma once
#include "Widget.hpp"
#include <vulkan/vulkan.h>
#include "../vulkan/SolidRenderer.hpp"

class VulkanApp;
class WaterRenderer;

class RenderPassDebugWidget : public Widget {
private:
    // No stored VulkanApp*: callers provide per-frame info via `setFrameInfo()`
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
    int cachedWidth = 0;
    int cachedHeight = 0;

public:
    RenderPassDebugWidget(WaterRenderer* waterRenderer, SolidRenderer* solidRenderer);
    ~RenderPassDebugWidget();

    // Per-frame update (do NOT store VulkanApp*)
    void setFrameInfo(uint32_t frameIndex, int width, int height);

    void render() override;
    void updateDescriptors(uint32_t frameIndex);
    void cleanup();
    void setWaterRenderer(WaterRenderer* renderer) { waterRenderer = renderer; }
    void setSolidRenderer(SolidRenderer* renderer) { solidRenderer = renderer; }
};
