#pragma once
#include "Widget.hpp"
#include "../Uniforms.hpp"
#include <vulkan/vulkan.h>

class VulkanApp;
class WaterRenderer;
class SolidRenderer;
class SkyRenderer;
class ShadowRenderer;

// Widget that displays render targets (Sky, Solid color/depth, Water depth,
// Shadow cascades) as ImGui image thumbnails.
class RenderTargetsWidget : public Widget {
private:
    WaterRenderer*  waterRenderer;
    SolidRenderer*  solidRenderer;
    SkyRenderer*    skyRenderer;
    ShadowRenderer* shadowMapper;

    // ImGui texture descriptors
    VkDescriptorSet skyDescriptor = VK_NULL_HANDLE;
    VkDescriptorSet solid360Descriptor = VK_NULL_HANDLE;
    VkDescriptorSet solidColorDescriptor = VK_NULL_HANDLE;
    VkDescriptorSet solidDepthDescriptor = VK_NULL_HANDLE;
    VkDescriptorSet waterColorDescriptor = VK_NULL_HANDLE;

    float previewScale = 0.25f;
    int currentFrame = 0;
    int cachedWidth = 0;
    int cachedHeight = 0;

public:
    RenderTargetsWidget(WaterRenderer* water, SolidRenderer* solid, SkyRenderer* sky,
                        ShadowRenderer* shadow = nullptr);
    ~RenderTargetsWidget();

    void setFrameInfo(uint32_t frameIndex, int width, int height);
    void render() override;
    void updateDescriptors(uint32_t frameIndex);
    void cleanup();
};
