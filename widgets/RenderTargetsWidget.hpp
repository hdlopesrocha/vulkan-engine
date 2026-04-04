#pragma once
#include "Widget.hpp"
#include <vulkan/vulkan.h>

class VulkanApp;
class WaterRenderer;
class SolidRenderer;
class SkyRenderer;

// Widget that displays the 3 main render targets (Sky equirect, Solid color, Water color)
// as ImGui image thumbnails. Accessible from the Windows menu.
class RenderTargetsWidget : public Widget {
private:
    WaterRenderer* waterRenderer;
    SolidRenderer* solidRenderer;
    SkyRenderer*   skyRenderer;

    // ImGui texture descriptors
    VkDescriptorSet skyDescriptor = VK_NULL_HANDLE;
    VkDescriptorSet solid360Descriptor = VK_NULL_HANDLE;
    VkDescriptorSet solidColorDescriptor = VK_NULL_HANDLE;
    VkDescriptorSet waterColorDescriptor = VK_NULL_HANDLE;

    float previewScale = 0.25f;
    int currentFrame = 0;
    int cachedWidth = 0;
    int cachedHeight = 0;

public:
    RenderTargetsWidget(WaterRenderer* water, SolidRenderer* solid, SkyRenderer* sky);
    ~RenderTargetsWidget();

    void setFrameInfo(uint32_t frameIndex, int width, int height);
    void render() override;
    void updateDescriptors(uint32_t frameIndex);
    void cleanup();
};
