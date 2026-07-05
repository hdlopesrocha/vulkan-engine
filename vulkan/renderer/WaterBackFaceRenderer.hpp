#pragma once
#include "../VulkanApp.hpp"
#include "IndirectRenderer.hpp"
#include <array>

class WaterBackFaceRenderer {
public:
    WaterBackFaceRenderer();
    ~WaterBackFaceRenderer();
    void init(VulkanApp* app);
    void cleanup(VulkanApp* app);

    void createPipelines(VulkanApp* app, VkPipelineLayout pipelineLayout);

    // Create/destroy per-frame depth targets
    void createRenderTargets(VulkanApp* app, uint32_t width, uint32_t height);
    void destroyRenderTargets(VulkanApp* app);

    // Execute back-face depth pre-pass. Caller provides indirect renderer
    // that actually draws the water geometry.
    void renderBackFacePass(VulkanApp* app, VkCommandBuffer cmd, uint32_t frameIndex,
                            IndirectRenderer& indirect, VkPipelineLayout pipelineLayout,
                                              VkDescriptorSet mainDs, VkDescriptorSet materialDs, VkDescriptorSet sceneDs,
                                              VkImage sceneDepthImage,
                                              VkBuffer compactIndirectBuffer = VK_NULL_HANDLE,
                                              VkBuffer visibleCountBuffer = VK_NULL_HANDLE);
    // Map the application frame index into the internal double-buffered arrays.
    // This allows callers to pass the swapchain frame index directly; the
    // implementation will use modulo mapping into the two back-face buffers.
    VkImage getBackFaceDepthImage(uint32_t frameIndex) const { return backFaceDepthImages[frameIndex % backFaceDepthImages.size()]; }
    VkImageView getBackFaceDepthView(uint32_t frameIndex) const { return backFaceDepthImageViews[frameIndex % backFaceDepthImageViews.size()]; }
    // Accessor for tracked per-frame layout (used by widgets to emit correct barriers)
    VkImageLayout getBackFaceDepthLayout(uint32_t frameIndex) const { return backFaceDepthImageLayouts[frameIndex % backFaceDepthImageLayouts.size()]; }
    void setBackFaceDepthLayout(uint32_t frameIndex, VkImageLayout layout) { if (frameIndex < backFaceDepthImageLayouts.size()) backFaceDepthImageLayouts[frameIndex] = layout; }

    // Add post-render barriers for the back-face depth image
    void postRenderBarrier(VkCommandBuffer cmd, uint32_t frameIndex);

private:
    VkPipeline backFacePipeline = VK_NULL_HANDLE;
    std::array<VkImage, 2> backFaceDepthImages = {VK_NULL_HANDLE, VK_NULL_HANDLE};
    std::array<VkDeviceMemory, 2> backFaceDepthMemories = {VK_NULL_HANDLE, VK_NULL_HANDLE};
    std::array<VkImageView, 2> backFaceDepthImageViews = {VK_NULL_HANDLE, VK_NULL_HANDLE};
    std::array<VkImageLayout, 2> backFaceDepthImageLayouts = {VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_UNDEFINED};
    uint32_t renderWidth = 0;
    uint32_t renderHeight = 0;
    VulkanApp* appPtr = nullptr;
};
