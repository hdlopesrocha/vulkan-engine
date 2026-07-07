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
    static constexpr uint32_t FRAMES = VulkanApp::MAX_FRAMES_IN_FLIGHT;
    std::array<VkImage, FRAMES> backFaceDepthImages = {};
    std::array<VmaAllocation, FRAMES> backFaceDepthAllocations = {};
    std::array<VkDeviceMemory, FRAMES> backFaceDepthMemories = {};
    std::array<VkImageView, FRAMES> backFaceDepthImageViews = {};
    std::array<VkImageLayout, FRAMES> backFaceDepthImageLayouts = {};
    uint32_t renderWidth = 0;
    uint32_t renderHeight = 0;
    VulkanApp* appPtr = nullptr;
};
