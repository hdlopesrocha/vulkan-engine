#pragma once
#include "../VulkanApp.hpp"
#include "../TrackedHandle.hpp"
#include "IndirectRenderer.hpp"
#include <array>
#include "CommandBufferState.hpp"

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
                            VkBuffer compactIndirectBuffer = VK_NULL_HANDLE,
                            VkBuffer visibleCountBuffer = VK_NULL_HANDLE);
    VkImage getBackFaceDepthImage(uint32_t frameIndex) const { return backFaceDepthImages[frameIndex % backFaceDepthImages.size()]; }
    VkImageView getBackFaceDepthView(uint32_t frameIndex) const { return backFaceDepthImageViews[frameIndex % backFaceDepthImageViews.size()]; }
    VkImageLayout getBackFaceDepthLayout(uint32_t frameIndex) const { return backFaceDepthImageLayouts[frameIndex % backFaceDepthImageLayouts.size()]; }
    void setBackFaceDepthLayout(uint32_t frameIndex, VkImageLayout layout) { if (frameIndex < backFaceDepthImageLayouts.size()) backFaceDepthImageLayouts[frameIndex] = layout; }

    // Dummy 1x1 depth image to avoid SYNC-HAZARD when binding #3 of set 2
    // (the back-face depth sampler) points to the same image that the back-face
    // pass writes as depth attachment. The back-face pass temporarily replaces
    // binding #3 with this dummy so the tessellation evaluation shader does not
    // read-from while the depth attachment writes-to the same image.
    void createDummyDepthView(VulkanApp* app);
    void destroyDummyDepthView(VulkanApp* app);
    VkImageView getDummyDepthView() const { return dummyDepthView; }

    // Patch binding #3 of a descriptor set (set 2) to point to `newView`.
    // Used to swap between the dummy and the real back-face depth.
    void patchBinding3(VkDescriptorSet ds, VkImageView newView);

private:
    TrackedHandle<VkPipeline> backFacePipeline;
    static constexpr uint32_t FRAMES = VulkanApp::MAX_FRAMES_IN_FLIGHT;
    std::array<VkImage, FRAMES> backFaceDepthImages = {};
    std::array<VmaAllocation, FRAMES> backFaceDepthAllocations = {};
    std::array<VkDeviceMemory, FRAMES> backFaceDepthMemories = {};
    std::array<VkImageView, FRAMES> backFaceDepthImageViews = {};
    std::array<VkImageLayout, FRAMES> backFaceDepthImageLayouts = {};
    uint32_t renderWidth = 0;
    uint32_t renderHeight = 0;
    VkImage dummyDepthImage = VK_NULL_HANDLE;
    VmaAllocation dummyDepthAllocation = VK_NULL_HANDLE;
    VkDeviceMemory dummyDepthMemory = VK_NULL_HANDLE;
    VkImageView dummyDepthView = VK_NULL_HANDLE;
    TrackedHandle<VkSampler> nearestSampler;
    VulkanApp* appPtr = nullptr;
public:
    CommandBufferState* cmdState = nullptr;
    void setCmdState(CommandBufferState* state) { cmdState = state; }
};
