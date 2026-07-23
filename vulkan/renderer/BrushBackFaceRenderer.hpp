#pragma once
#include "../VulkanApp.hpp"
#include "../TrackedHandle.hpp"
#include "IndirectRenderer.hpp"
#include <array>
#include "CommandBufferState.hpp"

class BrushBackFaceRenderer {
public:
    BrushBackFaceRenderer();
    ~BrushBackFaceRenderer();

    void init(VulkanApp* app);
    void cleanup(VulkanApp* app);

    void createPipelines(VulkanApp* app);
    void createRenderTargets(VulkanApp* app, uint32_t width, uint32_t height);
    void destroyRenderTargets(VulkanApp* app);

    void renderBackFacePass(VulkanApp* app, VkCommandBuffer cmd, uint32_t frameIndex,
                            IndirectRenderer& indirect,
                            VkDescriptorSet mainDs,
                            VkBuffer compactIndirectBuffer = VK_NULL_HANDLE,
                            VkBuffer visibleCountBuffer = VK_NULL_HANDLE);

    VkImage getBackFaceDepthImage(uint32_t frameIndex) const { return backFaceDepthImages[frameIndex % backFaceDepthImages.size()]; }
    VkImageView getBackFaceDepthView(uint32_t frameIndex) const { return backFaceDepthImageViews[frameIndex % backFaceDepthImageViews.size()]; }
    VkImageLayout getBackFaceDepthLayout(uint32_t frameIndex) const { return backFaceDepthImageLayouts[frameIndex % backFaceDepthImageLayouts.size()]; }
    void setBackFaceDepthLayout(uint32_t frameIndex, VkImageLayout layout) { if (frameIndex < backFaceDepthImageLayouts.size()) backFaceDepthImageLayouts[frameIndex] = layout; }

    void setCmdState(CommandBufferState* state) { cmdState = state; }
    VkPipelineLayout getPipelineLayout() const { return pipelineLayout; }

private:
    TrackedHandle<VkPipeline> backFacePipeline;
    TrackedHandle<VkPipelineLayout> pipelineLayout;
    static constexpr uint32_t FRAMES = VulkanApp::MAX_FRAMES_IN_FLIGHT;
    std::array<VkImage, FRAMES> backFaceDepthImages = {};
    std::array<VmaAllocation, FRAMES> backFaceDepthAllocations = {};
    std::array<VkDeviceMemory, FRAMES> backFaceDepthMemories = {};
    std::array<VkImageView, FRAMES> backFaceDepthImageViews = {};
    std::array<VkImageLayout, FRAMES> backFaceDepthImageLayouts = {};
    uint32_t renderWidth = 0;
    uint32_t renderHeight = 0;
    VulkanApp* appPtr = nullptr;
    CommandBufferState* cmdState = nullptr;
};
