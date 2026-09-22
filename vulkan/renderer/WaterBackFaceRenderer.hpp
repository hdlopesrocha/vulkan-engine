#pragma once
#include "Renderer.hpp"
#include "../VulkanApp.hpp"
#include "../TrackedHandle.hpp"
#include "IndirectRenderer.hpp"
#include <array>
#include "CommandBufferState.hpp"

class WaterBackFaceRenderer : public Renderer {
public:
    WaterBackFaceRenderer();
    ~WaterBackFaceRenderer();
    void init(VulkanApp* app);
    void cleanup(VulkanApp* app) override;

    void createPipelines(VulkanApp* app, VkPipelineLayout pipelineLayout);

    // Global tessellation toggle (Settings::tessellationEnabled): selects the
    // TRIANGLE_LIST + WATER_NO_TESS variant (no TCS/TES) when false. Both
    // variants are built by createPipelines() with the caller's layout, so
    // descriptor sets are shared and no pipeline is rebuilt at runtime.
    // Default true preserves the historical always-tessellated behavior.
    void setTessellationEnabled(bool enabled) { tessellationEnabled_ = enabled; }
    bool tessellationEnabled() const { return tessellationEnabled_; }

    // Create/destroy per-frame depth targets
    void createRenderTargets(VulkanApp* app, uint32_t width, uint32_t height);
    void destroyRenderTargets(VulkanApp* app);

    // Execute back-face depth pre-pass. Caller provides indirect renderer
    // that actually draws the water geometry.
    void render(VulkanApp* app, VkCommandBuffer cmd, uint32_t frameIndex,
                            IndirectRenderer& indirect, VkPipelineLayout pipelineLayout,
                            VkDescriptorSet mainDs, VkDescriptorSet sceneDs,
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
    void patchBinding0(VkDescriptorSet ds, VkImageView newView);

private:
    // Build one back-face pipeline variant (tess: PATCH_LIST + TCS/TES, else
    // TRIANGLE_LIST with the WATER_NO_TESS vertex module and no tessellation
    // state) with the caller-provided layout. `vertPath` selects the vertex
    // module (main_water.vert vs main_water_no_tess.vert).
    void createBackFacePipeline(VulkanApp* app, VkPipelineLayout pipelineLayout,
                                bool tess, const char* vertPath,
                                const char* debugName,
                                TrackedHandle<VkPipeline>& pipelineOut);

    // Active pipeline for the current tessellation toggle; falls back to the
    // tessellated pipeline if the no-tess variant failed to build.
    VkPipeline activePipeline() const {
        if (!tessellationEnabled_ && backFacePipelineNoTess != VK_NULL_HANDLE)
            return backFacePipelineNoTess.handle;
        return backFacePipeline.handle;
    }

    TrackedHandle<VkPipeline> backFacePipeline;
    // Non-tessellation variant (C1): TRIANGLE_LIST, no TCS/TES.
    TrackedHandle<VkPipeline> backFacePipelineNoTess;
    bool tessellationEnabled_ = true;
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
};
