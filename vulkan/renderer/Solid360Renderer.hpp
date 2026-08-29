#pragma once
#include "Renderer.hpp"
#include "../VulkanApp.hpp"
#include "../TrackedHandle.hpp"
#include "SkyRenderer.hpp"
#include "SolidRenderer.hpp"
#include "WaterRenderer.hpp"
#include "../ubo/UniformObject.hpp"
#include <array>
#include <vector>
#include "CommandBufferState.hpp"

// Per-face resources for parallel cubemap rendering. Each of the 6 cubemap faces
// gets its own graphics descriptor set (binding 0 = a distinct slot of the 6-slot
// cube UBO), its own solid/water compute descriptor sets (compact/visible outputs
// bound to per-face buffers) and its own compact/visible indirect buffers, so the
// 6 face rasterizations can run concurrently on different queues without sharing
// any writable resource (each face writes a distinct array layer of the cube image).
struct Cube360FaceResources {
    std::array<VkDescriptorSet, 6> gfxDs = {};
    std::array<VkDescriptorSet, 6> solidComputeDs = {};
    std::array<VkDescriptorSet, 6> waterComputeDs = {};
    std::array<VkBuffer, 6> compact = {};
    std::array<VkBuffer, 6> visible = {};
    std::array<VkBuffer, 6> waterCompact = {};
    std::array<VkBuffer, 6> waterVisible = {};
    // Materials SSBO set for the water pipeline's set 1 (the cube360 solid pass
    // binds brush depth at set 1, which is wrong for the water pipeline, so the
    // water draw explicitly rebinds this materials DS at set 1).
    VkDescriptorSet materialsDs = VK_NULL_HANDLE;
    // Fallback brush-depth DS for the solid pipeline's set 1. The caller may pass a
    // null brush depth DS (e.g. before the brush renderer's sets are allocated); bind
    // this dedicated DS instead so set 1 always carries the brush-depth layout.
    VkDescriptorSet brushDepthDs = VK_NULL_HANDLE;
};

class Solid360Renderer : public Renderer {
public:
    Solid360Renderer();
    ~Solid360Renderer();
    void init(VulkanApp* app);
    void cleanup(VulkanApp* app) override;

    void createSolid360Targets(VulkanApp* app, VkSampler linearSampler);
    void destroySolid360Targets(VulkanApp* app);

    void setWaterRenderer(WaterRenderer* wr) { waterRenderer = wr; }

    // Create depth-only and EQUAL-compare pipelines for deferred depth testing
    void createSolid360Pipelines(VulkanApp* app);

    // Render the 6 cubemap faces of the solid360 reflections in parallel. The 6 face
    // rasterizations are submitted as independent primary command buffers to distinct
    // graphics-family queues (app->getCubeQueue, round-robin); the indirect cull must
    // stay serial (it relies on a shared scratch buffer), so a single cull command
    // buffer feeds each face its own compact/visible buffers, then signals one
    // per-face semaphore (semCullFace[f]) that the raster CB waits on. A final join
    // command buffer waits all semFaceDone[f] and signals signalSolid360 so downstream
    // water/composite passes can sample the cubemap. The caller must pre-create
    // semCullFace[6] and semFaceDone[6]; faceRes holds the per-face descriptors and
    // indirect buffers (see Cube360FaceResources).
    void render(VulkanApp* app,
                        SkyRenderer* skyRenderer, SkySettings::Mode skyMode,
                        SolidRenderer* solidRenderer,
                        VkDescriptorSet brushDepthSet,
                        VkBuffer faceUboBuffer,
                        const Cube360FaceResources& faceRes,
                        const UniformObject& ubo,
                         bool renderSolid, bool renderWater,
                          VkSemaphore waitCullSolid360, uint64_t waitCullSolid360Value,
                          VkSemaphore waitShadowSolid360, uint64_t waitShadowSolid360Value,
                          VkSemaphore waitSolid360, uint64_t waitSolid360Value,
                          const VkSemaphore (&semCullFace)[6],
                         const VkSemaphore (&semFaceDone)[6],
                         VkSemaphore signalSolid360, uint64_t signalSolid360Value,
                         uint32_t frameIndex = 0);

    // Return the cubemap view for reflection sampling
    VkImageView getSolid360View() const { return cube360CubeView; }
    VkSampler getSolid360Sampler() const { return solid360Sampler; }
    VkImageView getDummyCubeView() const { return cube360DummyCubeView; }
    VkImageView getCube360FaceView(uint32_t face) const { return (face < 6) ? cube360FaceViews[face] : VK_NULL_HANDLE; }
    VkImageView getCube360DepthView(uint32_t face) const { return (face < 6) ? cube360DepthViews[face] : VK_NULL_HANDLE; }
    VkImage getCube360DepthImage() const { return cube360DepthImage; }

    // Per-face depth layout accessors (used by widgets to record correct barriers)
    VkImageLayout getCube360DepthLayout(uint32_t face) const {
        if (face < cube360DepthLayouts.size()) return cube360DepthLayouts[face];
        return VK_IMAGE_LAYOUT_UNDEFINED;
    }
    void setCube360DepthLayout(uint32_t face, VkImageLayout layout) {
        if (face < cube360DepthLayouts.size()) cube360DepthLayouts[face] = layout;
    }

private:
    WaterRenderer* waterRenderer = nullptr;
    static constexpr uint32_t CUBE360_FACE_SIZE = 512;

    // Deferred depth test pipelines
    TrackedHandle<VkPipeline> depthOnlyPipeline;
    TrackedHandle<VkPipelineLayout> depthOnlyPipelineLayout;
    TrackedHandle<VkPipeline> equalComparePipeline;
    TrackedHandle<VkPipelineLayout> equalComparePipelineLayout;

    VkImage cube360ColorImage = VK_NULL_HANDLE;
    VmaAllocation cube360ColorAllocation = VK_NULL_HANDLE;
    VkDeviceMemory cube360ColorMemory = VK_NULL_HANDLE;
    std::array<VkImageView, 6> cube360FaceViews = {};
    VkImageView cube360CubeView = VK_NULL_HANDLE;
    TrackedHandle<VkSampler> solid360Sampler;

    VkImage cube360DummyColorImage = VK_NULL_HANDLE;
    VmaAllocation cube360DummyColorAllocation = VK_NULL_HANDLE;
    VkDeviceMemory cube360DummyColorMemory = VK_NULL_HANDLE;
    VkImageView cube360DummyCubeView = VK_NULL_HANDLE;

    VkImage cube360DepthImage = VK_NULL_HANDLE;
    VmaAllocation cube360DepthAllocation = VK_NULL_HANDLE;
    VkDeviceMemory cube360DepthMemory = VK_NULL_HANDLE;
    std::array<VkImageView, 6> cube360DepthViews = {};

    // Track per-face color and depth image layouts
    std::array<VkImageLayout, 6> cube360ColorLayouts = {};
    std::array<VkImageLayout, 6> cube360DepthLayouts = {};

    // Equirectangular conversion removed: use cubemap directly for sampling

    // Persistently mapped staging buffers for UBO uploads via vkCmdCopyBuffer
    // (replaces vkCmdUpdateBuffer to avoid implicit FULL_QUEUE barrier).
    // Triple-buffered to avoid races with frames in flight.
    static constexpr uint32_t STAGING_FRAMES = 3;
    Buffer stagingUBOs[STAGING_FRAMES];
    mutable uint32_t stagingFrameIndex = 0;
};
