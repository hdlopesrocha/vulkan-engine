#pragma once

#include "../VulkanApp.hpp"
#include "../TrackedHandle.hpp"
#include "IndirectRenderer.hpp"
#include "SkyRenderer.hpp"
#include "SolidRenderer.hpp"
#include <glm/glm.hpp>
#include <memory>
#include <vector>
#include "../../utils/Scene.hpp"
#include "../ubo/UniformObject.hpp"
#include "../../widgets/SkySettings.hpp"
#include <unordered_map>
#include "../../utils/Model3DVersion.hpp"
#include "../../utils/WaterParams.hpp"
#include "../ubo/WaterParamsGPU.hpp"
#include "../ubo/WaterRenderUBO.hpp"
#include "../ubo/WaterUBO.hpp"
#include "CommandBufferState.hpp"

class WaterRenderer {
public:
    WaterRenderer();
    ~WaterRenderer();

    void init(VulkanApp* app, Buffer& waterParamsBuffer, const std::vector<WaterParams>& waterParams, uint32_t layerCount);
    void cleanup(VulkanApp* app);

    // Create offscreen render targets for water rendering
    void createRenderTargets(VulkanApp* app, uint32_t width, uint32_t height);
    void destroyRenderTargets(VulkanApp* app);

    // Get the indirect renderer for water meshes
    IndirectRenderer& getIndirectRenderer() { return waterIndirectRenderer; }

    // Begin water geometry pass (renders water depth/normals to offscreen target)
    void beginWaterGeometryPass(VkCommandBuffer cmd, uint32_t frameIndex);
    void endWaterGeometryPass(VkCommandBuffer cmd);

    // Back-face depth pre-pass (reversed winding for water volume thickness)
    // NOTE: back-face depth pre-pass is now owned by SceneRenderer. SceneRenderer
    // should provide the back-face depth view to WaterRenderer via
    // `updateSceneTexturesBinding` when available.

    // Execute the water offscreen geometry pass on the provided command buffer.
    // The solid render pass must have already ended on this same command buffer.
    void render(VulkanApp* app, VkCommandBuffer cmd, uint32_t frameIndex,
                VkImageView sceneColorView, VkImageView sceneDepthView,
                VkImageView skyView = VK_NULL_HANDLE);

    // Get water color/depth image view for post-process sampling
    VkImageView getWaterDepthView(uint32_t frameIndex) const { return waterDepthImageViews[frameIndex]; }
    // View that swizzles alpha into RGB so linear depth can be displayed easily
    VkImageView getWaterDepthAlphaView(uint32_t frameIndex) const { return waterDepthAlphaImageViews[frameIndex]; }
    // Depth image view used as the depth/stencil attachment for the water geometry pass
    VkImageView getWaterGeomDepthView(uint32_t frameIndex) const { return waterGeomDepthImageViews[frameIndex]; }
    // Expose the raw water geometry depth image (for layout transitions and sampling)
    VkImage getWaterGeomDepthImage(uint32_t frameIndex) const { return (frameIndex < 3) ? waterGeomDepthImages[frameIndex] : VK_NULL_HANDLE; }
    // Accessors for renderer-tracked layouts (used by widgets to record correct barriers)
    VkImageLayout getWaterGeomDepthLayout(uint32_t frameIndex) const;
    void setWaterGeomDepthLayout(uint32_t frameIndex, VkImageLayout layout);
    VkImageLayout getSceneDepthLayout(uint32_t frameIndex) const;
    
    // Get scene offscreen target views (for rendering scene before water)
    VkImage getSceneColorImage(uint32_t frameIndex) const { return sceneColorImages[frameIndex]; }
    VkImageView getSceneColorView(uint32_t frameIndex) const { return sceneColorImageViews[frameIndex]; }
    VkImage getSceneDepthImage(uint32_t frameIndex) const { return sceneDepthImages[frameIndex]; }
    VkImageView getSceneDepthView(uint32_t frameIndex) const { return sceneDepthImageViews[frameIndex]; }



    void updateGPUParamsForLayer(uint32_t layer, const WaterParams& params);

    // Initialize the per-frame water geometry depth image from the scene
    // depth image by copying depth values. This allows the water geometry
    // pass to depth-test against solid geometry so water is only rasterized
    // where it is visible in front of solids.
    
    // Get the water geometry pipeline (for rendering water to G-buffer)
    VkPipeline getWaterGeometryPipeline() const { return waterGeometryPipeline; }
    
    // Get the water geometry pipeline layout
    VkPipelineLayout getWaterGeometryPipelineLayout() const { return waterGeometryPipelineLayout; }

    // Get the descriptor set layout for scene textures (set 2)
    VkDescriptorSetLayout getWaterDepthDescriptorSetLayout() const { return waterDepthDescriptorSetLayout; }

    // Cubemap-compatible water pipeline (uses swapchain color format)
    VkPipeline getCubemapWaterPipeline() const { return cubemapWaterPipeline; }

    // Render water into a given color+depth attachment pair (used by Solid360Renderer).
    // The caller must ensure the attachments are in COLOR_ATTACHMENT_OPTIMAL /
    // DEPTH_STENCIL_ATTACHMENT_OPTIMAL layout before calling, and transition them
    // back after. descriptorSet0 must contain at least bindings 0, 5, 7, 10.
    // `frameIndex` selects the per-frame cubemap descriptor set so it is never
    // updated while a previous frame's command buffer is still pending (no UAB needed).
    void renderWaterIntoCubemap(VkCommandBuffer cmd,
                                VkImageView colorView, VkImageView depthView,
                                VkDescriptorSet descriptorSet0,
                                VkDescriptorSet materialDs,
                                uint32_t faceSize,
                                VkBuffer waterCompactBuffer, VkBuffer waterVisibleCountBuffer,
                                uint32_t frameIndex);

    VkDescriptorSet getCubemapWaterDepthDescriptorSet(uint32_t frameIndex) const {
        return (frameIndex < FRAMES) ? cubemapWaterDepthDS[frameIndex] : VK_NULL_HANDLE;
    }

    // Prepare render state (UBO upload, descriptor update, pre-barrier).
    // Call this before beginWaterGeometryPass when manually recording commands.
    void prepareRender(VulkanApp* app, VkCommandBuffer cmd, uint32_t frameIndex,
                       VkImageView sceneColorView, VkImageView sceneDepthView,
                       VkImageView skyView = VK_NULL_HANDLE);

    // Emit post-geometry-pass barrier for fragment shader sampling.
    // Ensures per-frame water images are available to fragment shaders.
    void postRenderBarrier(VkCommandBuffer cmd, uint32_t frameIndex);
    
    // Get water depth descriptor set (for binding scene depth texture)
    VkDescriptorSet getWaterDepthDescriptorSet(uint32_t frameIndex) const { return (frameIndex < FRAMES) ? waterDepthDescriptorSets[frameIndex] : VK_NULL_HANDLE; }
    
    // Get water params buffer
    Buffer& getWaterParamsBuffer() { return waterParamsBuffer; }
    
    // Ensure cubemap water rendering resources exist (pipeline, dummy textures, descriptor set).
    // Called lazily from renderWaterIntoCubemap.
    void ensureCubemapResources(VulkanApp* app, VkFormat colorFormat);

    // Get sampler for ImGui texture display
    VkSampler getLinearSampler() const { return linearSampler; }
    
    // Get image views for debug display
    VkImageView getSceneColorImageView(uint32_t frameIndex) const { return sceneColorImageViews[frameIndex]; }
    VkImageView getSceneDepthImageView(uint32_t frameIndex) const { return sceneDepthImageViews[frameIndex]; }
    
    // Update the scene textures binding (color + depth + sky) for refraction and edge foam.
    // Writes into `ds` (the caller chooses the per-command-buffer set so the set is
    // never shared between the async back-face task and the main command buffer).
    // `backFaceDepthView` and `cube360View` may be VK_NULL_HANDLE if those targets
    // are not present; SceneRenderer should pass them when available.
    void updateSceneTexturesBinding(VulkanApp* app, VkDescriptorSet ds, VkImageView colorImageView, VkImageView depthImageView, uint32_t frameIndex, VkImageView skyImageView = VK_NULL_HANDLE, VkImageView backFaceDepthView = VK_NULL_HANDLE, VkImageView cube360View = VK_NULL_HANDLE);

    // Allocate a fresh per-frame scene-texture descriptor set, free the previous
    // one, and update it with the given views. Returns the new set (or
    // VK_NULL_HANDLE on failure). The previous set is freed only after its command
    // buffer has completed (the caller must invoke this from preRenderPass, which
    // runs after the per-slot in-flight fence wait), so the set is never reused
    // while pending and never needs UPDATE_AFTER_BIND.
    VkDescriptorSet prepareSceneTexturesForFrame(VulkanApp* app, uint32_t frameIndex,
                                                 VkImageView colorImageView, VkImageView depthImageView,
                                                 VkImageView skyImageView = VK_NULL_HANDLE,
                                                 VkImageView backFaceDepthView = VK_NULL_HANDLE,
                                                 VkImageView cube360View = VK_NULL_HANDLE);

    // Dedicated pool + accessors for per-task descriptor sets used by the async
    // back-face task, so it never shares the per-frame set with the main CB.
    VkDescriptorPool getAsyncWaterDepthPool() const { return asyncWaterDepthPool; }

    // Clear per-frame render targets (color/depth) into default values.
    // Call this each frame when water rendering is disabled to avoid sampling
    // stale content from previous frames.
    void clearRenderTargets(VulkanApp* app, VkCommandBuffer cmd, uint32_t frameIndex);

    // Solid 360° cubemap reflection and back-face rendering are owned by SceneRenderer.
    // SceneRenderer must call `updateSceneTexturesBinding` to provide any required
    // views (back-face depth, cubemap/equirect) to WaterRenderer.

    // Register model version for water meshes (stored here)
    void registerModelVersion(NodeID id, const Model3DVersion& ver) { waterNodeModelVersions[id] = ver; }

    // Remove all registered water meshes from the indirect renderer and clear the map
    void removeAllRegisteredMeshes() {
        for (auto &entry : waterNodeModelVersions) {
            if (entry.second.meshId != UINT32_MAX) waterIndirectRenderer.removeMesh(entry.second.meshId);
        }
        waterNodeModelVersions.clear();
    }

private:

    void createWaterPipelines(VulkanApp* app, const std::vector<WaterParams>& waterParams);
    void initializeWaterParamsBuffer(const std::vector<WaterParams>& waterParams);
    void createSamplers(VulkanApp* app);

    
    // Indirect renderer for water geometry
    IndirectRenderer waterIndirectRenderer;

    // Scene offscreen render target (render main scene here before water)
    // Per-frame offscreen render targets for main scene (color + depth) - 2 frames in flight
    static constexpr uint32_t FRAMES = VulkanApp::MAX_FRAMES_IN_FLIGHT;
    std::array<VkImage, FRAMES> sceneColorImages = {};
    std::array<VmaAllocation, FRAMES> sceneColorAllocations = {};
    std::array<VkDeviceMemory, FRAMES> sceneColorMemories = {};
    std::array<VkImageView, FRAMES> sceneColorImageViews = {};
    std::array<VkImage, FRAMES> sceneDepthImages = {};
    std::array<VmaAllocation, FRAMES> sceneDepthAllocations = {};
    std::array<VkDeviceMemory, FRAMES> sceneDepthMemories = {};
    std::array<VkImageView, FRAMES> sceneDepthImageViews = {};
    std::array<VkImage, FRAMES> waterDepthImages = {};
    std::array<VmaAllocation, FRAMES> waterDepthAllocations = {};
    std::array<VkDeviceMemory, FRAMES> waterDepthMemories = {};
    std::array<VkImageView, FRAMES> waterDepthImageViews = {};
    std::array<VkImageView, FRAMES> waterDepthAlphaImageViews = {};
    std::array<VkImage, FRAMES> waterGeomDepthImages = {};
    std::array<VmaAllocation, FRAMES> waterGeomDepthAllocations = {};
    std::array<VkDeviceMemory, FRAMES> waterGeomDepthMemories = {};
    std::array<VkImageView, FRAMES> waterGeomDepthImageViews = {};

    // Pipelines
    TrackedHandle<VkPipeline> waterGeometryPipeline;
    TrackedHandle<VkPipeline> waterDepthPrePassPipeline;
    TrackedHandle<VkPipeline> cubemapWaterPipeline;
    
    // Water geometry pipeline layout (includes depth texture binding)
    TrackedHandle<VkPipelineLayout> waterGeometryPipelineLayout;

    // Descriptor set for water geometry (scene depth texture)
    TrackedHandle<VkDescriptorSetLayout> waterDepthDescriptorSetLayout;
    TrackedHandle<VkDescriptorPool> waterDepthDescriptorPool;
    // Per-frame descriptor sets for scene textures (3 frames in flight)
    // One scene-texture descriptor set per in-flight slot. Each slot's set is
    // freed and reallocated in prepareSceneTexturesForFrame() while the per-slot
    // in-flight fence guarantees the previous command buffer using that slot has
    // completed. This avoids both VUID-03047 (update/destroy while pending) and
    // the GPU-assisted false "length 0" seen with UPDATE_AFTER_BIND.
    std::array<VkDescriptorSet, FRAMES> waterDepthDescriptorSets{VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE};
    // Dedicated pool for per-task async back-face descriptor sets (never shared
    // with the main command buffer, so it can be updated without UPDATE_AFTER_BIND).
    TrackedHandle<VkDescriptorPool> asyncWaterDepthPool;

    // Cubemap water pass resources (per-frame to avoid updating a set that a
    // previous frame's command buffer still has pending)
    std::array<TrackedHandle<VkDescriptorSet>, FRAMES> cubemapWaterDepthDS{};
    VkImage cubemapDummyDepthImage = VK_NULL_HANDLE;
    VmaAllocation cubemapDummyDepthAllocation = VK_NULL_HANDLE;
    VkDeviceMemory cubemapDummyDepthMemory = VK_NULL_HANDLE;
    VkImageView cubemapDummyDepthView = VK_NULL_HANDLE;
    VkImage cubemapDummyCubeImage = VK_NULL_HANDLE;
    VmaAllocation cubemapDummyCubeAllocation = VK_NULL_HANDLE;
    VkDeviceMemory cubemapDummyCubeMemory = VK_NULL_HANDLE;
    VkImageView cubemapDummyCubeView = VK_NULL_HANDLE;

    // Storage buffer (SSBO) for per-layer WaterParamsGPU entries
    Buffer waterParamsBuffer;
    // Number of entries allocated in `waterParamsBuffer`
    uint32_t waterParamsCount = 0;
    // Back-pointer to app for mapping/unmapping buffer when updating GPU data
    VulkanApp* appPtr = nullptr;

    // Samplers
    TrackedHandle<VkSampler> linearSampler;
    TrackedHandle<VkSampler> nearestSampler;

    // Whether a cubemap reflection is currently available (set by SceneRenderer)
    bool cube360Available = false;
    // Last cubemap image view provided by SceneRenderer (preserve across updates)
    VkImageView currentCube360View = VK_NULL_HANDLE;

    uint32_t renderWidth = 0;
    uint32_t renderHeight = 0;

    // Cached frame index set by beginWaterGeometryPass, used by endWaterGeometryPass
    uint32_t activeWaterFrameIndex = 0;

    // Map of node -> model version for water geometry managed here
    std::unordered_map<NodeID, Model3DVersion> waterNodeModelVersions;
    CommandBufferState* cmdState = nullptr;
public:
    void setCmdState(CommandBufferState* state) { cmdState = state; }
};
