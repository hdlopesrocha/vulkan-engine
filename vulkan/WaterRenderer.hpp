#pragma once

#include "VulkanApp.hpp"
#include "IndirectRenderer.hpp"
#include "SkyRenderer.hpp"
#include "SolidRenderer.hpp"
#include <glm/glm.hpp>
#include <memory>
#include <vector>
#include "../utils/Scene.hpp"
#include "../Uniforms.hpp"
#include "../widgets/SkySettings.hpp"
#include <unordered_map>
#include "../utils/Model3DVersion.hpp"

#include "../utils/WaterParams.hpp"

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
    VkImage getWaterGeomDepthImage(uint32_t frameIndex) const { return (frameIndex < 2) ? waterGeomDepthImages[frameIndex] : VK_NULL_HANDLE; }
    // Accessors for renderer-tracked layouts (used by widgets to record correct barriers)
    VkImageLayout getWaterGeomDepthLayout(uint32_t frameIndex) const;
    VkImageLayout getSceneDepthLayout(uint32_t frameIndex) const;
    
    // Get scene offscreen target views (for rendering scene before water)
    VkImage getSceneColorImage(uint32_t frameIndex) const { return sceneColorImages[frameIndex]; }
    VkImageView getSceneColorView(uint32_t frameIndex) const { return sceneColorImageViews[frameIndex]; }
    VkImage getSceneDepthImage(uint32_t frameIndex) const { return sceneDepthImages[frameIndex]; }
    VkImageView getSceneDepthView(uint32_t frameIndex) const { return sceneDepthImageViews[frameIndex]; }
    VkFramebuffer getSceneFramebuffer(uint32_t frameIndex) const { return sceneFramebuffers[frameIndex]; }
    VkRenderPass getSceneRenderPass() const { return sceneRenderPass; }
        void beginScenePass(VkCommandBuffer cmd, uint32_t frameIndex, VkClearValue colorClear, VkClearValue depthClear);
        void endScenePass(VkCommandBuffer cmd);


    void updateGPUParamsForLayer(uint32_t layer, const WaterParams& params);

    
    // Get the water geometry pipeline (for rendering water to G-buffer)
    VkPipeline getWaterGeometryPipeline() const { return waterGeometryPipeline; }
    
    // Get the water geometry pipeline layout
    VkPipelineLayout getWaterGeometryPipelineLayout() const { return waterGeometryPipelineLayout; }

    // Get the water geometry render pass (for creating compatible wireframe pipelines)
    VkRenderPass getWaterRenderPass() const { return waterRenderPass; }

    // Get the descriptor set layout for scene textures (set 2)
    VkDescriptorSetLayout getWaterDepthDescriptorSetLayout() const { return waterDepthDescriptorSetLayout; }

    // Prepare render state (UBO upload, descriptor update, pre-barrier).
    // Call this before beginWaterGeometryPass when manually recording commands.
    void prepareRender(VulkanApp* app, VkCommandBuffer cmd, uint32_t frameIndex,
                       VkImageView sceneColorView, VkImageView sceneDepthView,
                       VkImageView skyView = VK_NULL_HANDLE);

    // Emit post-geometry-pass barrier for fragment shader sampling.
    // Ensures per-frame water images are available to fragment shaders.
    void postRenderBarrier(VkCommandBuffer cmd, uint32_t frameIndex);
    
    // Get water depth descriptor set (for binding scene depth texture)
    VkDescriptorSet getWaterDepthDescriptorSet(uint32_t frameIndex) const { return waterDepthDescriptorSets[frameIndex]; }
    
    // Get water params buffer
    Buffer& getWaterParamsBuffer() { return waterParamsBuffer; }
    
    // Get sampler for ImGui texture display
    VkSampler getLinearSampler() const { return linearSampler; }
    
    // Get image views for debug display
    VkImageView getSceneColorImageView(uint32_t frameIndex) const { return sceneColorImageViews[frameIndex]; }
    VkImageView getSceneDepthImageView(uint32_t frameIndex) const { return sceneDepthImageViews[frameIndex]; }
    
    // Update the scene textures binding (color + depth + sky) for refraction and edge foam.
    // `backFaceDepthView` and `cube360View` may be VK_NULL_HANDLE if those targets
    // are not present; SceneRenderer should pass them when available.
    void updateSceneTexturesBinding(VulkanApp* app, VkImageView colorImageView, VkImageView depthImageView, uint32_t frameIndex, VkImageView skyImageView = VK_NULL_HANDLE, VkImageView backFaceDepthView = VK_NULL_HANDLE, VkImageView cube360View = VK_NULL_HANDLE);

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
    void createWaterRenderPass(VulkanApp* app);
    void createSceneRenderPass(VulkanApp* app);
    void createWaterPipelines(VulkanApp* app, const std::vector<WaterParams>& waterParams);
    void initializeWaterParamsBuffer(const std::vector<WaterParams>& waterParams);
    void createSamplers(VulkanApp* app);

    
    // Indirect renderer for water geometry
    IndirectRenderer waterIndirectRenderer;

    // Scene offscreen render target (render main scene here before water)
    // Per-frame offscreen render targets for main scene (color + depth) - 2 frames in flight
    std::array<VkImage, 2> sceneColorImages = {VK_NULL_HANDLE, VK_NULL_HANDLE};
    std::array<VkDeviceMemory, 2> sceneColorMemories = {VK_NULL_HANDLE, VK_NULL_HANDLE};
    std::array<VkImageView, 2> sceneColorImageViews = {VK_NULL_HANDLE, VK_NULL_HANDLE};
    std::array<VkImage, 2> sceneDepthImages = {VK_NULL_HANDLE, VK_NULL_HANDLE};
    std::array<VkDeviceMemory, 2> sceneDepthMemories = {VK_NULL_HANDLE, VK_NULL_HANDLE};
    std::array<VkImageView, 2> sceneDepthImageViews = {VK_NULL_HANDLE, VK_NULL_HANDLE};
    std::array<VkFramebuffer, 2> sceneFramebuffers = {VK_NULL_HANDLE, VK_NULL_HANDLE};
    VkRenderPass sceneRenderPass = VK_NULL_HANDLE;

    // Offscreen render targets for water geometry pass
    std::array<VkImage, 2> waterDepthImages = {VK_NULL_HANDLE, VK_NULL_HANDLE};
    std::array<VkDeviceMemory, 2> waterDepthMemories = {VK_NULL_HANDLE, VK_NULL_HANDLE};
    std::array<VkImageView, 2> waterDepthImageViews = {VK_NULL_HANDLE, VK_NULL_HANDLE};
    // Alternate view that swizzles alpha (linear depth) into RGB for debug display
    std::array<VkImageView, 2> waterDepthAlphaImageViews = {VK_NULL_HANDLE, VK_NULL_HANDLE};

    // (Normals and mask images removed — water pass now only outputs a single color target)
    
    // Water geometry pass depth buffer (per-frame)
    std::array<VkImage, 2> waterGeomDepthImages = {VK_NULL_HANDLE, VK_NULL_HANDLE};
    std::array<VkDeviceMemory, 2> waterGeomDepthMemories = {VK_NULL_HANDLE, VK_NULL_HANDLE};
    std::array<VkImageView, 2> waterGeomDepthImageViews = {VK_NULL_HANDLE, VK_NULL_HANDLE};

    std::array<VkFramebuffer, 2> waterFramebuffers = {VK_NULL_HANDLE, VK_NULL_HANDLE};
    VkRenderPass waterRenderPass = VK_NULL_HANDLE;

    // Pipelines
    VkPipeline waterGeometryPipeline = VK_NULL_HANDLE;
    
    // Water geometry pipeline layout (includes depth texture binding)
    VkPipelineLayout waterGeometryPipelineLayout = VK_NULL_HANDLE;

    // Descriptor set for water geometry (scene depth texture)
    VkDescriptorSetLayout waterDepthDescriptorSetLayout = VK_NULL_HANDLE;
    VkDescriptorPool waterDepthDescriptorPool = VK_NULL_HANDLE;
    // Per-frame descriptor sets for scene textures (2 frames in flight)
    std::array<VkDescriptorSet, 2> waterDepthDescriptorSets = {VK_NULL_HANDLE, VK_NULL_HANDLE};

    // Storage buffer (SSBO) for per-layer WaterParamsGPU entries
    Buffer waterParamsBuffer;
    // Number of entries allocated in `waterParamsBuffer`
    uint32_t waterParamsCount = 0;
    // Back-pointer to app for mapping/unmapping buffer when updating GPU data
    VulkanApp* appPtr = nullptr;

    // Samplers
    VkSampler linearSampler = VK_NULL_HANDLE;
    VkSampler nearestSampler = VK_NULL_HANDLE;

    // Whether a cubemap reflection is currently available (set by SceneRenderer)
    bool cube360Available = false;
    // Last cubemap image view provided by SceneRenderer (preserve across updates)
    VkImageView currentCube360View = VK_NULL_HANDLE;

    uint32_t renderWidth = 0;
    uint32_t renderHeight = 0;

    // Map of node -> model version for water geometry managed here
    std::unordered_map<NodeID, Model3DVersion> waterNodeModelVersions;

};
