#pragma once

#include "Renderer.hpp"
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
#include "../../space/Model3DVersion.hpp"
#include "../../utils/WaterParams.hpp"
#include "../ubo/WaterParamsGPU.hpp"
#include "../ubo/WaterRenderUBO.hpp"
#include "../ubo/WaterUBO.hpp"
#include "CommandBufferState.hpp"

class BrushRenderer;
class WaterBackFaceRenderer;
class Solid360Renderer;
class WireframeRenderer;

class WaterRenderer : public Renderer {
public:
    WaterRenderer();
    ~WaterRenderer();

    void init(VulkanApp* app, Buffer& waterParamsBuffer_, const std::vector<WaterParams>& waterParams, uint32_t layerCount);
    void cleanup(VulkanApp* app) override;

    // Inject the scene sub-renderers the water pass samples from or draws
    // alongside (solid offscreen targets, brush liquid geometry, back-face
    // depth, 360° reflection cubemap, wireframe overlay). Called once by
    // SceneRenderer after all sub-renderers are created.
    void setSceneRenderers(SolidRenderer* solid, BrushRenderer* brush,
                           WaterBackFaceRenderer* backFace, Solid360Renderer* solid360,
                           WireframeRenderer* waterWireframe);

    // Full water pass orchestration: updates the water render UBO with the
    // active layer time, (re)allocates this slot's scene-texture descriptor
    // set, then records the offscreen water geometry pass (filled or wireframe
    // overlay) on the same command
    // buffer so the solid pass outputs are available for sampling.
    void renderPass(VulkanApp* app, VkCommandBuffer cmd, uint32_t frameIndex,
                    bool waterWireframeEnabled, float waterTime, VkImageView skyView,
                    VkDescriptorSet overrideWaterDs = VK_NULL_HANDLE,
                    bool drawBrushLiquid = true);

    // Brush-liquid overlay: draws the brush water geometry (secondaryIR) on top of
    // the already-rendered water targets on its own command buffer/queue, so it runs
    // in parallel with the main water pass's consumers. The water geometry pass is
    // re-entered with LOAD ops (preserving the main water EVSM + geom depth) and the
    // targets are restored to SHADER_READ_OPTIMAL for the composite. Must be called
    // after the main water pass has completed (waits on semWater externally).
    void renderBrushLiquid(VulkanApp* app, VkCommandBuffer cmd, uint32_t frameIndex,
                           VkImageView skyView, VkDescriptorSet overrideWaterDs = VK_NULL_HANDLE);

    // Water render time UBO (binding 10) — created and updated here, but
    // bound into the scene descriptor sets by SceneRenderer.
    Buffer& getWaterRenderUBO() { return waterRenderUBO_; }

    // Create offscreen render targets for water rendering
    void createRenderTargets(VulkanApp* app, uint32_t width, uint32_t height);
    void destroyRenderTargets(VulkanApp* app);

    // Get the indirect renderer for water meshes
    IndirectRenderer& getIndirectRenderer() { return waterIndirectRenderer; }

    // Begin water geometry pass (renders water depth/normals to offscreen target)
    void beginWaterGeometryPass(VkCommandBuffer cmd, uint32_t frameIndex, bool loadExisting = false);
    void endWaterGeometryPass(VkCommandBuffer cmd);
    // Merged end for the brush-liquid overlay path: water color + water
    // geometry depth → SHADER_READ_ONLY_OPTIMAL in a single barrier call
    // (was: endWaterGeometryPass plus a lone depth transition).
    void endWaterGeometryPassWithDepth(VkCommandBuffer cmd, uint32_t frameIndex);

    // Back-face depth pre-pass (reversed winding for water volume thickness)
    // NOTE: back-face depth pre-pass is now owned by SceneRenderer. SceneRenderer
    // should provide the back-face depth view to WaterRenderer via
    // `updateSceneTexturesBinding` when available.

    // Execute the water offscreen geometry pass on the provided command buffer.
    // The solid render pass must have already ended on this same command buffer.
    // `secondaryIR` is drawn with the same water pipeline, right after the main
    // water IR, inside the same geometry pass (used for brush liquid geometry —
    // brush water renders like main water but lives in its own IndirectRenderer).
    // `overrideWaterDs` (async path) is the caller-owned set-2 descriptor set
    // (binding 0 = real back-face depth, binding 1 = solid360 cubemap); when null
    // the per-frame set from prepareSceneTexturesForFrame() is bound instead.
    void render(VulkanApp* app, VkCommandBuffer cmd, uint32_t frameIndex,
                VkImageView sceneColorView,
                VkImageView skyView = VK_NULL_HANDLE,
                IndirectRenderer* secondaryIR = nullptr,
                VkDescriptorSet overrideWaterDs = VK_NULL_HANDLE);

    // Get water color/depth image view for post-process sampling
    VkImageView getWaterDepthView(uint32_t frameIndex) const { return waterDepthImageViews[frameIndex]; }
    // Depth image view used as the depth/stencil attachment for the water geometry pass
    VkImageView getWaterGeomDepthView(uint32_t frameIndex) const { return waterGeomDepthImageViews[frameIndex]; }
    // Expose the raw water geometry depth image (for layout transitions and sampling)
    VkImage getWaterGeomDepthImage(uint32_t frameIndex) const { return (frameIndex < 3) ? waterGeomDepthImages[frameIndex] : VK_NULL_HANDLE; }
    // Accessors for renderer-tracked layouts (used by widgets to record correct barriers)
    VkImageLayout getWaterGeomDepthLayout(uint32_t frameIndex) const;
    void setWaterGeomDepthLayout(uint32_t frameIndex, VkImageLayout layout);
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

    // Prepare render state (UBO upload, descriptor update, pre-barrier).
    // Call this before beginWaterGeometryPass when manually recording commands.
    void prepareRender(VulkanApp* app, VkCommandBuffer cmd, uint32_t frameIndex,
                       VkImageView sceneColorView,
                       VkImageView skyView = VK_NULL_HANDLE);

    // Get water depth descriptor set (for binding scene depth texture)
    VkDescriptorSet getWaterDepthDescriptorSet(uint32_t frameIndex) const { return (frameIndex < FRAMES) ? waterDepthDescriptorSets[frameIndex] : VK_NULL_HANDLE; }

    // Ensure cubemap reflection resources exist (the cubemap-compatible water
    // pipeline and its set-2 descriptor set bound to real resources).
    // Idempotent; called lazily from the solid 360 pass. The real cubemap
    // targets are created in SceneRenderer::init before first use.
    void ensureCubemapResources(VulkanApp* app, VkFormat colorFormat);

    // Draw water into one solid 360 cubemap face. Must be called INSIDE the face's
    // command buffer AFTER the solid color pass has ended; begins its own dynamic
    // rendering instance with LOAD ops so water composites over solid+sky, depth
    // tested against the prepassed solid depth (no depth writes). The face UBO
    // carries materialFlags.x == 1 (capture mode), so water.frag skips
    // reflection/refraction and never samples the cubemap it is rendering into;
    // set 2 is bound to the back-face dummy depth + real cube view instead.
    // `sceneDs0` is the per-face main-layout descriptor set (binding 0 = this
    // face's UBO slot).
    void renderWaterIntoCubemap(VkCommandBuffer cmd, VkDescriptorSet sceneDs0,
                                VkImageView colorView, VkImageView depthView,
                                uint32_t faceSize,
                                VkBuffer waterCompactBuffer, VkBuffer waterVisibleCountBuffer);

    // Get sampler for ImGui texture display
    VkSampler getLinearSampler() const { return linearSampler; }
    
    // Update the scene textures binding (color + depth + sky) for refraction and edge foam.
    // Writes into `ds` (the caller chooses the per-command-buffer set so the set is
    // never shared between the async back-face task and the main command buffer).
    // `backFaceDepthView` and `cube360View` may be VK_NULL_HANDLE if those targets
    // are not present; SceneRenderer should pass them when available.
    void updateSceneTexturesBinding(VulkanApp* app, VkDescriptorSet ds, uint32_t frameIndex, VkImageView backFaceDepthView = VK_NULL_HANDLE, VkImageView cube360View = VK_NULL_HANDLE);

    // Allocate a fresh per-frame scene-texture descriptor set, free the previous
    // one, and update it with the given views. Returns the new set (or
    // VK_NULL_HANDLE on failure). The previous set is freed only after its command
    // buffer has completed (the caller must invoke this from preRenderPass, which
    // runs after the per-slot in-flight fence wait), so the set is never reused
    // while pending and never needs UPDATE_AFTER_BIND.
    VkDescriptorSet prepareSceneTexturesForFrame(VulkanApp* app, uint32_t frameIndex,
                                                 VkImageView backFaceDepthView = VK_NULL_HANDLE,
                                                 VkImageView cube360View = VK_NULL_HANDLE);

    // Clear per-frame render targets (color/depth) into default values.
    // Call this each frame when water rendering is disabled to avoid sampling
    // stale content from previous frames.
    void clearRenderTargets(VulkanApp* app, VkCommandBuffer cmd, uint32_t frameIndex);

    // Solid 360° cubemap reflection and back-face rendering are owned by SceneRenderer.
    // SceneRenderer must call `updateSceneTexturesBinding` to provide any required
    // views (back-face depth, cubemap/equirect) to WaterRenderer.

private:

    // vkCmdEndRendering without barriers (shared by endWaterGeometryPass and
    // endWaterGeometryPassWithDepth, which emit their own batched barriers).
    void endWaterRendering(VkCommandBuffer cmd);

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
    std::array<VkImage, FRAMES> waterGeomDepthImages = {};
    std::array<VmaAllocation, FRAMES> waterGeomDepthAllocations = {};
    std::array<VkDeviceMemory, FRAMES> waterGeomDepthMemories = {};
    std::array<VkImageView, FRAMES> waterGeomDepthImageViews = {};

    // Pipelines
    TrackedHandle<VkPipeline> waterGeometryPipeline;

    // Water geometry pipeline layout (includes depth texture binding)
    TrackedHandle<VkPipelineLayout> waterGeometryPipelineLayout;

    // Descriptor set for water geometry (scene depth texture)
    TrackedHandle<VkDescriptorSetLayout> waterDepthDescriptorSetLayout;
    TrackedHandle<VkDescriptorPool> waterDepthDescriptorPool;
    // Per-frame descriptor sets for scene textures (3 frames in flight).
    // Each slot's set is allocated once and updated in place via
    // vkUpdateDescriptorSets — never freed or deferred-destroyed in the
    // render loop (zero vkFreeDescriptorSets). With VK_EXT_descriptor_buffer
    // the same update becomes a plain host memory write (vkGetDescriptorEXT)
    // into descriptor-buffer memory: no set allocation/free, no cache needed,
    // so the bindings are rewritten unconditionally on every call.
    std::array<VkDescriptorSet, FRAMES> waterDepthDescriptorSets{VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE};

    // Cubemap water pass: dedicated pipeline (swapchain color format so it can
    // render into the solid 360 cube faces; the main water geometry pipeline
    // targets R32G32B32A32_SFLOAT and is format-incompatible). The pipeline owns
    // a layout built from the same 3 set layouts as waterGeometryPipelineLayout,
    // so descriptor binding stays compatible. The set-2 descriptor set binds
    // the back-face dummy depth + real cube view (rewritten only when the
    // cube view is recreated on swapchain resize), so a single set (not
    // per-frame) is sufficient; it lives in its own pool so the
    // waterDepthDescriptorPool reset on swapchain recreate cannot invalidate it.
    TrackedHandle<VkPipeline> cubemapWaterPipeline;
    TrackedHandle<VkPipelineLayout> cubemapWaterPipelineLayout;
    TrackedHandle<VkDescriptorPool> cubemapWaterDescPool;
    VkDescriptorSet cubemapWaterDS = VK_NULL_HANDLE;
    VkFormat cubemapWaterPipelineFormat = VK_FORMAT_UNDEFINED;
    // Views currently bound into cubemapWaterDS (to detect swapchain-resize
    // staleness: the real cube view is recreated on resize, so the write-once
    // set must be rewritten when the handles change).
    VkImageView cubemapWaterBoundDepthView = VK_NULL_HANDLE;
    VkImageView cubemapWaterBoundCubeView = VK_NULL_HANDLE;

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

    // Scene sub-renderers injected via setSceneRenderers
    SolidRenderer* solidRenderer_ = nullptr;
    BrushRenderer* brushRenderer_ = nullptr;
    WaterBackFaceRenderer* backFaceRenderer_ = nullptr;
    Solid360Renderer* solid360Renderer_ = nullptr;
    WireframeRenderer* waterWireframe_ = nullptr;

    // Water render time UBO (binding 10)
    Buffer waterRenderUBO_;
};
