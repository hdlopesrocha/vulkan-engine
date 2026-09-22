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
class WireframeRenderer;

class WaterRenderer : public Renderer {
public:
    WaterRenderer();
    ~WaterRenderer();

    void init(VulkanApp* app, Buffer& waterParamsBuffer_, const std::vector<WaterParams>& waterParams, uint32_t layerCount);
    void cleanup(VulkanApp* app) override;

    // Inject the scene sub-renderers the water pass samples from or draws
    // alongside (solid offscreen targets, brush liquid geometry, back-face
    // depth, wireframe overlay). Called once by SceneRenderer after all
    // sub-renderers are created. (The legacy 360° cubemap injector was removed
    // with Solid360Renderer — water reflections are hardware ray tracing.)
    void setSceneRenderers(SolidRenderer* solid, BrushRenderer* brush,
                           WaterBackFaceRenderer* backFace,
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
    // (binding 0 = real back-face depth, bindings 1-2 = RT outputs, binding 3 =
    // sky); when null the per-frame set from prepareSceneTexturesForFrame() is
    // bound instead.
    void render(VulkanApp* app, VkCommandBuffer cmd, uint32_t frameIndex,
                VkImageView sceneColorView,
                VkImageView skyView = VK_NULL_HANDLE,
                IndirectRenderer* secondaryIR = nullptr,
                VkDescriptorSet overrideWaterDs = VK_NULL_HANDLE);

    // Get water color/depth image view for post-process sampling
    VkImageView getWaterDepthView(uint32_t frameIndex) const { return waterDepthImageViews[frameIndex]; }
    // Depth image view used as the depth/stencil attachment for the water geometry pass
    VkImageView getWaterGeomDepthView(uint32_t frameIndex) const { return waterGeomDepthImageViews[frameIndex]; }
    // Water body attachment (color attachment 1 of the water geometry pass):
    // RGBA16F, RGB = refraction + tint body (pre-reflection, sharp), A = body
    // weight = composite coverage * (1 - reflection mix). The final composite
    // blurs the body with a depth-scaled kernel and re-inserts it with this
    // weight, so the reflection lobe and surface effects stay sharp.
    VkImageView getWaterBodyView(uint32_t frameIndex) const {
        return (frameIndex < FRAMES) ? waterBodyImageViews[frameIndex] : VK_NULL_HANDLE;
    }
    VkImage getWaterBodyImage(uint32_t frameIndex) const {
        return (frameIndex < FRAMES) ? waterBodyImages[frameIndex] : VK_NULL_HANDLE;
    }
    // Water column attachment (color attachment 2 of the water geometry pass):
    // RG16F, R = measured water depth (m), G = per-material blur radius in
    // pixels (0 = crisp, computed from the layer's blur params). Drives the
    // depth-guided water blur in the final composite.
    VkImageView getWaterColumnView(uint32_t frameIndex) const {
        return (frameIndex < FRAMES) ? waterColumnImageViews[frameIndex] : VK_NULL_HANDLE;
    }
    VkImage getWaterColumnImage(uint32_t frameIndex) const {
        return (frameIndex < FRAMES) ? waterColumnImages[frameIndex] : VK_NULL_HANDLE;
    }
    VkImageLayout getWaterBodyLayout(uint32_t frameIndex) const;
    void setWaterBodyLayout(uint32_t frameIndex, VkImageLayout layout);
    VkImageLayout getWaterColumnLayout(uint32_t frameIndex) const;
    void setWaterColumnLayout(uint32_t frameIndex, VkImageLayout layout);
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
    
    // Get the water geometry pipeline (for rendering water to G-buffer).
    // Returns the RT fragment variant only while RT shading is enabled; with
    // every ray path off, the cheaper non-RT variant is bound instead (same
    // layout/descriptors, sky fallbacks), so a disabled configuration does
    // not pay the register/occupancy cost of the ray-query shader.
    VkPipeline getWaterGeometryPipeline() const { return activeGeometryPipeline(); }
    
    // Get the water geometry pipeline layout
    VkPipelineLayout getWaterGeometryPipelineLayout() const { return waterGeometryPipelineLayout; }

    // Water-in-main blend pipeline (Phase-1 migration): identical stages /
    // layout to the geometry pipeline, but alpha-blended, depth-write off and
    // targeting the main solid color format. Used by renderMainTargets().
    VkPipeline getWaterMainPipeline() const { return activeMainPipeline(); }

    // Runtime RT-shading selector (from the settings ray-path toggles + the
    // per-material water layer flags). Recreating pipelines is avoided: both
    // variants are built at init and swapped per draw.
    void setRtShadingEnabled(bool enabled) { rtShadingEnabled_ = enabled; }
    bool rtShadingEnabled() const { return rtShadingEnabled_; }

    // Per-op RT profiling selector (RT_PROFILE frag+TES variant, built only
    // when the device supports VK_KHR_shader_clock). Opt-in: the instrumented
    // shaders carry atomics + device-clock reads.
    void setRtProfilingEnabled(bool enabled) { rtProfilingEnabled_ = enabled; }
    bool rtProfilingEnabled() const { return rtProfilingEnabled_; }

    // Global ray-path gates (from Settings) forwarded to the water shader via
    // the water render UBO so they apply in BOTH fragment variants (the non-RT
    // variant cannot see the RT params block, which is why the gates are not
    // read from `rt.*`). Writers go through immediately so the water-in-main
    // path (which does not rewrite the UBO) sees them too.
    void setRtFeatureFlags(bool reflections, bool refractions);

    // Get the descriptor set layout for scene textures (set 2)
    VkDescriptorSetLayout getWaterDepthDescriptorSetLayout() const { return waterDepthDescriptorSetLayout; }

    // Prepare render state (UBO upload, descriptor update, pre-barrier).
    // Call this before beginWaterGeometryPass when manually recording commands.
    void prepareRender(VulkanApp* app, VkCommandBuffer cmd, uint32_t frameIndex,
                       VkImageView sceneColorView,
                       VkImageView skyView = VK_NULL_HANDLE);

    // Get water depth descriptor set (for binding scene depth texture)
    VkDescriptorSet getWaterDepthDescriptorSet(uint32_t frameIndex) const { return (frameIndex < FRAMES) ? waterDepthDescriptorSets[frameIndex] : VK_NULL_HANDLE; }

    // Get sampler for ImGui texture display
    VkSampler getLinearSampler() const { return linearSampler; }
    
    // Update the scene textures binding (back-face depth + RT outputs + sky).
    // Writes into `ds` (the caller chooses the per-command-buffer set so the set is
    // never shared between the async back-face task and the main command buffer).
    // Missing RT views fall back to internal 1x1 dummies (never NULL — the
    // shader statically uses every binding).
    void updateSceneTexturesBinding(VulkanApp* app, VkDescriptorSet ds, uint32_t frameIndex,
                                     VkImageView backFaceDepthView = VK_NULL_HANDLE,
                                     VkImageView rtReflectView = VK_NULL_HANDLE,
                                     VkImageView rtRefractView = VK_NULL_HANDLE,
                                     VkImageView skyView = VK_NULL_HANDLE,
                                     VkImageView solidColorView = VK_NULL_HANDLE,
                                     VkImageView solidDepthView = VK_NULL_HANDLE,
                                     VkImageView vegColorView = VK_NULL_HANDLE,
                                     VkImageView vegDepthView = VK_NULL_HANDLE);

    // Allocate a fresh per-frame scene-texture descriptor set, free the previous
    // one, and update it with the given views. Returns the new set (or
    // VK_NULL_HANDLE on failure). The previous set is freed only after its command
    // buffer has completed (the caller must invoke this from preRenderPass, which
    // runs after the per-slot in-flight fence wait), so the set is never reused
    // while pending and never needs UPDATE_AFTER_BIND.
    VkDescriptorSet prepareSceneTexturesForFrame(VulkanApp* app, uint32_t frameIndex,
                                                 VkImageView backFaceDepthView = VK_NULL_HANDLE,
                                                 VkImageView rtReflectView = VK_NULL_HANDLE,
                                                 VkImageView rtRefractView = VK_NULL_HANDLE,
                                                 VkImageView skyView = VK_NULL_HANDLE,
                                                 VkImageView solidColorView = VK_NULL_HANDLE,
                                                 VkImageView solidDepthView = VK_NULL_HANDLE,
                                                 VkImageView vegColorView = VK_NULL_HANDLE,
                                                 VkImageView vegDepthView = VK_NULL_HANDLE);

    // Clear per-frame render targets (color/depth) into default values.
    // Call this each frame when water rendering is disabled to avoid sampling
    // stale content from previous frames.
    void clearRenderTargets(VulkanApp* app, VkCommandBuffer cmd, uint32_t frameIndex);

    // Phase-1 water-in-main: draw the water geometry directly into the given
    // color/depth targets (the main solid pass outputs) with alpha blending,
    // instead of the separate water color/geometry-depth pair. LOAD ops
    // preserve the opaque scene; depth test on, depth write off. The caller
    // must have bound per-frame scene textures (back-face depth, sky, and the
    // PREVIOUS frame's solid color/depth + vegetation, since the current
    // frames are the attachments here). No-op when the blend pipeline is
    // unavailable. Does not touch the water color/geom-depth targets, so the
    // composite must skip the water branch while this path is active.
    void renderMainTargets(VulkanApp* app, VkCommandBuffer cmd, uint32_t frameIndex,
                           VkImage colorImage, VkImageView colorView,
                           VkImage depthImage, VkImageView depthView,
                           VkImageView skyView, VkDescriptorSet overrideWaterDs);

    // 1x1 zeroed color view (alpha 0) bound to the composite's water input
    // while water is drawn directly into the main color target, so the
    // composite's water blend collapses to the base color.
    VkImageView getDummyWaterColorView() const { return dummyWaterView_; }

    // Hybrid RT resources (set after SceneRenderer creates them; may be null
    // when RT is unsupported — the non-async prepare path then binds dummies
    // for the RT outputs and the sky view passed by the caller).
    void setRTResources(class RayTracingResources* rt) { rtResources_ = rt; }

private:

    // Active pipeline variant (profiling RT while RT profiling is on, else RT
    // while RT shading is on and the RT variant exists, else the cheaper
    // non-RT variant).
    VkPipeline activeGeometryPipeline() const {
        if (rtShadingEnabled_ && rtProfilingEnabled_ && waterGeometryPipelineRtProf != VK_NULL_HANDLE)
            return waterGeometryPipelineRtProf.handle;
        return (rtShadingEnabled_ && waterGeometryPipelineRt != VK_NULL_HANDLE)
            ? waterGeometryPipelineRt.handle : waterGeometryPipeline.handle;
    }
    VkPipeline activeMainPipeline() const {
        if (rtShadingEnabled_ && rtProfilingEnabled_ && waterMainPipelineRtProf != VK_NULL_HANDLE)
            return waterMainPipelineRtProf.handle;
        return (rtShadingEnabled_ && waterMainPipelineRt != VK_NULL_HANDLE)
            ? waterMainPipelineRt.handle : waterMainPipeline.handle;
    }

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
    // Water body attachment (color attachment 1): refraction + tint body (RGB)
    // and body weight (A, see getWaterBodyView).
    std::array<VkImage, FRAMES> waterBodyImages = {};
    std::array<VmaAllocation, FRAMES> waterBodyAllocations = {};
    std::array<VkDeviceMemory, FRAMES> waterBodyMemories = {};
    std::array<VkImageView, FRAMES> waterBodyImageViews = {};
    // Water column attachment (color attachment 2): measured water depth (m)
    // in R and the per-material blur radius (px) in G.
    std::array<VkImage, FRAMES> waterColumnImages = {};
    std::array<VmaAllocation, FRAMES> waterColumnAllocations = {};
    std::array<VkDeviceMemory, FRAMES> waterColumnMemories = {};
    std::array<VkImageView, FRAMES> waterColumnImageViews = {};

    // Pipelines
    TrackedHandle<VkPipeline> waterGeometryPipeline;
    TrackedHandle<VkPipeline> waterMainPipeline; // alpha-blended, main-pass targets
    // RT fragment-shader variants (same layout/descriptors; bound instead of
    // the non-RT pair while any ray path is enabled).
    TrackedHandle<VkPipeline> waterGeometryPipelineRt;
    TrackedHandle<VkPipeline> waterMainPipelineRt;
    // RT_PROFILE variants (frag + TES with counters + device clock; only
    // created when VK_KHR_shader_clock is supported).
    TrackedHandle<VkPipeline> waterGeometryPipelineRtProf;
    TrackedHandle<VkPipeline> waterMainPipelineRtProf;
    bool rtShadingEnabled_ = true;
    bool rtProfilingEnabled_ = false;
    bool rtReflectionsEnabled_ = true;
    bool rtRefractionsEnabled_ = true;

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

    // NOTE (hybrid RT §12): the cubemap water pass (dedicated pipeline +
    // set-2 dummy set for solid-360 faces) was deleted with Solid360Renderer.

    // Storage buffer (SSBO) for per-layer WaterParamsGPU entries
    Buffer waterParamsBuffer;
    // Number of entries allocated in `waterParamsBuffer`
    uint32_t waterParamsCount = 0;
    // Back-pointer to app for mapping/unmapping buffer when updating GPU data
    VulkanApp* appPtr = nullptr;

    // Samplers
    TrackedHandle<VkSampler> linearSampler;
    TrackedHandle<VkSampler> nearestSampler;

    uint32_t renderWidth = 0;
    uint32_t renderHeight = 0;

    // Cached frame index set by beginWaterGeometryPass, used by endWaterGeometryPass
    uint32_t activeWaterFrameIndex = 0;

    // Scene sub-renderers injected via setSceneRenderers
    SolidRenderer* solidRenderer_ = nullptr;
    BrushRenderer* brushRenderer_ = nullptr;
    WaterBackFaceRenderer* backFaceRenderer_ = nullptr;
    WireframeRenderer* waterWireframe_ = nullptr;
    class RayTracingResources* rtResources_ = nullptr;

    // 1x1 dummy views bound when RT outputs/sky are unavailable (never NULL).
    VkImage dummyRTImage_ = VK_NULL_HANDLE;
    VmaAllocation dummyRTAlloc_ = VK_NULL_HANDLE;
    VkDeviceMemory dummyRTMem_ = VK_NULL_HANDLE;
    VkImageView dummyRTView_ = VK_NULL_HANDLE; // GENERAL layout (RT outputs)
    VkImage dummySkyImage_ = VK_NULL_HANDLE;
    VmaAllocation dummySkyAlloc_ = VK_NULL_HANDLE;
    VkDeviceMemory dummySkyMem_ = VK_NULL_HANDLE;
    VkImageView dummySkyView_ = VK_NULL_HANDLE; // SHADER_READ layout (sky)
    VkImage dummyWaterImage_ = VK_NULL_HANDLE;
    VmaAllocation dummyWaterAlloc_ = VK_NULL_HANDLE;
    VkDeviceMemory dummyWaterMem_ = VK_NULL_HANDLE;
    VkImageView dummyWaterView_ = VK_NULL_HANDLE; // zeroed RGBA, SHADER_READ
    void ensureDummyViews(VulkanApp* app);

    // Water render time UBO (binding 10)
    Buffer waterRenderUBO_;
};
