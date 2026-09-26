#pragma once

#include "Renderer.hpp"
#include "../VulkanApp.hpp"
#include "../TrackedHandle.hpp"
#include "IndirectRenderer.hpp"
#include "SkyRenderer.hpp"
#include "SolidRenderer.hpp"
#include <glm/glm.hpp>
#include <array>
#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>
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
    // Body/column aux attachment selection for beginWaterGeometryPass.
    //   Auto     - follow the H4 blur gate, so the pass always matches
    //              activeGeometryPipeline().
    //   ForceOn  - always attach body+column (the 3-attachment pipelines).
    //   ForceOff - never attach them (the single-attachment no-body
    //              pipelines). Used by the wireframe overlay, which is drawn in
    //              its own single-attachment scope: its fragment stage declares
    //              one output and, without the independentBlend device feature,
    //              every attachment must share attachment 0's blend state, so
    //              the overlay cannot mask off the aux attachments it does not
    //              write.
    enum class BodyAttachments { Auto, ForceOn, ForceOff };

    // Begins the water geometry pass. Returns false when the pass could not be
    // begun (no pipeline / missing target), in which case the caller must not
    // record any draw.
    bool beginWaterGeometryPass(VkCommandBuffer cmd, uint32_t frameIndex, bool loadExisting = false,
                                BodyAttachments bodyAttachments = BodyAttachments::Auto);
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

    // Runtime body/column (final-pass blur) gate (H4, perf report 19): true
    // when at least one water layer has enableBlur && blurRadius > 0, i.e.
    // the composite's depth-guided blur can run. When false the geometry pass
    // selects the single-color-attachment WATER_NO_BODY variant and skips the
    // body/column transitions/clears, and the composite must not fetch them.
    // Tracked from the layer params (initializeWaterParamsBuffer /
    // updateGPUParamsForLayer), so no new MyApp setter is required.
    bool waterBlurNeeded() const { return waterBlurNeeded_; }

    // L15: true when the allocated body/column targets disagree with current
    // need (blur/pipeline selection flipped since creation). Consumed by the
    // per-frame recreate check to regrow/shrink them on the existing idle
    // path. Allocation is all-or-nothing per creation, so slot 0 represents
    // the set; a missing set with need (or vice versa) triggers one idle
    // recreate, after which begin/end/clear/composite NULL-guards hold.
    bool waterBodyTargetsStale() const {
        const bool need = geometryBodyAttachmentsActive();
        const bool have = waterBodyImages[0] != VK_NULL_HANDLE;
        return need != have;
    }

    // Runtime tessellation selector (Settings::tessellationEnabled). Both
    // pipeline families are built at init — the historical PATCH_LIST + TCS/TES
    // family and a TRIANGLE_LIST family with the WATER_NO_TESS vertex module
    // and no tessellation stages — and the active*() selectors swap per draw,
    // so no pipeline is ever recreated from the render loop. Default true
    // preserves the always-tessellated behavior on first frame.
    void setTessellationEnabled(bool enabled) { tessellationEnabled_ = enabled; }
    bool tessellationEnabled() const { return tessellationEnabled_; }

    // Per-op RT profiling selector (RT_PROFILE frag+TES variant, built only
    // when the device supports VK_KHR_shader_clock). Opt-in: the instrumented
    // shaders carry atomics + device-clock reads.
    void setRtProfilingEnabled(bool enabled) { rtProfilingEnabled_ = enabled; }
    bool rtProfilingEnabled() const { return rtProfilingEnabled_; }

    // Global feature gates (from Settings) forwarded to the water shader via
    // the water render UBO so they apply in BOTH fragment variants (the non-RT
    // variant cannot see the RT params block, which is why the gates are not
    // read from `rt.*`). Pure state store (M7): the flags plus a dirty bit are
    // kept here — no immediate UBO write. The offscreen path folds them into
    // renderPass()'s single per-frame UBO write; the water-in-main path (which
    // never calls renderPass) flushes them lazily from renderMainTargets().
    void setRtFeatureFlags(bool reflections, bool refractions, bool blur);

    // Declare whether the solid depth bound in set 2 is THIS frame's (true, the
    // offscreen water pass, which runs after the solid pass) or a stale one
    // (false, the water-in-main variant, which binds the previous frame's depth
    // for its in-trace lookups). Gates the shader-side solid-occlusion
    // rejection (perf report 20 C5). Bumps the UBO dirty bit on change so the
    // water-in-main flush picks it up.
    void setSolidDepthCurrentFrame(bool current);

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
    // shader statically uses every binding). Inputs are cached per set (M7):
    // the DescriptorWriter update is skipped when all eight effective views and
    // samplers are unchanged. INVARIANT: external sets passed by the caller
    // must be stable for the process lifetime (verified in MyApp: the
    // per-ring-slot async sets are allocated once and reused); renderer-owned
    // sets are invalidated on allocation/free via
    // invalidateSceneTexturesBinding(), which every other writer/freer of a
    // cached set must call first.
    void updateSceneTexturesBinding(VulkanApp* app, VkDescriptorSet ds, uint32_t frameIndex,
                                     VkImageView backFaceDepthView = VK_NULL_HANDLE,
                                     VkImageView rtReflectView = VK_NULL_HANDLE,
                                     VkImageView rtRefractView = VK_NULL_HANDLE,
                                     VkImageView skyView = VK_NULL_HANDLE,
                                     VkImageView solidColorView = VK_NULL_HANDLE,
                                     VkImageView solidDepthView = VK_NULL_HANDLE,
                                     VkImageView vegColorView = VK_NULL_HANDLE,
                                     VkImageView vegDepthView = VK_NULL_HANDLE);

    // Drop the cached binding key for `ds` (and the back-face binding-0 patch
    // cache entry, since its binding 0 is written by the same update). Call
    // whenever `ds` is freed, reallocated, or its bindings are written outside
    // updateSceneTexturesBinding(), so a recycled VkDescriptorSet handle can
    // never hit a stale cache entry.
    void invalidateSceneTexturesBinding(VkDescriptorSet ds);

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
    // non-RT variant). The tessellation family is selected first: when
    // tessellation is off the non-tess (TRIANGLE_LIST, no TCS/TES) family is
    // used, with the RT-prof variant falling back to the plain RT non-tess
    // variant (and RT to non-RT) when a variant was not built.
    //
    // H4: when no layer needs the final-pass blur, the single-attachment
    // WATER_NO_BODY variants are preferred (RT-prof falls back to the RT
    // no-body variant; a missing no-body module falls back to the matching
    // blur-capable variant). geometryBodyAttachmentsActive() mirrors this
    // selection so the pass begin/end always agree with the bound pipeline.
    VkPipeline activeGeometryPipeline() const {
        if (!waterBlurNeeded_) {
            if (!tessellationEnabled_) {
                if (rtShadingEnabled_ && waterGeometryPipelineRtNoTessNoBody != VK_NULL_HANDLE)
                    return waterGeometryPipelineRtNoTessNoBody.handle;
                if (waterGeometryPipelineNoTessNoBody != VK_NULL_HANDLE)
                    return waterGeometryPipelineNoTessNoBody.handle;
            } else {
                if (rtShadingEnabled_ && waterGeometryPipelineRtNoBody != VK_NULL_HANDLE)
                    return waterGeometryPipelineRtNoBody.handle;
                if (waterGeometryPipelineNoBody != VK_NULL_HANDLE)
                    return waterGeometryPipelineNoBody.handle;
            }
            // No-body variant unavailable: fall through to the blur-capable
            // variants (waterBlurNeeded_ false then only skips the composite
            // blur path, which has no effect since blurPx is 0 anyway).
        }
        if (!tessellationEnabled_) {
            if (rtShadingEnabled_ && rtProfilingEnabled_ && waterGeometryPipelineRtProfNoTess != VK_NULL_HANDLE)
                return waterGeometryPipelineRtProfNoTess.handle;
            if (rtShadingEnabled_ && waterGeometryPipelineRtNoTess != VK_NULL_HANDLE)
                return waterGeometryPipelineRtNoTess.handle;
            return waterGeometryPipelineNoTess.handle;
        }
        if (rtShadingEnabled_ && rtProfilingEnabled_ && waterGeometryPipelineRtProf != VK_NULL_HANDLE)
            return waterGeometryPipelineRtProf.handle;
        return (rtShadingEnabled_ && waterGeometryPipelineRt != VK_NULL_HANDLE)
            ? waterGeometryPipelineRt.handle : waterGeometryPipeline.handle;
    }

    // Whether the single-attachment (WATER_NO_BODY) geometry variants were
    // built. ForceOff requires one of them, or the pass would declare one
    // colour attachment while a three-format pipeline is bound.
    bool noBodyVariantsAvailable() const {
        if (!tessellationEnabled_) {
            return (rtShadingEnabled_ && waterGeometryPipelineRtNoTessNoBody != VK_NULL_HANDLE)
                || waterGeometryPipelineNoTessNoBody != VK_NULL_HANDLE;
        }
        return (rtShadingEnabled_ && waterGeometryPipelineRtNoBody != VK_NULL_HANDLE)
            || waterGeometryPipelineNoBody != VK_NULL_HANDLE;
    }

    // Whether the currently selectable geometry pipeline writes the body and
    // column aux attachments. Mirrors activeGeometryPipeline()'s no-body
    // selection (H4) so begin/endWaterGeometryPass and the pipeline always
    // agree, including the missing-module fallback.
    bool geometryBodyAttachmentsActive() const {
        if (waterBlurNeeded_) return true;
        if (!tessellationEnabled_) {
            if (rtShadingEnabled_ && waterGeometryPipelineRtNoTessNoBody != VK_NULL_HANDLE) return false;
            if (waterGeometryPipelineNoTessNoBody != VK_NULL_HANDLE) return false;
        } else {
            if (rtShadingEnabled_ && waterGeometryPipelineRtNoBody != VK_NULL_HANDLE) return false;
            if (waterGeometryPipelineNoBody != VK_NULL_HANDLE) return false;
        }
        return true;
    }

    // Recompute waterBlurNeeded_ from the tracked per-layer gates.
    void refreshWaterBlurNeeded();

    // Single writer of waterRenderUBO_ (M7): builds the full WaterRenderUBO
    // (timeParams.x = waterTime or the preserved current value, y/z/w from the
    // stored gates) and memcpys it in one map/unmap, then clears
    // waterRenderUboDirty_. `preserveTime` keeps the existing timeParams.x: the
    // water-in-main flush has no per-frame time source and must not zero the
    // wave clock written by an earlier renderPass().
    void flushWaterRenderUBO(float waterTime, bool preserveTime);

    VkPipeline activeMainPipeline() const {
        if (!tessellationEnabled_) {
            if (rtShadingEnabled_ && rtProfilingEnabled_ && waterMainPipelineRtProfNoTess != VK_NULL_HANDLE)
                return waterMainPipelineRtProfNoTess.handle;
            if (rtShadingEnabled_ && waterMainPipelineRtNoTess != VK_NULL_HANDLE)
                return waterMainPipelineRtNoTess.handle;
            return waterMainPipelineNoTess.handle;
        }
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

    static constexpr uint32_t FRAMES = VulkanApp::MAX_FRAMES_IN_FLIGHT;
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
    // Non-tessellation family (C1, perf report 19): TRIANGLE_LIST topology, no
    // TCS/TES and no tessellation state, using the WATER_NO_TESS vertex
    // module. Same fragment modules and pipeline layout as the tessellated
    // family, so descriptor sets are shared. Selected when
    // tessellationEnabled_ is false.
    TrackedHandle<VkPipeline> waterGeometryPipelineNoTess;
    TrackedHandle<VkPipeline> waterMainPipelineNoTess;
    TrackedHandle<VkPipeline> waterGeometryPipelineRtNoTess;
    TrackedHandle<VkPipeline> waterMainPipelineRtNoTess;
    TrackedHandle<VkPipeline> waterGeometryPipelineRtProfNoTess;
    TrackedHandle<VkPipeline> waterMainPipelineRtProfNoTess;
    // H4 (perf report 19): single-color-attachment geometry variants built
    // from the WATER_NO_BODY fragment modules (no outWaterBody/outWaterColumn).
    // Selected while waterBlurNeeded_ is false; there are no water-in-main
    // counterparts (that pipeline already targets one attachment).
    TrackedHandle<VkPipeline> waterGeometryPipelineNoBody;
    TrackedHandle<VkPipeline> waterGeometryPipelineRtNoBody;
    TrackedHandle<VkPipeline> waterGeometryPipelineNoTessNoBody;
    TrackedHandle<VkPipeline> waterGeometryPipelineRtNoTessNoBody;
    bool tessellationEnabled_ = true;
    bool rtShadingEnabled_ = true;
    bool rtProfilingEnabled_ = false;
    bool rtReflectionsEnabled_ = true;
    bool rtRefractionsEnabled_ = true;
    // Global gate for the per-material refraction/tint blur (Settings::
    // blurEnabled): delivered as WaterRenderUBO.timeParams.w.
    bool blurEnabled_ = true;
    // M7: true while the UBO does not yet carry the current feature gates.
    // Starts true so the first frame of either path writes the (initially
    // zeroed) UBO. renderPass() clears it after its unconditional write;
    // renderMainTargets() flushes + clears it when setRtFeatureFlags()
    // changed a gate.
    bool waterRenderUboDirty_ = true;

    // H4 per-layer blur gate: layerBlurNeeded_[i] = layer i's enableBlur &&
    // blurRadius > 0, kept in sync with the SSBO upload; waterBlurNeeded_ is
    // the OR over all tracked layers. The vector is sized to the larger of
    // waterParamsCount and the uploaded CPU vector so SSBO entries that are
    // never written count as no-blur.
    std::vector<bool> layerBlurNeeded_;
    bool waterBlurNeeded_ = true; // conservative default until params init

    // Body/column mode of the pass currently being recorded, cached by
    // beginWaterGeometryPass so both end functions skip the aux transitions in
    // the no-body (H4) mode. Default true = blur-capable.
    bool activePassBodyAttachments_ = true;

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
    // into descriptor-buffer memory; either way the per-set input cache below
    // skips the write when no input changed (M7).
    std::array<VkDescriptorSet, FRAMES> waterDepthDescriptorSets{VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE};

    // M7 CPU-only cache: last scene-texture bindings written per descriptor
    // set, so an unchanged set (same eight effective image views + samplers)
    // skips the DescriptorWriter/descriptor-buffer write entirely. Pure state;
    // no GPU work. Entries are dropped by invalidateSceneTexturesBinding()
    // whenever a cached set is freed, reallocated, or written elsewhere.
    struct SceneTexturesBindingKey {
        std::array<VkImageView, 8> imageViews{};
        std::array<VkSampler, 8> samplers{};
        bool operator==(const SceneTexturesBindingKey& other) const {
            return imageViews == other.imageViews && samplers == other.samplers;
        }
    };
    // Keyed by the raw VkDescriptorSet handle value (stable uint64_t key).
    // Guarded because the update can run on the async water task thread while
    // a resize/cleanup path invalidates from another thread.
    std::unordered_map<uint64_t, SceneTexturesBindingKey> sceneTexturesBindingCache_;
    std::mutex sceneTexturesCacheMutex_;

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

    // True while the solid depth bound in set 2 belongs to the current frame
    // (see setSolidDepthCurrentFrame). The offscreen path is the default.
    bool solidDepthCurrentFrame_ = true;
};
