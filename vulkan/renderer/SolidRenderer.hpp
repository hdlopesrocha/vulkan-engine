#pragma once

#include "Renderer.hpp"
#include "../VulkanApp.hpp"
#include "../TrackedHandle.hpp"
#include "IndirectRenderer.hpp"
#include "WireframeRenderer.hpp"
#include "../../space/Model3DVersion.hpp"
#include "../ShaderStage.hpp"
#include "../../math/Vertex.hpp"
#include "../../utils/Scene.hpp"
#include <unordered_map>
#include <vector>
#include <array>
#include "CommandBufferState.hpp"

class SolidRenderer : public Renderer {
public:
    explicit SolidRenderer();
    ~SolidRenderer();

    void init();
    void createPipelines(VulkanApp* app);
    void createWireframe(VulkanApp* app);
    void createRenderTargets(VulkanApp* app, uint32_t width, uint32_t height);
    void destroyRenderTargets(VulkanApp* app);
    void beginPass(VkCommandBuffer cmd, uint32_t frameIndex, VkClearValue colorClear, VkClearValue depthClear, VulkanApp* app);
    void endPass(VkCommandBuffer cmd, uint32_t frameIndex, VulkanApp* app);
    void cleanup(VulkanApp* app) override;

    // Draw the solid wireframe overlay on top of the existing solid render.
    // Must be called inside a compatible render pass.
    void drawWireframeOverlay(VkCommandBuffer& commandBuffer, VulkanApp* app, VkDescriptorSet perTextureDescriptorSet);

    // Draw main solid geometry: bind pipeline and draw
    void render(VkCommandBuffer &commandBuffer, VulkanApp* app, VkDescriptorSet perTextureDescriptorSet, VkDescriptorSet brushDepthSet = VK_NULL_HANDLE);
    // Draw depth-only pre-pass to populate depth buffer without writing color
    void renderDepthPrepass(VkCommandBuffer &commandBuffer, VulkanApp* app, VkDescriptorSet perTextureDescriptorSet, VkDescriptorSet brushDepthSet = VK_NULL_HANDLE);

    // Access for adding meshes
    IndirectRenderer& getIndirectRenderer() { return indirectRenderer; }


    // Offscreen solid pass outputs
    VkImageView getColorView(uint32_t frameIndex) const { return solidColorImageViews[frameIndex % SOLID_FRAMES]; }
    VkImage getColorImage(uint32_t frameIndex) const { return solidColorImages[frameIndex % SOLID_FRAMES]; }
    VkImageView getDepthView(uint32_t frameIndex) const { return solidDepthImageViews[frameIndex % SOLID_FRAMES]; }
    VkImage getDepthImage(uint32_t frameIndex) const { return solidDepthImages[frameIndex % SOLID_FRAMES]; }
    uint32_t getRenderWidth() const { return renderWidth; }
    uint32_t getRenderHeight() const { return renderHeight; }

public:

    // Per-frame depth image layout accessors (for external widgets to record barriers)
    VkImageLayout getDepthLayout(uint32_t frameIndex) const {
        if (frameIndex < solidDepthImageLayouts.size()) return solidDepthImageLayouts[frameIndex];
        return VK_IMAGE_LAYOUT_UNDEFINED;
    }
    void setDepthLayout(uint32_t frameIndex, VkImageLayout layout) {
        if (frameIndex < solidDepthImageLayouts.size()) solidDepthImageLayouts[frameIndex] = layout;
    }
    VkPipeline getGraphicsPipeline() const { return activeGraphicsPipeline(); }
    VkPipelineLayout getGraphicsPipelineLayout() const { return graphicsPipelineLayout; }

    // Runtime RT-shading selector (solid reflections / local shadows). With
    // both off, the cheap non-RT fragment variant is bound instead of the
    // ray-query shader, so raster-only configurations never pay its register
    // pressure/occupancy cost. Both variants are built at init.
    void setRtShadingEnabled(bool enabled) { rtShadingEnabled_ = enabled; }
    bool rtShadingEnabled() const { return rtShadingEnabled_; }

    // Runtime tessellation selector (Settings::tessellationEnabled, perf
    // report 21 C1). Only the deferred-depth pass has a no-tess twin (its
    // fragment shader samples no materials, so the twin is exact); the color
    // passes keep the tessellated family to preserve the TCS material blend.
    // RT/profiling/brush/wireframe paths never had a twin. All fallbacks bind
    // the tessellated pipelines, which is always correct.
    void setTessellationEnabled(bool enabled) { tessellationEnabled_ = enabled; }
    bool tessellationEnabled() const { return tessellationEnabled_; }

    // Per-op RT profiling selector (RT_PROFILE variant, built only when the
    // device supports VK_KHR_shader_clock). Opt-in: instrumented shaders carry
    // atomics + device-clock reads, so they are only bound while the user has
    // RT profiling enabled in the overlay.
    void setRtProfilingEnabled(bool enabled) { rtProfilingEnabled_ = enabled; }
    bool rtProfilingEnabled() const { return rtProfilingEnabled_; }

    // Deferred depth test: draw only depth (no color)
    void drawDepth(VkCommandBuffer &commandBuffer, VulkanApp* app, VkDescriptorSet descSet);
    // Deferred depth test: draw only color with LESS_OR_EQUAL compare, no depth write
    void drawColor(VkCommandBuffer &commandBuffer, VulkanApp* app, VkDescriptorSet descSet, VkDescriptorSet brushDepthSet = VK_NULL_HANDLE);
    // Draw depth using an external IndirectRenderer (e.g. separate brush mesh buffer)
    void drawDepthExternal(VkCommandBuffer &cmd, VkDescriptorSet descSet, IndirectRenderer& indirect);
    // Draw color using an external IndirectRenderer
    void drawColorExternal(VkCommandBuffer &cmd, VkDescriptorSet descSet, IndirectRenderer& indirect, VkDescriptorSet brushDepthSet = VK_NULL_HANDLE);
    // Draw brush color (main_brush.frag, no shadows) using an external IndirectRenderer
    void drawBrushColorExternal(VkCommandBuffer &cmd, VkDescriptorSet descSet, IndirectRenderer& indirect);
    // Draw brush color with alpha blending at the given opacity
    void drawBrushColor(VkCommandBuffer &cmd, VkDescriptorSet descSet, IndirectRenderer& indirect, float opacity);
    // Draw brush overlay (opaque, no blending) into scene_color
    void drawBrushOverlay(VkCommandBuffer &cmd, VkDescriptorSet descSet, IndirectRenderer& indirect);

private:
    
    // Active solid color pipeline variant (profiling RT fragment while RT
    // profiling is on; otherwise the RT variant while RT shading is on and it
    // exists; otherwise the non-RT variant).
    VkPipeline activeGraphicsPipeline() const {
        if (rtShadingEnabled_ && rtProfilingEnabled_ && graphicsPipelineRtProf != VK_NULL_HANDLE)
            return graphicsPipelineRtProf.handle;
        return (rtShadingEnabled_ && graphicsPipelineRt != VK_NULL_HANDLE)
            ? graphicsPipelineRt.handle : graphicsPipeline.handle;
    }
    VkPipeline activeDeferredColorPipeline() const {
        if (rtShadingEnabled_ && rtProfilingEnabled_ && deferredColorPipelineRtProf != VK_NULL_HANDLE)
            return deferredColorPipelineRtProf.handle;
        return (rtShadingEnabled_ && deferredColorPipelineRt != VK_NULL_HANDLE)
            ? deferredColorPipelineRt.handle : deferredColorPipeline.handle;
    }
    // Active deferred-depth pipeline: the no-tess twin while tessellation is
    // off (same depth_only.frag, no TCS/TES/displacement sampling).
    VkPipeline activeDeferredDepthPipeline() const {
        if (!tessellationEnabled_ && deferredDepthPipelineNoTess != VK_NULL_HANDLE)
            return deferredDepthPipelineNoTess.handle;
        return deferredDepthPipeline.handle;
    }

    IndirectRenderer indirectRenderer;
    TrackedHandle<VkPipeline> graphicsPipeline;
    TrackedHandle<VkPipelineLayout> graphicsPipelineLayout;
    TrackedHandle<VkPipeline> depthPrePassPipeline;
    TrackedHandle<VkPipelineLayout> depthPrePassPipelineLayout;

    // Deferred depth test pipelines
    TrackedHandle<VkPipeline> deferredDepthPipeline;
    TrackedHandle<VkPipelineLayout> deferredDepthPipelineLayout;
    TrackedHandle<VkPipeline> deferredColorPipeline;
    TrackedHandle<VkPipelineLayout> deferredColorPipelineLayout;
    // RT fragment-shader variants (same configs/layouts; bound only while
    // solid RT paths are enabled).
    TrackedHandle<VkPipeline> graphicsPipelineRt;
    TrackedHandle<VkPipeline> deferredColorPipelineRt;
    // RT_PROFILE variants (counters + device clock; only created when
    // VK_KHR_shader_clock is supported).
    TrackedHandle<VkPipeline> graphicsPipelineRtProf;
    TrackedHandle<VkPipeline> deferredColorPipelineRtProf;
    // No-tessellation twin (perf report 21 C1): TRIANGLE_LIST, SOLID_NO_TESS
    // vertex shader, no TCS/TES. Only the deferred-depth pipeline has one:
    // depth_only.frag samples no materials, so the twin is exact there. The
    // color pipelines keep the tessellated family because the TCS 3-corner
    // material compression cannot be reproduced in a VS and flat shading it
    // visibly breaks slope/height-band blending. Layout is a matching
    // duplicate of the tessellated pipeline's layout (same setLayouts), so
    // the bind sites keep using deferredDepthPipelineLayout.
    TrackedHandle<VkPipeline> deferredDepthPipelineNoTess;
    bool rtShadingEnabled_ = true;
    bool rtProfilingEnabled_ = false;
    // Tessellation family selector (Settings::tessellationEnabled, perf
    // report 21 C1). True = bind the PATCH_LIST TCS/TES pipelines; false =
    // bind the TRIANGLE_LIST no-tess twin where one exists (deferred depth).
    bool tessellationEnabled_ = true;
    // Brush color pipeline (alpha blending enabled)
    TrackedHandle<VkPipeline> brushDeferredColorPipeline;
    TrackedHandle<VkPipelineLayout> brushDeferredColorPipelineLayout;
    // Brush overlay pipeline (opaque, no blending) for scene_color rendering
    TrackedHandle<VkPipeline> brushOverlayPipeline;
    TrackedHandle<VkPipelineLayout> brushOverlayPipelineLayout;
    bool deferredPipelinesCreated = false;

    // Offscreen framebuffer resources matching MAX_FRAMES_IN_FLIGHT
    static constexpr uint32_t SOLID_FRAMES = VulkanApp::MAX_FRAMES_IN_FLIGHT;
    std::array<VkImage, SOLID_FRAMES> solidColorImages = {};
    std::array<VmaAllocation, SOLID_FRAMES> solidColorAllocations = {};
    std::array<VkDeviceMemory, SOLID_FRAMES> solidColorMemories = {};
    std::array<VkImageView, SOLID_FRAMES> solidColorImageViews = {};
    std::array<VkImage, SOLID_FRAMES> solidDepthImages = {};
    std::array<VmaAllocation, SOLID_FRAMES> solidDepthAllocations = {};
    std::array<VkDeviceMemory, SOLID_FRAMES> solidDepthMemories = {};
    std::array<VkImageView, SOLID_FRAMES> solidDepthImageViews = {};
    std::array<VkImageLayout, SOLID_FRAMES> solidDepthImageLayouts = {};
    uint32_t renderWidth = 0;
    uint32_t renderHeight = 0;

    // Solid wireframe overlay (owned by this renderer so wireframe mode works
    // uniformly for the solid pass)
    WireframeRenderer wireframe;
public:
    void setCmdState(CommandBufferState* state) override {
        Renderer::setCmdState(state);
        wireframe.setCmdState(state);
        // The solid IndirectRenderer is main-thread-only (main/shadow passes),
        // so it is safe to wire into the shared per-frame tracker.
        indirectRenderer.setCmdState(state);
    }
};
