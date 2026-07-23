#pragma once

#include "../VulkanApp.hpp"
#include "../TrackedHandle.hpp"
#include "IndirectRenderer.hpp"
#include "../../utils/Model3DVersion.hpp"
#include "../ShaderStage.hpp"
#include "../../math/Vertex.hpp"
#include "../../utils/Scene.hpp"
#include <unordered_map>
#include <vector>
#include <array>
#include "CommandBufferState.hpp"

class SolidRenderer {
public:
    explicit SolidRenderer();
    ~SolidRenderer();

    void init();
    void createPipelines(VulkanApp* app);
    void createRenderTargets(VulkanApp* app, uint32_t width, uint32_t height);
    void destroyRenderTargets(VulkanApp* app);
    void beginPass(VkCommandBuffer cmd, uint32_t frameIndex, VkClearValue colorClear, VkClearValue depthClear, VulkanApp* app);
    void endPass(VkCommandBuffer cmd, uint32_t frameIndex, VulkanApp* app);
    void cleanup(VulkanApp* app);

    // Draw main solid geometry: bind pipeline and draw
    void render(VkCommandBuffer &commandBuffer, VulkanApp* app, VkDescriptorSet perTextureDescriptorSet);
    // Draw depth-only pre-pass to populate depth buffer without writing color
    void renderDepthPrepass(VkCommandBuffer &commandBuffer, VulkanApp* app, VkDescriptorSet perTextureDescriptorSet);

    // Access for adding meshes
    IndirectRenderer& getIndirectRenderer() { return indirectRenderer; }


    // Offscreen solid pass outputs
    VkImageView getColorView(uint32_t frameIndex) const { return (frameIndex < solidColorImageViews.size()) ? solidColorImageViews[frameIndex] : (solidColorImageViews.empty() ? VK_NULL_HANDLE : solidColorImageViews.back()); }
    VkImage getColorImage(uint32_t frameIndex) const { return (frameIndex < solidColorImages.size()) ? solidColorImages[frameIndex] : (solidColorImages.empty() ? VK_NULL_HANDLE : solidColorImages.back()); }
    VkImageView getDepthView(uint32_t frameIndex) const { return (frameIndex < solidDepthImageViews.size()) ? solidDepthImageViews[frameIndex] : (solidDepthImageViews.empty() ? VK_NULL_HANDLE : solidDepthImageViews.back()); }
    VkImage getDepthImage(uint32_t frameIndex) const { return (frameIndex < solidDepthImages.size()) ? solidDepthImages[frameIndex] : (solidDepthImages.empty() ? VK_NULL_HANDLE : solidDepthImages.back()); }

public:
    // Public accessor for nodeModelVersions (read-only)
    const std::unordered_map<NodeID, Model3DVersion>& getNodeModelVersions() const { return solidChunks; }
public:

    // Per-frame depth image layout accessors (for external widgets to record barriers)
    VkImageLayout getDepthLayout(uint32_t frameIndex) const {
        if (frameIndex < solidDepthImageLayouts.size()) return solidDepthImageLayouts[frameIndex];
        return VK_IMAGE_LAYOUT_UNDEFINED;
    }
    void setDepthLayout(uint32_t frameIndex, VkImageLayout layout) {
        if (frameIndex < solidDepthImageLayouts.size()) solidDepthImageLayouts[frameIndex] = layout;
    }
    VkPipeline getGraphicsPipeline() const { return graphicsPipeline; }
    VkPipelineLayout getGraphicsPipelineLayout() const { return graphicsPipelineLayout; }

    // Deferred depth test: draw only depth (no color)
    void drawDepth(VkCommandBuffer &commandBuffer, VulkanApp* app, VkDescriptorSet descSet);
    // Deferred depth test: draw only color with LESS_OR_EQUAL compare, no depth write
    void drawColor(VkCommandBuffer &commandBuffer, VulkanApp* app, VkDescriptorSet descSet);
    // Draw depth using an external IndirectRenderer (e.g. separate brush mesh buffer)
    void drawDepthExternal(VkCommandBuffer &cmd, VkDescriptorSet descSet, IndirectRenderer& indirect);
    // Draw color using an external IndirectRenderer
    void drawColorExternal(VkCommandBuffer &cmd, VkDescriptorSet descSet, IndirectRenderer& indirect);
    // Draw brush color (brush.frag, no shadows) using an external IndirectRenderer
    void drawBrushColorExternal(VkCommandBuffer &cmd, VkDescriptorSet descSet, IndirectRenderer& indirect);
    // Draw brush color with alpha blending at the given opacity
    void drawBrushColor(VkCommandBuffer &cmd, VkDescriptorSet descSet, IndirectRenderer& indirect, float opacity);
    // Draw brush overlay (opaque, no blending) into scene_color
    void drawBrushOverlay(VkCommandBuffer &cmd, VkDescriptorSet descSet, IndirectRenderer& indirect);

private:
    
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
    // Brush color pipeline (alpha blending enabled)
    TrackedHandle<VkPipeline> brushDeferredColorPipeline;
    TrackedHandle<VkPipelineLayout> brushDeferredColorPipelineLayout;
    // Brush overlay pipeline (opaque, no blending) for scene_color rendering
    TrackedHandle<VkPipeline> brushOverlayPipeline;
    TrackedHandle<VkPipelineLayout> brushOverlayPipelineLayout;
    bool deferredPipelinesCreated = false;

    std::unordered_map<NodeID, Model3DVersion> solidChunks;

    // Offscreen framebuffer resources (2 frames in flight, clamp index for 3-frame safety)
    std::array<VkImage, 2> solidColorImages = {VK_NULL_HANDLE, VK_NULL_HANDLE};
    std::array<VmaAllocation, 2> solidColorAllocations = {VK_NULL_HANDLE, VK_NULL_HANDLE};
    std::array<VkDeviceMemory, 2> solidColorMemories = {VK_NULL_HANDLE, VK_NULL_HANDLE};
    std::array<VkImageView, 2> solidColorImageViews = {VK_NULL_HANDLE, VK_NULL_HANDLE};
    std::array<VkImage, 2> solidDepthImages = {VK_NULL_HANDLE, VK_NULL_HANDLE};
    std::array<VmaAllocation, 2> solidDepthAllocations = {VK_NULL_HANDLE, VK_NULL_HANDLE};
    std::array<VkDeviceMemory, 2> solidDepthMemories = {VK_NULL_HANDLE, VK_NULL_HANDLE};
    std::array<VkImageView, 2> solidDepthImageViews = {VK_NULL_HANDLE, VK_NULL_HANDLE};
    std::array<VkImageLayout, 2> solidDepthImageLayouts = {};
    uint32_t renderWidth = 0;
    uint32_t renderHeight = 0;
    CommandBufferState* cmdState = nullptr;
public:
    void setCmdState(CommandBufferState* state) { cmdState = state; }
};
