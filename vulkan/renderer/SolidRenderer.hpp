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
    void drawColor(VkCommandBuffer &commandBuffer, VulkanApp* app, VkDescriptorSet descSet, VkDescriptorSet brushDepthSet = VK_NULL_HANDLE);
    // Draw depth using an external IndirectRenderer (e.g. separate brush mesh buffer)
    void drawDepthExternal(VkCommandBuffer &cmd, VkDescriptorSet descSet, IndirectRenderer& indirect);
    // Draw color using an external IndirectRenderer
    void drawColorExternal(VkCommandBuffer &cmd, VkDescriptorSet descSet, IndirectRenderer& indirect, VkDescriptorSet brushDepthSet = VK_NULL_HANDLE);
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
    CommandBufferState* cmdState = nullptr;
public:
    void setCmdState(CommandBufferState* state) { cmdState = state; }
};
