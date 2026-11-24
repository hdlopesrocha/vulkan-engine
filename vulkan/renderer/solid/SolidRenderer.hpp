#pragma once

#include "../Renderer.hpp"
#include "../../core/VulkanApp.hpp"
#include "../indirect/IndirectRenderer.hpp"
#include <array>

// Offscreen solid color/depth targets plus the cull/debug command streams.
//
// Terrain rasterization is gone: the octree SDF raymarch writes these targets,
// so this class no longer owns vertex/index pipelines, chunk draws or a
// wireframe overlay. The embedded IndirectRenderer survives only because the
// vegetation and debug-overlay culls are folded into its command stream and it
// supplies the GPU culling resources those passes use.
class SolidRenderer : public Renderer {
public:
    SolidRenderer();
    ~SolidRenderer();

    void init();
    void createRenderTargets(VulkanApp* app, uint32_t width, uint32_t height);
    void destroyRenderTargets(VulkanApp* app);
    void cleanup(VulkanApp* app) override;

    IndirectRenderer& getIndirectRenderer() { return indirectRenderer; }

    // Offscreen terrain targets (written by OctreeRaymarchRenderer).
    VkImageView getColorView(uint32_t frameIndex) const { return solidColorImageViews[frameIndex % SOLID_FRAMES]; }
    VkImage getColorImage(uint32_t frameIndex) const { return solidColorImages[frameIndex % SOLID_FRAMES]; }
    VkImageView getDepthView(uint32_t frameIndex) const { return solidDepthImageViews[frameIndex % SOLID_FRAMES]; }
    VkImage getDepthImage(uint32_t frameIndex) const { return solidDepthImages[frameIndex % SOLID_FRAMES]; }
    uint32_t getRenderWidth() const { return renderWidth; }
    uint32_t getRenderHeight() const { return renderHeight; }

    VkImageLayout getDepthLayout(uint32_t frameIndex) const {
        if (frameIndex < solidDepthImageLayouts.size()) return solidDepthImageLayouts[frameIndex];
        return VK_IMAGE_LAYOUT_UNDEFINED;
    }
    void setDepthLayout(uint32_t frameIndex, VkImageLayout layout) {
        if (frameIndex < solidDepthImageLayouts.size()) solidDepthImageLayouts[frameIndex] = layout;
    }

    void setCmdState(CommandBufferState* state) override {
        Renderer::setCmdState(state);
        indirectRenderer.setCmdState(state);
    }

private:
    IndirectRenderer indirectRenderer;

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
};
