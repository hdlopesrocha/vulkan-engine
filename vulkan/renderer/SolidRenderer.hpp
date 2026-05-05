#pragma once

#include "../VulkanApp.hpp"
#include "IndirectRenderer.hpp"
#include "../../utils/Model3DVersion.hpp"
#include "../ShaderStage.hpp"
#include "../../math/Vertex.hpp"
#include "../../utils/Scene.hpp"
#include <unordered_map>
#include <vector>
#include <array>

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

    // Access for adding meshes
    IndirectRenderer& getIndirectRenderer() { return indirectRenderer; }


    // Offscreen solid pass outputs
    VkImageView getColorView(uint32_t frameIndex) const { return solidColorImageViews[frameIndex]; }
    VkImageView getDepthView(uint32_t frameIndex) const { return solidDepthImageViews[frameIndex]; }
    VkImage getDepthImage(uint32_t frameIndex) const { return solidDepthImages[frameIndex]; }
    VkRenderPass getRenderPass() const { return solidRenderPass; }

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
private:
    
    IndirectRenderer indirectRenderer;
    VkPipeline graphicsPipeline = VK_NULL_HANDLE;
    VkPipelineLayout graphicsPipelineLayout = VK_NULL_HANDLE;
    VkPipeline depthPrePassPipeline = VK_NULL_HANDLE;
    VkPipelineLayout depthPrePassPipelineLayout = VK_NULL_HANDLE;

    std::unordered_map<NodeID, Model3DVersion> solidChunks;

    // Offscreen framebuffer resources (2 frames in flight)
    std::array<VkImage, 2> solidColorImages = {VK_NULL_HANDLE, VK_NULL_HANDLE};
    std::array<VkDeviceMemory, 2> solidColorMemories = {VK_NULL_HANDLE, VK_NULL_HANDLE};
    std::array<VkImageView, 2> solidColorImageViews = {VK_NULL_HANDLE, VK_NULL_HANDLE};
    std::array<VkImage, 2> solidDepthImages = {VK_NULL_HANDLE, VK_NULL_HANDLE};
    std::array<VkDeviceMemory, 2> solidDepthMemories = {VK_NULL_HANDLE, VK_NULL_HANDLE};
    std::array<VkImageView, 2> solidDepthImageViews = {VK_NULL_HANDLE, VK_NULL_HANDLE};
    std::array<VkImageLayout, 2> solidDepthImageLayouts = {};
    std::array<VkFramebuffer, 2> solidFramebuffers = {VK_NULL_HANDLE, VK_NULL_HANDLE};
    VkRenderPass solidRenderPass = VK_NULL_HANDLE;
    uint32_t renderWidth = 0;
    uint32_t renderHeight = 0;
};
