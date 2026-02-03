#pragma once

#include "VulkanApp.hpp"
#include "IndirectRenderer.hpp"
#include "Model3DVersion.hpp"
#include "ShaderStage.hpp"
#include "../math/Vertex.hpp"
#include "../utils/Scene.hpp"
#include <unordered_map>
#include <vector>
#include <array>

class SolidRenderer {
public:
    explicit SolidRenderer(VulkanApp* app_ = nullptr);
    ~SolidRenderer();

    void init(VulkanApp* app_);
    void createPipelines();
    void createRenderTargets(uint32_t width, uint32_t height);
    void destroyRenderTargets();
    void beginPass(VkCommandBuffer cmd, uint32_t frameIndex, VkClearValue colorClear, VkClearValue depthClear);
    void endPass(VkCommandBuffer cmd);
    void cleanup();

    // Depth pre-pass that binds depthPrePassPipeline and draws indirect only
    void depthPrePass(VkCommandBuffer &commandBuffer, VkQueryPool queryPool);

    // Draw main solid geometry: bind pipeline (wireframe or normal) and draw
    void draw(VkCommandBuffer &commandBuffer, VulkanApp* app, VkDescriptorSet perTextureDescriptorSet, bool wireframeEnabled);

    // Access for adding meshes
    IndirectRenderer& getIndirectRenderer() { return indirectRenderer; }


    // Offscreen solid pass outputs
    VkImageView getColorView(uint32_t frameIndex) const { return solidColorImageViews[frameIndex]; }
    VkImageView getDepthView(uint32_t frameIndex) const { return solidDepthImageViews[frameIndex]; }
    VkRenderPass getRenderPass() const { return solidRenderPass; }

public:
    // Public accessor for nodeModelVersions (read-only)
    const std::unordered_map<NodeID, Model3DVersion>& getNodeModelVersions() const { return solidChunks; }
public:
    VkPipelineLayout getGraphicsPipelineLayout() const { return graphicsPipelineLayout; }
private:
    VulkanApp* app = nullptr;
    IndirectRenderer indirectRenderer;
    VkPipeline graphicsPipeline = VK_NULL_HANDLE;
    VkPipelineLayout graphicsPipelineLayout = VK_NULL_HANDLE;
    VkPipeline graphicsPipelineWire = VK_NULL_HANDLE;
    VkPipelineLayout graphicsPipelineWireLayout = VK_NULL_HANDLE;
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
    std::array<VkFramebuffer, 2> solidFramebuffers = {VK_NULL_HANDLE, VK_NULL_HANDLE};
    VkRenderPass solidRenderPass = VK_NULL_HANDLE;
    uint32_t renderWidth = 0;
    uint32_t renderHeight = 0;
};
