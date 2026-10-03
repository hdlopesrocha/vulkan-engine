#pragma once

#include "../VulkanApp.hpp"
#include <vulkan/vulkan.h>
#include <atomic>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>
#include <array>
#include <glm/glm.hpp>

// ─── Shared utility functions used by multiple renderers ──────────────────────
// All functions are header-only inlines to avoid a separate translation unit.

namespace RendererUtils {

// Corner-mesh tables for the 6-plane billboard mesh shared by ImpostorCapture
// and VegetationRenderer (24 vertices = 6 planes × 4 corners): the base tangent
// per plane, and the outward tilt direction for the first four planes.
inline const glm::vec3 kBillboardBaseTangents[6] = {
    {0,0,1}, {-1,0,0}, {0,0,-1}, {1,0,0}, {1,0,0}, {0,0,1}
};
inline const glm::vec3 kBillboardOutwardDirs[4] = {
    {1,0,0}, {0,0,1}, {-1,0,0}, {0,0,-1}
};

// Create a 2D image + VMA-backed device-local memory + image view.
// Registers all three with VulkanResourceManager. Uses VMA for suballocation.
inline void createImage2DWithVma(
    VkDevice       device,
    VulkanApp*     app,
    uint32_t       width,
    uint32_t       height,
    VkFormat       format,
    VkImageUsageFlags usage,
    VkImageAspectFlags aspect,
    const char*    name,
    VkImage&       outImage,
    VmaAllocation& outAllocation,
    VkDeviceMemory& outMemory,
    VkImageView&   outView)
{
    VkImageCreateInfo imageInfo{};
    imageInfo.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType     = VK_IMAGE_TYPE_2D;
    imageInfo.extent        = {width, height, 1};
    imageInfo.mipLevels     = 1;
    imageInfo.arrayLayers   = 1;
    imageInfo.format        = format;
    imageInfo.tiling        = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imageInfo.usage         = usage;
    imageInfo.samples       = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
    imageInfo.queueFamilyIndexCount = 0;
    imageInfo.pQueueFamilyIndices = nullptr;

    VmaAllocationCreateInfo allocCI{};
    allocCI.usage = VMA_MEMORY_USAGE_AUTO;
    allocCI.preferredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

    VmaAllocationInfo allocInfo;
    if (vmaCreateImage(app->getVmaAllocator(), &imageInfo, &allocCI, &outImage, &outAllocation, &allocInfo) != VK_SUCCESS)
        throw std::runtime_error(std::string("Failed to create image with VMA: ") + name);
    outMemory = allocInfo.deviceMemory;
    app->resources.addImageVma(outImage, outAllocation, name);
    std::cerr << "[RendererUtils::createImage2DWithVma] created image=" << (void*)outImage << " name=" << name << std::endl;

    VkImageViewCreateInfo viewInfo{};
    viewInfo.sType                           = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image                           = outImage;
    viewInfo.viewType                        = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format                          = format;
    viewInfo.subresourceRange.aspectMask     = aspect;
    viewInfo.subresourceRange.baseMipLevel   = 0;
    viewInfo.subresourceRange.levelCount     = 1;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount     = 1;

    if (vkCreateImageView(device, &viewInfo, nullptr, &outView) != VK_SUCCESS)
        throw std::runtime_error(std::string("Failed to create image view: ") + name);
    app->resources.addImageView(outView, name);
}

// Options for buildFullscreenPipeline.  All defaults match a standard
// fullscreen-triangle pass: fill, no cull, CCW, depth disabled, no blend.
struct FullscreenPipelineOpts {
    VkPolygonMode  polygonMode     = VK_POLYGON_MODE_FILL;
    VkCullModeFlags cullMode       = VK_CULL_MODE_NONE;
    VkFrontFace    frontFace       = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    bool           depthTestEnable  = false;
    bool           depthWriteEnable = false;
    VkCompareOp    depthCompareOp   = VK_COMPARE_OP_LESS_OR_EQUAL;
    uint32_t       colorAttachmentCount = 1;
    bool           blendEnable      = false;
    // VK_EXT_descriptor_buffer: when true the pipeline is created with
    // VK_PIPELINE_CREATE_DESCRIPTOR_BUFFER_BIT_EXT (VUID-vkCmdDraw-None-08117
    // requires it for vkCmdSetDescriptorBufferOffsetsEXT binds).
    bool           descriptorBuffer = false;
};

// Build a simple graphics pipeline with no vertex-input state and dynamic
// viewport/scissor.  The caller is responsible for providing a pre-created
// pipelineLayout and attachment formats. Returns the created VkPipeline and
// registers it with the VulkanResourceManager.
// Throws std::runtime_error on failure.
inline VkPipeline buildFullscreenPipeline(
    VkDevice                    device,
    VulkanApp*                  app,
    VkFormat                    colorFormat,
    VkFormat                    depthFormat,
    VkPipelineLayout            layout,
    const std::vector<VkPipelineShaderStageCreateInfo>& stages,
    const FullscreenPipelineOpts& opts,
    const char*                 name)
{
    VkPipelineVertexInputStateCreateInfo vi{};
    vi.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

    VkPipelineInputAssemblyStateCreateInfo ia{};
    ia.sType    = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkPipelineViewportStateCreateInfo vp{};
    vp.sType         = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    vp.viewportCount = 1;
    vp.scissorCount  = 1;

    VkPipelineRasterizationStateCreateInfo rs{};
    rs.sType       = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rs.polygonMode = opts.polygonMode;
    rs.lineWidth   = 1.0f;
    rs.cullMode    = opts.cullMode;
    rs.frontFace   = opts.frontFace;

    VkPipelineMultisampleStateCreateInfo ms{};
    ms.sType                 = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    ms.rasterizationSamples  = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineDepthStencilStateCreateInfo ds{};
    ds.sType             = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    ds.depthTestEnable   = opts.depthTestEnable  ? VK_TRUE : VK_FALSE;
    ds.depthWriteEnable  = opts.depthWriteEnable ? VK_TRUE : VK_FALSE;
    ds.depthCompareOp    = opts.depthCompareOp;

    VkPipelineColorBlendAttachmentState ca{};
    ca.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                        VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    ca.blendEnable    = opts.blendEnable ? VK_TRUE : VK_FALSE;
    std::vector<VkPipelineColorBlendAttachmentState> colorAtts(opts.colorAttachmentCount, ca);

    VkPipelineColorBlendStateCreateInfo cb{};
    cb.sType           = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    cb.attachmentCount = static_cast<uint32_t>(colorAtts.size());
    cb.pAttachments    = colorAtts.data();

    std::array<VkDynamicState, 2> dynStates = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dyn{};
    dyn.sType             = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dyn.dynamicStateCount = static_cast<uint32_t>(dynStates.size());
    dyn.pDynamicStates    = dynStates.data();

    VkGraphicsPipelineCreateInfo pi{};
    pi.sType               = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pi.flags               = opts.descriptorBuffer ? VK_PIPELINE_CREATE_DESCRIPTOR_BUFFER_BIT_EXT : 0;
    pi.stageCount          = static_cast<uint32_t>(stages.size());
    pi.pStages             = stages.data();
    pi.pVertexInputState   = &vi;
    pi.pInputAssemblyState = &ia;
    pi.pViewportState      = &vp;
    pi.pRasterizationState = &rs;
    pi.pMultisampleState   = &ms;
    pi.pDepthStencilState  = &ds;
    pi.pColorBlendState    = &cb;
    pi.pDynamicState       = &dyn;
    pi.layout              = layout;
    pi.subpass             = 0;

    VkPipelineRenderingCreateInfo renderingInfo{};
    renderingInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    renderingInfo.colorAttachmentCount = opts.colorAttachmentCount;
    renderingInfo.pColorAttachmentFormats = (opts.colorAttachmentCount > 0) ? &colorFormat : nullptr;
    renderingInfo.depthAttachmentFormat = depthFormat;
    renderingInfo.stencilAttachmentFormat = VK_FORMAT_UNDEFINED;

    pi.pNext = &renderingInfo;
    pi.renderPass = VK_NULL_HANDLE;

    VkPipeline pipeline = VK_NULL_HANDLE;
    if (vkCreateGraphicsPipelines(device, app->getPipelineCache(), 1, &pi, nullptr, &pipeline) != VK_SUCCESS)
        throw std::runtime_error(std::string("Failed to create pipeline: ") + name);
    app->resources.addPipeline(pipeline, name);
    return pipeline;
}

// ─── Barrier statistics: per-frame vkCmdPipelineBarrier2 accounting ─────────
// Goal: <20 vkCmdPipelineBarrier2 calls per frame (currently 40+). Every
// barrier emitted through RendererUtils helpers or VulkanApp::
// recordTransitionBatch is counted here. Call beginFrame() at the start of
// each frame and endFrameReport() at the end to get a throttled log line.
struct BarrierStats {
    inline static std::atomic<uint64_t> totalCalls{0};
    inline static std::atomic<uint64_t> totalImageBarriers{0};
    inline static std::atomic<uint64_t> frameCalls{0};
    inline static std::atomic<uint64_t> frameImageBarriers{0};
    inline static std::atomic<uint64_t> frameIndex{0};

    static void noteBarrier(uint64_t imageBarrierCount) {
        totalCalls.fetch_add(1, std::memory_order_relaxed);
        totalImageBarriers.fetch_add(imageBarrierCount, std::memory_order_relaxed);
        frameCalls.fetch_add(1, std::memory_order_relaxed);
        frameImageBarriers.fetch_add(imageBarrierCount, std::memory_order_relaxed);
    }
    static void beginFrame() {
        frameCalls.store(0, std::memory_order_relaxed);
        frameImageBarriers.store(0, std::memory_order_relaxed);
    }
    // Logs one line per frame when `VULKAN_BARRIER_STATS=1` is set. Warns when
    // the frame exceeds `warnThreshold` barrier calls (default 20).
    static void endFrameReport(uint64_t warnThreshold = 20);
};

// Record a single-image layout transition barrier into a command buffer.
// Wraps VkImageMemoryBarrier2 + vkCmdPipelineBarrier2 for the common case
// of transitioning a single subresource (mip 0, layer 0).
inline void transitionImageLayout(
    VkCommandBuffer           cmd,
    VkImage                   image,
    VkImageLayout             oldLayout,
    VkImageLayout             newLayout,
    VkAccessFlags2            srcAccess,
    VkAccessFlags2            dstAccess,
    VkPipelineStageFlags2     srcStage,
    VkPipelineStageFlags2     dstStage,
    VkImageAspectFlags        aspect     = VK_IMAGE_ASPECT_COLOR_BIT,
    uint32_t                  baseLayer  = 0,
    uint32_t                  layerCount = 1)
{
    VkImageMemoryBarrier2 barrier{};
    barrier.sType                = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    barrier.oldLayout            = oldLayout;
    barrier.newLayout            = newLayout;
    barrier.srcQueueFamilyIndex  = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex  = VK_QUEUE_FAMILY_IGNORED;
    barrier.image                = image;
    barrier.subresourceRange     = { aspect, 0, 1, baseLayer, layerCount };
    barrier.srcAccessMask        = srcAccess;
    barrier.dstAccessMask        = dstAccess;
    barrier.srcStageMask         = srcStage;
    barrier.dstStageMask         = dstStage;

    VkDependencyInfo depInfo{};
    depInfo.sType                    = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    depInfo.imageMemoryBarrierCount = 1;
    depInfo.pImageMemoryBarriers    = &barrier;

    vkCmdPipelineBarrier2(cmd, &depInfo);
    BarrierStats::noteBarrier(1);
}

// ─── Dynamic-rendering pass setup ────────────────────────────────────────────
// Begin a depth-only pass: the depth attachment is CLEARed to `clearDepth` and
// a full-target viewport/scissor is set. Pair with endDepthOnlyPass(). The
// attachment/rendering structs are consumed by vkCmdBeginRendering before this
// returns, so the locals do not need to outlive the call.
inline void beginDepthOnlyPass(
    VkCommandBuffer cmd,
    VkImageView     depthImageView,
    uint32_t        width,
    uint32_t        height,
    float           clearDepth)
{
    VkClearValue clear{};
    clear.depthStencil = {clearDepth, 0};

    VkRenderingAttachmentInfo depthAtt{};
    depthAtt.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    depthAtt.imageView = depthImageView;
    depthAtt.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    depthAtt.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depthAtt.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    depthAtt.clearValue = clear;

    VkRenderingInfo renderingInfo{};
    renderingInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
    renderingInfo.renderArea.offset = {0, 0};
    renderingInfo.renderArea.extent = {width, height};
    renderingInfo.layerCount = 1;
    renderingInfo.colorAttachmentCount = 0;
    renderingInfo.pDepthAttachment = &depthAtt;

    vkCmdBeginRendering(cmd, &renderingInfo);

    VkViewport viewport{};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = static_cast<float>(width);
    viewport.height = static_cast<float>(height);
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    vkCmdSetViewport(cmd, 0, 1, &viewport);
    VkRect2D scissor{};
    scissor.offset = {0, 0};
    scissor.extent = {width, height};
    vkCmdSetScissor(cmd, 0, 1, &scissor);
}

inline void endDepthOnlyPass(VkCommandBuffer cmd) {
    vkCmdEndRendering(cmd);
}

// Begin a color+depth pass: both attachments CLEARed with the caller's values,
// full-target viewport/scissor set. Callers end with vkCmdEndRendering.
inline void beginColorDepthPass(
    VkCommandBuffer cmd,
    VkImageView     colorImageView,
    VkImageView     depthImageView,
    uint32_t        width,
    uint32_t        height,
    const VkClearValue& colorClear,
    const VkClearValue& depthClear)
{
    VkRenderingAttachmentInfo colorAtt{};
    colorAtt.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    colorAtt.imageView = colorImageView;
    colorAtt.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    colorAtt.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAtt.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    colorAtt.clearValue = colorClear;

    VkRenderingAttachmentInfo depthAtt{};
    depthAtt.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    depthAtt.imageView = depthImageView;
    depthAtt.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    depthAtt.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depthAtt.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    depthAtt.clearValue = depthClear;

    VkRenderingInfo ri{};
    ri.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
    ri.renderArea.offset = {0, 0};
    ri.renderArea.extent = {width, height};
    ri.layerCount = 1;
    ri.colorAttachmentCount = 1;
    ri.pColorAttachments = &colorAtt;
    ri.pDepthAttachment = &depthAtt;
    vkCmdBeginRendering(cmd, &ri);

    VkViewport viewport{};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = static_cast<float>(width);
    viewport.height = static_cast<float>(height);
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    vkCmdSetViewport(cmd, 0, 1, &viewport);
    VkRect2D scissor{};
    scissor.offset = {0, 0};
    scissor.extent = {width, height};
    vkCmdSetScissor(cmd, 0, 1, &scissor);
}

// Fill the standard dynamic viewport/scissor pipeline state used by the
// water and wireframe pipelines: one viewport/scissor, VIEWPORT + SCISSOR
// dynamic. pDynamicStates points at a function-local constexpr table that
// stays valid for the duration of the pipeline create call.
inline void fillViewportScissorState(
    VkPipelineViewportStateCreateInfo& viewportState,
    VkPipelineDynamicStateCreateInfo&  dynamicState)
{
    viewportState = {};
    viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.scissorCount = 1;

    static constexpr VkDynamicState dynamicStates[] = {
        VK_DYNAMIC_STATE_VIEWPORT,
        VK_DYNAMIC_STATE_SCISSOR
    };
    dynamicState = {};
    dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamicState.dynamicStateCount = 2;
    dynamicState.pDynamicStates = dynamicStates;
}

} // namespace RendererUtils
