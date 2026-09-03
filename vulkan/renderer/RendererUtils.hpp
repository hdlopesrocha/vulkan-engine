#pragma once

#include "../VulkanApp.hpp"
#include <vulkan/vulkan.h>
#include <atomic>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>
#include <array>

// ─── Shared utility functions used by multiple renderers ──────────────────────
// All functions are header-only inlines to avoid a separate translation unit.

namespace RendererUtils {

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

// ─── Explicit no-op barrier ─────────────────────────────────────────────────
// Emits a VK_ACCESS_2_NONE / VK_PIPELINE_STAGE_2_NONE memory barrier. Use for
// "no-op" dependency points where the image layout is unchanged and no memory
// hazard exists, so the barrier documents the frame-graph edge without
// stalling the pipeline. Prefer dropping pure no-ops from batched barrier
// calls entirely; use this helper only where an explicit ordering point aids
// readability or debugging.
inline void emitNoOpBarrier(VkCommandBuffer cmd, const char* reason = nullptr) {
    VkMemoryBarrier2 mb{};
    mb.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
    mb.srcStageMask  = VK_PIPELINE_STAGE_2_NONE;
    mb.dstStageMask  = VK_PIPELINE_STAGE_2_NONE;
    mb.srcAccessMask = VK_ACCESS_2_NONE;
    mb.dstAccessMask = VK_ACCESS_2_NONE;

    VkDependencyInfo depInfo{};
    depInfo.sType             = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    depInfo.memoryBarrierCount = 1;
    depInfo.pMemoryBarriers    = &mb;

    vkCmdPipelineBarrier2(cmd, &depInfo);
    BarrierStats::noteBarrier(0);
    (void)reason;
}

// ─── Minimal frame graph for barrier batching ───────────────────────────────
// Nodes = render passes, edges = resource dependencies (shared VkImages).
// Each pass declares its explicit resource accesses (read/write, stage,
// layout). buildMergedBarriers() computes, per distinct (image, layer range),
// the single transition covering all uses in the frame (first-seen layout →
// last-seen layout). emitMergedBarriers() records all of them in ONE
// vkCmdPipelineBarrier2 call. Resources whose layout never changes across
// uses become VK_ACCESS_2_NONE / VK_PIPELINE_STAGE_2_NONE no-op entries
// inside the same call (no extra barrier, no stall).
struct FrameGraphResourceAccess {
    VkImage               image      = VK_NULL_HANDLE;
    VkFormat              format     = VK_FORMAT_UNDEFINED;
    VkImageLayout         layout     = VK_IMAGE_LAYOUT_UNDEFINED;
    VkAccessFlags2        access     = 0;
    VkPipelineStageFlags2 stage      = 0;
    uint32_t              baseLayer  = 0;
    uint32_t              layerCount = 1;
    bool                  isWrite    = false;
};

struct FrameGraphPassNode {
    std::string name;
    std::vector<FrameGraphResourceAccess> accesses;
};

class FrameGraph {
public:
    // Add a render-pass node; returns its index for addAccess().
    uint32_t addPass(const std::string& name);
    void addAccess(uint32_t pass, const FrameGraphResourceAccess& access);
    size_t passCount() const { return passes_.size(); }
    void clear() { passes_.clear(); }

    // One merged transition per distinct (image, layer range): first-seen
    // layout → last-seen layout. Entries with unchanged layouts are flagged
    // as no-ops (caller should emit them with NONE/NONE masks).
    struct MergedTransition {
        VkImage       image      = VK_NULL_HANDLE;
        VkFormat      format     = VK_FORMAT_UNDEFINED;
        VkImageLayout oldLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
        VkImageLayout newLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
        uint32_t      baseLayer  = 0;
        uint32_t      layerCount = 1;
        bool          isWrite    = false;
        bool          isNoOp     = false; // oldLayout == newLayout && !isWrite
    };
    std::vector<MergedTransition> buildMergedBarriers() const;

    // Record all merged transitions in a single vkCmdPipelineBarrier2 via
    // VulkanApp::recordTransitionBatch (which applies the same stage/access
    // mapping as single transitions, so semantics are unchanged — only the
    // call count drops). No-op entries are emitted with NONE/NONE masks.
    // Returns the number of vkCmdPipelineBarrier2 calls emitted (0 or 1).
    // `app` must be non-null; `cmd` must be in the recording state.
    uint32_t emitMergedBarriers(VkCommandBuffer cmd, VulkanApp* app) const;

private:
    std::vector<FrameGraphPassNode> passes_;
};

} // namespace RendererUtils
