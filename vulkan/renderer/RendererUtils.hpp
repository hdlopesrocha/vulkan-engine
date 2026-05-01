#pragma once

#include "../VulkanApp.hpp"
#include <vulkan/vulkan.h>
#include <stdexcept>
#include <string>
#include <vector>
#include <array>

// ─── Shared utility functions used by multiple renderers ──────────────────────
// All functions are header-only inlines to avoid a separate translation unit.

namespace RendererUtils {

// Create a 2D image + device-local memory + image view, registering all three
// with the VulkanResourceManager. Throws std::runtime_error on any failure.
inline void createImage2D(
    VkDevice       device,
    VulkanApp*     app,
    uint32_t       width,
    uint32_t       height,
    VkFormat       format,
    VkImageUsageFlags usage,
    VkImageAspectFlags aspect,
    const char*    name,
    VkImage&       outImage,
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

    if (vkCreateImage(device, &imageInfo, nullptr, &outImage) != VK_SUCCESS)
        throw std::runtime_error(std::string("Failed to create image: ") + name);
    app->resources.addImage(outImage, name);

    VkMemoryRequirements memReq;
    vkGetImageMemoryRequirements(device, outImage, &memReq);

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize  = memReq.size;
    allocInfo.memoryTypeIndex = app->findMemoryType(memReq.memoryTypeBits,
                                                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    if (vkAllocateMemory(device, &allocInfo, nullptr, &outMemory) != VK_SUCCESS)
        throw std::runtime_error(std::string("Failed to allocate image memory: ") + name);
    vkBindImageMemory(device, outImage, outMemory, 0);
    app->resources.addDeviceMemory(outMemory, name);

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
// pipelineLayout and renderPass.  Returns the created VkPipeline and
// registers it with the VulkanResourceManager.
// Throws std::runtime_error on failure.
inline VkPipeline buildFullscreenPipeline(
    VkDevice                    device,
    VulkanApp*                  app,
    VkRenderPass                renderPass,
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
    pi.renderPass          = renderPass;
    pi.subpass             = 0;

    VkPipeline pipeline = VK_NULL_HANDLE;
    if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pi, nullptr, &pipeline) != VK_SUCCESS)
        throw std::runtime_error(std::string("Failed to create pipeline: ") + name);
    app->resources.addPipeline(pipeline, name);
    return pipeline;
}

} // namespace RendererUtils
