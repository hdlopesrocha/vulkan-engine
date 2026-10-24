#pragma once

#include <vulkan/vulkan.h>
#include <vector>

// Fill/raster state used by VulkanApp::createGraphicsPipelineWithLayout().
struct GraphicsPipelineConfig {
    VkPolygonMode polygonMode = VK_POLYGON_MODE_FILL;
    VkCullModeFlagBits cullMode = VK_CULL_MODE_BACK_BIT;
    VkFrontFace frontFace = VK_FRONT_FACE_CLOCKWISE;
    bool depthTestEnable = true;
    bool depthWriteEnable = true;
    bool colorWrite = true;
    VkCompareOp depthCompareOp = VK_COMPARE_OP_LESS;
    VkPrimitiveTopology topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    bool depthClampEnable = false;
    std::vector<VkFormat> colorFormats = {};
    VkFormat depthFormat = VK_FORMAT_D32_SFLOAT;
    bool noColorAttachment = false;
    bool depthBiasEnable = false;
    bool blendEnable = false;
    // Blend factors applied when blendEnable is true. Straight alpha by
    // default; pipelines with a different accumulation model (e.g.
    // premultiplied coverage) override these.
    VkBlendFactor blendSrcColorFactor = VK_BLEND_FACTOR_SRC_ALPHA;
    VkBlendFactor blendDstColorFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    VkBlendFactor blendSrcAlphaFactor = VK_BLEND_FACTOR_ONE;
    VkBlendFactor blendDstAlphaFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
};
