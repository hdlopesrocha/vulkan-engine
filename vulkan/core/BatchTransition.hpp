#pragma once

#include <vulkan/vulkan.h>
#include <cstdint>

// One image layout transition in a batched vkCmdPipelineBarrier2 call.
struct BatchTransition {
    VkImage       image          = VK_NULL_HANDLE;
    VkFormat      format         = VK_FORMAT_UNDEFINED;
    VkImageLayout oldLayout      = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImageLayout newLayout      = VK_IMAGE_LAYOUT_UNDEFINED;
    uint32_t      mipLevels      = 1;
    uint32_t      baseArrayLayer = 0;
    uint32_t      layerCount     = 1;
    // When true and oldLayout == newLayout, the entry is emitted as a
    // VK_ACCESS_2_NONE / VK_PIPELINE_STAGE_2_NONE no-op barrier inside
    // the same vkCmdPipelineBarrier2 call (documents the frame-graph
    // edge without stalling). Entries without isNoOp whose resolved
    // layouts are equal are skipped entirely (same as the single
    // transition early-out).
    bool          isNoOp         = false;
};
