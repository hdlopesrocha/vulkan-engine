#pragma once

#include <vulkan/vulkan.h>
#include <cstdint>

// Deferred image-layout transition recorded while a command buffer is open.
struct PendingLayoutUpdate {
    VkImage image;
    VkImageLayout newLayout;
    uint32_t baseArrayLayer;
    uint32_t layerCount;
    bool isBarrier;
};
