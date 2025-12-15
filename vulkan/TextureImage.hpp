// Auto-generated wrapper header for TextureImage
#pragma once

#include "vulkan.hpp"

struct TextureImage {
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    uint32_t mipLevels = 1;
};
