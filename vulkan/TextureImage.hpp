#pragma once

#include "vulkan.hpp"
#include "../third_party/VulkanMemoryAllocator/include/vk_mem_alloc.h"

struct TextureImage {
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    VmaAllocation allocation = VK_NULL_HANDLE;
    uint32_t mipLevels = 1;
};
