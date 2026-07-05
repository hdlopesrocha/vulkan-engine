#pragma once

#include <vulkan/vulkan.h>

#include "../third_party/VulkanMemoryAllocator/include/vk_mem_alloc.h"

// Small helper types used across the Vulkan module
struct Buffer {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VmaAllocation allocation = VK_NULL_HANDLE;
};

