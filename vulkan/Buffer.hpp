#pragma once

#include <vulkan/vulkan.h>

// Small helper types used across the Vulkan module
struct Buffer {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
};

