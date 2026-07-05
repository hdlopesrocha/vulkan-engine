#pragma once

#include <vulkan/vulkan.h>

#include "../third_party/VulkanMemoryAllocator/include/vk_mem_alloc.h"

// Small helper types used across the Vulkan module
struct Buffer {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VmaAllocation allocation = VK_NULL_HANDLE;
    void* mappedData = nullptr;

    // Convenience: return mapped data at a byte offset (null if not mapped)
    void* map(VkDeviceSize offset = 0) const {
        return mappedData ? static_cast<char*>(mappedData) + offset : nullptr;
    }
    // No-op unmap (VMA persistent mapping) — kept for symmetry with old API
    void unmap() const {}
};

