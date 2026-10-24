#pragma once

#include <vulkan/vulkan.h>

// GPU memory budget information (VK_EXT_memory_budget).
struct MemoryHeapBudget {
    VkMemoryHeapFlags flags;
    VkDeviceSize      usage;   // bytes currently allocated (estimated by driver)
    VkDeviceSize      budget;  // bytes available before paging/system memory (0 = unknown)
    VkDeviceSize      size;    // total heap size
};
