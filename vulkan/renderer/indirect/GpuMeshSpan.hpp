#pragma once

#include <vulkan/vulkan.h>
#include <cstdint>

// A reserved span in the renderer's packed vertex/index pools.
struct GpuMeshSpan {
    uint32_t slotIndex = UINT32_MAX;
    uint32_t baseVertex = 0;     // pool element index the pass writes at
    uint32_t vertexCapacity = 0; // element budget reserved for the pass
    uint32_t firstIndex = 0;     // index pool element index the pass writes at
    uint32_t indexCapacity = 0;  // index element budget reserved for the pass
    VkBuffer vertexBuffer = VK_NULL_HANDLE;
    VkBuffer indexBuffer = VK_NULL_HANDLE;
};
