#pragma once

#include <vulkan/vulkan.h>
#include <cstdint>

// One submitted queue segment shown by the queue-timeline widget.
struct QueueSegment {
    VkQueue  queue    = VK_NULL_HANDLE;
    VkFence  fence    = VK_NULL_HANDLE;
    uint64_t frame    = 0;   // drawFrame index at submit time
    uint64_t submitId = 0;
    uint64_t startNs  = 0;   // steady_clock at submit
    uint64_t endNs    = 0;   // steady_clock when fence signaled (0 = ongoing)
};
