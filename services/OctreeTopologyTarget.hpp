#pragma once

#include <vulkan/vulkan.h>
#include <cstdint>

// Destination of the GPU topology path: the renderer's packed element pools.
// The finalize pass writes one vertex per host-assigned surface-node slot into
// `vertexBuffer[baseVertex .. baseVertex+vertexCapacity)` in the renderer's
// 64-byte Vertex layout and the chunk-local index stream into
// `indexBuffer[firstIndex .. firstIndex+indexCapacity)`. The host never
// uploads or reads vertex data — it only reserves the span and publishes the
// draw entry (indexCount = index stream length, vertexOffset = baseVertex).
struct OctreeTopologyTarget {
    VkBuffer vertexBuffer = VK_NULL_HANDLE;
    uint32_t baseVertex = 0;
    uint32_t vertexCapacity = 0;
    VkBuffer indexBuffer = VK_NULL_HANDLE;
    uint32_t firstIndex = 0;
    uint32_t indexCapacity = 0;
    uint32_t slotIndex = UINT32_MAX; // opaque, echoed back in the result
};
