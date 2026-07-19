#pragma once

#include "../Buffer.hpp"
#include "../../space/ThreadPool.hpp"   // not used here directly, kept for symmetry
#include <vulkan/vulkan.h>
#include "../VmaContext.hpp"
#include <vector>
#include <memory>
#include <mutex>
#include <cstddef>

class VulkanApp;   // global namespace (matches vulkan/VulkanApp.hpp)

namespace streaming {

// Pre-allocated, reusable pool of device-local GPU buffer *slots*. One slot
// holds the vertex + index buffers for a single streamed chunk. Slots are
// created on the MAIN thread (vkCreateBuffer / vmaCreateBuffer touch no queue),
// so handing a slot to a worker thread never requires the worker to perform a
// Vulkan operation — it only receives opaque buffer handles to fill a job with.
//
// Buffers are created EXCLUSIVE on the graphics queue family. Uploads are routed
// through geometryTransferQueue() (same family, separate queue object), which
// matches the engine's existing RADV-safe transfer path and avoids the
// ownership-transfer barrier dance required for a distinct transfer family.
struct ChunkGPUBuffers {
    Buffer        vertexBuffer{};
    Buffer        indexBuffer{};
    VkDeviceSize  vertexCapacity = 0;   // bytes
    VkDeviceSize  indexCapacity  = 0;   // bytes
    uint64_t      ownerChunk     = 0;
    bool          inUse          = false;
};

class ChunkBufferPool {
public:
    void init(VulkanApp* app,
              VkDeviceSize chunkVertexBytes,
              VkDeviceSize chunkIndexBytes,
              uint32_t initialSlots);

    // Acquire a free slot (main thread only). Grows the pool — allocating fresh
    // GPU buffers — if none are free. Returns nullptr only on allocation failure.
    ChunkGPUBuffers* acquire(uint64_t chunkId);

    // Return a slot to the free list (main thread only). Called by the manager
    // after the GPU transfer for that slot has retired.
    void release(ChunkGPUBuffers* slot);

    void destroy();

    VkDeviceSize vertexCapacity() const { return vCap_; }
    VkDeviceSize indexCapacity()  const { return iCap_; }

private:
    void grow();

    VulkanApp* app_ = nullptr;
    VmaAllocator     vma_ = VK_NULL_HANDLE;
    uint32_t         gfxFamily_ = 0;

    VkDeviceSize     vCap_ = 0;
    VkDeviceSize     iCap_ = 0;

    std::mutex mtx_;
    std::vector<std::unique_ptr<ChunkGPUBuffers>> slots_;
    std::vector<ChunkGPUBuffers*>                 free_;
};

} // namespace streaming
