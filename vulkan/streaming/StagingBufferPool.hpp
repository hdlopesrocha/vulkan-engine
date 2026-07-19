#pragma once

#include <vulkan/vulkan.h>
#include "../VmaContext.hpp"
#include <vector>
#include <functional>
#include <cstddef>

namespace streaming {

// One reusable staging resource. A staging slot owns everything needed to move
// one chunk's worth of mesh data to the GPU:
//   - a host-visible, persistently-mapped staging buffer (TRANSFER_SRC)
//   - a dedicated command buffer (reset, not reallocated, between uses)
//   - a fence used for non-blocking completion polling
//   - a binary semaphore signaled by the transfer submission so the render
//     frame can wait on it (cross-queue hazard tracking)
//
// A slot is reused ONLY after the GPU has finished the transfer that used it
// (fence signaled / timeline value reached). While busy it is not handed out.
struct StagingSlot {
    VkBuffer      buffer    = VK_NULL_HANDLE;
    VmaAllocation alloc     = VK_NULL_HANDLE;
    void*         mapped    = nullptr;
    VkDeviceSize  size      = 0;

    VkCommandBuffer cmd     = VK_NULL_HANDLE;
    VkFence         fence   = VK_NULL_HANDLE;
    VkSemaphore     signalSem = VK_NULL_HANDLE;   // binary, for frame waits

    uint64_t timelineValue  = 0;   // value this slot's submission signaled (timeline path)
    bool     busy           = false;
    bool     waitRegistered = false; // binary signalSem already added as a frame wait
                                     // (a binary semaphore may be waited on only
                                     //  once; re-adding it on later frames is illegal)

    std::function<void()> onComplete; // main-thread callback fired when GPU done
    void*    chunkSlot     = nullptr; // ChunkGPUBuffers* handed back to ChunkBufferPool
};

class StagingBufferPool {
public:
    void init(VmaAllocator vma, VkDevice device,
              uint32_t queueFamily, uint32_t slotCount, VkDeviceSize slotSize);

    // Returns a free slot, or nullptr if every slot is currently busy. This is
    // the ONLY back-pressure in the system: when all slots are busy the manager
    // simply stops submitting and continues next frame.
    StagingSlot* acquireFree();

    // Recycle a slot whose transfer has completed: reset fence + command buffer,
    // clear state. Does NOT run onComplete (the manager does that)..
    void reset(StagingSlot& s);

    std::vector<StagingSlot>& slots() { return slots_; }

    void destroy();

private:
    void createSlot(StagingSlot& s, VkDevice device, uint32_t queueFamily, VkDeviceSize slotSize);

    VmaAllocator            vma_ = VK_NULL_HANDLE;
    VkDevice                device_ = VK_NULL_HANDLE;
    VkCommandPool           cmdPool_ = VK_NULL_HANDLE;
    std::vector<StagingSlot> slots_;
};

} // namespace streaming
