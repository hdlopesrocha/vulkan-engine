#pragma once

#include "StreamCommon.hpp"
#include "LockFreeQueue.hpp"
#include "StagingBufferPool.hpp"

#include <vulkan/vulkan.h>
#include <vector>
#include <array>
#include <memory>
#include <mutex>
#include <cstdint>

class VulkanApp;
class ThreadPool;

namespace streaming {

// Consumes UploadJobs from the lock-free per-category queues and drives the GPU
// transfer engine. Designed so that:
//   * It imposes NO fixed uploads-per-frame cap. The only limiter is the number
//     of staging slots (GPU/CPU memory) and GPU completion.
//   * It runs entirely on the main thread and never blocks (no vkQueueWaitIdle /
//     vkDeviceWaitIdle). Completed slots are detected with non-blocking fence /
//     timeline polling.
//   * Transfers are submitted to the engine's RADV-safe upload queue
//     (geometryTransferQueue(), i.e. a graphics-family queue distinct from the
//     render queue) so uploading and rendering overlap.
class UploadManager {
public:
    void init(VulkanApp* app,
              VkDeviceSize chunkVertexBytes,
              VkDeviceSize chunkIndexBytes,
              uint32_t stagingSlots = 32);

    // Worker threads call this (lock-free push). `job` must reference
    // destination buffers and contain only CPU data.
    void enqueue(UploadJob&& job);

    // Byte capacity of a single staging slot. UploadManager is the ONLY upload
    // path: a job whose total footprint exceeds this is rejected by the caller
    // (assert + return false). Size slots to cover the largest chunk
    // (currently 2 MiB vertex + 2 MiB index = 4 MiB per slot).
    VkDeviceSize slotSize() const { return slotSize_; }

    // --- Called once per frame from the render loop -----------------------
    // Registers outstanding upload completion semaphores with the frame submit
    // so rendering waits for uploads without stalling the CPU.
    void prepareFrameWaits(VulkanApp* app);
    // Recycles finished slots and submits as many queued jobs as staging
    // resources allow. Returns immediately when slots are exhausted or the
    // queues drain — it never waits on the GPU.
    void processUploads();

    // Blocking drain: submits every queued UploadJob and waits until all
    // in-flight transfers retire (onComplete fired, slots recycled). After this
    // returns no queued or in-flight job references any destination buffer, so
    // the caller may safely destroy/recreate buffers that were upload targets.
    // Intended for rare, already-heavy events (e.g. a full IndirectRenderer
    // rebuild that grows and reallocates the merged vertex/index buffers).
    void flush();

    void destroy();

private:
    void submitJob(StagingSlot& slot, UploadJob&& job);
    bool isComplete(const StagingSlot& slot) const;
    std::mutex& pickMutex();
    VkSemaphore makeBinarySemaphore();

    VulkanApp*     app_ = nullptr;
    VmaAllocator   vma_ = VK_NULL_HANDLE;
    VkDevice       device_ = VK_NULL_HANDLE;
    VkQueue        queue_ = VK_NULL_HANDLE;      // geometryTransferQueue() or graphicsQueue
    uint32_t       queueFamily_ = 0;             // graphics family (staging is EXCLUSIVE on it)

    StagingBufferPool staging_;
    VkDeviceSize   slotSize_ = 0;

    std::array<MPSCQueue<UploadJob>, (size_t)StreamCategory::Count> queues_;

    // Timeline-semaphore completion path (preferred when supported).
    VkSemaphore   m_timeline = VK_NULL_HANDLE;
    uint64_t      m_timelineSignal = 0;
    bool          m_timelineSupported = false;
};

// Orchestrates the whole subsystem: the UploadManager and two independent
// worker ThreadPools (solid / water). Each category's meshing runs on its own
// pool of threads, so solid and water geometry are generated as parallel as the
// hardware allows; finished jobs stream through the shared UploadManager
// without a per-frame budget.
class TerrainStreamer {
public:
    void init(VulkanApp* app,
              VkDeviceSize chunkVertexBytes,
              VkDeviceSize chunkIndexBytes,
              uint32_t stagingSlots = 32,
              uint32_t workersPerCategory = 2);

    // Call ONCE per frame, before drawFrame's submit:
    void update(VulkanApp* app) {
        upload_.prepareFrameWaits(app);
        upload_.processUploads();
    }

    UploadManager&       uploadManager()  { return upload_; }

    void destroy();

private:
    UploadManager upload_;
    std::array<std::unique_ptr<class ThreadPool>, (size_t)StreamCategory::Count> pools_;
};

} // namespace streaming
