#pragma once

#include "StreamCommon.hpp"
#include "LockFreeQueue.hpp"
#include "ChunkBufferPool.hpp"
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
              uint32_t stagingSlots = 4,
              uint32_t initialChunkSlots = 64);

    // Worker threads call this (lock-free push). `job` must reference
    // destination buffers acquired from chunkPool() and contain only CPU data.
    void enqueue(UploadJob&& job);

    ChunkBufferPool& chunkPool() { return chunkPool_; }
    const ChunkBufferPool& chunkPool() const { return chunkPool_; }

    // Byte capacity of a single staging slot. A job whose total footprint
    // (jobStagingFootprint) exceeds this cannot be serviced and the caller
    // must fall back to its own (larger) staging path.
    VkDeviceSize slotSize() const { return slotSize_; }

    // --- Called once per frame from the render loop -----------------------
    // Registers outstanding upload completion semaphores with the frame submit
    // so rendering waits for uploads without stalling the CPU.
    void prepareFrameWaits(VulkanApp* app);
    // Recycles finished slots and submits as many queued jobs as staging
    // resources allow. Returns immediately when slots are exhausted or the
    // queues drain — it never waits on the GPU.
    void processUploads();

    // Latest timeline value the frame should wait on (timeline path). 0 if the
    // timeline path is unavailable (use binary semaphores instead).
    uint64_t frameTimelineValue() const { return m_timelineWaitValue; }

    // Timeline semaphore handle (timeline path). The frame submit should add a
    // wait on this semaphore at frameTimelineValue() so rendering does not start
    // until the relevant uploads have landed on the GPU.
    VkSemaphore timeline() const { return m_timeline; }
    bool        timelineSupported() const { return m_timelineSupported; }

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

    ChunkBufferPool chunkPool_;
    StagingBufferPool staging_;
    VkDeviceSize   slotSize_ = 0;

    std::array<MPSCQueue<UploadJob>, (size_t)StreamCategory::Count> queues_;

    // Timeline-semaphore completion path (preferred when supported).
    VkSemaphore   m_timeline = VK_NULL_HANDLE;
    uint64_t      m_timelineSignal = 0;
    uint64_t        m_timelineWaitValue = 0;
    bool          m_timelineSupported = false;
};

// Orchestrates the whole subsystem: a ChunkBufferPool, the UploadManager, and
// three independent worker ThreadPools (solid / water / brush). Each category's
// meshing runs on its own pool of threads, so solid, water and brush geometry
// are generated as parallel as the hardware allows; finished jobs stream through
// the shared UploadManager without a per-frame budget.
class TerrainStreamer {
public:
    void init(VulkanApp* app,
              VkDeviceSize chunkVertexBytes,
              VkDeviceSize chunkIndexBytes,
              uint32_t stagingSlots = 4,
              uint32_t initialChunkSlots = 64,
              uint32_t workersPerCategory = 2);

    // Schedule generation of one chunk's mesh for `category`. Acquires the GPU
    // slot on the MAIN thread, then dispatches a CPU-only task to that category's
    // worker pool. `generator` performs the meshing (Tesselator / water SDF /
    // brush) and fills `job` (CPU data + destination buffers + onComplete).
    // The task pushes the job to the upload queue; no Vulkan call happens on the
    // worker thread.
    void requestMesh(StreamCategory category,
                     uint64_t chunkId,
                     int lod,
                     std::function<void(ChunkGPUBuffers&, UploadJob&)> generator);

    // Call ONCE per frame, before drawFrame's submit:
    void update(VulkanApp* app) {
        upload_.prepareFrameWaits(app);
        upload_.processUploads();
    }

    UploadManager&       uploadManager()  { return upload_; }
    ChunkBufferPool&     chunkPool()      { return upload_.chunkPool(); }

    void destroy();

private:
    VulkanApp* app_ = nullptr;
    UploadManager upload_;
    std::array<std::unique_ptr<class ThreadPool>, (size_t)StreamCategory::Count> pools_;
};

} // namespace streaming
