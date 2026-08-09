#include "UploadManager.hpp"
#include "../VulkanApp.hpp"
#include "../SubmissionTracker.hpp"
#include "../../space/ThreadPool.hpp"

#include <stdexcept>
#include <iostream>
#include <array>
#include <cstring>

namespace {

// A binary semaphore's signaled state can only be consumed by a QUEUE wait:
// vkWaitSemaphores() accepts only timeline semaphores
// (VUID-VkSemaphoreWaitInfo-pSemaphores-03256). Submit an empty drain batch
// whose wait list contains the semaphore, then idle the queue. All signal
// operations have already completed (their fences signaled) before this runs,
// so the drain batch completes instantly. Returns true when the signal was
// consumed and the semaphore is safe to destroy. On queue-submit failure (e.g.
// device lost) the semaphore may still be referenced by an in-flight batch:
// destroying it would violate VUID-vkDestroySemaphore-semaphore-05149, so the
// caller must NOT destroy it (a leak at teardown is acceptable — the OS
// reclaims everything when the process exits).
static bool consumeBinarySignal(VkDevice dev, VkQueue queue, VkSemaphore sem) {
    if (sem == VK_NULL_HANDLE || queue == VK_NULL_HANDLE) return false;
    VkSemaphoreSubmitInfo wait{ VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO };
    wait.semaphore = sem;
    wait.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    VkSubmitInfo2 si{ VK_STRUCTURE_TYPE_SUBMIT_INFO_2 };
    si.waitSemaphoreInfoCount = 1;
    si.pWaitSemaphoreInfos = &wait;
    if (vkQueueSubmit2(queue, 1, &si, VK_NULL_HANDLE) != VK_SUCCESS) {
        std::cerr << "[UploadManager] consumeBinarySignal: drain submit failed; skipping semaphore destroy (degraded shutdown)" << std::endl;
        return false;
    }
    if (vkQueueWaitIdle(queue) != VK_SUCCESS) {
        std::cerr << "[UploadManager] consumeBinarySignal: queue idle failed; skipping semaphore destroy (degraded shutdown)" << std::endl;
        return false;
    }
    return true;
}

} // namespace

namespace streaming {

// ----------------------------------------------------------------------------
// UploadManager
// ----------------------------------------------------------------------------

void UploadManager::init(VulkanApp* app,
                         VkDeviceSize chunkVertexBytes,
                         VkDeviceSize chunkIndexBytes,
                         uint32_t stagingSlots,
                         uint32_t initialChunkSlots) {
    app_    = app;
    vma_    = app->getVmaAllocator();
    device_ = app->getDevice();

    // Upload queue: prefer the distinct graphics-family queue (overlaps render),
    // otherwise fall back to the main graphics queue. NEVER the dedicated
    // transfer-family queue (RADV/RENOIR GPUVM faults, per engine policy).
    // NOTE: the app no longer requests a distinct geometry queue (it stalls on
    // RADV/Rembrandt under burst load), so this falls back to the main
    // graphics queue on this hardware by design.
    queue_ = app->geometryTransferQueue();
    if (queue_ == VK_NULL_HANDLE) queue_ = app->graphicsQueue;

    auto qfi = app->findQueueFamilies(app->getPhysicalDevice());
    queueFamily_ = qfi.graphicsFamily.value();

    chunkPool_.init(app, chunkVertexBytes, chunkIndexBytes, initialChunkSlots);

    const VkDeviceSize slotSize = chunkVertexBytes + chunkIndexBytes;
    slotSize_ = slotSize;
    staging_.init(vma_, device_, queueFamily_, stagingSlots, slotSize);

    // Timeline semaphore for completion tracking when the device supports it
    // (core in Vulkan 1.2+). Falls back to per-slot fences via isComplete().
    VkSemaphoreTypeCreateInfo tci{};
    tci.sType         = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
    tci.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
    tci.initialValue  = 0;
    VkSemaphoreCreateInfo sci{};
    sci.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    sci.pNext = &tci;
    if (vkCreateSemaphore(device_, &sci, nullptr, &m_timeline) == VK_SUCCESS) {
        m_timelineSupported = true;
        std::cout << "[UploadManager] timeline-semaphore completion path enabled\n";
    } else {
        m_timelineSupported = false;
        std::cout << "[UploadManager] timeline unavailable; using per-slot fences\n";
    }
}

void UploadManager::enqueue(UploadJob&& job) {
    queues_[(size_t)job.category].push(std::move(job));
}

std::mutex& UploadManager::pickMutex() {
    // Must serialize vkQueueSubmit2 with the SAME mutex the app uses for the
    // chosen queue, or we race with the app's own submissions to that queue.
    if (queue_ == app_->transferQueue)        return app_->transferSubmitMutex;
    if (queue_ == app_->geometryQueue &&
        app_->geometryQueue != app_->graphicsQueue) return app_->geometrySubmitMutex;
    return app_->graphicsSubmitMutex;
}

bool UploadManager::isComplete(const StagingSlot& s) const {
    // Completion is tracked by the binary submit fence (s.fence), which is
    // signaled at the end of the same vkQueueSubmit2 that also signals m_timeline.
    // Use the fence directly instead of vkGetSemaphoreCounterValue(m_timeline):
    // querying a timeline semaphore from the host triggers the validation layer's
    // UNASSIGNED-VkSemaphore-state-timeout false positive
    // (KhronosGroup/Vulkan-ValidationLayers#4968 / #8461), which aborts the app.
    // The fence is the reliable, validation-clean completion signal.
    return vkGetFenceStatus(device_, s.fence) == VK_SUCCESS;
}

VkSemaphore UploadManager::makeBinarySemaphore() {
    VkSemaphoreCreateInfo si{ VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
    VkSemaphore s = VK_NULL_HANDLE;
    if (vkCreateSemaphore(device_, &si, nullptr, &s) != VK_SUCCESS)
        throw std::runtime_error("UploadManager: vkCreateSemaphore failed");
    return s;
}

void UploadManager::submitJob(StagingSlot& s, UploadJob&& job) {
    char* base = static_cast<char*>(s.mapped);
    VkDeviceSize off = 0;

    if (vkResetCommandBuffer(s.cmd, 0) != VK_SUCCESS)
        throw std::runtime_error("UploadManager: failed to reset upload command buffer");
    VkCommandBufferBeginInfo bi{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (vkBeginCommandBuffer(s.cmd, &bi) != VK_SUCCESS)
        throw std::runtime_error("UploadManager: failed to begin upload command buffer");

    // WRITE-AFTER-READ/WRITE guard: destination buffers (e.g. IndirectRenderer's
    // merged vertex/index buffers) may still be read by a previous frame's
    // draw (vertex/index/indirect input) submitted earlier on THIS queue, or
    // written by a previous transfer into the same region (in-place slot
    // updates). The upload queue is the graphics-family geometry queue (same
    // queue as draws on this device), so an execution+memory barrier here
    // correctly orders those prior reads/writes before the transfer writes.
    // Harmless for fresh chunk buffers that had no prior access.
    //
    // srcStageMask must be ALL_COMMANDS (not just VERTEX_ATTRIBUTE_INPUT):
    // sync validation attributes a draw's vertex-attribute reads to the whole
    // pipeline span of the draw (TESS_EVAL, GEOMETRY, FRAGMENT, COLOR_ATTACHMENT
    // OUTPUT, ... — observed for the tessellated solid pipeline), so a barrier
    // scoped only to VERTEX_ATTRIBUTE_INPUT|INDEX_INPUT|DRAW_INDIRECT leaves
    // the transfer write unsynchronized against those reads
    // (SYNC-HAZARD-WRITE-AFTER-READ, aborting GPU-assisted runs). The access
    // masks still restrict the ordering to actual read accesses; ALL_COMMANDS
    // only widens the stage span.
    {
        VkMemoryBarrier2 mb{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
        mb.srcStageMask  = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
        mb.srcAccessMask = VK_ACCESS_2_VERTEX_ATTRIBUTE_READ_BIT |
                           VK_ACCESS_2_INDEX_READ_BIT |
                           VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT |
                           VK_ACCESS_2_TRANSFER_WRITE_BIT |
                           VK_ACCESS_2_TRANSFER_READ_BIT;
        mb.dstStageMask  = VK_PIPELINE_STAGE_2_COPY_BIT;
        mb.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;

        VkDependencyInfo dep{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
        dep.memoryBarrierCount = 1;
        dep.pMemoryBarriers    = &mb;
        vkCmdPipelineBarrier2(s.cmd, &dep);
    }

    for (auto& u : job.uploads) {
        const VkDeviceSize sz = u.cpuData.size();
        if (sz == 0) continue;
        std::memcpy(base + off, u.cpuData.data(), sz);
        // Flush the written range so the GPU observes the bytes (covers
        // non-coherent host-visible memory).
        if (vmaFlushAllocation(vma_, s.alloc, off, sz) != VK_SUCCESS)
            throw std::runtime_error("UploadManager: vmaFlushAllocation failed");

        VkBufferCopy copy{};
        copy.srcOffset = off;
        copy.dstOffset = u.dstOffset;
        copy.size      = sz;
        vkCmdCopyBuffer(s.cmd, s.buffer, u.dst.buffer, 1, &copy);
        off += sz;
    }
    if (vkEndCommandBuffer(s.cmd) != VK_SUCCESS)
        throw std::runtime_error("UploadManager: failed to end upload command buffer");

    // Signal the timeline (used for recycling/ordering) AND always a fresh binary
    // semaphore for cross-queue frame synchronization. We always register the
    // binary semaphore via addExtraWaitSemaphore so the integration needs NO
    // modification to drawFrame; the timeline remains an internal optimization.
    // A fresh binary semaphore per submission is required: reusing one would stay
    // permanently signaled and let a later frame render before the slot's NEW
    // transfer finished.
    uint64_t tlValue = 0;
    if (m_timelineSupported) tlValue = ++m_timelineSignal;

    std::array<VkSemaphoreSubmitInfo, 2> sig{};
    uint32_t sigCount = 0;

    s.signalSem = makeBinarySemaphore();
    VkSemaphoreSubmitInfo bin{ VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO };
    bin.semaphore = s.signalSem;
    bin.value     = 0;
    bin.stageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
    sig[sigCount++] = bin;

    if (m_timelineSupported) {
        VkSemaphoreSubmitInfo tl{ VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO };
        tl.semaphore   = m_timeline;
        tl.value       = tlValue;
        tl.stageMask   = VK_PIPELINE_STAGE_2_COPY_BIT;
        tl.deviceIndex = 0;
        sig[sigCount++] = tl;
    }

    VkCommandBufferSubmitInfo cbs{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO };
    cbs.commandBuffer = s.cmd;

    VkSubmitInfo2 si{ VK_STRUCTURE_TYPE_SUBMIT_INFO_2 };
    si.commandBufferInfoCount    = 1;
    si.pCommandBufferInfos       = &cbs;
    si.signalSemaphoreInfoCount  = sigCount;
    si.pSignalSemaphoreInfos     = sig.data();

    {
        std::lock_guard<std::mutex> lk(pickMutex());
        SubmissionTracker::record("upload");
        VkResult r = vkQueueSubmit2(queue_, 1, &si, s.fence);
        if (r != VK_SUCCESS)
            throw std::runtime_error("UploadManager: vkQueueSubmit2 failed");
    }

    s.busy          = true;
    s.waitRegistered = false;   // fresh binary semaphore: needs one frame-wait registration
    s.onComplete    = std::move(job.onComplete);
    s.chunkSlot     = job.chunkSlot;
}

void UploadManager::prepareFrameWaits(VulkanApp* app) {
    // Always register each in-flight upload's (fresh) binary semaphore with the
    // frame submit via the engine's cross-queue hazard mechanism. This makes the
    // integration drop-in: no drawFrame modification is required. (The timeline,
    // when available, is used only for recycling/ordering.)
    for (auto& s : staging_.slots()) {
        // A binary semaphore may be waited on exactly once. A slot can stay busy
        // for several frames (transfer not yet complete / not yet recycled); only
        // the FIRST frame after submission must wait on it. Subsequent frames on
        // the same graphics queue are naturally ordered after that frame, so they
        // need no additional wait. Re-adding the (already-consumed, never
        // re-signaled) binary semaphore would violate VUID-vkQueueSubmit2-semaphore-03873.
        if (s.busy && !s.waitRegistered && s.signalSem != VK_NULL_HANDLE) {
            app->addExtraWaitSemaphore(s.signalSem,
                VK_PIPELINE_STAGE_2_VERTEX_INPUT_BIT |
                VK_PIPELINE_STAGE_2_INDEX_INPUT_BIT |
                VK_PIPELINE_STAGE_2_VERTEX_ATTRIBUTE_INPUT_BIT |
                VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT);
            s.waitRegistered = true;
        }
    }
}

void UploadManager::processUploads() {
    // 1) Recycle completed staging slots on the MAIN thread. Non-blocking.
    for (auto& s : staging_.slots()) {
        if (!s.busy) continue;
        if (isComplete(s)) {
            // onComplete publishes the new buffers into the live scene and is
            // responsible for returning the PREVIOUS chunk's slot to chunkPool_
            // via app->deferDestroyUntilAllPending (so it is not reused while a
            // frame in flight still reads it). The manager never touches the
            // chunk slot — the slot just uploaded is now the chunk's live data.
            if (s.onComplete) s.onComplete();
            // Destroy the binary signal semaphore the manager created for this
            // job IF the app never took ownership of it via a frame-wait
            // registration (prepareFrameWaits). That is exactly the case for
            // jobs submitted AND completed inside flush(), which recycles slots
            // but never registers waits — without this, staging reset() nulls
            // the handle and it leaks at device teardown
            // (VUID-vkDestroyDevice-device-05137). Registered semaphores are
            // owned by the app's pending-destroy list and MUST NOT be destroyed
            // here (double free). Deferred so any GPU work still completes first.
            if (s.signalSem != VK_NULL_HANDLE && !s.waitRegistered) {
                VkSemaphore ss = s.signalSem;
                app_->deferDestroyUntilAllPending(
                    [ss, dev = device_, uploadQueue = queue_]() {
                        // The upload fence has signaled (isComplete() above), so
                        // this binary semaphore's signal operation has completed
                        // on the GPU. A binary semaphore's signaled state can
                        // only be consumed by a QUEUE wait: vkWaitSemaphores() is
                        // illegal for binary semaphores — it accepts only timeline
                        // semaphores (VUID-VkSemaphoreWaitInfo-pSemaphores-03256).
                        // We therefore submit an empty drain batch on the upload
                        // queue that waits on the semaphore and idle the queue;
                        // because the signal already completed, the drain
                        // completes instantly and destroying the semaphore is
                        // legal (VUID-vkDestroySemaphore-semaphore-05149). If the
                        // drain submit fails (device lost), the semaphore may still
                        // be referenced by a cancelled batch — skip the destroy
                        // (leak at teardown; the OS reclaims it).
                        if (consumeBinarySignal(dev, uploadQueue, ss))
                            vkDestroySemaphore(dev, ss, nullptr);
                    });
                s.signalSem = VK_NULL_HANDLE;
            }
            staging_.reset(s);
        }
    }

    // 2) Submit as many queued jobs as free staging slots allow. No per-frame cap.
    //    Each slot, pick the HIGHEST-PRIORITY ready job across all three category
    //    queues (nearest-first streaming) while keeping categories independent in
    //    generation. Losers are re-pushed to the back of their own queue (lock-free).
    for (;;) {
        StagingSlot* s = staging_.acquireFree();
        if (!s) break;                                     // all slots busy → stop, continue next frame

        // Pull one candidate from every non-empty category queue.
        UploadJob best;
        float     bestPri = std::numeric_limits<float>::lowest();
        int       bestCat = -1;
        bool      any = false;
        for (size_t c = 0; c < (size_t)StreamCategory::Count; ++c) {
            UploadJob cand;
            if (queues_[c].tryPop(cand)) {
                any = true;
                if (cand.priority >= bestPri) {            // >= keeps first-seen as tie-break
                    if (bestCat >= 0) queues_[bestCat].push(std::move(best)); // re-push loser
                    best = std::move(cand);
                    bestPri = cand.priority;
                    bestCat = (int)c;
                } else {
                    queues_[c].push(std::move(cand));       // re-push lower-priority
                }
            }
        }
        if (!any) break;                                   // all queues drained → stop

        // Guard against a malformed job with no data: complete it inline and
        // return the acquired chunk slot to the pool (it never became live data).
        bool hasData = false;
        for (auto& u : best.uploads) if (!u.cpuData.empty()) { hasData = true; break; }
        if (!hasData) {
            if (best.onComplete) best.onComplete();
            if (best.chunkSlot) chunkPool_.release(static_cast<ChunkGPUBuffers*>(best.chunkSlot));
            continue;
        }
        submitJob(*s, std::move(best));
    }
}

void UploadManager::flush() {
    // Guarantees no queued/in-flight UploadJob still references a destination
    // buffer the caller is about to destroy. Blocking is acceptable: this is
    // only invoked on a full rebuild that reallocates the target buffers.
    for (;;) {
        // Recycle any finished slots (fires onComplete) and admit as many
        // queued jobs as free staging slots allow.
        processUploads();

        bool queuedRemain = false;
        for (auto& q : queues_) if (!q.empty()) { queuedRemain = true; break; }
        bool busyRemain = false;
        for (auto& s : staging_.slots()) if (s.busy) { busyRemain = true; break; }

        if (!queuedRemain && !busyRemain) break;

        // Block on the in-flight transfers so the next iteration can recycle
        // their slots and admit the still-queued jobs.
        for (auto& s : staging_.slots())
            if (s.busy) VulkanApp::waitFence(device_, s.fence);
    }
}

void UploadManager::destroy() {
    // Wait for in-flight transfers before tearing down staging resources.
    for (auto& s : staging_.slots()) {
        if (s.busy) {
            VulkanApp::waitFence(device_, s.fence);
            if (s.onComplete) s.onComplete();
            if (s.chunkSlot) chunkPool_.release(static_cast<ChunkGPUBuffers*>(s.chunkSlot));
        }
        // Resolve binary-signal-semaphore ownership before StagingBufferPool::
        // destroy() (which unconditionally destroys any non-null signalSem):
        //  - registered → owned by the app's pending-destroy list, destroyed in
        //    VulkanApp::cleanup(); drop our handle so it is not double-freed.
        //  - unregistered → the app never learned about it; destroy it here (the
        //    fences above guarantee the GPU is done with it).
        if (s.signalSem != VK_NULL_HANDLE) {
            if (!s.waitRegistered) {
                // The fence waits above guarantee the signal operation has
                // completed, but the signal state was never consumed by a wait
                // (this slot was never registered with the frame submit). A
                // binary semaphore's signaled state can only be consumed by a
                // QUEUE wait: vkWaitSemaphores() is illegal for binary
                // semaphores — it accepts only timeline semaphores
                // (VUID-VkSemaphoreWaitInfo-pSemaphores-03256). Drain the
                // upload queue with an empty batch waiting on the semaphore and
                // idle it; the drain completes instantly because the signal
                // already completed, making destruction legal
                // (VUID-vkDestroySemaphore-semaphore-05149). If the drain
                // submit fails (device lost), the semaphore may still be
                // referenced by a cancelled batch — skip the destroy (leak at
                // teardown; the OS reclaims it).
                if (consumeBinarySignal(device_, queue_, s.signalSem))
                    vkDestroySemaphore(device_, s.signalSem, nullptr);
            }
            s.signalSem = VK_NULL_HANDLE;
        }
    }
    // Drain any remaining queued jobs so the MPSCQueue destructors find empty
    // queues. Without this, ~MPSCQueue() would try to destroy UploadJobs whose
    // BufferUpload data references chunk buffers we are about to free, causing
    // heap corruption during shutdown.
    for (auto& q : queues_) {
        UploadJob job;
        while (q.tryPop(job)) {}
    }
    staging_.destroy();
    chunkPool_.destroy();
    if (m_timeline) { vkDestroySemaphore(device_, m_timeline, nullptr); m_timeline = VK_NULL_HANDLE; }
}

// ----------------------------------------------------------------------------
// TerrainStreamer
// ----------------------------------------------------------------------------

void TerrainStreamer::init(VulkanApp* app,
                           VkDeviceSize chunkVertexBytes,
                           VkDeviceSize chunkIndexBytes,
                           uint32_t stagingSlots,
                           uint32_t initialChunkSlots,
                           uint32_t workersPerCategory) {
    upload_.init(app, chunkVertexBytes, chunkIndexBytes, stagingSlots, initialChunkSlots);
    for (auto& p : pools_) {
        // ThreadPool owns a std::mutex → not movable; store via unique_ptr.
        p = std::make_unique<ThreadPool>(workersPerCategory);
    }
}

void TerrainStreamer::requestMesh(StreamCategory category,
                                  uint64_t chunkId,
                                  int lod,
                                  std::function<void(ChunkGPUBuffers&, UploadJob&)> generator) {
    // Acquire the destination GPU slot on the MAIN thread (Vulkan allocation
    // happens here, never on a worker). The worker only does CPU meshing.
    ChunkGPUBuffers* slot = upload_.chunkPool().acquire(chunkId);

    pools_[(size_t)category]->enqueueDetached(
        [this, category, chunkId, lod, slot, generator]() {
            UploadJob job;
            job.category  = category;
            job.chunkId   = chunkId;
            job.lod       = lod;
            job.chunkSlot = slot;
            // CPU-only mesh generation (Tesselator / water SDF / brush system).
            // Worker never touches Vulkan — it only fills `job.cpuData`.
            generator(*slot, job);
            upload_.enqueue(std::move(job));
        });
}

void TerrainStreamer::destroy() {
    for (auto& p : pools_) p.reset();   // ThreadPool destructor joins workers
    upload_.destroy();
}

} // namespace streaming
