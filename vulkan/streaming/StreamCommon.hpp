#pragma once

#include <vulkan/vulkan.h>
#include "../Buffer.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>
#include <memory>
#include <functional>

namespace streaming {

// Which subsystem produced a given upload. Keeping categories separate lets
// solid / water / brush meshes be generated and streamed fully in parallel.
enum class StreamCategory : uint8_t {
    Solid = 0,
    Water = 1,
    Brush = 2,
    Count = 3
};

// One GPU destination buffer plus the exact CPU bytes that must be copied into it.
// dst is a device-local buffer whose handle was created on the MAIN thread
// (ChunkBufferPool), so worker threads only ever touch `cpuData` (plain RAM).
struct BufferUpload {
    Buffer        dst{};              // destination device-local buffer (EXCLUSIVE, graphics family)
    VkDeviceSize  dstOffset = 0;      // byte offset inside dst
    std::vector<std::byte> cpuData;   // CPU mesh data; moved through the queue, never copied on the hot path
};

// A fully-specified upload job. Constructed on a worker thread (CPU only) and
// pushed into a lock-free queue; consumed later by UploadManager on the main
// thread, which copies `cpuData` into staging and records the transfer.
struct UploadJob {
    StreamCategory category = StreamCategory::Solid;
    uint64_t       chunkId  = 0;      // metadata for completion callback (swap into scene graph)
    int            lod      = 0;

    // Scheduling priority: HIGHER uploads sooner. Terrain streaming typically
    // sets this to -distance² from the camera so the nearest chunks stream in
    // first regardless of category. Generation order from the worker pools is
    // non-deterministic, so the manager uses this to pick the next job globally.
    float          priority = 0.0f;

    // One per destination buffer (typically: vertex buffer + index buffer).
    std::vector<BufferUpload> uploads;

    // Identifies the ChunkBufferPool slot this job fills. The manager releases
    // the slot back to the pool once the GPU transfer retires.
    void* chunkSlot = nullptr;

    // Invoked ONCE on the main thread when the GPU copy is complete. Use it to
    // publish the new buffers into the live scene and retire the old ones via
    // app->deferDestroyUntilFence / app->deferDestroyUntilAllPending.
    std::function<void()> onComplete;
};

// Total bytes a job will need inside a staging buffer (vertex + index).
inline VkDeviceSize jobStagingFootprint(const UploadJob& job) {
    VkDeviceSize total = 0;
    for (const auto& u : job.uploads) total += u.cpuData.size();
    return total;
}

} // namespace streaming
