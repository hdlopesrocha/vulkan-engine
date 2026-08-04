#pragma once

#include <cstdint>

// Describes the lifecycle of a chunk mesh from dirtied through
// CPU generation, GPU upload, and finally being swapped into the
// live render set. Transitions are driven by the async job pipeline
// and are thread-safe via atomics/queues (not a global lock).
enum class ChunkState : uint8_t {
    // No pending work; the chunk's current mesh (if any) is live.
    Clean,

    // A change was detected; the chunk has been added to the dirty
    // queue but no worker has picked it up yet.
    Queued,

    // A worker thread is currently running Surface Nets meshing on
    // the chunk's octree node. The CPU result is not yet ready.
    BuildingCPU,

    // Mesh data has been produced and a GPU upload is in flight
    // (either via the UploadManager or the legacy staging path).
    UploadingGPU,

    // The new mesh is fully built (CPU + GPU) and ready to
    // be swapped into the live scene. The main thread
    // will perform the swap on its next call to processDirtyChunks.
    ReadyToSwap
};
