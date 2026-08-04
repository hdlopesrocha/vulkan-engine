#pragma once

#include <cstdint>
#include <memory>
#include <atomic>

// A single terrain chunk owned by the World.
//
// Chunk has NO knowledge of GPU resources, Vulkan, or rendering internals.
// It stores only:
//   - A unique identifier
//   - World-space bounds
//   - Layer (opaque/transparent)
//   - A dirty flag for the world to track changes
//
// The renderer keeps GPU/mesh data for a chunk in its own slot state
// (IndirectRenderer + ChunkManager); Chunk stays GPU-free.
class Chunk {
public:
    using ChunkId = uint64_t;

    Chunk() = default;

    explicit Chunk(ChunkId id_)
        : id(id_) {}

    // Chunks are move-only.
    Chunk(const Chunk&) = delete;
    Chunk& operator=(const Chunk&) = delete;
    Chunk(Chunk&&) = default;
    Chunk& operator=(Chunk&&) = default;

    // ── Accessors ──────────────────────────────────────────────────────────

    ChunkId id = 0;

    // World-space center (set by the world when the chunk is created).
    float centerX = 0.0f;
    float centerY = 0.0f;
    float centerZ = 0.0f;

    // Layer hint for the renderer.
    // 0 = opaque/solid, 1 = transparent/water
    uint32_t layer = 0;

    // Octree version at last rebuild (for staleness detection).
    uint32_t version = 0;
};
