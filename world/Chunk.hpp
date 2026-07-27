#pragma once

#include <cstdint>
#include <memory>
#include <atomic>

class RenderProxy;

// A single terrain chunk owned by the World.
//
// Chunk has NO knowledge of GPU resources, Vulkan, or rendering internals.
// It stores only:
//   - A unique identifier
//   - A pointer to its current immutable RenderProxy (set by the renderer)
//   - World-space bounds
//   - Layer (opaque/transparent)
//   - A dirty flag for the world to track changes
//
// The RenderProxy pointer is atomically swappable so the render thread can
// read it without locking while the world/upload thread writes it.
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

    // The currently active RenderProxy (immutable, may be null).
    // Set atomically by the renderer after GPU upload completes.
    std::shared_ptr<const RenderProxy> currentProxy;

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
