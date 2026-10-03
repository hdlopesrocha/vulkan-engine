#pragma once

#include "../vulkan/renderer/ChunkManager.hpp"
#include "../utils/LocalScene.hpp"
#include <memory>

// The World owns all terrain data and chunk state.
//
// Responsibility boundary:
//   WORLD owns: Octrees (SDF), ChunkManager state machine
//   RENDERER owns: GPU resources, indirect draw, slot data
//
// The World knows nothing about Vulkan.
class World {
public:
    // Unique identifier for a chunk (typically the octree node ID cast to uint64).
    using ChunkId = uint64_t;

    World();
    ~World();

    // No copying.
    World(const World&) = delete;
    World& operator=(const World&) = delete;

    // Explicitly stop all scene thread pools.  Must be called before any
    // objects captured by enqueued pool tasks are destroyed (e.g. before
    // the SceneRenderer that registered change-handler callbacks).
    void stopPools();

    // ── Scene/octree access ─────────────────────────────────────────────────

    // The underlying scene with octrees.
    LocalScene& scene() { return *scene_; }
    const LocalScene& scene() const { return *scene_; }

    // ── Brush scene (separate scene for editing previews) ────────────────────
    // The brush scene has its own octrees for brush preview geometry. It is
    // managed separately from the main scene and uses its own IndirectRenderer.
    // Returns nullptr until createBrushScene() is called.
    LocalScene* brushScene() { return brushScene_.get(); }
    const LocalScene* brushScene() const { return brushScene_.get(); }

    // Create (or recreate) the brush scene. Call once during setup.
    // The brush octrees are reset on each rebuildBrushScene call.
    void createBrushScene() {
        brushScene_ = std::make_unique<LocalScene>();
    }

    // ── Chunk manager (state machine for the async rebuild pipeline) ────────

    ChunkManager& chunkManager() { return chunkManager_; }
    const ChunkManager& chunkManager() const { return chunkManager_; }

private:
    // The main scene with octrees (SDF storage + meshing).
    std::unique_ptr<LocalScene> scene_;

    // The brush editing scene (separate octrees, no chunk tracking).
    std::unique_ptr<LocalScene> brushScene_;

    // Chunk state machine driving the async rebuild pipeline.
    ChunkManager chunkManager_;
};
