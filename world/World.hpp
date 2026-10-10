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

    // The scene the renderer consumes: the remote scene while connected,
    // otherwise the local one. MyApp sets this once at startup; it never
    // changes mid-frame (set before any dispatch, read any time after).
    void setActiveScene(Scene* scene) { activeScene_ = scene; }
    Scene& activeScene() { return activeScene_ ? *activeScene_ : *scene_; }
    const Scene& activeScene() const { return activeScene_ ? *activeScene_ : *scene_; }

    // ── Chunk manager (state machine for the async rebuild pipeline) ────────

    ChunkManager& chunkManager() { return chunkManager_; }
    const ChunkManager& chunkManager() const { return chunkManager_; }

private:
    // The main scene with octrees (SDF storage + meshing).
    std::unique_ptr<LocalScene> scene_;

    // Non-owning view of the rendered scene (remote or local). Never null
    // when read through activeScene() (falls back to scene_).
    Scene* activeScene_ = nullptr;

    // Chunk state machine driving the async rebuild pipeline.
    ChunkManager chunkManager_;
};
