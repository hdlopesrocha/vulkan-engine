#pragma once

// Publish routing for background-tessellated chunk meshes: where finished
// meshes are queued for main-thread GPU upload. Moved out of MyApp.cpp so
// the app entry stays orchestration-only; the space-change handler factory
// (still in MyApp.cpp) is the only consumer.

#include <mutex>
#include <unordered_map>

#include "../../space/Model3DVersion.hpp"
#include "../../space/Octree.hpp"
#include "../renderer/SceneRenderer.hpp"

struct PublishTarget {
    // Where finished meshes are queued for main-thread GPU upload. ONE shared
    // queue for both streams (solid + water); each entry is keyed by its
    // emitting octree node id and carries its own single LoDMesh (no ladder
    // structures anywhere).
    std::unordered_map<NodeID, SceneRenderer::PendingMeshData>& meshData;
    std::mutex& queueMutex;
    // Chunk registry whose entries are removed on delete (solid vs water).
    std::unordered_map<NodeID, Model3DVersion>& chunks;
    // Mutex guarding [chunks] (per-space, chosen by the caller).
    std::recursive_mutex& chunksMutex;
    // Deferred slot-registry on delete (solid/water) used in slotted mode.
    std::unordered_map<NodeID, SceneRenderer::PendingDeleteEntry>& deferredSlots;
};
