#pragma once

#include <algorithm>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>

#include "../../space/Model3DVersion.hpp"
#include "../../space/Octree.hpp"
#include "../../space/ThreadPool.hpp"
#include "PendingDeleteEntry.hpp"

// One entry per Scene layer (index = Layer). Each layer tracks its own chunk
// registry + deferred-delete slots + generation pool, while the two shared
// IndirectRenderers (solid/water) are selected per layer via the Scene's
// LayerRendererType (LayersWidget). States are heap-allocated so PublishTarget
// references stay stable across vector growth.
struct SceneLayerState {
    std::unordered_map<NodeID, Model3DVersion> chunks;
    std::unordered_map<NodeID, PendingDeleteEntry> pendingDeletes;
    std::unique_ptr<std::recursive_mutex> mutex;
    std::unique_ptr<ThreadPool> genPool;
    SceneLayerState()
        : mutex(std::make_unique<std::recursive_mutex>()),
          genPool(std::make_unique<ThreadPool>(std::max(2u, std::thread::hardware_concurrency() / 2))) {}
};
