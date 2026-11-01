#pragma once

#include "ShapeSdfCache.hpp"

// Per-thread state for one Octree::apply: the shape-SDF value cache. A context
// is reused across applies (the apply thread and the thread_local worker
// contexts), so the cache is generation-tagged; see ShapeSdfCache for the
// exactness and lifetime rules.
//
// The chunk-level work-unit history is NOT per-thread: it lives in the
// octree's OctreeSessionContext (space/OctreeSessionContext.hpp), because it
// spans applies and must describe the whole tree.
class OctreeThreadContext {
public:
    ShapeSdfCache shapeSdfCache;
};
