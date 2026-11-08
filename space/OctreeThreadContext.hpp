#pragma once

#include "ShapeSdfCache.hpp"

// Per-thread state for one Octree::apply: the shape-SDF value cache. A context
// is reused across applies (thread_local worker contexts), so the cache is
// generation-tagged; see ShapeSdfCache for the exactness and lifetime rules.
class OctreeThreadContext {
public:
    ShapeSdfCache shapeSdfCache;
};
