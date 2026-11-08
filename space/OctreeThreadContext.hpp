#pragma once

#include "ShapeSdfCache.hpp"
#include "ChunkHistory.hpp"

// Per-thread state for one Octree::apply: the shape-SDF value cache and the
// chunk-level work-unit history. A context is reused across applies (the
// apply owner's context and the thread_local worker contexts), so both
// structures are generation/refresh tagged:
//   - ShapeSdfCache is generation-tagged; see its exactness and lifetime rules.
//   - ChunkHistory is refreshed structurally at the start of each apply, so it
//     never hands a stale node pointer to a different (or reset) tree.
class OctreeThreadContext {
public:
    ShapeSdfCache shapeSdfCache;
    ChunkHistory chunkHistory;
};
