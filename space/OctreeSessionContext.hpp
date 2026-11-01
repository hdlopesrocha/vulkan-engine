#pragma once

#include "ChunkHistory.hpp"

// State that lives for one octree editing session, i.e. across a sequence of
// Octree::apply calls. apply() runs under the octree write lock, so the same
// context can serve whichever thread performs an apply; nothing here is
// touched by the parallel shape tasks.
//
// The chunk history is session-scoped (not per-thread): it must survive
// applies and must describe the whole tree, so it cannot be split across
// worker threads. The per-thread ShapeSdfCache stays in OctreeThreadContext,
// where the parallel evaluations can each keep their own values.
class OctreeSessionContext {
public:
    ChunkHistory chunkHistory;
};
