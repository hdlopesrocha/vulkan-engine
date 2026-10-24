#pragma once

#include <cstdint>

// Sizing demand handed to the target provider once the walk has run. The walk
// is submitted alone first because the index stream has exactly one entry per
// raw (pre-dedup) vertex and its span is reserved from that exact count.
struct OctreeTopologyDemand {
    uint32_t rawVertexCount = 0;      // 3 per emitted triangle (== index count)
    uint32_t maxUniqueVertices = 0;   // nodeCount: the dedup key space (one key per node)
    uint32_t suggestedVertexCapacity = 0; // min(rawVertexCount, maxUniqueVertices)
};
