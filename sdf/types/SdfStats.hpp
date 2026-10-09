#pragma once

// CPU-side SDF renderer diagnostics (CPU-only, no GPU twin).
#include <cstdint>

struct SdfStats {
    uint32_t containerCount = 0;
    uint32_t definitionCount = 0;
    uint32_t materialCount = 0;
    uint32_t gridCellCount = 0;
    uint32_t lastDrawInstances = 0;
    // GPU march counters (SdfProfileCounters, read back with a 3-frame slot
    // latency while the SDF march counters toggle is on; 0 otherwise).
    uint64_t lastCellVisits = 0;     // DDA cells with candidates
    uint64_t lastFragments = 0;      // fragment shader invocations (proxy raster pixels)
    uint64_t lastMarchSteps = 0;     // march loop iterations
    uint64_t lastCandidateEvals = 0; // sdfEvalInstance evaluations
    uint64_t lastHits = 0;           // shaded surface hits
    uint32_t lavaAnchors = 0;    // flame anchors from brush-4 lava chunks
    uint32_t lavaChunks = 0;     // lava-bearing chunks currently tracked
    uint32_t rockAnchors = 0;    // boulder anchors from brush-7 rock chunks
    uint32_t rockChunks = 0;     // rock-bearing chunks currently tracked
    uint32_t grassAnchors = 0;   // grass clumps derived from vegetation instances
    uint32_t grassChunks = 0;    // vegetation chunks currently tracked
};
