#pragma once

// CPU-side SDF renderer diagnostics (CPU-only, no GPU twin).
#include <cstdint>

struct SdfStats {
    uint32_t containerCount = 0;
    uint32_t definitionCount = 0;
    uint32_t materialCount = 0;
    uint32_t gridCellCount = 0;
    uint32_t lastDrawInstances = 0;
    uint64_t lastCellVisits = 0; // stub: needs query pool
    uint64_t lastFragments = 0;  // stub: needs query pool
    uint32_t lavaAnchors = 0;    // flame anchors from brush-4 lava chunks
    uint32_t lavaChunks = 0;     // lava-bearing chunks currently tracked
};
