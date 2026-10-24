#pragma once

#include <cstdint>

// Result of one GPU topology build. Only counts come back to the host — the
// Cartesian vertex stream stays in the renderer's device-local pools.
struct OctreeTopologyResidentResult {
    uint32_t poolVertexCount = 0; // deduped vertices written into the pool span
    uint32_t indexCount = 0;      // index stream length (== raw vertex count)
    uint32_t slotIndex = UINT32_MAX;
    bool ok = false;              // pool span fully written: safe to draw
};
