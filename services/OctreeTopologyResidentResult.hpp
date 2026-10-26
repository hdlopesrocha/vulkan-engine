#pragma once

#include <cstdint>

// Result of one GPU topology build. Only counts come back to the host — the
// Cartesian vertex stream stays in the renderer's device-local pools.
struct OctreeTopologyResidentResult {
    uint32_t poolVertexCount = 0;     // highest referenced slot + 1 (the published extent)
    uint32_t writtenVertexCount = 0;  // distinct vertex payloads actually written
    uint32_t indexCount = 0;          // index stream length actually appended
    uint32_t slotIndex = UINT32_MAX;
    uint32_t baseVertex = 0;      // pool element index of the first span vertex
    uint32_t firstIndex = 0;      // pool element index of the first span index
    // Capacity plan and overflow telemetry for the per-chunk profile.
    uint32_t crossEdgeCount = 0;
    uint32_t maxSegmentsPerCrossEdge = 1;
    uint32_t vertexCapacity = 0;  // reserved vertex span (dense slot bound)
    uint32_t indexCapacity = 0;   // reserved index span (cross-edge bound)
    uint32_t vertexExtent = 0;    // max referenced slot + 1 (before trimming)
    uint32_t indexOverflow = 0;   // triangles that did not fit the index span
    uint32_t vertexOverflow = 0;  // triangles whose vertex slot did not fit
    uint32_t invalidSlots = 0;    // corners whose node had no host-assigned slot
    uint32_t attempts = 1;        // dispatches submitted (1 = no overflow growth)
    bool grew = false;            // the span was released and re-reserved bigger
    bool ok = false;              // pool span fully written: safe to draw
};
