#pragma once

#include <cstdint>

// Rolling stats for the GPU topology pass (dispatch + counters for one chunk
// is one "call"). Logged periodically so the pass's cost is visible.
struct OctreeTopologyStats {
    uint64_t calls = 0;
    double avgMs = 0.0;
    double avgMsLast = 0.0;
    double avgCells = 0.0;
    double avgNodes = 0.0;
    double avgTriangles = 0.0;
};
