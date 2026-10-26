#pragma once

#include <cstdint>

// Sizing demand handed to the target provider before the single geometry
// dispatch. Both capacities are safe upper bounds derived on the CPU while the
// chunk topology is flattened, so the GPU pass never needs a separate counting
// walk: it appends indices directly and writes each referenced vertex payload
// once.
struct OctreeTopologyDemand {
    uint32_t crossEdgeCount = 0;           // (scan cell, crossing cube edge) occurrences
    uint32_t maxSegmentsPerCrossEdge = 1;  // largest per-edge dyadic interval bound
    uint32_t suggestedVertexCapacity = 0;  // dense surface-node slot address space
    uint32_t suggestedIndexCapacity = 0;   // cross-edge topology bound (indices)
};
