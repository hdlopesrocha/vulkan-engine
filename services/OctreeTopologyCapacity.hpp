#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

// CPU-side sizing rules for the GPU topology pass.
//
// A "cross edge" is one scanned cube edge whose two stored corner samples have
// opposite signs. The walker emits one segment per interval the adaptive break
// collector produces along that edge, and each segment expands to a ring of at
// most four surface cells -> at most two triangles.
//
// The break collector only inserts boundaries of cells returned by findCellAt.
// A scan cell at stored ladder level `targetLod` stops that descent at nodes
// with lod <= targetLod, i.e. cells no smaller than the chunk's target-rung
// cell. The finest such cell in the flattened neighborhood therefore bounds
// how many dyadic intervals one target-cell edge can contain:
//
//   segments(cell) = cellSize / smallestFlattenedCellSize
//
// (a power of two, clamped to the shader's break-list capacity). This is a
// topology bound, not a tuned multiplier: every break coordinate is a cell
// boundary on that dyadic lattice, so no edge can produce more intervals. The
// per-cell span is summed over the chunk's crossing edges (rather than using
// the chunk's largest scan cell for every edge), which keeps the reservation
// tight when the rung contains a mix of cell sizes.
namespace octree_topology {

constexpr uint32_t kTrianglesPerSegment = 2;   // triangle fan of a 3/4 ring
constexpr uint32_t kIndicesPerTriangle = 3;
// OctreeTriangles.comp's global break list holds 64 entries including the two
// span endpoints. More intervals than that are reported as a walk overflow and
// the chunk is rejected before it can be drawn, so capacities need not grow
// beyond this bound.
constexpr uint32_t kMaxSegmentsPerCrossEdge = 64;

inline uint32_t segmentBound(uint32_t levelSpan) {
    if (levelSpan >= 6u) return kMaxSegmentsPerCrossEdge;
    return 1u << levelSpan;
}

// Number of dyadic intervals one scan cell's edge can be split into. Every
// break coordinate collectBreaks() can add is an octree-cube boundary, and all
// cubes are aligned to the smallest cell in the flattened neighborhood. For a
// scan cell of size `cellSize` that leaves at most `cellSize / smallestCellSize`
// intervals (a power of two because every cube size is a dyadic fraction of
// the root).
inline uint32_t segmentsForCellSize(float cellSize, float smallestCellSize) {
    if (!(cellSize > 0.0f) || !(smallestCellSize > 0.0f)) return 1u;
    const double ratio = static_cast<double>(cellSize) /
                         static_cast<double>(smallestCellSize);
    if (ratio <= 1.0) return 1u;
    // ceil(log2(ratio)); the tiny epsilon keeps an exact power of two from
    // rounding up to the next rung through log2's last-bit error.
    const int span = static_cast<int>(std::ceil(std::log2(ratio) - 1e-6));
    return segmentBound(static_cast<uint32_t>(std::max(span, 0)));
}

// Safe index-stream capacity for a pre-summed segment budget. Each segment
// expands to at most two triangles of three indices. The vertex side needs no
// equivalent formula here: its capacity is the dense surface-node slot address
// space (OctreeTopologyInput::activeVertexCount), which is the exact upper bound
// for the slot indices the shader can reference.
inline uint32_t indexCapacityFromSegments(uint64_t segmentCount) {
    const uint64_t value = segmentCount * kTrianglesPerSegment * kIndicesPerTriangle;
    return static_cast<uint32_t>(std::min<uint64_t>(
        value, std::numeric_limits<uint32_t>::max()));
}

} // namespace octree_topology
