#pragma once
#include "../math/Geometry.hpp"
#include <cstdint>

class BoundingCube;

// Decimate a triangle mesh by vertex clustering: interior vertices are welded
// onto a coarse grid of `clusterSize` (positions averaged per cell), while
// vertices within `borderStrip` of the chunk's AABB boundary keep their exact
// positions. Border preservation keeps adjacent chunks watertight at ANY LoD
// level: every level of a chunk shares the exact same boundary vertices, and
// neighbors do too, so seams match regardless of which levels meet.
//
// Triangles that collapse to fewer than 3 distinct clusters are dropped
// (degenerate slivers — invisible in practice for terrain at distance).
//
// Returns true and fills `out` when at least one triangle survives; returns
// false (out left untouched) when the mesh is already coarser than the grid
// (nothing left to weld), so callers can stop the ladder early.
bool decimateVertexCluster(const Geometry& in,
                           float clusterSize,
                           float borderStrip,
                           const BoundingCube& chunkBounds,
                           Geometry& out);
