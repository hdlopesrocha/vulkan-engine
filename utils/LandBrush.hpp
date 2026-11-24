#pragma once

#include <glm/glm.hpp>

// CPU twin of shaders/includes/octree/LandBrush.glsl.
//
// Terrain nodes are painted with the LandBrush sentinel (LAND_BRUSH_INDEX,
// 255): the actual material depends on the solved surface position/normal and
// must be classified per vertex. The GPU chunk path (OctreeTriangles.comp ->
// nodeResolvedBrush) does this before writing the vertex payload. The runtime
// no longer builds GPU terrain chunks, but vegetation placement still walks the
// octree on the CPU, so it needs the same classification to know which
// triangles are grass-bearing.
namespace landbrush {

constexpr int kLandBrushIndex = 255;
constexpr int kNormalBrushIndex = 254;
constexpr int kUnderground = -1;
constexpr int kRock = 7;
constexpr int kSand = 8;
constexpr int kSnow = 9;
constexpr int kSoftSand = 10;

// Material index for a LAND_BRUSH_INDEX vertex, from its world position and
// (unnormalized) surface normal.
int materialIndex(const glm::vec3& pos, const glm::vec3& normal);

// Resolve a stored node brush into the value the GPU writes into the vertex:
//   - LAND_BRUSH_INDEX  -> classified terrain material (underground -> rock)
//   - NORMAL_BRUSH_INDEX-> 0 when the normal faces +Y, DISCARD otherwise
//   - anything else     -> unchanged
int resolveVertexMaterial(int brush, const glm::vec3& pos,
                          const glm::vec3& normal);

} // namespace landbrush
