#pragma once

#include "OctreeNodeTriangleHandler.hpp"
#include "OctreeThreadContext.hpp"
#include "../math/geometry/Geometry.hpp"
#include "../math/geometry/Vertex.hpp"

// Collects the triangles of ONE tessellated node (one chunk or one ladder
// ancestor). Each node gets its own Tesselator instance, so the class stays a
// plain single-geometry collector. Used only by the offline Blender exporter;
// the engine runtime's chunk vertices are written on the GPU.
class Tesselator : public OctreeNodeTriangleHandler{
public:
    Geometry geometry;
    Tesselator(long * count);
    void handle(OctreeNode &o0, OctreeNode &o1, OctreeNode &o2,
                const CellVertex &v0, const CellVertex &v1, const CellVertex &v2,
                float eps) override;
};
