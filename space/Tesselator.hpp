#pragma once

#include "OctreeNodeTriangleHandler.hpp"
#include "ThreadContext.hpp"
#include "../math/Geometry.hpp"
#include "../math/Vertex.hpp"

// Collects the triangles of ONE tessellated node (one chunk or one ladder
// ancestor). Each node gets its own Tesselator instance, so the class stays
// a plain single-geometry collector: per-node meshes are bucketed by level
// at the call sites (Processor for the chunk, LocalScene for the ancestors).
class Tesselator : public OctreeNodeTriangleHandler{
public:
    Geometry geometry;
    Tesselator(long * count);
    void handle(Vertex &v0, Vertex &v1, Vertex &v2) override;
};
