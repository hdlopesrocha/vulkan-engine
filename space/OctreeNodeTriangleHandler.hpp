#pragma once
#include "OctreeNode.hpp"
#include "EdgeCell.hpp"

// Receives the triangles the octree walk emits. A triangle is passed as its
// three corner NODES plus the surface point/normal resolved for each corner
// (CellVertex). The walk never hands over a Vertex: the material (brush index,
// HSV tint, texture-coordinate technique, UV scale) lives on the node, and the
// point/normal are either the GPU-computed vertices or, on GPU-less paths, the
// cell's SDF solution. The handler is the one place that assembles the output
// Vertex (into a Geometry).
class OctreeNodeTriangleHandler {
public:
    long * count;
    OctreeNodeTriangleHandler(long * count);
    virtual void handle(OctreeNode &o0, OctreeNode &o1, OctreeNode &o2,
                        const CellVertex &v0, const CellVertex &v1, const CellVertex &v2,
                        float eps) = 0;
};
