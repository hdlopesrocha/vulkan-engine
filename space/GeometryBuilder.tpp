#pragma once

template <typename T> class GeometryBuilder {
public:
    virtual InstanceGeometry<T> * build(Octree * tree, OctreeNodeData &params, ThreadContext * context) = 0;
};