#pragma once

template <typename T> class InstanceBuilderHandler {
public:
    virtual void handle(const Octree &tree, OctreeNodeData &data, std::vector<T> * instances, ThreadContext * context) = 0;
};

