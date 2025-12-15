#pragma once

template <typename T> class InstanceBuilder : public IteratorHandler{
    InstanceBuilderHandler<T> * handler;
    std::vector<T> * instances;
    ThreadContext * context;
public:
    InstanceBuilder(InstanceBuilderHandler<T> * handler, std::vector<T> * instances, ThreadContext * context);
    void before(const Octree &tree, OctreeNodeData &params);
    void after(const Octree &tree, OctreeNodeData &params);
    bool test(const Octree &tree, OctreeNodeData &params);
    void getOrder(const Octree &tree, OctreeNodeData &params, uint8_t order[8]);
};