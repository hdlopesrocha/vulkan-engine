#ifndef SPACE_ITERATOR_HPP
#define SPACE_ITERATOR_HPP

#include "ThreadPool.hpp"
#include "IteratorHandler.hpp"
#include <stack>

template <typename T> class GeometryBuilder {
public:
    virtual InstanceGeometry<T> * build(Octree * tree, OctreeNodeData &params, ThreadContext * context) = 0;
};

template <typename T> class InstanceBuilderHandler {
public:
    virtual void handle(const Octree &tree, OctreeNodeData &data, std::vector<T> * instances, ThreadContext * context) = 0;
};

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

#include "IteratorHandler.hpp"
#include "Tesselator.hpp"
#include "Processor.hpp"
#include "OctreeFile.hpp"
#include "OctreeNodeFile.hpp"
#include "OctreeVisibilityChecker.hpp"

#endif
