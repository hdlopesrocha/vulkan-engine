#pragma once
#include <stack>
#include <functional>

#include "OctreeNodeData.hpp"
#include "Octree.hpp"

class ThreadPool;

class IteratorHandler {
    std::stack<OctreeNodeData> flatData;

public:
    void iterate(const Octree &tree, OctreeNodeData &params,
        const Octree::IterateHandler &iterateHandler, const Octree::IterateOrderHandler &getOrderHandler);
    void iterateMultiThreaded(
        const Octree &tree, 
        OctreeNodeData &params, 
        ThreadPool& pool,
        const Octree::IterateHandler &iterateHandler, 
        const Octree::IterateOrderHandler &getOrderHandler,
        const Octree::IterateThreadedHandler &iterateThreadedHandler
    );
};
