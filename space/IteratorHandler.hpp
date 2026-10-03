#pragma once
#include <functional>

#include "OctreeNodeData.hpp"
#include "Octree.hpp"

class ThreadPool;

class IteratorHandler {
public:
    void iterateMultiThreaded(
        const Octree &tree, 
        OctreeNodeData &params, 
        ThreadPool& pool,
        const Octree::IterateHandler &iterateHandler, 
        const Octree::IterateOrderHandler &getOrderHandler,
        const Octree::IterateThreadedHandler &iterateThreadedHandler
    );
};
