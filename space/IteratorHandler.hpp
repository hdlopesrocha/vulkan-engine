#pragma once
#include <stack>
#include <functional>

#include "OctreeNodeData.hpp"
#include "StackFrame.hpp"
#include "StackFrameOut.hpp"
#include "Octree.hpp"

class ThreadPool;

class IteratorHandler {
    std::stack<OctreeNodeData> flatData;
    std::stack<StackFrameOut> stackOut;

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

    void iterateFlatIn(const Octree &tree, OctreeNodeData &params,
        const Octree::IterateHandler &iterateHandler, const Octree::IterateOrderHandler &getOrderHandler);
    void iterateFlatOut(const Octree &tree, OctreeNodeData &params,
        const Octree::IterateHandler &iterateHandler, const Octree::IterateOrderHandler &getOrderHandler);
    void iterateBFS(const Octree &tree, OctreeNodeData &rootParams,
        const Octree::IterateHandler &iterateHandler, const Octree::IterateOrderHandler &getOrderHandler);
    void iterateParallelBFS(const Octree &tree, OctreeNodeData &rootParams, ThreadPool& pool,
        const Octree::IterateHandler &iterateHandler, const Octree::IterateOrderHandler &getOrderHandler);
};
