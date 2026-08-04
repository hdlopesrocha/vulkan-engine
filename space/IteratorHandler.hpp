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
    void iterateOctree(const Octree &tree, OctreeNodeData &params,
        Octree::IterateHandler iterateHandler, Octree::IterateOrderHandler getOrderHandler);
    void iterateMultiThreaded(const Octree &tree, OctreeNodeData &params, ThreadPool& pool,
        Octree::IterateHandler iterateHandler, Octree::IterateOrderHandler getOrderHandler);

    void iterateFlatIn(const Octree &tree, OctreeNodeData &params,
        Octree::IterateHandler iterateHandler, Octree::IterateOrderHandler getOrderHandler);
    void iterateFlatOut(const Octree &tree, OctreeNodeData &params,
        Octree::IterateHandler iterateHandler, Octree::IterateOrderHandler getOrderHandler);
    void iterateBFS(const Octree &tree, OctreeNodeData &rootParams,
        Octree::IterateHandler iterateHandler, Octree::IterateOrderHandler getOrderHandler);
    void iterateParallelBFS(const Octree &tree, OctreeNodeData &rootParams, ThreadPool& pool,
        Octree::IterateHandler iterateHandler, Octree::IterateOrderHandler getOrderHandler);
};
