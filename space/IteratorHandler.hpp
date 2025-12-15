#ifndef SPACE_ITERATOR_HANDLER_HPP
#define SPACE_ITERATOR_HANDLER_HPP

#include <stack>

#include "OctreeNodeData.hpp"
#include "StackFrame.hpp"
#include "StackFrameOut.hpp"

class Octree;
class ThreadPool;

class IteratorHandler {
    std::stack<OctreeNodeData> flatData;
    std::stack<StackFrame> stack;
    std::stack<StackFrameOut> stackOut;

public:
    virtual bool test(const Octree &tree, OctreeNodeData &params) = 0;
    virtual void before(const Octree &tree, OctreeNodeData &params) = 0;
    virtual void after(const Octree &tree, OctreeNodeData &params) = 0;
    virtual void getOrder(const Octree &tree, OctreeNodeData &params, uint8_t order[8]) = 0;
    void iterate(const Octree &tree, OctreeNodeData &params);
    void iterateMultiThreaded(const Octree &tree, OctreeNodeData &params);

    void iterateFlatIn(const Octree &tree, OctreeNodeData &params);
    void iterateFlatOut(const Octree &tree, OctreeNodeData &params);
    void iterateFlat(const Octree &tree, OctreeNodeData &params);
    void iterateBFS(const Octree &tree, OctreeNodeData &rootParams);
    void iterateParallelBFS(const Octree &tree, OctreeNodeData &rootParams, ThreadPool& pool);
};

#endif
