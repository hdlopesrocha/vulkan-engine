#pragma once
#include "IteratorHandler.hpp"
#include "Octree.hpp"
#include "ThreadPool.hpp"
#include "ForwardingHandler.hpp"
#include <vector>
#include <unordered_set>

class Processor : public IteratorHandler {
    ThreadPool &threadPool;
    ThreadContext * context;
    long * count;
    ForwardingHandler fh;
    int targetLod = -1;
    float firstCellSize = 0.0f;
    float * cellSizeOut = nullptr;

public:
    Processor(long * count, ThreadPool &threadPool, ThreadContext * context, std::vector<OctreeNodeTriangleHandler*> * handlers, int targetLod_ = -1, float * cellSizeOut_ = nullptr);
    bool iterate(const Octree &tree, OctreeNodeData &params) override;
    void getOrder(const Octree &tree, OctreeNodeData &params, uint8_t order[8]) override;
    void virtualize(Octree * tree, const BoundingCube &cube, float * sdf, uint level, uint levels);
    void resetCellSizeCapture() { firstCellSize = 0.0f; }
    float getFirstCellSize() const { return firstCellSize; }
};

 
