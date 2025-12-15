#ifndef SPACE_PROCESSOR_HPP
#define SPACE_PROCESSOR_HPP

#include "IteratorHandler.hpp"
#include "Octree.hpp"
#include "ThreadPool.hpp"
#include <unordered_set>
#include "../math/BoundingCubeHasher.hpp"

class Processor : public IteratorHandler {
    ThreadPool &threadPool;
    ThreadContext * context;
    std::vector<OctreeNodeTriangleHandler*> * handlers;
    std::unordered_set<BoundingCube,BoundingCubeHasher> iteratedCubes;

public:
    Processor(long * count, ThreadPool &threadPool, ThreadContext * context, std::vector<OctreeNodeTriangleHandler*> * handlers);
    void iterate(const Octree &tree, OctreeNodeData &params);
    void before(const Octree &tree, OctreeNodeData &params) override;
    void after(const Octree &tree, OctreeNodeData &params) override;
    bool test(const Octree &tree, OctreeNodeData &params) override;
    void getOrder(const Octree &tree, OctreeNodeData &params, uint8_t order[8]) override;
    void virtualize(Octree * tree, const BoundingCube &cube, float * sdf, uint level, uint levels);
};

#endif
