#pragma once
#include "IteratorHandler.hpp"
#include "Octree.hpp"
#include "ThreadPool.hpp"
#include "../math/Geometry.hpp"
#include <functional>
#include <vector>
#include <unordered_set>

class Processor : public IteratorHandler {
    ThreadPool &threadPool;
    ThreadContext * context;
    long * count;
    // Cube of the chunk this request tessellates: the root walk prunes every
    // node off this chunk's root path, so only the chunk and its ancestors
    // emit meshes.
    const BoundingCube targetCube;
public:
    Processor(
        long * count, 
        ThreadPool &threadPool, 
        ThreadContext * context, 
        const BoundingCube &targetCube, 
        float * cellSizeOut_, 
        const GeometryLodCallback& ph_
    );
    // One Tesselator per node: called once per tessellated node with its
    // ladder level (the node's chunkLod), the node itself and the node's own
    // root-descended bounding cube (used for the tessellation bounds — no
    // reconstructed cubes), plus the node's geometry.
    
    bool iterate(const Octree &tree, OctreeNodeData &params) override;
    void getOrder(const Octree &tree, OctreeNodeData &params, uint8_t order[8]) override;
    void virtualize(Octree * tree, const BoundingCube &cube, float * sdf, uint level, uint levels);

private:
    GeometryLodCallback onGeometry;

};
