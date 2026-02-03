#pragma once
#include "Simplifier.hpp"
#include "ThreadPool.hpp"
#include <unordered_set>
#include "ThreadContext.hpp"
#include "OctreeNodeFrame.hpp"
#include "OctreeNodeLevel.hpp"
#include "OctreeAllocator.hpp"
#include "OctreeChangeHandler.hpp"
#include "OctreeNodeTriangleHandler.hpp"
#include "ShapeArgs.hpp"
#include "OctreeSerialized.hpp"
#include "../sdf/WrappedSignedDistanceFunction.hpp"
#include <functional>
#include "../math/BoundingCube.hpp"

class IteratorHandler;

class Octree: public BoundingCube {
    using BoundingCube::BoundingCube;
public:
    float chunkSize;

    OctreeNode * root;
    typedef unsigned int uint;

    using IterateBorderHandler = std::function<void(const BoundingCube &childCube, const float sdf[8], uint level)>;
    OctreeAllocator * allocator;
    int threadsCreated;
    std::shared_ptr<std::atomic<int>> shapeCounter;
    tsl::robin_map<glm::vec3, ThreadContext> chunks;
    ThreadPool threadPool = ThreadPool(std::thread::hardware_concurrency());
    std::mutex mutex;

    Octree(BoundingCube minCube, float chunkSize);
    Octree();

    void expand(const ShapeArgs &args);
    void add(WrappedSignedDistanceFunction *function, const Transformation model, glm::vec4 translate, glm::vec4 scale, const TexturePainter &painter, float minSize, Simplifier &simplifier, OctreeChangeHandler &changeHandler);
    void del(WrappedSignedDistanceFunction *function, const Transformation model, glm::vec4 translate, glm::vec4 scale, const TexturePainter &painter, float minSize, Simplifier &simplifier, OctreeChangeHandler &changeHandler);
    void reset();
    NodeOperationResult shape(OctreeNodeFrame frame, const ShapeArgs &args, ThreadContext * threadContext, ContainmentType parentContainment);
    void iterate(IteratorHandler &handler);
    void iterateFlat(IteratorHandler &handler);
    void iterateParallel(IteratorHandler &handler);
    OctreeNodeLevel getNodeAt(const glm::vec3 &pos, int level, bool simplification) const;
    OctreeNode* getNodeAt(const glm::vec3 &pos, bool simplification) const;
    float getSdfAt(const glm::vec3 &pos);
    void handleQuadNodes(const BoundingCube &cube, uint level, const float sdf[8], std::vector<OctreeNodeTriangleHandler*> * handlers, bool simplification, ThreadContext * context) const;
    OctreeNodeLevel fetch(glm::vec3 pos, uint level, bool simplification, ThreadContext * context) const;
    void iterateBorder(
        const OctreeNode * from,
        const BoundingCube &fromCube,
        const float fromSDF[8],
        const uint fromLevel,
        const OctreeNode *to,
        const BoundingCube &toCube,
        const float toSDF[8],
        const uint toLevel,
        bool &nodeIterated,
        const IterateBorderHandler &func,
        ThreadContext * context) const;
    bool isChunkNode(float length) const;
    bool isThreadNode(float length, float minSize, int threadSize) const;
    void exportOctreeSerialization(OctreeSerialized * octree);
    void exportNodesSerialization(std::vector<OctreeNodeCubeSerialized> * nodes);
private:
    void buildSDF(const ShapeArgs &args, BoundingCube &cube, float shapeSDF[8], float resultSDF[8], float existingResultSDF[8], ThreadContext * threadContext) const;
    float evaluateSDF(const ShapeArgs &args, tsl::robin_map<glm::vec3, float> * threadContext, glm::vec3 p) const;
};


// Simplifier is declared in Simplifier.hpp

 
