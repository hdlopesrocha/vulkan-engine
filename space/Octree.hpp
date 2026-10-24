#pragma once
#include "Simplifier.hpp"
#include "ThreadPool.hpp"
#include <unordered_set>
#include "OctreeThreadContext.hpp"
#include "OctreeNodeFrame.hpp"
#include "OctreeNodeLevel.hpp"
#include "OctreeAllocator.hpp"
#include "../math/geometry/Vertex.hpp"
#include <functional>
#include "ShapeArgs.hpp"
#include "../sdf/SignedDistanceFunction.hpp"
#include <functional>
#include <shared_mutex>
#include "../math/geometry/BoundingCube.hpp"
#include "../math/geometry/Ray.hpp"
#include <string>
// Node identifier used by change handlers/collectors to key per-node state
// (previously defined in the removed OctreeChangeHandler.hpp).
typedef uintptr_t NodeID;



class Octree: public BoundingCube {
    public:
    // Published chunk metadata lives in vulkan/LodMesh.hpp (struct LoDMesh).

    float chunkSize;

    OctreeNode * root;
    typedef unsigned int uint;

    using IterateBorderHandler = std::function<void(const BoundingCube&, const float[8], uint)>;
    using OctreeNodeDataHandler = std::function<void(const OctreeNodeData&)>;
    using IterateHandler = std::function<bool(const Octree&, OctreeNodeData&)>;
    using IterateOrderHandler = std::function<void(const Octree&, OctreeNodeData&, uint8_t[8])>;
    using IterateThreadedHandler = std::function<bool(const Octree&, OctreeNodeData&)>;

    OctreeAllocator * allocator;
    std::atomic<int> inFlightShapeOps{0};
    ThreadPool threadPool = ThreadPool(std::thread::hardware_concurrency());
    // Read/write guard for tree structure + node data. iterate* take a shared
    // (read) lock; apply takes a unique (write) lock so traversal threads never
    // walk nodes that a concurrent brush/mesh op is mutating.
    mutable std::shared_mutex treeMutex;

    Octree(const BoundingCube &minCube, float chunkSize);
    Octree();
    ~Octree();

    void expand(const ShapeArgs &args);
    void apply(
        const SignedDistanceOperation &operation, 
        const SignedDistanceFunction &function, 
        const Transformation &model, 
        const TexturePositionPainter &painter, 
        const TextureCoordPainter &coordPainter,
        float minSize, 
        const Simplifier &simplifier, 
        OctreeNodeDataHandler &updateHandler,
        OctreeNodeDataHandler &deleteHandler
    );
    void reset();
    void shape(
        NodeOperationResult &r,
        OctreeNodeFrame frame, 
        const ShapeArgs &args, 
        OctreeThreadContext * threadContext,
        OctreeNodeDataHandler &updateHandler,
        OctreeNodeDataHandler &deleteHandler
    );
    bool intersect(const Ray& ray, glm::vec3& outPos) const;
    OctreeNodeLevel getNodeAt(const glm::vec3 &pos, int level, bool simplification) const;
    OctreeNode* getNodeAt(const glm::vec3 &pos, bool simplification) const;
    float getSdfAt(const glm::vec3 &pos);
    OctreeNodeLevel fetch(glm::vec3 pos, uint level, bool simplification, OctreeThreadContext * context) const;

    // Triangle topology (the adaptive walker) no longer lives here: it moved to
    // OctreeTriangleTessellator (space/OctreeTriangleTessellator.hpp) — the
    // GPU port is services/OctreeTopologyCompute + OctreeTriangles.comp, and
    // this class owns only the SDF/octree data itself.

    // HeightRootToChunk(N): how many LoD levels a chunk can hold above its
    // tessellation frontier before reaching the chunk-size boundary, i.e.
    // floor(log2(chunkSize / minSize)) - N. >= 0 means a cell at LoD N is
    // valid: its ladder (levels 0..N) fits inside its own size band.
    // Negative means the frontier sits deeper than the chunk can represent.
    int heightRootToChunk(int lod, float minSize) const;

    bool isChunkNode(float nodeLength) const;
    bool isThreadNode(float nodeLength, float minSize, int threadSize) const;
private:
    void buildShapeSDF(const bool isLeaf,const ShapeArgs &args, OctreeNodeFrame &frame, NodeOperationResult &r, NodeOperationResult children[8], OctreeThreadContext * threadContext, bool force) const;
    void buildResultSDF(const bool isLeaf,const ShapeArgs &args, OctreeNodeFrame &frame, NodeOperationResult &r, NodeOperationResult children[8], OctreeThreadContext * threadContext) const;
    float evaluateSDF(const ShapeArgs &args, tsl::robin_map<glm::vec3, float> * threadContext, glm::vec3 p) const;
    void shapeChildren(
        const OctreeNodeFrame &frame, 
        const ShapeArgs &args, 
        OctreeThreadContext * threadContext, 
        NodeOperationResult childResult[8],
        OctreeNodeDataHandler &updateHandler,
        OctreeNodeDataHandler &deleteHandler
    );
};


// Simplifier is declared in Simplifier.hpp
