#pragma once
#include "Simplifier.hpp"
#include "ThreadPool.hpp"
#include <unordered_set>
#include "ThreadContext.hpp"
#include "OctreeNodeFrame.hpp"
#include "OctreeNodeLevel.hpp"
#include "OctreeAllocator.hpp"
#include "OctreeNodeTriangleHandler.hpp"
#include "ShapeArgs.hpp"
#include "OctreeSerialized.hpp"
#include "../sdf/SignedDistanceFunction.hpp"
#include <functional>
#include <shared_mutex>
#include "../math/BoundingCube.hpp"
#include "../math/Ray.hpp"
#include <string>
#include "../math/Geometry.hpp"
// Node identifier used by change handlers/collectors to key per-node state
// (previously defined in the removed OctreeChangeHandler.hpp).
typedef uintptr_t NodeID;
class IteratorHandler;



class Octree: public BoundingCube {
    public:
    // ── Parallel scene loading ─────────────────────────────────────────────────
    // CPU mesh-generation results are pushed here from the background loading
    // thread; the main (render) thread drains the queue each frame and performs
    // the actual Vulkan uploads.
    struct LoDMesh {
        Geometry geom;
        uint8_t  lod = 0;     // LoD level of this mesh (= node's chunkLod, 0 = chunks)
        unsigned int     version = 0; // snapshot of node->version at generation time
        float    cellSize = 0;  // this level's cell size, used by the GPU band test
        uint8_t  maxLevel = 0;  // scene-wide band clamp (LocalScene::maxChunkLod); 0 = always keep
    };


    float chunkSize;

    OctreeNode * root;
    typedef unsigned int uint;

    using IterateBorderHandler = std::function<void(const BoundingCube&, const float[8], uint)>;
    using OctreeNodeDataHandler = std::function<void(const OctreeNodeData&)>;
    using IterateHandler = std::function<bool(const Octree&, OctreeNodeData&)>;
    using IterateOrderHandler = std::function<void(const Octree&, OctreeNodeData&, uint8_t[8])>;
    using IterateThreadedHandler = std::function<bool(const Octree&, OctreeNodeData&)>;

    OctreeAllocator * allocator;
    int threadsCreated;
    int prunedEmptyNodes;
    int prunedSolidNodes;
    std::shared_ptr<std::atomic<int>> shapeCounter;
    std::atomic<int> inFlightShapeOps{0};
    tsl::robin_map<glm::vec3, ThreadContext> chunks;
    ThreadPool threadPool = ThreadPool(std::thread::hardware_concurrency());
    std::mutex mutex;
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
        const TexturePainter &painter, 
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
        ThreadContext * threadContext,
        OctreeNodeDataHandler &updateHandler,
        OctreeNodeDataHandler &deleteHandler
    );
    void iterate(OctreeNodeData &data, const IterateHandler &iterateHandler, const IterateOrderHandler &getOrderHandler);
    void iterateFlat(OctreeNodeData &data, const IterateHandler &iterateHandler, const IterateOrderHandler &getOrderHandler);
    void iterate(const IterateHandler &iterateHandler, const IterateOrderHandler &getOrderHandler);
    void iterateFlat(const IterateHandler &iterateHandler, const IterateOrderHandler &getOrderHandler);
    void iterateMultiThreaded(const IterateHandler &iterateHandler, const IterateOrderHandler &getOrderHandler, const IterateThreadedHandler &iterateThreadedHandler);
    void iterateParallel(OctreeNodeData &data, const IterateHandler &iterateHandler, const IterateOrderHandler &getOrderHandler);
    void iterateParallel(const IterateHandler &iterateHandler, const IterateOrderHandler &getOrderHandler);
    bool intersect(const Ray& ray, glm::vec3& outPos) const;
    OctreeNodeLevel getNodeAt(const glm::vec3 &pos, int level, bool simplification) const;
    OctreeNode* getNodeAt(const glm::vec3 &pos, bool simplification) const;
    float getSdfAt(const glm::vec3 &pos);
    OctreeNodeLevel fetch(glm::vec3 pos, uint level, bool simplification, ThreadContext * context) const;

    void iterateTriangles(OctreeNode * from,
        const BoundingCube &fromCube,
        int fromLevel,
        OctreeNodeTriangleHandler &func,
        ThreadContext * context,
        // targetLod uses the +1-shifted STORED ladder level:
        // 0 = no LoD (legacy full-walk mode), 1 = frontier, k = ancestor.
        int targetLod = 0) const;

    // HeightRootToChunk(N): how many LoD levels a chunk can hold above its
    // tessellation frontier before reaching the chunk-size boundary, i.e.
    // floor(log2(chunkSize / minSize)) - N. >= 0 means a chunk at LoD N is
    // valid: its ladder (levels 0..N) fits inside its own size band, and the
    // renderer clamps the chunk's maxLevel to this so coarse levels never
    // exceed the chunk. Negative means the frontier sits deeper than the
    // chunk can represent.
    int heightRootToChunk(int lod, float minSize) const;

    bool isChunkNode(float nodeLength) const;
    bool isThreadNode(float nodeLength, float minSize, int threadSize) const;
    void exportOctreeSerialization(OctreeSerialized * octree);
    void exportNodesSerialization(std::vector<OctreeNodeCubeSerialized> * nodes);
    void exportToJson(const std::string &filename) const;
    void exportToBson(const std::string &filename) const;
private:
    void buildShapeSDF(const ShapeArgs &args, OctreeNodeFrame &frame, NodeOperationResult &r, NodeOperationResult children[8], ThreadContext * threadContext, bool force) const;
    void buildResultSDF(const ShapeArgs &args, OctreeNodeFrame &frame, NodeOperationResult &r, NodeOperationResult children[8], ThreadContext * threadContext) const;
    float evaluateSDF(const ShapeArgs &args, tsl::robin_map<glm::vec3, float> * threadContext, glm::vec3 p) const;
    void shapeChildren(
        const OctreeNodeFrame &frame, 
        const ShapeArgs &args, 
        ThreadContext * threadContext, 
        NodeOperationResult childResult[8],
        OctreeNodeDataHandler &updateHandler,
        OctreeNodeDataHandler &deleteHandler
    );
};


// Simplifier is declared in Simplifier.hpp
