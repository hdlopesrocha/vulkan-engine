#pragma once
#include "ThreadPool.hpp"
#include "OctreeThreadContext.hpp"
#include "OctreeSessionContext.hpp"
#include "OctreeNodeFrame.hpp"
#include "OctreeAllocator.hpp"
#include <functional>
#include "ShapeArgs.hpp"
#include <shared_mutex>
#include "../math/geometry/BoundingCube.hpp"
#include <cstdint>
// Node identifier used by change handlers/collectors to key per-node state
// (previously defined in the removed OctreeChangeHandler.hpp).
typedef uintptr_t NodeID;

class Ray;



class Octree: public BoundingCube {
    public:
    // Published chunk metadata lives in vulkan/LodMesh.hpp (struct LoDMesh).

    float chunkSize;

    OctreeNode * root;
    typedef unsigned int uint;

    using OctreeNodeDataHandler = std::function<void(const OctreeNodeData&)>;
    using IterateHandler = std::function<bool(const Octree&, OctreeNodeData&)>;
    using IterateOrderHandler = std::function<void(const Octree&, OctreeNodeData&, uint8_t[8])>;
    using IterateThreadedHandler = std::function<bool(const Octree&, OctreeNodeData&)>;

    OctreeAllocator * allocator;
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
        const BrushPainter &painter, 
        const TextureCoordPainter &coordPainter,
        float minSize, 
        const Simplifier &simplifier, 
        OctreeNodeDataHandler &updateHandler,
        OctreeNodeDataHandler &deleteHandler
    );
    void reset();
    // `allowChildren` lets the chunk-driven apply run one reached chunk's own
    // node logic without descending into its subtree: when false the general
    // path takes the pruned branch exactly as if function.check() had reported
    // Disjoint, while the early classification branches and the materialization
    // of this node still run. The top-down apply always passes true.
    void shape(
        NodeOperationResult &r,
        OctreeNodeFrame frame, 
        const ShapeArgs &args, 
        OctreeThreadContext * threadContext,
        OctreeNodeDataHandler &updateHandler,
        OctreeNodeDataHandler &deleteHandler,
        bool allowChildren = true
    );
    bool intersect(const Ray& ray, glm::vec3& outPos) const;
    float getSdfAt(const glm::vec3 &pos);

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
    // Session state that outlives a single apply (the chunk-level history).
    // apply() owns the tree write lock, so no extra synchronization is needed.
    OctreeSessionContext sessionContext;

    // Chunk-driven apply. Starts at the work units kept in the session's chunk
    // history instead of the world root, does not descend into chunk subtrees
    // the shape cannot reach, and then refreshes the ancestors bottom-up so
    // their SDF corners/type/LoD still describe the whole subtree. Falls back
    // to the full root descent (shape()) whenever no chunk is hit or the stored
    // tree does not satisfy the invariants the bottom-up refresh relies on.
    void shapeChunks(
        const ShapeArgs &args,
        OctreeSessionContext &session,
        OctreeThreadContext * threadContext,
        OctreeNodeDataHandler &updateHandler,
        OctreeNodeDataHandler &deleteHandler
    );
    void buildShapeSDF(const ShapeArgs &args, OctreeNodeFrame &frame, NodeOperationResult &r, NodeOperationResult children[8], OctreeThreadContext * threadContext, bool leafPass) const;
    void buildResultSDF(const bool isLeaf,const ShapeArgs &args, OctreeNodeFrame &frame, NodeOperationResult &r, NodeOperationResult children[8], OctreeThreadContext * threadContext) const;
    float evaluateSDF(const ShapeArgs &args, OctreeThreadContext * threadContext, glm::vec3 p) const;
    // Fills the per-apply lattice fields of args (see ShapeArgs/ShapeSdfCache).
    void configureShapeCache(ShapeArgs &args) const;
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
