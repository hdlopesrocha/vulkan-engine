#include "Octree.hpp"
#include <array>
#include <atomic>
#include <future>
#include <memory>
#include <mutex>
#include <string>
#include "../math/BrushMode.hpp"
#include "NodeOperationResult.hpp"
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <vector>
#include <cstdint>
#include <algorithm>
#include "OctreeAllocator.hpp"
#include "OctreeNode.hpp"
#include "OctreeNodeData.hpp"
#include "../sdf/SDF.hpp"
#include "../math/geometry/Ray.hpp"


//      6-----7
//     /|    /|
//    4z+---5 |
//    | 2y--+-3
//    |/    |/
//    0-----1x

static std::vector<glm::ivec4> TESSELATION_ORDERS;
static std::vector<glm::ivec2> TESSELATION_EDGES;
static bool initialized = false;

static void initialize() {
    if(!initialized) {
        TESSELATION_ORDERS.push_back(glm::ivec4(0,1,3,2));
        TESSELATION_EDGES.push_back(SDF_EDGES[11]);
        
        TESSELATION_ORDERS.push_back(glm::ivec4(0,2,6,4));
        TESSELATION_EDGES.push_back(SDF_EDGES[6]);
        
        TESSELATION_ORDERS.push_back(glm::ivec4(0,4,5,1));
        TESSELATION_EDGES.push_back(SDF_EDGES[5]);
        initialized = true;
    }
}
Octree::Octree(const BoundingCube &minCube, float chunkSize_) : BoundingCube(minCube), allocator(new OctreeAllocator()) {
    this->chunkSize = chunkSize_;
	this->root = allocator->allocate()->init(glm::vec3(minCube.getCenter()));
	initialize();
}

Octree::Octree() : Octree(glm::vec3(0.0f), 1.0f) {
    this->chunkSize = 1.0f;
    this->root = NULL;
    initialize();
}

int getNodeIndex(const glm::vec3 &vec, const BoundingCube &cube) {
    return cube.getChildIndex(vec);
}

SpaceType childToParent(bool childSolid, bool childEmpty) {
    if(childSolid) {
        return SpaceType::Solid;
    } else if(childEmpty) {
        return SpaceType::Empty;
    } else {
        return SpaceType::Surface;
    }
}

bool Octree::intersect(const Ray& ray, glm::vec3& outPos) const {
    float tNear, tFar;
    if (!ray.intersects(*this, &tNear, &tFar))
        return false;

    struct Entry { OctreeNode* node; BoundingCube cube; float tNear; };
    std::vector<Entry> stack;
    stack.reserve(64); // one allocation per pick instead of one per level
    stack.push_back({root, *this, tNear});

    float bestT = tFar;
    bool found = false;

    while (!stack.empty()) {
        Entry e = stack.back();
        stack.pop_back();

        if (e.tNear >= bestT) continue;

        if (e.node->isLeaf()) {
            if (e.node->getType() == SpaceType::Surface) {
                float tn, tf;
                if (ray.intersects(e.cube, &tn, &tf) && tn < bestT) {
                    bestT = tn;
                    found = true;
                }
            }
            continue;
        }

        ChildBlock* block = e.node->getBlock(*allocator);
        if (!block) continue;

        struct ChildE { OctreeNode* node; int idx; float tNear; };
        ChildE children[8];
        int childCount = 0;
        for (int i = 0; i < 8; ++i) {
            OctreeNode* child = block->get(i, *allocator);
            if (!child) continue;
            BoundingCube childCube = e.cube.getChild(i);
            float tn, tf;
            if (ray.intersects(childCube, &tn, &tf) && tf >= 0.0f && tn < bestT) {
                children[childCount++] = {child, i, tn};
            }
        }

        std::sort(children, children + childCount,
                  [](const ChildE& a, const ChildE& b) { return a.tNear < b.tNear; });
        for (int i = childCount - 1; i >= 0; --i) {
            stack.push_back({children[i].node, e.cube.getChild(children[i].idx), children[i].tNear});
        }
    }

    if (found) {
        outPos = ray.pointAt(bestT);
        return true;
    }
    return false;
}




float Octree::getSdfAt(const glm::vec3 &pos) {
    OctreeNode * candidate = root;
    OctreeNode * node = candidate;
    BoundingCube candidateCube = *this;
    BoundingCube nodeCube = candidateCube;

	if(!contains(pos)) {
		return INFINITY;
	}
    while (candidate) {
        node = candidate;
        nodeCube = candidateCube;
        int i = getNodeIndex(pos, candidateCube);
        candidateCube = nodeCube.getChild(i);
        ChildBlock * block = node->getBlock(*allocator);
        candidate = block != NULL ? block->get(i, *allocator) : NULL;
    }

    if(node) {
        return SDF::interpolate(node->sdf, pos, nodeCube);
    }
    return INFINITY;
}






void Octree::expand(const ShapeArgs &args) {
    while (!args.function.isContained(*this)) {
        glm::vec3 point = args.function.getCenter();
        unsigned int i = getNodeIndex(point, *this) ^ 0x7;

        setMin(getMin() - Octree::getShift(i) * getLengthX());
        setLength(getLengthX()*2);

        OctreeNode* oldRoot = root;
        OctreeNode* newRoot = allocator->allocate()->init(getCenter());
        ChildBlock* newBlock = newRoot->allocate(*allocator)->init();

        if (oldRoot != NULL) {
            ChildBlock* oldBlock = oldRoot->getBlock(*allocator);
            bool emptyNode = oldRoot->getType() == SpaceType::Empty;
            bool emptyBlock = (oldBlock == NULL || oldBlock->isEmpty());

            if (emptyNode && emptyBlock) {
                if (oldBlock != NULL) {
                    oldBlock = oldRoot->clear(*allocator, oldBlock);
                }
                oldRoot = allocator->deallocate(oldRoot);
            }
        }
        if (newRoot == oldRoot) {
            throw std::runtime_error("Infinite recursion!");
        }

        if (oldRoot != NULL) {
            newBlock->set(i, oldRoot, *allocator);
        }
        root = newRoot;
    }
}

// Bit-exact float3 comparison (not operator==: -0.0f and +0.0f must not share
// a cache entry, since the SDF could in principle see the sign).
static inline bool sameFloatBits(const glm::vec3 &a, const glm::vec3 &b) {
    return std::bit_cast<uint32_t>(a.x) == std::bit_cast<uint32_t>(b.x)
        && std::bit_cast<uint32_t>(a.y) == std::bit_cast<uint32_t>(b.y)
        && std::bit_cast<uint32_t>(a.z) == std::bit_cast<uint32_t>(b.z);
}

// One generation per Octree::apply, process-wide: a recycled (thread_local)
// context can serve different Octrees, so the tag must not be per-tree.
static uint32_t nextShapeCacheGeneration() {
    static std::atomic<uint32_t> generation{0};
    uint32_t next = generation.fetch_add(1, std::memory_order_relaxed) + 1;
    if (next == 0) {
        // 2^32 applies; skip the reserved "no generation" value.
        next = generation.fetch_add(1, std::memory_order_relaxed) + 1;
    }
    return next;
}

float Octree::evaluateSDF(const ShapeArgs &args, OctreeThreadContext * threadContext, glm::vec3 p) const {
    ShapeSdfCache &cache = threadContext->shapeSdfCache;
    cache.beginGeneration(args.cacheGeneration);

    // Fast path: canonical lattice points. The rounded index is accepted only
    // when it reconstructs p bit-for-bit, which makes the packed key injective
    // on the queried floats: a hit can only return the value that was computed
    // for exactly these coordinates in this apply. Anything else -- a query
    // off this apply's lattice (the stored tree can hold finer nodes from an
    // earlier apply with a smaller minSize, and the recursion follows them
    // below its own leaf length), or a lattice point whose accumulated
    // subdivision rounding did not reproduce the canonical bits -- falls
    // through to the float-keyed map, so behavior is unchanged for it.
    if (args.cacheSpacing > 0.0f && args.cacheGeneration != 0) {
        const glm::vec3 rel = (p - args.cacheOrigin) / args.cacheSpacing;
        const float limit = float(ShapeSdfCache::kAxisLimit);
        if (rel.x >= 0.0f && rel.x < limit &&
            rel.y >= 0.0f && rel.y < limit &&
            rel.z >= 0.0f && rel.z < limit) {
            const int64_t nx = std::llround(rel.x);
            const int64_t ny = std::llround(rel.y);
            const int64_t nz = std::llround(rel.z);
            const glm::vec3 canonical = args.cacheOrigin
                + glm::vec3(float(nx), float(ny), float(nz)) * args.cacheSpacing;
            if (sameFloatBits(canonical, p)) {
                const uint64_t key = ShapeSdfCache::pack(nx, ny, nz);
                float cached = 0.0f;
                if (cache.find(key, cached)) {
                    if (args.cacheStats) args.cacheStats->hits.fetch_add(1, std::memory_order_relaxed);
                    return cached;
                }
                if (args.cacheStats) args.cacheStats->misses.fetch_add(1, std::memory_order_relaxed);
                const float d = args.function.distance(p);
                cache.insert(key, d);
                return d;
            }
        }
    }

    tsl::robin_map<glm::vec3, float> &fallback = cache.fallback();
    auto it = fallback.find(p);
    if (it != fallback.end()) {
        if (args.cacheStats) args.cacheStats->hits.fetch_add(1, std::memory_order_relaxed);
        return it->second;
    }

    if (args.cacheStats) args.cacheStats->misses.fetch_add(1, std::memory_order_relaxed);
    float d = args.function.distance(p);
    fallback.try_emplace(p, d);
    return d;
}

void Octree::buildShapeSDF(const ShapeArgs &args, OctreeNodeFrame &frame, NodeOperationResult &r, NodeOperationResult children[8], OctreeThreadContext * threadContext, bool leafPass) const {
    const glm::vec3 cubeMin = frame.cube.getMin();
    const glm::vec3 cubeLength = frame.cube.getLength();

    if(leafPass) {
        for (uint i = 0; i < 8; ++i) {
            r.shapeSDF[i] = evaluateSDF(args, threadContext, cubeMin + cubeLength * Octree::getShift(i));
        }
        r.shapeType = SDF::eval(r.shapeSDF);
    } else {
        bool childShapeSolid = true;
        bool childShapeEmpty = true;
        for(uint i = 0; i < 8; ++i) {
            NodeOperationResult &child = children[i];
            childShapeEmpty &= child.shapeType == SpaceType::Empty;
            childShapeSolid &= child.shapeType == SpaceType::Solid;
            r.shapeSDF[i] = child.shapeSDF[i];
        }
        r.shapeType = childToParent(childShapeSolid, childShapeEmpty);
    }
}

void Octree::buildResultSDF(const bool isLeaf, const ShapeArgs &args, OctreeNodeFrame &frame, NodeOperationResult &r, NodeOperationResult children[8], OctreeThreadContext * threadContext) const {
    if(isLeaf) {
        for (uint i = 0; i < 8; ++i) {
            r.resultSDF[i] = args.operation->combine(frame.sdf[i], r.shapeSDF[i]);
        }
        r.resultType = SDF::eval(r.resultSDF);
    } else {
        bool childResultSolid = true;
        bool childResultEmpty = true;
        for(uint i = 0; i < 8; ++i) {
            NodeOperationResult &child = children[i];
            childResultEmpty &= child.resultType == SpaceType::Empty;
            childResultSolid &= child.resultType == SpaceType::Solid;
            r.resultSDF[i] = child.resultSDF[i];
        }
        r.resultType = childToParent(childResultSolid, childResultEmpty);
    }
}

void Octree::configureShapeCache(ShapeArgs &args) const {
    // The cache lattice is this apply's own leaf lattice: corners are
    // origin + n*L_leaf and centers are its half steps, where L_leaf is the
    // first halving of the (already expanded) root length that is <= minSize
    // -- the same halving `shape` performs, so the value is bit-identical to
    // the one the recursion uses. The recursion can still visit finer cubes
    // when the stored tree holds nodes from an earlier apply with a smaller
    // minSize; those queries are off-lattice and take the cache's float-keyed
    // fallback, so the fast path is an optimization, never a correctness
    // requirement.
    float leafLength = getLengthX();
    for (int guard = 0; guard < 128 && leafLength > args.minSize && leafLength > 0.0f; ++guard) {
        leafLength *= 0.5f;
    }
    const float spacing = leafLength * 0.5f;
    if (args.minSize > 0.0f && spacing > 0.0f && std::isfinite(spacing)) {
        args.cacheOrigin = getMin();
        args.cacheSpacing = spacing;
    }
    args.cacheGeneration = nextShapeCacheGeneration();
}

void Octree::apply(
        const SignedDistanceOperation &operation,
        const SignedDistanceFunction &function,
        const Transformation &model,
        const TexturePositionPainter &painter,
        const TextureCoordPainter &coordPainter,
        float minSize,
        const Simplifier &simplifier,
        OctreeNodeDataHandler &updateHandler,
        OctreeNodeDataHandler &deleteHandler
    ) {
    std::unique_lock<std::shared_mutex> writeLock(treeMutex);

    ShapeArgs args = ShapeArgs(operation, function, painter, coordPainter, model, simplifier, minSize);	
    expand(args);
    // Must run after expand(): the root cube (and therefore the lattice
    // origin, leaf length and packed index range) is final then.
    configureShapeCache(args);

#ifndef NDEBUG
    // Perf report 29 M7: one line per apply under VULKAN_SETUP_DIAG. The
    // environment is read once; with the variable unset the counters stay
    // nullptr and only the branch in evaluateSDF remains.
    static const bool setupDiag = std::getenv("VULKAN_SETUP_DIAG") != nullptr;
    static std::atomic<uint64_t> applyIndex{0};
    ShapeCacheStats cacheStats;
    const uint64_t thisApply = setupDiag ? applyIndex.fetch_add(1, std::memory_order_relaxed) : 0;
    const auto applyStart = setupDiag ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point();
    if (setupDiag) args.cacheStats = &cacheStats;
#endif

    OctreeNodeFrame frame = OctreeNodeFrame(root, NULL, *this, root ? root->getType() : SpaceType::Empty, 0, root ? root->sdf : nullptr, DISCARD_BRUSH_INDEX, *this);
    OctreeThreadContext localChunkContext;
    NodeOperationResult r = NodeOperationResult();
    shape(r, frame, args, &localChunkContext, updateHandler, deleteHandler);

#ifndef NDEBUG
    if (setupDiag) {
        const double ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - applyStart).count();
        const uint64_t hits = cacheStats.hits.load(std::memory_order_relaxed);
        const uint64_t misses = cacheStats.misses.load(std::memory_order_relaxed);
        const uint64_t probes = hits + misses;
        const double rate = probes > 0 ? (100.0 * double(hits) / double(probes)) : 0.0;
        std::cout << "[octree-diag] apply " << thisApply << ": " << ms << " ms, cache hits "
                  << hits << ", misses " << misses << ", hit rate " << rate << "%" << std::endl;
    }
#endif
}

int Octree::heightRootToChunk(int lod, float minSize) const {
    // Number of subdivision levels a chunk-size node can hold above minSize.
    // Guarded against degenerate configurations (chunkSize < minSize).
    const float ratio = std::max(1.0f, chunkSize / std::max(minSize, 1.0f));
    const int maxLevels = static_cast<int>(std::floor(std::log2(ratio)));
    return maxLevels - lod;
}


bool Octree::isChunkNode(float nodeLength) const {
    return chunkSize*0.5f < nodeLength && nodeLength <= chunkSize;
}

bool Octree::isThreadNode(float nodeLength, float minSize, int threadSize) const {
    // Fork band, chosen as (minSize*threadSize/4, minSize*threadSize] with
    // threadSize = 16: a 16*minSize node forks eight 8*minSize children and
    // each of those forks eight 4*minSize grandchildren, for up to 64 tasks
    // per fork (perf report 29, M5). The 4*minSize nodes are below the band
    // and do not fork again, which keeps the fork depth at two levels. Only
    // granularity changes: every value is a pure function of its position, so
    // which thread computes a node cannot affect the result.
    return minSize*threadSize*0.25f < nodeLength && nodeLength <= minSize*threadSize;
}

// Detached shape tasks carry their exception back to the parent's ForkState
// (the parent waits on the latch before reading it), so a failed child cannot
// silently leave a partially written result that the parent commits.

void Octree::shapeChildren(
    const OctreeNodeFrame &frame, 
    const ShapeArgs &args, 
    OctreeThreadContext * threadContext, 
    NodeOperationResult childResult[8],
    OctreeNodeDataHandler &updateHandler,
    OctreeNodeDataHandler &deleteHandler) {
    float childLength = frame.cube.getLengthX()*0.5f;
    bool isChildThread = isThreadNode(childLength, args.minSize, 16);
    bool isChildChunk = isChunkNode(childLength);

    OctreeNode * children[8] = { NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL };
    OctreeNode * node = frame.node;
    if(node != NULL) {
        node->getChildren(*allocator, children);
    }

    int brushIndex = node ? node->brushIndex : frame.brushIndex;
    if(brushIndex == DISCARD_BRUSH_INDEX) {
        brushIndex = frame.brushIndex;
    }
    glm::vec3 hsv = node ? node->hsv : frame.hsv;

    // Threaded children write into a shared result block owned by the tasks:
    // if an enqueue fails the parent may unwind while already-enqueued tasks
    // still run, so the block must not live on this stack. Frames and ShapeArgs
    // are copied into each task for the same reason.
    std::shared_ptr<std::array<NodeOperationResult, 8>> threadedResults;
    std::array<std::future<void>, 8> futures;
    int futureCount = 0;
    if (isChildThread) {
        threadedResults = std::make_shared<std::array<NodeOperationResult, 8>>();
    }
    const ShapeArgs argsCopy = args;
    // Child frames are stored in place (optional::emplace) rather than copied
    // into an array: OctreeNodeFrame has no assignment operator.
    std::optional<OctreeNodeFrame> childFrames[8];

    // Join every enqueued future on both the normal and the unwind path. The
    // wait helps drain queued work, then blocks on the future: once runOneTask()
    // finds an empty queue the future's task is running, and the fork band's
    // deepest task never forks, so blocking cannot starve the pool. wait() (not
    // get()) keeps child exceptions from escaping; the shape path has no
    // exception contract.
    struct FutureJoin {
        ThreadPool& pool;
        std::array<std::future<void>, 8>& futures;
        int& count;
        void wait() {
            for (int i = 0; i < count; ++i) {
                while (futures[i].wait_for(std::chrono::milliseconds(0)) != std::future_status::ready) {
                    if (!pool.runOneTask())
                        futures[i].wait();
                }
            }
        }
        ~FutureJoin() { wait(); }
    } join{threadPool, futures, futureCount};

    // Validate the child links before anything is enqueued.
    if(node != NULL) {
        for (uint i = 0; i < 8; ++i) {
            if(children[i] == node) {
                throw std::runtime_error("Infinite loop " + std::to_string((long)children[i]) + " " + std::to_string((long)node));
            }
        }
    }

    // Iterate nodes and submit threaded children to the pool
    for (uint i = 0; i < 8; ++i) {
        OctreeNode * child = children[i];

        float childSDF[8] = {INFINITY,INFINITY,INFINITY,INFINITY,INFINITY,INFINITY,INFINITY,INFINITY};
        int childBrushIndex = child ? child->brushIndex : brushIndex;
        if(childBrushIndex == DISCARD_BRUSH_INDEX) {
            childBrushIndex = brushIndex;
        }
        glm::vec3 childHsv = child ? child->hsv : hsv;

        if(child != NULL) {
            SDF::copySDF(child->sdf, childSDF);
        } else {
            SDF::getChildSDF(frame.sdf, i, childSDF);
        }
        BoundingCube childCube = frame.cube.getChild(i);
        childFrames[i].emplace(
            child,
            child ? frame.iteratedNode : frame.node,
            childCube,
            child ? child->getType() : frame.type,
            frame.level + 1,
            childSDF,
            childBrushIndex,
            isChildChunk ? childCube : frame.chunkCube
        );
        childFrames[i]->hsv = childHsv;

        if(isChildThread) {
            futures[futureCount++] = threadPool.enqueue(
                [this, frame = childFrames[i].value(), results = threadedResults, i,
                 argsCopy, &updateHandler, &deleteHandler] {
                    thread_local OctreeThreadContext workerContext;
                    shape((*results)[i], frame, argsCopy, &workerContext,
                          updateHandler, deleteHandler);
                });
        } else {
            shape(childResult[i], childFrames[i].value(), args, threadContext, updateHandler, deleteHandler);
        }
    }
    if(isChildThread) {
        join.wait();
        for (int i = 0; i < 8; ++i)
            childResult[i] = (*threadedResults)[i];
    }
}


// Report every published chunk rung in a subtree that is about to be freed,
// plus the node that stops being a surface chunk. The scene's change collector
// keys pending updates by node address; the octree allocator recycles freed
// nodes, so without the matching deletes a node allocated at a recycled
// address would inherit the old pending add (whose cube belongs to the freed
// occupant) and its own add would be dropped or built against the wrong cube.
// Must run before OctreeNode::clear() releases the children to the allocator.
static void reportDeletedSubtree(OctreeNode* node, const BoundingCube &cube, uint level,
                                 OctreeAllocator &allocator,
                                 const Octree::OctreeNodeDataHandler &deleteHandler) {
    if (node == nullptr) return;
    if (node->getChunkLod() > 0) {
        deleteHandler(OctreeNodeData(level, node, cube, nullptr));
    }
    OctreeNode * children[8] = { NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL };
    node->getChildren(allocator, children);
    for (uint i = 0; i < 8; ++i) {
        if (children[i] != NULL) {
            reportDeletedSubtree(children[i], cube.getChild(i), level + 1, allocator, deleteHandler);
        }
    }
}

void Octree::shape(
    NodeOperationResult &r,
    OctreeNodeFrame frame, 
    const ShapeArgs &args, 
    OctreeThreadContext * threadContext,
    OctreeNodeDataHandler &updateHandler,
    OctreeNodeDataHandler &deleteHandler
) {    
    r.node = frame.node;
    const float nodeLength = frame.cube.getLengthX();
    const bool isShapeLeaf = nodeLength <= args.minSize;
    const bool isNodeLeaf = r.node == NULL || r.node->isLeaf();
    r.brushIndex = r.node ? r.node->brushIndex : frame.brushIndex;
    r.brushHsv = r.node ? r.node->hsv : frame.hsv;
    // Mapping state: start from the node's existing values so an untouched node
    // keeps what an earlier apply painted. A subdivided node then overwrites
    // these with its dominant painted child below (same rule as the brush).
    r.textureType = r.node ? r.node->textureType : 0;
    r.data = r.node ? r.node->data : glm::vec2(0.0f);
    bool isChunk = isChunkNode(nodeLength);
    bool isLeaf = isShapeLeaf && isNodeLeaf;
    r.selectedLod = isLeaf ? 1u : 0u;
    r.selectedChunkLod = isChunk ? 1u : 0u;

    NodeOperationResult children[8] = { 
        NodeOperationResult(), NodeOperationResult(), 
        NodeOperationResult(), NodeOperationResult(),
        NodeOperationResult(), NodeOperationResult(), 
        NodeOperationResult(), NodeOperationResult() 
    };

    buildShapeSDF(args, frame, r, children, threadContext, true);

    const glm::vec3 center = frame.cube.getCenter();
    float shapeSdfCenter = evaluateSDF(args, threadContext, center);

    bool process = true;
    bool processed = false;
    if(!isLeaf) {
        const float halfDiagonal = nodeLength * 0.866025403784439f;

        if(!processed && isNodeLeaf) {
            bool allInfinity = true;
            for(int i = 0; i < 8; ++i)
                if(frame.sdf[i] != INFINITY) { allInfinity = false; break; }
            if(allInfinity) {
                if(args.operation->propagatesFromInfinity()) {
                    if(shapeSdfCenter < -halfDiagonal) {
                        SDF::copySDF(r.shapeSDF, r.resultSDF);
                        r.shapeType = SpaceType::Solid;
                        r.resultType = SpaceType::Solid;
                        r.selectedLod = 0;
                        processed = true;
                    } else if(shapeSdfCenter > halfDiagonal) {
                        SDF::copySDF(r.shapeSDF, r.resultSDF);
                        r.shapeType = SpaceType::Empty;
                        r.resultType = SpaceType::Empty;
                        r.selectedLod = 0;
                        processed = true;
                    }
                } else {
                    r.shapeType = SDF::eval(r.shapeSDF);
                    r.resultType = SpaceType::Empty;
                    r.selectedLod = 0;
                    processed = true;
                }
            }
            
        }

        if(!processed && frame.type == SpaceType::Solid &&
           args.operation->preservesSolid()) {
            bool allFinite = true;
            for(int i = 0; i < 8; ++i)
                if(frame.sdf[i] == INFINITY) { allFinite = false; break; }
            if(allFinite) {

            if(args.operation->paintsVertices() && r.shapeType != SpaceType::Empty) {
                if(r.node != NULL) {
                    // Painters work on the node directly; the octree build never
                    // computes the surface position/normal (no SDF::getPosition).
                    r.brushIndex = args.painter.paint(*r.node);
                    r.brushHsv = args.painter.paintHSV(*r.node);
                    r.textureType = static_cast<int>(args.coordPainter.textureType());
                    r.data = args.coordPainter.paintUV(*r.node);
                    r.node->hsv = r.brushHsv;
                    r.node->brushIndex = r.brushIndex;
                    r.node->textureType = r.textureType;
                    r.node->data = r.data;
                    r.node->setBrush(r.brushIndex);
                }
            }
            for(uint i = 0; i < 8; ++i) {
                r.resultSDF[i] = args.operation->combine(frame.sdf[i], r.shapeSDF[i]);
            }
            r.resultType = SpaceType::Solid;
            processed = true;
            }
        }

        if(!processed && frame.type == SpaceType::Empty && args.operation->preservesEmpty()) {
            bool allFinite = true;
            for(int i = 0; i < 8; ++i)
                if(frame.sdf[i] == INFINITY) { allFinite = false; break; }
            if(allFinite) {
                for(uint i = 0; i < 8; ++i) {
                    r.resultSDF[i] = args.operation->combine(frame.sdf[i], r.shapeSDF[i]);
                }
                r.shapeType = SDF::eval(r.shapeSDF);
                r.resultType = SpaceType::Empty;
                r.brushIndex = frame.brushIndex;
                r.brushHsv = frame.hsv;
                processed = true;
            }
        }

        if(!processed && frame.type == SpaceType::Solid) {
            bool allFinite = true;
            for(int i = 0; i < 8; ++i)
                if(frame.sdf[i] == INFINITY) { allFinite = false; break; }
            if(allFinite) {

            const float existingSdfCenter = SDF::interpolate(frame.sdf, center, frame.cube);
            const float resultSdfCenter = args.operation->combine(existingSdfCenter, shapeSdfCenter);
            if(resultSdfCenter < -halfDiagonal) {
                for(uint i = 0; i < 8; ++i) {
                    r.resultSDF[i] = args.operation->combine(frame.sdf[i], r.shapeSDF[i]);
                }
                r.shapeType = SDF::eval(r.shapeSDF);
                r.resultType = SpaceType::Solid;
                if(args.operation->paintsVertices() && r.shapeType != SpaceType::Empty) {
                    if(r.node != NULL) {
                        r.brushIndex = args.painter.paint(*r.node);
                        r.brushHsv = args.painter.paintHSV(*r.node);
                        r.textureType = static_cast<int>(args.coordPainter.textureType());
                        r.data = args.coordPainter.paintUV(*r.node);
                        r.node->hsv = r.brushHsv;
                        r.node->brushIndex = r.brushIndex;
                        r.node->textureType = r.textureType;
                        r.node->data = r.data;
                        r.node->setBrush(r.brushIndex);
                    }
                }
                processed = true;
            }
            }
            
        }

        if(!processed) {
            process = shapeSdfCenter <= halfDiagonal;

            if(process) {
                const ContainmentType check = args.function.check(frame.cube);
                process = check != ContainmentType::Disjoint;
            }
            if(!process && frame.node == NULL && frame.type == SpaceType::Surface) {
                process = true;
            }
            if(process) {    
                shapeChildren(frame, args, threadContext, children, updateHandler, deleteHandler);
            } else {
                r.shapeType = SDF::eval(r.shapeSDF);
                r.resultType = frame.type;
                for(uint i = 0; i < 8; ++i) {
                    r.resultSDF[i] = args.operation->combine(frame.sdf[i], r.shapeSDF[i]);
                }
            }
        }
    }

    if(isLeaf) {
        // The leafPass buildShapeSDF() at the top of this function already
        // filled r.shapeSDF with these eight corners: for a leaf it takes the
        // same `isLeaf || force` branch with the same cubeMin/cubeLength, and
        // nothing below touches r.shapeSDF on the leaf path. buildResultSDF
        // only reads those values, so the second corner pass is redundant
        // (perf report 29, C1).
        buildResultSDF(true, args, frame, r, children, threadContext);
    } else if(process && !processed) {
        buildShapeSDF(args, frame, r, children, threadContext, false);
        buildResultSDF(false, args, frame, r, children, threadContext);
    }

    bool interpolatedSurface = (frame.node == NULL)
                                && SDF::eval(frame.sdf) == SpaceType::Surface 
                                && frame.type == SpaceType::Surface
                                ;


    if(r.shapeType != SpaceType::Empty || interpolatedSurface || r.node != NULL) {
        if(r.resultType == SpaceType::Surface) {
            // Create nodes for surface results if they don't exist
            if(r.node == NULL) {
                r.node = allocator->allocate()->init(Vertex(frame.cube.getCenter()));   
            }

            if(r.node!= NULL) {
                // Simplification & Painting
                if(isLeaf) {
                    if(r.shapeType != SpaceType::Empty) {
                        // Painters work on the node directly; no SDF::getPosition.
                        r.brushIndex = args.painter.paint(*r.node);
                        r.node->hsv = args.painter.paintHSV(*r.node);
                        r.node->brushIndex = r.brushIndex;
                        r.node->textureType = static_cast<int>(args.coordPainter.textureType());
                        r.node->data = args.coordPainter.paintUV(*r.node);
                        r.textureType = r.node->textureType;
                        r.data = r.node->data;
                        r.brushHsv = r.node->hsv;
                    }  
                } else  {
                
                    if(process) {
                        if (!isChunk) {
                            // Pass frame.chunkCube so the simplifier can guard chunk borders.
                            SimplificationResult simplificationResult = args.simplifier.simplify(frame.cube, r.resultSDF, children, frame.chunkCube);
                            if(simplificationResult.isSimplified) {
                                r.selectedLod = 1u;
                            }
                        }
                        OctreeNode * childNodes[8] = {NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL};
                        for(uint i =0 ; i < 8 ; ++i) {
                            NodeOperationResult &child = children[i];
                            OctreeNode * childNode = child.node;

                            if(child.resultType != SpaceType::Surface) {
                                if(childNode == NULL) {
                                    BoundingCube childCube = frame.cube.getChild(i);
                                    childNode = allocator->allocate()->init(Vertex(childCube.getCenter()));
                                    children[i].node = childNode;
                                }
                                childNode->setType(child.resultType);
                                childNode->setSDF(child.resultSDF);
                                childNode->setChunk(false);
                                childNode->setBrush(r.brushIndex);
                                childNode->textureType = r.textureType;
                                childNode->data = r.data;
                                childNode->setLod(child.selectedLod);
                                childNode->setChunkLod(child.selectedChunkLod);
                                childNode->hsv = child.brushHsv;
                            }

                            if(frame.node != NULL && childNode == r.node) {
                                throw std::runtime_error("Infinite recursion! " + std::to_string((long) childNode) + " " + std::to_string((long)r.node) );
                            }        
                            childNodes[i] = childNode;
                        }
                        r.node->setChildren(*allocator, childNodes);
                    }
                }
            }
        }
    }

    if(!isLeaf) {
        uint8_t selectedLod = 0;
        uint8_t selectedChunkLod = 0;

        for(uint i =0 ; i < 8 ; ++i) {
            NodeOperationResult &child = children[i];
            OctreeNode * childNode = child.node;
            // Brush tally. The original std::unordered_map was declared here,
            // inside the per-child loop, so it never held more than one entry:
            // every surface child with a valid brush won with count 1 and the
            // last one's mapping state stayed on the parent. The tally is kept
            // per child (reset here) to preserve that exact result; hoisting it
            // across the eight children would turn it into a majority vote and
            // change r.brushIndex/hsv/textureType/data, hence the mesh.
            struct BrushTally { int brush; int count; };
            BrushTally brushCounts[8];
            int brushCountSize = 0;
            int bestCount = 0;
            if(childNode != NULL && childNode->getType() == SpaceType::Surface) {
                const uint8_t childLod = childNode->getLod();
                const uint8_t childChunkLod = childNode->getChunkLod();
                if(childLod > 0u) {
                    selectedLod = selectedLod == 0u ?
                        childLod : glm::min(selectedLod, childLod);
                }
                if(childChunkLod > 0u) {
                    selectedChunkLod = selectedChunkLod == 0u ?
                        childChunkLod : glm::min(selectedChunkLod, childChunkLod);
                }
                const int childBrush = childNode->getBrush();
                if(childBrush > DISCARD_BRUSH_INDEX) {
                    // Linear scan over at most eight entries (perf report 29,
                    // H4): no hash table construction per child.
                    int slot = 0;
                    while(slot < brushCountSize && brushCounts[slot].brush != childBrush) {
                        ++slot;
                    }
                    if(slot == brushCountSize) {
                        brushCounts[brushCountSize++] = BrushTally{childBrush, 0};
                    }
                    const int count = ++brushCounts[slot].count;
                    if(count > bestCount) {
                        bestCount = count;
                        r.brushIndex = childBrush;
                        r.brushHsv = childNode->hsv;
                        r.textureType = childNode->textureType;
                        r.data = childNode->data;
                    }
                }
            }    
        }
        if(selectedLod > 0u) {
            r.selectedLod = selectedLod + 1u;
        }
        if(selectedChunkLod > 0u) {
            r.selectedChunkLod = selectedChunkLod + 1u;
        }
    }

    // Materialize this node's properties for both subdivided and terminal nodes.
    if(process && r.node != NULL) {
        if(r.resultType != SpaceType::Surface) {
            // The node stops being a surface chunk and its children are about
            // to be recycled: report the whole subtree deleted first so the
            // scene drops its pending add / tessellation dedupe records for
            // these addresses.
            if (deleteHandler)
                reportDeletedSubtree(r.node, frame.cube, frame.level, *allocator, deleteHandler);
            r.node->clear(*allocator, NULL);
        }
        r.node->setType(r.resultType);
        r.node->setSDF(r.resultSDF);
        r.node->setChunk(isChunk);
        r.node->setLod(r.selectedLod);
        r.node->setChunkLod(r.selectedChunkLod);
        r.node->setBrush(r.brushIndex);
        r.node->hsv = r.brushHsv;
        // Mapping state must land on the node too: the tessellators read it
        // back per vertex (Vertex::texCoord = UV scale, Vertex::textureType).
        // Internal cells are never painted (painting runs at the shape leaf),
        // so without this every chunkLod > 1 mesh — which is built from cells
        // at lod > 1 — sampled its textures with a zero UV scale, collapsing
        // to one texel. This is the dominant painted child's mapping state.
        r.node->textureType = r.textureType;
        r.node->data = r.data;

        if(r.node->getChunkLod() > 0 && r.resultType == SpaceType::Surface) {
            ++r.node->version;
            OctreeNodeData data = OctreeNodeData(frame.level, r.node, frame.cube, nullptr);
            updateHandler(data);
        }
    }
}

Octree::~Octree() {
    // stop() drains the queue and joins every worker, so no task can still
    // reference the tree or allocator. apply() is synchronous (every fork
    // waits for its children before returning), so no separate in-flight
    // counter is needed.
    threadPool.stop();
    delete allocator;
}

void Octree::reset() {
    // The caller serializes against apply() (the undo path holds treeMutex;
    // scene resets join the tessellation thread first). apply() joins all of
    // its fork tasks before returning, so none can be using the allocator.
    if(root != NULL) {
        allocator->childAllocator.reset();
        allocator->nodeAllocator.reset();
        this->root = allocator->allocate()->init(glm::vec3(getCenter()));
    }
}

