#include "Octree.hpp"
#include <memory>
#include <atomic>
#include <thread>
#include "../math/BrushMode.hpp"
#include "NodeOperationResult.hpp"
#include "OctreeNodeCubeSerialized.hpp"
#include <cmath>
#include <fstream>
#include <iomanip>
#include <vector>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <limits>
#include "OctreeAllocator.hpp"
#include "OctreeNode.hpp"
#include "IteratorHandler.hpp"
#include "../sdf/SDF.hpp"
#include "../math/BrushMode.hpp"


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
Octree::Octree(const BoundingCube &minCube, float chunkSize) : BoundingCube(minCube), allocator(new OctreeAllocator()) {
    this->chunkSize = chunkSize;
	this->root = allocator->allocate()->init(glm::vec3(minCube.getCenter()));
    this->shapeCounter = std::make_shared<std::atomic<int>>(0);
	initialize();
}

Octree::Octree() : Octree(glm::vec3(0.0f), 1.0f) {
    this->chunkSize = 1.0f;
    this->root = NULL;
    this->shapeCounter = std::make_shared<std::atomic<int>>(0);
    initialize();
}

int getNodeIndex(const glm::vec3 &vec, const BoundingCube &cube) {
    return cube.getChildIndex(vec);
}




OctreeNodeLevel Octree::getNodeAt(const glm::vec3 &pos, int level, bool simplification) const{
    OctreeNode * candidate = root;
    OctreeNode* node = candidate;
    BoundingCube cube = *this;
    uint currentLevel = 0;
	if(!contains(pos)) {
		return OctreeNodeLevel(NULL, 0);
	}
    while (candidate != NULL && level-- > 0 ) {
        if (simplification && node->isSimplified()) {
            break;
        }
        int i = getNodeIndex(pos, cube);
        cube = cube.getChild(i);
        ChildBlock * block = node->getBlock(*allocator);
        candidate = block != NULL ? block->get(i, *allocator) : NULL;
        if(candidate != NULL) {
            node = candidate;
            ++currentLevel;
        }
    }
    return OctreeNodeLevel(node, currentLevel);
}

OctreeNode* Octree::getNodeAt(const glm::vec3 &pos, bool simplification) const {
    OctreeNode * candidate = root;
    OctreeNode* node = candidate;
    BoundingCube cube = *this;
	if(!contains(pos)) {
		return NULL;
	}
    while (candidate != NULL) {
        if (simplification && node->isSimplified()) {
            break;
        }
        int i = getNodeIndex(pos, cube);
        cube = cube.getChild(i);
        ChildBlock * block = node->getBlock(*allocator);
        candidate = block != NULL ? block->get(i, *allocator) : NULL;
        if(candidate != NULL) {
            node = candidate;
        }
    }
    return node;
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

void Octree::iterateTriangles(
        OctreeNode * from,
            const BoundingCube &fromCube,
            int fromLevel,
            OctreeNodeTriangleHandler &func,
            ThreadContext * context) const {
    (void)fromLevel;

    struct EdgeCell {
        OctreeNode *node = NULL;
        BoundingCube cube;
        int level = 0;

        bool isSurface() const {
            return node != NULL && node->getType() == SpaceType::Surface && node->isSimplified();
        }
    };

    // Anchor for the upper-traversal cache: findCellAt climbs from a cached
    // ancestor of this cell (via context->parentOf) instead of the root. The
    // parent links (and their child indices) are registered/seeded by the tree
    // traversal (IteratorHandler), so they are NOT stored inside the nodes.
    EdgeCell hint;

    // Cache for the parent-chain rebuild below. The chain (and therefore its
    // reversed node/index arrays) depends ONLY on hint.node, never on the
    // queried position, so we rebuild it at most once per distinct anchor and
    // reuse it across findCellAt calls. This avoids O(depth) repeated
    // parentOf hash-map lookups on every findCellAt invocation.
    static constexpr int MAX_CHAIN = 64;
    OctreeNode *cachedChainNode = NULL;
    OctreeNode *cachedChainNodes[MAX_CHAIN] = {};
    int cachedChainIndices[MAX_CHAIN] = {};
    int cachedChainLen = 0;
    bool cachedChainOk = false;

    // Per-node cache for findCellAt. The 12 edges and their segments repeatedly
    // query identical quadrant/face sample positions (the same edge midpoint is
    // sampled by both collectBreaks and emitSegment, and adjacent edges share
    // corners), so memoizing avoids repeated O(depth) descents. findCellAt is a
    // pure function of pos (the octree/context are constant during this call),
    // so cached results are reusable and bit-identical to a fresh lookup.
    tsl::robin_map<glm::vec3, EdgeCell> cellCache;

    struct EdgeSpan {
        int axis = 0;
        int u = 1;
        int v = 2;
        float fixedU = 0.0f;
        float fixedV = 0.0f;
        float start = 0.0f;
        float end = 0.0f;
        float eps = 1e-6f;
    };

    auto samePosition = [](const glm::vec3 &a, const glm::vec3 &b, float eps) {
        glm::vec3 d = a - b;
        return glm::dot(d, d) <= eps * eps;
    };

    auto edgeAxes = [](int axis, int &u, int &v) {
        if (axis == 0) {
            u = 1; v = 2;
        } else if (axis == 1) {
            u = 0; v = 2;
        } else {
            u = 0; v = 1;
        }
    };

    auto findCellAt = [this, context, &hint, &fromCube, &cachedChainNode, &cachedChainNodes,
            &cachedChainIndices, &cachedChainLen, &cachedChainOk, &cellCache](const glm::vec3 &pos) {
        auto cacheHit = cellCache.find(pos);
        if(cacheHit != cellCache.end()) {
            return cacheHit->second;
        }
        EdgeCell result;
        if(root == NULL || !contains(pos)) {
            cellCache[pos] = result;
            return result;
        }

        // Default to the original root descent; the cache below only shortens it.
        OctreeNode *startNode = root;
        BoundingCube startCube = *this;

        if(context != NULL && hint.node != NULL) {
            // Rebuild root-consistent cubes by replaying the parent-index chain
            // stored in context->parentOf. This is essential: the cached cubes
            // (fromCube / data.cube) drift from the root-descended cubes by
            // floating-point error, and at a chunk-boundary face that drift
            // flips getNodeIndex, sending descent into the wrong (cross-chunk)
            // cell and leaving holes. Rebuilding from *this keeps every cube
            // bit-identical to a root descent, so the result matches HEAD.
            // Rebuild root-consistent cubes without any heap allocation: the
            // parent chain is a single path from root, so we store it in fixed
            // stack arrays and pick the deepest ancestor whose cube still
            // contains pos. (Depth is bounded; MAX_CHAIN matches the
            // collectBreaks recursion cap. If exceeded we simply fall through to
            // the default root descent below — correct, just less optimal.)
            // The chain is cached in cachedChain* and only rebuilt when the
            // anchor (hint.node) changes; it is independent of pos.
            if(hint.node != cachedChainNode) {
                cachedChainNode = hint.node;
                cachedChainLen = 0;
                cachedChainOk = true;
                OctreeNode *n = hint.node;
                while(n != NULL && n != root) {
                    auto it = context->parentOf.find(n);
                    if(it == context->parentOf.end()) { cachedChainOk = false; break; }
                    if(cachedChainLen >= MAX_CHAIN) { cachedChainOk = false; break; }
                    cachedChainNodes[cachedChainLen] = n;
                    cachedChainIndices[cachedChainLen] = it->second.second;
                    ++cachedChainLen;
                    n = it->second.first;
                }
                if(!(cachedChainOk && n == root && cachedChainLen > 0)) {
                    cachedChainOk = false;
                } else {
                    for(int i = 0; i < cachedChainLen / 2; ++i) {
                        std::swap(cachedChainNodes[i], cachedChainNodes[cachedChainLen - 1 - i]);
                        std::swap(cachedChainIndices[i], cachedChainIndices[cachedChainLen - 1 - i]);
                    }
                }
            }
            if(cachedChainOk) {
                BoundingCube c = *this;
                int chosen = -1;
                for(int i = 0; i < cachedChainLen; ++i) {
                    c = c.getChild(cachedChainIndices[i]);
                    if(c.contains(pos)) chosen = i;
                    else break;
                }
                if(chosen >= 0) {
                    BoundingCube cc = *this;
                    for(int i = 0; i <= chosen; ++i) cc = cc.getChild(cachedChainIndices[i]);
                    startNode = cachedChainNodes[chosen];
                    startCube = cc;
                }
            }
        }

        OctreeNode *node = startNode;
        BoundingCube cube = startCube;

        while(node != NULL && !node->isSimplified() && !node->isLeaf()) {
            ChildBlock *block = node->getBlock(*allocator);
            if(block == NULL) {
                node = NULL;
                break;
            }

            int childIndex = getNodeIndex(pos, cube);
            OctreeNode *child = block->get(childIndex, *allocator);
            if(child == NULL) {
                node = NULL;
                break;
            }

            cube = cube.getChild(childIndex);
            node = child;
        }

        if(node != NULL) {
            result.node = node;
            result.cube = cube;
        }
        if(result.node != NULL) {
            hint = result;
        }
        cellCache[pos] = result;
        return result;
    };

    // Anchor the cache on `from`: its ancestor chain (seeded by the traversal,
    // rooted at the world root) is already in context->parentOf, so neighbor
    // lookups climb from a cached parent instead of the root.
    hint.node = from;
    hint.cube = fromCube;

    auto sideCoordinate = [](float value, int side) {
        return std::nextafter(value, side < 0 ? -std::numeric_limits<float>::infinity()
                                               :  std::numeric_limits<float>::infinity());
    };

    auto quadrantSigns = [](int axis, int quadrant, int &su, int &sv) {
        static const int SIGNS[3][4][2] = {
            {{-1, -1}, {-1,  1}, { 1,  1}, { 1, -1}},
            {{-1, -1}, { 1, -1}, { 1,  1}, {-1,  1}},
            {{-1, -1}, {-1,  1}, { 1,  1}, { 1, -1}}
        };
        su = SIGNS[axis][quadrant][0];
        sv = SIGNS[axis][quadrant][1];
    };

    auto edgePoint = [](const EdgeSpan &edge, float t) {
        glm::vec3 p(0.0f);
        p[edge.axis] = t;
        p[edge.u] = edge.fixedU;
        p[edge.v] = edge.fixedV;
        return p;
    };

    auto edgeSamplePoint = [&](const EdgeSpan &edge, float t, int quadrant) {
        int su, sv;
        quadrantSigns(edge.axis, quadrant, su, sv);

        glm::vec3 p(0.0f);
        p[edge.axis] = t;
        p[edge.u] = sideCoordinate(edge.fixedU, su);
        p[edge.v] = sideCoordinate(edge.fixedV, sv);
        return p;
    };

    auto addBreak = [](std::vector<float> &breaks, float value, float start, float end, float eps) {
        if(value <= start + eps || value >= end - eps) {
            return false;
        }
        for(float existing : breaks) {
            if(std::fabs(existing - value) <= eps) {
                return false;
            }
        }
        breaks.push_back(value);
        return true;
    };

    auto sortUnique = [](std::vector<float> &values, float eps) {
        std::sort(values.begin(), values.end());
        values.erase(std::unique(values.begin(), values.end(), [eps](float a, float b) {
            return std::fabs(a - b) <= eps;
        }), values.end());
    };

    // Fixed-buffer variant of addBreak/sortUnique to avoid a heap allocation in
    // the per-recursion localBreaks (its size is bounded; see collectBreaks).
    auto addToBuffer = [](float *buf, int &count, int cap, float value, float start, float end, float eps) {
        if(value <= start + eps || value >= end - eps) {
            return false;
        }
        for(int k = 0; k < count; ++k) {
            if(std::fabs(buf[k] - value) <= eps) {
                return false;
            }
        }
        if(count < cap) {
            buf[count++] = value;
            return true;
        }
        return false;
    };

    auto sortUniqueBuf = [](float *buf, int &count, float eps) {
        std::sort(buf, buf + count);
        float *newEnd = std::unique(buf, buf + count, [eps](float a, float b) {
            return std::fabs(a - b) <= eps;
        });
        count = int(newEnd - buf);
    };

    std::function<void(const EdgeSpan&, float, float, std::vector<float>&, int)> collectBreaks;
    collectBreaks = [&](const EdgeSpan &edge, float start, float end, std::vector<float> &breaks, int depth) {
        if(depth > 64 || end - start <= edge.eps * 2.0f) {
            return;
        }

        float mid = start + (end - start) * 0.5f;
        // Bounded stack buffer: at most the 4 quadrant cells contribute their cube
        // min/max faces on the axis (<=8 interior values) plus start/end (<=10
        // total), so 32 has wide margin and avoids a per-recursion heap alloc.
        float localBreaks[32];
        int localCount = 0;
        localBreaks[localCount++] = start;
        localBreaks[localCount++] = end;

        for(int q = 0; q < 4; ++q) {
            EdgeCell cell = findCellAt(edgeSamplePoint(edge, mid, q));
            if(cell.node == NULL) {
                continue;
            }

            addToBuffer(localBreaks, localCount, 32, cell.cube.getMin()[edge.axis], start, end, edge.eps);
            addToBuffer(localBreaks, localCount, 32, cell.cube.getMax()[edge.axis], start, end, edge.eps);
        }

        sortUniqueBuf(localBreaks, localCount, edge.eps);
        if(localCount <= 2) {
            return;
        }

        for(int i = 1; i + 1 < localCount; ++i) {
            addBreak(breaks, localBreaks[i], start, end, edge.eps);
        }

        for(int i = 1; i < localCount; ++i) {
            collectBreaks(edge, localBreaks[i - 1], localBreaks[i], breaks, depth + 1);
        }
    };

    auto makeEdgeSpan = [&](int edgeIndex) {
        glm::ivec2 edgeCorners = SDF_EDGES[edgeIndex];
        glm::vec3 p0 = fromCube.getCorner(edgeCorners.x);
        glm::vec3 p1 = fromCube.getCorner(edgeCorners.y);
        glm::vec3 d = glm::abs(p1 - p0);

        EdgeSpan edge;
        if(d.x >= d.y && d.x >= d.z) {
            edge.axis = 0;
        } else if(d.y >= d.x && d.y >= d.z) {
            edge.axis = 1;
        } else {
            edge.axis = 2;
        }
        edgeAxes(edge.axis, edge.u, edge.v);
        edge.fixedU = p0[edge.u];
        edge.fixedV = p0[edge.v];
        edge.start = glm::min(p0[edge.axis], p1[edge.axis]);
        edge.end = glm::max(p0[edge.axis], p1[edge.axis]);
        edge.eps = glm::max(getLengthX(), fromCube.getLengthX()) * 1e-6f;
        return edge;
    };

    auto ownerLess = [](const EdgeCell &a, const EdgeCell &b) {
        float al = a.cube.getLengthX();
        float bl = b.cube.getLengthX();
        if(al != bl) return al < bl;

        glm::vec3 amin = a.cube.getMin();
        glm::vec3 bmin = b.cube.getMin();
        if(amin.x != bmin.x) return amin.x < bmin.x;
        if(amin.y != bmin.y) return amin.y < bmin.y;
        if(amin.z != bmin.z) return amin.z < bmin.z;
        return std::less<OctreeNode*>()(a.node, b.node);
    };

    auto emitTriangle = [&](Vertex *a, Vertex *b, Vertex *c, float eps) {
        if(a == NULL || b == NULL || c == NULL) return;
        if(samePosition(a->position, b->position, eps)) return;
        if(samePosition(b->position, c->position, eps)) return;
        if(samePosition(c->position, a->position, eps)) return;
        func.handle(*a, *b, *c);
    };

    auto emitSegment = [&](const EdgeSpan &edge, float start, float end) {
        float mid = start + (end - start) * 0.5f;
        EdgeCell cells[4];
        for(int q = 0; q < 4; ++q) {
            cells[q] = findCellAt(edgeSamplePoint(edge, mid, q));
            if(!cells[q].isSurface()) {
                return;
            }
        }

        EdgeCell owner = cells[0];
        for(int q = 1; q < 4; ++q) {
            if(ownerLess(cells[q], owner)) {
                owner = cells[q];
            }
        }

        if(owner.node != from) {
            return;
        }

        glm::vec3 p0 = edgePoint(edge, start);
        glm::vec3 p1 = edgePoint(edge, end);
        float d0 = SDF::interpolate(owner.node->sdf, p0, owner.cube);
        float d1 = SDF::interpolate(owner.node->sdf, p1, owner.cube);
        if((d0 < 0.0f) == (d1 < 0.0f)) {
            return;
        }

        // Bounded stack buffer: built from the 4 quadrant cells (deduped), so it
        // never exceeds 4 elements — avoids a per-segment heap allocation.
        EdgeCell plist[4];
        int pcount = 0;
        for(int q = 0; q < 4; ++q) {
            bool duplicate = (pcount > 0)
                && (plist[pcount - 1].node == cells[q].node
                    || samePosition(plist[pcount - 1].node->vertex.position, cells[q].node->vertex.position, edge.eps));
            if(!duplicate && pcount < 4) {
                plist[pcount++] = cells[q];
            }
        }

        if(pcount > 1) {
            const EdgeCell &first = plist[0];
            const EdgeCell &last = plist[pcount - 1];
            if(first.node == last.node || samePosition(first.node->vertex.position, last.node->vertex.position, edge.eps)) {
                --pcount;
            }
        }

        for(int i = 0; i < pcount; ++i) {
            for(int j = i + 1; j < pcount; ++j) {
                if(plist[i].node == plist[j].node
                    || samePosition(plist[i].node->vertex.position, plist[j].node->vertex.position, edge.eps)) {
                    return;
                }
            }
        }

        // Rotate polygon so `from` (owner/finest cell) is at index 0 for reliable UV.
        {
            int it = -1;
            for(int k = 0; k < pcount; ++k) {
                if(plist[k].node == from) { it = k; break; }
            }
            if(it > 0 && it < pcount) {
                EdgeCell tmp[4];
                int t = 0;
                for(int k = it; k < pcount; ++k) tmp[t++] = plist[k];
                for(int k = 0; k < it; ++k) tmp[t++] = plist[k];
                for(int k = 0; k < pcount; ++k) plist[k] = tmp[k];
            }
        }

        // Determine winding from the SDF sign change direction — authoritative and
        // independent of vertex normals, which are unreliable for coarse LOD cells.
        // d0 < 0: solid at the lower-axis end → surface faces the positive axis → emit as-is.
        // d0 > 0: empty at the lower-axis end → surface faces the negative axis → reverse.
        // Reversal keeps plist[0] (= `from`) as the pivot so Tesselator UV is consistent.
        const bool solidAtStart = (d0 < 0.0f);
        if(pcount == 3) {
            if(solidAtStart)
                emitTriangle(&plist[0].node->vertex, &plist[1].node->vertex, &plist[2].node->vertex, edge.eps);
            else
                emitTriangle(&plist[0].node->vertex, &plist[2].node->vertex, &plist[1].node->vertex, edge.eps);
        } else if(pcount == 4) {
            if(solidAtStart) {
                emitTriangle(&plist[0].node->vertex, &plist[1].node->vertex, &plist[2].node->vertex, edge.eps);
                emitTriangle(&plist[0].node->vertex, &plist[2].node->vertex, &plist[3].node->vertex, edge.eps);
            } else {
                emitTriangle(&plist[0].node->vertex, &plist[3].node->vertex, &plist[2].node->vertex, edge.eps);
                emitTriangle(&plist[0].node->vertex, &plist[2].node->vertex, &plist[1].node->vertex, edge.eps);
            }
        }
    };

    if(from == NULL || from->getType() != SpaceType::Surface || !from->isSimplified()) {
        return;
    }

    for(int edgeIndex = 0; edgeIndex < 12; ++edgeIndex) {
        glm::ivec2 edgeCorners = SDF_EDGES[edgeIndex];
        bool sign0 = from->sdf[edgeCorners.x] < 0.0f;
        bool sign1 = from->sdf[edgeCorners.y] < 0.0f;
        if(sign0 == sign1) {
            continue;
        }

        EdgeSpan edge = makeEdgeSpan(edgeIndex);
        if(edge.end - edge.start <= edge.eps * 2.0f) {
            continue;
        }

        std::vector<float> breaks;
        breaks.reserve(64);
        breaks.push_back(edge.start);
        breaks.push_back(edge.end);
        collectBreaks(edge, edge.start, edge.end, breaks, 0);
        sortUnique(breaks, edge.eps);

        for(size_t i = 1; i < breaks.size(); ++i) {
            if(breaks[i] - breaks[i - 1] > edge.eps * 2.0f) {
                emitSegment(edge, breaks[i - 1], breaks[i]);
            }
        }
    }
}



OctreeNodeLevel Octree::fetch(glm::vec3 pos, uint level, bool simplification, ThreadContext * context) const {
    glm::vec4 key = glm::vec4(pos, level);
    if(context->nodeCache.find(key) != context->nodeCache.end()) {
        return context->nodeCache[key];
    } else {
        OctreeNodeLevel nodeLevel = getNodeAt(pos, level, simplification);
        context->nodeCache[key] = nodeLevel;
        return nodeLevel;
    }
}



void Octree::expand(const ShapeArgs &args) {
    while (!args.function->isContained(*this, args.model, args.minSize)) {
        glm::vec3 point = args.function->getCenter(args.model);
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

float Octree::evaluateSDF(const ShapeArgs &args, tsl::robin_map<glm::vec3, float> *cache, glm::vec3 p) const {
    auto it = cache->find(p);
    if (it != cache->end())
        return it->second;

    float d = args.function->distance(p, args.model);
    cache->try_emplace(p, d);
    return d;
}

void Octree::buildShapeSDF(const ShapeArgs &args, OctreeNodeFrame &frame, float shapeSDF[8], ThreadContext * threadContext) const {
    const glm::vec3 min = frame.cube.getMin();
    const glm::vec3 length = frame.cube.getLength();
    tsl::robin_map<glm::vec3, float> * shapeSdfCache = &threadContext->shapeSdfCache;

    for (uint i = 0; i < 8; ++i) {
        shapeSDF[i] = evaluateSDF(args, shapeSdfCache, min + length * Octree::getShift(i));
    }
}

void Octree::buildResultSDF(const ShapeArgs &args, OctreeNodeFrame &frame, float shapeSDF[8], float resultSDF[8], ThreadContext * threadContext) const {
    for (uint i = 0; i < 8; ++i) {
        resultSDF[i] = args.operation(frame.sdf[i], shapeSDF[i]);
    }
}

void Octree::apply(
        float (*operation)(float, float),
        WrappedSignedDistanceFunction *function,
        const Transformation model,
        glm::vec4 translate,
        glm::vec4 scale,
        const TexturePainter &painter,
        float minSize,
        Simplifier &simplifier,
        const OctreeChangeHandler &changeHandler
    ) {
    threadsCreated = 0;
    *shapeCounter = 0;
    ShapeArgs args = ShapeArgs(operation, function, painter, model, translate, scale, simplifier, changeHandler, minSize);	
  	expand(args);
    OctreeNodeFrame frame = OctreeNodeFrame(root, NULL, *this, root ? root->getType() : SpaceType::Empty, 0, root ? root->sdf : nullptr, DISCARD_BRUSH_INDEX, *this);
    ThreadContext localChunkContext = ThreadContext(*this);
    NodeOperationResult r = NodeOperationResult();
    shape(r, frame, args, &localChunkContext);
#ifdef DEBUG
    std::cout << "\t\tOctree::apply Ok! threads=" << threadsCreated << ", works=" << *shapeCounter << std::endl;
#endif
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

bool Octree::isChunkNode(float length) const {
    return chunkSize*0.5f < length && length <= chunkSize;
}

bool Octree::isThreadNode(float length, float minSize, int threadSize) const {
    return minSize*threadSize < length;
}

void Octree::shapeChildren(const OctreeNodeFrame &frame, const ShapeArgs &args, ThreadContext * threadContext, NodeOperationResult childResult[8], bool fromPool) {
    float childLength = frame.cube.getLengthX()*0.5f;
    bool isChildThread = !fromPool && isThreadNode(childLength, args.minSize, 16);
    bool isChildChunk = isChunkNode(childLength);
    std::vector<std::future<void>> futures;

    OctreeNode * children[8] = { NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL };
    OctreeNode * node = frame.node;
    if(node != NULL) {
        node->getChildren(*allocator, children);
    }

    int brushIndex = node ? node->vertex.brushIndex : frame.brushIndex;

    // Iterate nodes and submit threaded children to the pool
    for (uint i = 0; i < 8; ++i) {
        OctreeNode * child = children[i];
        if(node!=NULL && child == node) {
            throw std::runtime_error("Infinite loop " + std::to_string((long)child) + " " + std::to_string((long)node));
        }

        float childSDF[8] = {INFINITY,INFINITY,INFINITY,INFINITY,INFINITY,INFINITY,INFINITY,INFINITY};
        int childBrushIndex = child ? child->vertex.brushIndex : brushIndex;

        if(child != NULL) {
            SDF::copySDF(child->sdf, childSDF);
        } else {
            SDF::getChildSDF(frame.sdf, i, childSDF);
        }
        BoundingCube childCube = frame.cube.getChild(i);
        OctreeNodeFrame childFrame = OctreeNodeFrame(
            child,
            child ? frame.iteratedNode : frame.node,
            childCube,
            child ? child->getType() : frame.type,
            frame.level + 1,
            childSDF,
            childBrushIndex,
            isChildChunk ? childCube : frame.chunkCube
        );

    
        if(isChildThread) {
            ++threadsCreated;
            NodeOperationResult * result = &childResult[i];
            inFlightShapeOps.fetch_add(1);
            futures.push_back(threadPool.enqueue([this, childFrame, args, result]() {
                ThreadContext localThreadContext(childFrame.cube);
                shape(*result, childFrame, args, &localThreadContext, true);
                inFlightShapeOps.fetch_sub(1);
            }));
        } else {
            shape(childResult[i], childFrame, args, threadContext);
        }
        (*shapeCounter)++;
    
    }
    if(isChildThread) {
        for(auto &f : futures) {
            f.wait();
        }
    }
}

void Octree::shape(NodeOperationResult &r,OctreeNodeFrame frame, const ShapeArgs &args, ThreadContext * threadContext, bool fromPool) {    
    r.node = frame.node;
    const float length = frame.cube.getLengthX();
    const bool isShapeLeaf = length <= args.minSize;
    const bool isNodeLeaf = r.node == NULL || r.node->isLeaf();
    r.brushIndex = r.node ? r.node->vertex.brushIndex : frame.brushIndex;
    r.isChunk = isChunkNode(length);
    r.isLeaf = isShapeLeaf && isNodeLeaf;
    r.isSimplified = r.isLeaf;

    NodeOperationResult children[8] = { 
        NodeOperationResult(), NodeOperationResult(), 
        NodeOperationResult(), NodeOperationResult(),
        NodeOperationResult(), NodeOperationResult(), 
        NodeOperationResult(), NodeOperationResult() 
    };

    buildShapeSDF(args, frame, r.shapeSDF, threadContext);

    const glm::vec3 center = frame.cube.getCenter();
    r.shapeSdfCenter = evaluateSDF(args, &threadContext->shapeSdfCache, center);

    bool process = true;
    if(r.isLeaf) {
        buildResultSDF(args, frame, r.shapeSDF, r.resultSDF, threadContext);
        r.shapeType = SDF::eval(r.shapeSDF);
        r.resultType = SDF::eval(r.resultSDF);
    }
    else {
        const float halfDiagonal = length * 0.866025403784439f;
        process = r.shapeSdfCenter <= halfDiagonal;

        if(process) {
            const ContainmentType check = args.function->check(frame.cube, args.model, args.minSize);
            process = check != ContainmentType::Disjoint;
        }
        if(process) {    
            shapeChildren(frame, args, threadContext, children, fromPool);
            bool childResultSolid = true;
            bool childResultEmpty = true;
            bool childShapeSolid = true;
            bool childShapeEmpty = true;
            for(uint i = 0; i < 8; ++i) {
                NodeOperationResult &child = children[i];
                childResultEmpty &= child.resultType == SpaceType::Empty;
                childResultSolid &= child.resultType == SpaceType::Solid;
                childShapeEmpty &= child.shapeType == SpaceType::Empty;
                childShapeSolid &= child.shapeType == SpaceType::Solid;
                r.shapeSDF[i] = child.shapeSDF[i];
                r.resultSDF[i] = child.resultSDF[i];
            }
            r.shapeType = childToParent(childShapeSolid, childShapeEmpty);
            r.resultType = childToParent(childResultSolid, childResultEmpty);
        } else {
            r.shapeType = SDF::eval(r.shapeSDF);
            r.resultType = frame.type;
            SDF::copySDF(frame.sdf, r.resultSDF);
        }
        
    }
    bool interpolatedSurface = (frame.node == NULL)
                                && SDF::eval(frame.sdf) == SpaceType::Surface 
                                && frame.type == SpaceType::Surface
                                ;

    if(r.shapeType != SpaceType::Empty || interpolatedSurface) {
        if(r.resultType == SpaceType::Surface) {
            // Create nodes for surface results if they don't exist
            if(r.node == NULL) {
                r.node = allocator->allocate()->init(Vertex(frame.cube.getCenter()));   
            }

            if(r.node!= NULL) {
                r.node->vertex.position = SDF::getPosition(r.resultSDF, frame.cube);
                r.node->vertex.normal = SDF::getNormalFromPosition(r.resultSDF, frame.cube, r.node->vertex.position);
                // Simplification & Painting
                if(r.isLeaf) {
                    if(r.shapeType != SpaceType::Empty) {
                        r.brushIndex = args.painter.paint(r.node->vertex, args.translate, args.scale);
                    }  
                } else {    
                    if (!r.isChunk) {
                        // Pass frame.chunkCube so the simplifier can guard chunk borders.
                        SimplificationResult simplificationResult = args.simplifier.simplify(frame.cube, r.resultSDF, children, frame.chunkCube);
                        r.isSimplified = simplificationResult.isSimplified;
                        if(r.isSimplified) {
                            r.brushIndex = simplificationResult.brushIndex;
                            // Fall back to inherited brush if simplifier had no surface children
                            if(r.brushIndex == DISCARD_BRUSH_INDEX) {
                                r.brushIndex = frame.brushIndex;
                            }
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
                            childNode->setSimplified(child.isSimplified);
                            childNode->setChunk(child.isChunk);
                            childNode->setBrush(child.brushIndex);                        
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

        if(r.node != NULL) {
            r.node->setType(r.resultType);
            r.node->setSDF(r.resultSDF);
            r.node->setSimplified(r.isSimplified);
            r.node->setChunk(r.isChunk);
            r.node->setBrush(r.brushIndex);
            if (r.isChunk) {
                ++r.node->version;
                OctreeNodeData nodeData(frame.level, r.node, frame.cube, nullptr);
                r.resultType == SpaceType::Surface ? args.changeHandler.onNodeAdded(nodeData) : 
                                                args.changeHandler.onNodeDeleted(nodeData);
            }
            if(r.resultType != SpaceType::Surface) {
                r.node->clear(*allocator, NULL);
            }
        }
    } else {
        SDF::copySDF(frame.sdf, r.resultSDF);
        r.resultType = frame.type;
    }
}

void Octree::iterate(IteratorHandler &handler, OctreeNodeData data) {
	handler.iterateOctree(*this, data);
}

void Octree::iterate(IteratorHandler &handler) {
    OctreeNodeData data(0, root, *this, nullptr);
	handler.iterateOctree(*this, data);
}

void Octree::iterateFlat(IteratorHandler &handler, OctreeNodeData data) {
    handler.iterateFlatIn(*this, data);
}


void Octree::iterateFlat(IteratorHandler &handler) {
    OctreeNodeData data(0, root, *this, nullptr);
    handler.iterateFlatIn(*this, data);
}

void Octree::iterateParallel(IteratorHandler &handler) {
    OctreeNodeData data(0, root, *this, nullptr);
    handler.iterateBFS(*this, data);
    //handler.iterateParallelBFS(*this, data, threadPool);
}

void Octree::exportOctreeSerialization(OctreeSerialized * node) {
    std::cout << "exportOctreeSerialization()" << std::endl;
    for(int i = 0; i < 3; ++i) {
        node->min[i] = this->min[i];
        std::cout << "\tmin["<<std::to_string(i) <<"]: " << std::to_string(node->min[i]) << std::endl;
    }
    node->length = this->length;
    node->chunkSize = this->chunkSize;

    std::cout << "\tlength: " << std::to_string(node->length) << std::endl;
    std::cout << "\tchunkSize: " << std::to_string(node->chunkSize) << std::endl;
}

void Octree::exportNodesSerialization(std::vector<OctreeNodeCubeSerialized> * nodes) {
	std::cout << "exportNodesSerialization()" << std::endl;
    nodes->clear();
    int leafNodes = 0;
    root->exportSerialization(*allocator, nodes, &leafNodes, *this, *this, 0u);
	std::cout << "exportNodesSerialization Ok!" << std::endl;
}

Octree::~Octree() {
    while (inFlightShapeOps.load() > 0) {
        std::this_thread::yield();
    }
    delete allocator;
}

void Octree::reset() {
    while (inFlightShapeOps.load() > 0) {
        std::this_thread::yield();
    }
    if(root != NULL) {
        allocator->childAllocator.reset();
        allocator->nodeAllocator.reset();
        this->root = allocator->allocate()->init(glm::vec3(getCenter()));
    }
}

void Octree::exportToJson(const std::string &filename) const {
    std::ofstream file(filename);
    if (!file) {
        std::cerr << "Octree::exportToJson() Error opening file: " << filename << std::endl;
        return;
    }
    file.setf(std::ios::fixed);
    file << std::setprecision(6);

    std::function<void(const OctreeNode*, const BoundingCube&)> writeNode;
    writeNode = [&](const OctreeNode* node, const BoundingCube &cube) {
        if (node == NULL) {
            file << "null";
            return;
        }
        file << "{";
        const Vertex &v = node->vertex;
        file << "\"position\":[" << v.position.x << "," << v.position.y << "," << v.position.z << "],";
        file << "\"normal\":[" << v.normal.x << "," << v.normal.y << "," << v.normal.z << "],";
        file << "\"texCoord\":[" << v.texCoord.x << "," << v.texCoord.y << "],";
        file << "\"brushIndex\":" << v.brushIndex << ",";
        file << "\"bits\":" << (int)node->bits << ",";
        glm::vec3 min = cube.getMin();
        glm::vec3 len = cube.getLength();
        file << "\"min\":[" << min.x << "," << min.y << "," << min.z << "],";
        file << "\"length\":[" << len.x << "," << len.y << "," << len.z << "],";

        ChildBlock * block = node->getBlock(*allocator);
        if (block == NULL) {
            file << "\"children\": null";
        } else {
            OctreeNode * children[8] = { NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL };
            node->getChildren(*allocator, children);
            file << "\"children\":";
            file << "[";
            for (int i = 0; i < 8; ++i) {
                writeNode(children[i], cube.getChild(i));
                if (i < 7) file << ",";
            }
            file << "]";
        }
        file << "}";
    };

    file << "{ \"root\": ";
    writeNode(root, *this);
    file << " }\n";
    file.close();
    std::cout << "Octree exported to JSON: " << filename << std::endl;
}

void Octree::exportToBson(const std::string &filename) const {
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Octree::exportToBson() Error opening file: " << filename << std::endl;
        return;
    }

    OctreeAllocator * alloc = this->allocator;

    auto appendInt32 = [](std::vector<uint8_t> &buf, int32_t v) {
        buf.push_back((uint8_t)(v & 0xff));
        buf.push_back((uint8_t)((v >> 8) & 0xff));
        buf.push_back((uint8_t)((v >> 16) & 0xff));
        buf.push_back((uint8_t)((v >> 24) & 0xff));
    };
    auto appendInt64 = [](std::vector<uint8_t> &buf, int64_t v) {
        for (int i = 0; i < 8; ++i) buf.push_back((uint8_t)((v >> (8*i)) & 0xff));
    };
    auto appendDouble = [&](std::vector<uint8_t> &buf, double d) {
        uint64_t u = 0;
        std::memcpy(&u, &d, sizeof(double));
        for (int i = 0; i < 8; ++i) buf.push_back((uint8_t)((u >> (8*i)) & 0xff));
    };
    auto appendCString = [&](std::vector<uint8_t> &buf, const std::string &s) {
        buf.insert(buf.end(), s.begin(), s.end());
        buf.push_back(0x00);
    };

    std::function<std::vector<uint8_t>(const std::vector<double>&)> makeDoubleArrayDoc;
    makeDoubleArrayDoc = [&](const std::vector<double> &arr) -> std::vector<uint8_t> {
        std::vector<uint8_t> doc;
        appendInt32(doc, 0); // placeholder
        for (size_t i = 0; i < arr.size(); ++i) {
            doc.push_back(0x01); // double
            appendCString(doc, std::to_string(i));
            appendDouble(doc, arr[i]);
        }
        doc.push_back(0x00); // terminator
        int32_t size = (int32_t)doc.size();
        doc[0] = (uint8_t)(size & 0xff);
        doc[1] = (uint8_t)((size >> 8) & 0xff);
        doc[2] = (uint8_t)((size >> 16) & 0xff);
        doc[3] = (uint8_t)((size >> 24) & 0xff);
        return doc;
    };

    std::function<std::vector<uint8_t>(const OctreeNode*, const BoundingCube&)> buildNodeDoc;
    buildNodeDoc = [&](const OctreeNode* node, const BoundingCube &cube) -> std::vector<uint8_t> {
        std::vector<uint8_t> doc;
        appendInt32(doc, 0); // placeholder

        if (node == nullptr) {
            // empty document
            doc.push_back(0x00);
            int32_t size = (int32_t)doc.size();
            doc[0] = (uint8_t)(size & 0xff);
            doc[1] = (uint8_t)((size >> 8) & 0xff);
            doc[2] = (uint8_t)((size >> 16) & 0xff);
            doc[3] = (uint8_t)((size >> 24) & 0xff);
            return doc;
        }

        // position (array)
        std::vector<double> pos = { node->vertex.position.x, node->vertex.position.y, node->vertex.position.z };
        std::vector<uint8_t> posArr = makeDoubleArrayDoc(pos);
        doc.push_back(0x04); appendCString(doc, "position"); doc.insert(doc.end(), posArr.begin(), posArr.end());

        // normal
        std::vector<double> nrm = { node->vertex.normal.x, node->vertex.normal.y, node->vertex.normal.z };
        std::vector<uint8_t> nrmArr = makeDoubleArrayDoc(nrm);
        doc.push_back(0x04); appendCString(doc, "normal"); doc.insert(doc.end(), nrmArr.begin(), nrmArr.end());

        // texCoord
        std::vector<double> tex = { node->vertex.texCoord.x, node->vertex.texCoord.y };
        std::vector<uint8_t> texArr = makeDoubleArrayDoc(tex);
        doc.push_back(0x04); appendCString(doc, "texCoord"); doc.insert(doc.end(), texArr.begin(), texArr.end());

        // brushIndex (int32)
        doc.push_back(0x10); appendCString(doc, "brushIndex"); appendInt32(doc, node->vertex.brushIndex);

        // bits (int32)
        doc.push_back(0x10); appendCString(doc, "bits"); appendInt32(doc, (int32_t)node->bits);

        // min
        glm::vec3 minv = cube.getMin();
        std::vector<double> minArr = { minv.x, minv.y, minv.z };
        std::vector<uint8_t> minDoc = makeDoubleArrayDoc(minArr);
        doc.push_back(0x04); appendCString(doc, "min"); doc.insert(doc.end(), minDoc.begin(), minDoc.end());

        // length
        glm::vec3 lenv = cube.getLength();
        std::vector<double> lenArr = { lenv.x, lenv.y, lenv.z };
        std::vector<uint8_t> lenDoc = makeDoubleArrayDoc(lenArr);
        doc.push_back(0x04); appendCString(doc, "length"); doc.insert(doc.end(), lenDoc.begin(), lenDoc.end());

        // children
        ChildBlock * block = node->getBlock(*alloc);
        if (block == NULL) {
            doc.push_back(0x0A); appendCString(doc, "children"); // null
        } else {
            OctreeNode * children[8] = { NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL };
            node->getChildren(*alloc, children);
            std::vector<uint8_t> arr;
            appendInt32(arr, 0); // placeholder
            for (int i = 0; i < 8; ++i) {
                OctreeNode *child = children[i];
                std::string key = std::to_string(i);
                if (child == NULL) {
                    arr.push_back(0x0A); appendCString(arr, key); // null
                } else {
                    std::vector<uint8_t> childDoc = buildNodeDoc(child, cube.getChild(i));
                    arr.push_back(0x03); appendCString(arr, key); // embedded document
                    arr.insert(arr.end(), childDoc.begin(), childDoc.end());
                }
            }
            arr.push_back(0x00);
            int32_t arrSize = (int32_t)arr.size();
            arr[0] = (uint8_t)(arrSize & 0xff);
            arr[1] = (uint8_t)((arrSize >> 8) & 0xff);
            arr[2] = (uint8_t)((arrSize >> 16) & 0xff);
            arr[3] = (uint8_t)((arrSize >> 24) & 0xff);

            doc.push_back(0x04); appendCString(doc, "children"); doc.insert(doc.end(), arr.begin(), arr.end());
        }

        doc.push_back(0x00);
        int32_t size = (int32_t)doc.size();
        doc[0] = (uint8_t)(size & 0xff);
        doc[1] = (uint8_t)((size >> 8) & 0xff);
        doc[2] = (uint8_t)((size >> 16) & 0xff);
        doc[3] = (uint8_t)((size >> 24) & 0xff);
        return doc;
    };

    // top-level document
    std::vector<uint8_t> top;
    appendInt32(top, 0);
    if (root == NULL) {
        top.push_back(0x0A); appendCString(top, "root");
    } else {
        std::vector<uint8_t> rootDoc = buildNodeDoc(root, *this);
        top.push_back(0x03); appendCString(top, "root"); top.insert(top.end(), rootDoc.begin(), rootDoc.end());
    }
    top.push_back(0x00);
    int32_t topSize = (int32_t)top.size();
    top[0] = (uint8_t)(topSize & 0xff);
    top[1] = (uint8_t)((topSize >> 8) & 0xff);
    top[2] = (uint8_t)((topSize >> 16) & 0xff);
    top[3] = (uint8_t)((topSize >> 24) & 0xff);

    file.write(reinterpret_cast<const char*>(top.data()), top.size());
    file.close();
    std::cout << "Octree exported to BSON: " << filename << std::endl;
}
