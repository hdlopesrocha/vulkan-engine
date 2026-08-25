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
#include <unordered_map>
#include <limits>
#include <set>
#include <array>
#include "OctreeAllocator.hpp"
#include "OctreeNode.hpp"
#include "IteratorHandler.hpp"
#include "../sdf/SDF.hpp"
#include "../math/BrushMode.hpp"

// Read guard for Octree::treeMutex. iterate* can nest (an iterate handler may
// re-enter the octree, e.g. LocalScene::requestModel3D calls iterateTriangles
// from inside an iterateFlat handler), and std::shared_mutex is not recursive,
// so only the OUTERMOST iterate acquires the lock; inner ones (same thread)
// piggyback on the outer read. apply takes the exclusive path (it is the only
// writer) and is never re-entered from an iterate handler.
class OctreeSharedLock {
public:
    explicit OctreeSharedLock(std::shared_mutex &m) : m_(m) {
        if (depth_++ == 0) m_.lock_shared();
    }
    ~OctreeSharedLock() {
        if (--depth_ == 0) m_.unlock_shared();
    }
    OctreeSharedLock(const OctreeSharedLock&) = delete;
    OctreeSharedLock& operator=(const OctreeSharedLock&) = delete;
private:
    static inline thread_local int depth_ = 0;
    std::shared_mutex &m_;
};


//      6-----7
//     /|    /|
//    4z+---5 |
//    | 2y--+-3
//    |/    |/
//    0-----1x

// Ladder base cell size: the frontier (stored lod 1) is one chunk divided
// by 16 per axis (chunkLod levels 1..5). With the standard 512-chunk this
// anchors at 32, i.e. the heightmap's 30^3 minSize frontier; the log2
// rounding below maps every 30*2^k ladder cell to its level exactly
// (30^3→1, 60^3→2, 120^3→3, 240^3→4, 480^3→5, ...).
static uint8_t lodForCellSize(float nodeLength, float chunkSize) {
    const float frontierSize = chunkSize / static_cast<float>(1u << (5u - 1u));
    const int levelsAboveFrontier = std::lround(std::log2(nodeLength / frontierSize));
    return static_cast<uint8_t>(std::max(1, levelsAboveFrontier + 1));
}

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
    this->shapeCounter = std::make_shared<std::atomic<int>>(0);
    this->prunedEmptyNodes = 0;
    this->prunedSolidNodes = 0;
	initialize();
}

Octree::Octree() : Octree(glm::vec3(0.0f), 1.0f) {
    this->chunkSize = 1.0f;
    this->root = NULL;
    this->shapeCounter = std::make_shared<std::atomic<int>>(0);
    this->prunedEmptyNodes = 0;
    this->prunedSolidNodes = 0;
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

        struct ChildE { int idx; float tNear; };
        std::vector<ChildE> children;
        for (int i = 0; i < 8; ++i) {
            OctreeNode* child = block->get(i, *allocator);
            if (!child) continue;
            BoundingCube childCube = e.cube.getChild(i);
            float tn, tf;
            if (ray.intersects(childCube, &tn, &tf) && tf >= 0.0f && tn < bestT) {
                children.push_back({i, tn});
            }
        }

        std::sort(children.begin(), children.end(),
                  [](const ChildE& a, const ChildE& b) { return a.tNear < b.tNear; });
        for (int i = static_cast<int>(children.size()) - 1; i >= 0; --i) {
            int idx = children[i].idx;
            OctreeNode* child = block->get(idx, *allocator);
            stack.push_back({child, e.cube.getChild(idx), children[i].tNear});
        }
    }

    if (found) {
        outPos = ray.pointAt(bestT);
        return true;
    }
    return false;
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
        if (simplification && node->getLod() == 1u) {
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
        if (simplification && node->getLod() == 1u) {
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
            ThreadContext * context,
            int targetLod) const {
    (void)fromLevel;
    OctreeSharedLock lock(treeMutex);

    struct EdgeCell {
        OctreeNode *node = NULL;
        BoundingCube cube;
        int level = 0;

        // A cell is "surface at the walk's resolution": either a frontier
        // simplified cell (targetLod == 0 — legacy full-walk mode) or a ladder
        // cell exactly at targetLod. With targetLod >= 1 the descent already
        // stops at cells with lod == targetLod, so requiring lod equality here
        // keeps coarse ladder levels (internal, non-simplified nodes) emitting
        // their own cells while neighbors one level finer/coarser stay out.
        // lod and targetLod are both in the +1-shifted STORED space.
        bool isSurface(int targetLod) const {
            return node != NULL && node->getType() == SpaceType::Surface &&
                (targetLod == 0 ? node->getLod() == 1u : node->getLod() == targetLod);
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

    // Bit-pattern key for a vertex position (exact, no hashing of floats).
    auto vertexKey = [](const glm::vec3 &p) {
        uint32_t b[3];
        std::memcpy(&b, &p, sizeof(b));
        return (uint64_t)b[0] ^ ((uint64_t)b[1] << 21) ^ ((uint64_t)b[2] << 42);
    };

    // Emitted-triangle dedup set (see emitTriangle): every ladder mesh is built
    // by walking each cell's 12 edges, and a segment on a cell boundary line is
    // visited once per flanking cell. Meshes hold a few hundred triangles.
    std::set<std::array<uint64_t, 3>> emitted;

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
            &cachedChainIndices, &cachedChainLen, &cachedChainOk, &cellCache, targetLod](const glm::vec3 &pos) {
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

        while(node != NULL && !node->isLeaf() &&
              ((targetLod == 0) ? (node->getLod() == 0u) : (node->getLod() > targetLod))) {
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

    auto makeEdgeSpan = [&](int edgeIndex, const BoundingCube &cellCube) {
        glm::ivec2 edgeCorners = SDF_EDGES[edgeIndex];
        glm::vec3 p0 = cellCube.getCorner(edgeCorners.x);
        glm::vec3 p1 = cellCube.getCorner(edgeCorners.y);
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
        edge.eps = glm::max(getLengthX(), cellCube.getLengthX()) * 1e-6f;
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
        // Tesselator::handle drops triangles with a DISCARD brush; mirror that
        // check here so a triangle is only ever emitted once with a valid brush.
        if(a->brushIndex <= DISCARD_BRUSH_INDEX || b->brushIndex <= DISCARD_BRUSH_INDEX
            || c->brushIndex <= DISCARD_BRUSH_INDEX) return;
        // A surface segment lying on the line shared by two (or four) flanking
        // cells at the same level is walked once per flanking cell, so the
        // same triangle can be submitted up to 4x with bit-identical vertices.
        // Dedup by the sorted bit patterns of the 3 vertex positions: identical
        // inputs produce identical UVs/normals/brushes (all derived from the
        // positions here), so dropping the duplicates is exact and safe. Meshes
        // are small (~hundreds of tris), so a set is cheap.
        std::array<uint64_t, 3> key = { vertexKey(a->position), vertexKey(b->position), vertexKey(c->position) };
        std::sort(key.begin(), key.end());
        if(!emitted.insert(key).second) {
            return;
        }
        func.handle(*a, *b, *c);
    };

    auto emitSegment = [&](const EdgeSpan &edge, float start, float end, bool sign0, bool sign1) {
        float mid = start + (end - start) * 0.5f;
        EdgeCell cells[4];
        for(int q = 0; q < 4; ++q) {
            cells[q] = findCellAt(edgeSamplePoint(edge, mid, q));
        }

        // Owner = positionally-smallest SURFACE ring cell. Non-surface
        // quadrants (Empty/Solid space, cells of a neighboring LOD level)
        // never own the segment, but they do NOT abort it either: the segment
        // still closes the mesh at the surface boundary (shape faces, chunk
        // ladder seams, world edge) with the polygon of the surface quadrants
        // alone. Aborting there left open edge chains along every face
        // aligned with the cell grid.
        EdgeCell owner = cells[0];
        bool ownerValid = false;
        for(int q = 0; q < 4; ++q) {
            if(cells[q].isSurface(targetLod) && (!ownerValid || ownerLess(cells[q], owner))) {
                owner = cells[q];
                ownerValid = true;
            }
        }
        if(!ownerValid) {
            return;
        }

        // Attribute the segment to the walk root `from`: it is emitted iff
        // its owner cell lies inside from's cube. For per-leaf walks (legacy
        // targetLod < 0 mode) this is exactly the old `owner == from` test —
        // adjacent cells' centers are never inside from's cube — so behavior
        // is unchanged. For per-node ladder walks (from = a chunk or ladder
        // ancestor, targetLod >= 0) the owner is a finer lod cell inside
        // from's cube, so the WHOLE node tessellates in ONE call instead of
        // one call per frontier leaf, and boundary segments are emitted by
        // exactly the node that contains their owner.
        if(owner.node != from && !fromCube.contains(owner.cube.getCenter())) {
            return;
        }

        glm::vec3 p0 = edgePoint(edge, start);
        glm::vec3 p1 = edgePoint(edge, end);
        float d0 = SDF::interpolate(owner.node->sdf, p0, owner.cube);
        float d1 = SDF::interpolate(owner.node->sdf, p1, owner.cube);
        // The owner's field may carry INFINITY sentinels (an interpolated surface
        // from an ancestor, or a node whose corners were never materialized), so
        // trilinear interpolation returns INFINITY at the endpoints. INFINITY < 0.0f
        // is false for both ends, which made the test below report "no sign change"
        // and drop the segment — a mesh hole right at the boundary of an edited or
        // interpolated surface. Fall back to the authoritative crossing signs from
        // the producing cell (scanCell only emits edges whose own corners differ in
        // sign), which are always finite.
        bool s0 = (d0 == INFINITY) ? sign0 : (d0 < 0.0f);
        bool s1 = (d1 == INFINITY) ? sign1 : (d1 < 0.0f);
        if(s0 == s1) {
            return;
        }

        // Bounded stack buffer: built from the surface quadrant cells
        // (deduped), so it never exceeds 4 elements — avoids a per-segment
        // heap allocation. Non-surface quadrants are skipped: their nodes
        // have no usable vertex and their cells never tessellate.
        EdgeCell plist[4];
        int pcount = 0;
        for(int q = 0; q < 4; ++q) {
            if(!cells[q].isSurface(targetLod)) {
                continue;
            }
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
        const bool solidAtStart = s0;
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

    // Accept any ladder cell as the `from` anchor: a frontier simplified leaf
    // (simplification 1, developed raw lod == 1 in the stored space) or an
    // internal ancestor (simplification 0 but developed > 0). Reject only
    // non-ladder leaves (stored lod 0).
    if(from == NULL || from->getType() != SpaceType::Surface ||
       (from->getLod() == 0u && from->getLod() == 0)) {
        return;
    }

    // Scan ONE Surface-Nets cell: walk the 12 edges of cellCube with the
    // cell's OWN stored corner samples and emit the segments whose owner
    // (finest of the four quadrant cells at the walk's resolution) lies
    // inside the outer from-cube. Cells one level finer/coarser stay out of
    // the segment via the isSurface(targetLod) quadrant filter; the polygon
    // then closes with the remaining surface quadrants instead of leaving
    // open edges at LOD band boundaries and shape faces.
    auto scanCell = [&](OctreeNode *cellNode, const BoundingCube &cellCube) {
        for(int edgeIndex = 0; edgeIndex < 12; ++edgeIndex) {
            glm::ivec2 edgeCorners = SDF_EDGES[edgeIndex];
            bool sign0 = cellNode->sdf[edgeCorners.x] < 0.0f;
            bool sign1 = cellNode->sdf[edgeCorners.y] < 0.0f;

            if(sign0 == sign1) {
                continue;
            }

            EdgeSpan edge = makeEdgeSpan(edgeIndex, cellCube);
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
                    emitSegment(edge, breaks[i - 1], breaks[i], sign0, sign1);
                }
            }
        }
    };

    if(targetLod == 0) {
        // Legacy mode: `from` IS the frontier cell.
        scanCell(from, fromCube);
        return;
    }

    // Ladder mode: the level-k mesh is the AGGREGATE of the cells at lod k
    // inside the anchor (each cell's own stored corner samples, which are
    // correct). A single Surface-Nets cell over the whole anchor would miss
    // interior surface detail — the anchor's own corners have no zero
    // crossing when the surface is inside it — so every level emits its own
    // cell resolution (cell size frontierCell*2^k), which is what the
    // distance bands consume.
    std::function<void(OctreeNode*, const BoundingCube&)> walkLadder;
    walkLadder = [&](OctreeNode *node, const BoundingCube &cube) {
        if(node == NULL) return;
        // Stored (+1-shifted → uint8) lod: 0 = unset, 1 = frontier, k+1 = parent.
        const uint8_t lod = node->getLod();
        if(lod == targetLod) {
            scanCell(node, cube);
            // Its children are one level finer and belong to other LOD levels,
            // so stop here rather than scanning them too (avoids overlap).
            return;
        }
        ChildBlock *block = node->getBlock(*allocator);
        if(block == NULL) {
            // Finest available cell in this subregion and it is NOT at the
            // requested level: emit it as a fallback so the level-k mesh has no
            // hole where the ladder never produced a lod==targetLod cell (a coarse
            // surface leaf that was never subdivided, or a node whose child block
            // was dropped). A descendant at lod==targetLod would have been scanned
            // and stopped the descent above, so this only fires for bare regions.
            if(node->getType() == SpaceType::Surface) {
                scanCell(node, cube);
            }
            return;
        }
        // Descend to find target-level cells (or finer fallback leaves).
        for(uint i = 0; i < 8; ++i) {
            OctreeNode *child = block->get(i, *allocator);
            if(child != NULL) {
                walkLadder(child, cube.getChild(i));
            }
        }
    };
    walkLadder(from, fromCube);
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

float Octree::evaluateSDF(const ShapeArgs &args, tsl::robin_map<glm::vec3, float> *cache, glm::vec3 p) const {
    auto it = cache->find(p);
    if (it != cache->end())
        return it->second;

    float d = args.function.distance(p);
    cache->try_emplace(p, d);
    return d;
}

void Octree::buildShapeSDF(const ShapeArgs &args, OctreeNodeFrame &frame, NodeOperationResult &r, NodeOperationResult children[8], ThreadContext * threadContext, bool force) const {
    const glm::vec3 cubeMin = frame.cube.getMin();
    const glm::vec3 cubeLength = frame.cube.getLength();
    tsl::robin_map<glm::vec3, float> * shapeSdfCache = &threadContext->shapeSdfCache;

    if(r.isLeaf || force) {
        for (uint i = 0; i < 8; ++i) {
            r.shapeSDF[i] = evaluateSDF(args, shapeSdfCache, cubeMin + cubeLength * Octree::getShift(i));
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

void Octree::buildResultSDF(const ShapeArgs &args, OctreeNodeFrame &frame, NodeOperationResult &r, NodeOperationResult children[8], ThreadContext * threadContext) const {
    if(r.isLeaf) {
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

void Octree::apply(
        const SignedDistanceOperation &operation,
        const SignedDistanceFunction &function,
        const Transformation &model,
        const TexturePainter &painter,
        float minSize,
        const Simplifier &simplifier,
        OctreeNodeDataHandler &updateHandler,
        OctreeNodeDataHandler &deleteHandler
    ) {
    std::unique_lock<std::shared_mutex> writeLock(treeMutex);
    threadsCreated = 0;
    prunedEmptyNodes = 0;
    prunedSolidNodes = 0;

    *shapeCounter = 0;
    ShapeArgs args = ShapeArgs(operation, function, painter, model, simplifier, minSize);	
    expand(args);
    OctreeNodeFrame frame = OctreeNodeFrame(root, NULL, *this, root ? root->getType() : SpaceType::Empty, 0, root ? root->sdf : nullptr, DISCARD_BRUSH_INDEX, *this);
    ThreadContext localChunkContext = ThreadContext(*this);
    NodeOperationResult r = NodeOperationResult();
    shape(r, frame, args, &localChunkContext, updateHandler, deleteHandler);
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
    return minSize*threadSize*0.5f < nodeLength && nodeLength <= minSize*threadSize;
}

void Octree::shapeChildren(
    const OctreeNodeFrame &frame, 
    const ShapeArgs &args, 
    ThreadContext * threadContext, 
    NodeOperationResult childResult[8],
    OctreeNodeDataHandler &updateHandler,
    OctreeNodeDataHandler &deleteHandler) {
    float childLength = frame.cube.getLengthX()*0.5f;
    bool isChildThread = isThreadNode(childLength, args.minSize, 16);
    bool isChildChunk = isChunkNode(childLength);
    std::vector<std::future<void>> futures;

    OctreeNode * children[8] = { NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL };
    OctreeNode * node = frame.node;
    if(node != NULL) {
        node->getChildren(*allocator, children);
    }

    int brushIndex = node ? node->vertex.brushIndex : frame.brushIndex;
    if(brushIndex == DISCARD_BRUSH_INDEX) {
        brushIndex = frame.brushIndex;
    }
    glm::vec3 hsv = node ? node->vertex.hsv : frame.hsv;

    // Iterate nodes and submit threaded children to the pool
    for (uint i = 0; i < 8; ++i) {
        OctreeNode * child = children[i];
        if(node!=NULL && child == node) {
            throw std::runtime_error("Infinite loop " + std::to_string((long)child) + " " + std::to_string((long)node));
        }

        float childSDF[8] = {INFINITY,INFINITY,INFINITY,INFINITY,INFINITY,INFINITY,INFINITY,INFINITY};
        int childBrushIndex = child ? child->vertex.brushIndex : brushIndex;
        if(childBrushIndex == DISCARD_BRUSH_INDEX) {
            childBrushIndex = brushIndex;
        }
        glm::vec3 childHsv = child ? child->vertex.hsv : hsv;

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
        childFrame.hsv = childHsv;

    
        if(isChildThread) {
            ++threadsCreated;
            NodeOperationResult * result = &childResult[i];
            inFlightShapeOps.fetch_add(1);
            futures.push_back(threadPool.enqueue([this, childFrame, args, result, &updateHandler, &deleteHandler]() {
                ThreadContext localThreadContext(childFrame.cube);
                shape(*result, childFrame, args, &localThreadContext, updateHandler, deleteHandler);
                inFlightShapeOps.fetch_sub(1);
            }));
        } else {
            shape(childResult[i], childFrame, args, threadContext, updateHandler, deleteHandler);
        }
        (*shapeCounter)++;
    
    }
    if(isChildThread) {
        for(auto &f : futures) {
            f.wait();
        }
    }
}


void Octree::shape(
    NodeOperationResult &r,
    OctreeNodeFrame frame, 
    const ShapeArgs &args, 
    ThreadContext * threadContext,
    OctreeNodeDataHandler &updateHandler,
    OctreeNodeDataHandler &deleteHandler
) {    
    r.node = frame.node;
    const float nodeLength = frame.cube.getLengthX();
    const bool isShapeLeaf = nodeLength <= args.minSize;
    const bool isNodeLeaf = r.node == NULL || r.node->isLeaf();
    r.brushIndex = r.node ? r.node->vertex.brushIndex : frame.brushIndex;
    r.brushHsv = r.node ? r.node->vertex.hsv : frame.hsv;
    r.isChunk = isChunkNode(nodeLength);
    r.isLeaf = isShapeLeaf && isNodeLeaf;
    r.selectedLod = r.isLeaf ? 1 : 0;

    NodeOperationResult children[8] = { 
        NodeOperationResult(), NodeOperationResult(), 
        NodeOperationResult(), NodeOperationResult(),
        NodeOperationResult(), NodeOperationResult(), 
        NodeOperationResult(), NodeOperationResult() 
    };

    buildShapeSDF(args, frame, r, children, threadContext, true);

    const glm::vec3 center = frame.cube.getCenter();
    float shapeSdfCenter = evaluateSDF(args, &threadContext->shapeSdfCache, center);

    bool process = true;
    bool processed = false;
    if(!r.isLeaf) {
        const float halfDiagonal = nodeLength * 0.866025403784439f;

        // No existing SDF data (all INFINITY) — result is purely the shape.
        // Operations that propagate from infinity: need the Lipschitz center
        // check for safety. Others: result is INFINITY = Empty.
        // The allInfinity guard is mandatory: a cell that ALREADY has SDF
        // data (e.g. an interpolated surface from a parent, or a previous
        // shape op) must NOT be pruned by this new shape's center alone —
        // pruning it would wipe the existing field.
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
                        r.selectedLod = 1;
                        ++prunedSolidNodes;
                        processed = true;
                    } else if(shapeSdfCenter > halfDiagonal) {
                        SDF::copySDF(r.shapeSDF, r.resultSDF);
                        r.shapeType = SpaceType::Empty;
                        r.resultType = SpaceType::Empty;
                        r.selectedLod = 1;
                        ++prunedEmptyNodes;
                        processed = true;
                    }
                } else {
                    // Non-propagating (Paint, Delete): INFINITY op anything is
                    // INFINITY = Empty, so the result stays the default
                    // (all-INFINITY). The shape's own values must NOT leak
                    // into the parent aggregation — there is no geometry here.
                    r.shapeType = SDF::eval(r.shapeSDF);
                    r.resultType = SpaceType::Empty;
                    r.selectedLod = 1;
                    ++prunedEmptyNodes;
                    processed = true;
                }
            }
            
        }

        // Operations that preserve Solid — existing Solid → always Solid everywhere
        // The allFinite guard: only prune when the existing field is fully
        // resolved (no INFINITY corners). An interpolated/partial field must
        // descend so its corners get real values.
        if(!processed && frame.type == SpaceType::Solid &&
           args.operation->preservesSolid()) {
            bool allFinite = true;
            for(int i = 0; i < 8; ++i)
                if(frame.sdf[i] == INFINITY) { allFinite = false; break; }
            if(allFinite) {

            // Painting prunes the whole subtree here (no descent to leaves
            // where the painter normally runs), so apply the painter at the
            // node center to keep the paint stroke visible in solid regions.
            if(args.operation->paintsVertices() && r.shapeType != SpaceType::Empty) {
                if(r.node != NULL) {
                    r.brushIndex = args.painter.paint(r.node->vertex);
                    r.brushHsv = args.painter.paintHSV(r.node->vertex);
                    r.node->vertex.normal = SDF::getNormalFromPosition(r.resultSDF, frame.cube, r.node->vertex.position);
                    r.node->vertex.hsv = r.brushHsv;
                    r.node->vertex.brushIndex = r.brushIndex;
                    r.node->setBrush(r.brushIndex);
                }
            }
            // The parent aggregates its corners from this child's resultSDF,
            // so an unresolved default (all INFINITY, Empty) here would poison
            // every ancestor's SDF while returning. Combine the existing field
            // with the shape so the propagated values match the true result
            // (Add keeps the min; Paint's opPaint is the identity on the SDF).
            for(uint i = 0; i < 8; ++i) {
                r.resultSDF[i] = args.operation->combine(frame.sdf[i], r.shapeSDF[i]);
            }
            r.resultType = SpaceType::Solid;
            ++prunedSolidNodes;
            processed = true;
            }
        }

        // Operations that preserve Empty — existing Empty → always Empty everywhere
        // allFinite guard: an unresolved (INFINITY-corner) field must descend.
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
                ++prunedEmptyNodes;
                processed = true;
            }
        }

        // For remaining operations (or types): check if the shape center is far enough
        // that the entire cube is definitely Solid/Empty.
        // allFinite guard: interpolating an unresolved field (INFINITY corners)
        // would compare garbage — the cell must descend instead.
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
                // Same painting concern as the preserves-Solid prune above:
                // apply the painter at the node center so painted operations
                // pruned here still leave a visible stroke.
                if(args.operation->paintsVertices() && r.shapeType != SpaceType::Empty) {
                    if(r.node != NULL) {
                        r.brushIndex = args.painter.paint(r.node->vertex);
                        r.brushHsv = args.painter.paintHSV(r.node->vertex);
                        r.node->vertex.normal = SDF::getNormalFromPosition(r.resultSDF, frame.cube, r.node->vertex.position);
                        r.node->vertex.hsv = r.brushHsv;
                        r.node->vertex.brushIndex = r.brushIndex;
                        r.node->setBrush(r.brushIndex);
                    }
                }
                ++prunedSolidNodes;
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

            // A node-less cell whose frame carries a Surface type holds a field
            // interpolated from an ancestor (or a stale inherited type). It must
            // descend to the shape frontier so the field is materialized as real
            // nodes at every level: the leaf combine is exact at the actual
            // corners for every operation, and resolved children prune cheaply.
            // Stopping here would instead keep a single coarse simplified node
            // tessellated directly from the trilinear field — the shape would
            // never iterate down to the leaves of interpolated surfaces.
            if(!process && frame.node == NULL && frame.type == SpaceType::Surface) {
                process = true;
            }
            if(process) {    
                shapeChildren(frame, args, threadContext, children, updateHandler, deleteHandler);
            } else {
                // Shape does not reach this cell (center beyond half-diagonal,
                // so the shape is positive at every corner) — but it may still
                // be NEARER than the existing field (e.g. a sphere surface just
                // outside an existing box: min(existing, shape) < existing).
                // The result is the exact combined field at the corners; the
                // pre-existing field alone would poison the parent aggregation
                // with stale values (and the parent's setChildren copies them
                // into nodes).
                r.shapeType = SDF::eval(r.shapeSDF);
                r.resultType = frame.type;
                if(frame.type == SpaceType::Empty) ++prunedEmptyNodes;
                for(uint i = 0; i < 8; ++i) {
                    r.resultSDF[i] = args.operation->combine(frame.sdf[i], r.shapeSDF[i]);
                }
            }
        }
    }

    // Leaf: combine shape with the existing field; non-leaf with computed
    // children: aggregate upward. Pruned/disjoint cells were fully resolved
    // above and must NOT be re-aggregated from default (Empty) children.
    if(r.isLeaf || (process && !processed && !r.isLeaf)) {
        buildShapeSDF(args, frame, r, children, threadContext, false);
        buildResultSDF(args, frame, r, children, threadContext);
    }

    bool interpolatedSurface = (frame.node == NULL)
                                && SDF::eval(frame.sdf) == SpaceType::Surface 
                                && frame.type == SpaceType::Surface
                                ;

    // Gate: write/refresh the node when the result has geometry (non-Empty
    // shape or interpolated surface) OR when a node already exists. The last
    // clause is essential: a node whose corners the shape alone sees as Empty
    // (e.g. a tiny sphere inside a large box cell — all sphere corners
    // positive, so shapeType==Empty) still needs its combined resultSDF
    // written back, otherwise it keeps stale pre-combine corners.
    if(r.shapeType != SpaceType::Empty || interpolatedSurface || r.node != NULL) {
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
                        r.brushIndex = args.painter.paint(r.node->vertex);
                        r.node->vertex.hsv = args.painter.paintHSV(r.node->vertex);
                        r.node->vertex.brushIndex = r.brushIndex;
                        r.brushHsv = r.node->vertex.hsv;
                    }  
                } else if(process) {
                    // Only manage children when this cell actually descended
                    // (process=true): the children[] array then holds the real
                    // per-child results. With process=false the array is empty
                    // (default Empty results), so the simplifier and the
                    // childNodes loop below would fabricate eight Empty
                    // children and overwrite the existing subtree with
                    // INFINITY-corner dummies. The cell still gets its
                    // combined corners written below.
                    if (!r.isChunk) {
                        // Pass frame.chunkCube so the simplifier can guard chunk borders.
                        SimplificationResult simplificationResult = args.simplifier.simplify(frame.cube, r.resultSDF, children, frame.chunkCube);
                        r.selectedLod = simplificationResult.isSimplified ? 1 : 0;
                    }
                    OctreeNode * childNodes[8] = {NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL};
                    for(uint i =0 ; i < 8 ; ++i) {
                        NodeOperationResult &child = children[i];
                        OctreeNode * childNode = child.node;
                        if(child.resultType != SpaceType::Surface || childNode == NULL) {
                            // A Surface result normally owns its node (created
                            // by its own shape() run). Degenerate cells — e.g.
                            // a delete rim exactly touching a corner, whose
                            // -0.0 corner classifies the cell as Surface with
                            // all-zero-or-negative corners — may return without
                            // one; never store a NULL slot for them.
                            if(childNode == NULL) {
                                BoundingCube childCube = frame.cube.getChild(i);
                                childNode = allocator->allocate()->init(Vertex(childCube.getCenter()));
                                children[i].node = childNode;
                            }
                            childNode->setType(child.resultType);
                            childNode->setSDF(child.resultSDF);
                            childNode->setChunk(child.isChunk);
                            childNode->setBrush(r.brushIndex);
                            childNode->setChunkLod(child.selectedChunkLod);
                            childNode->setLod(child.selectedLod);
                            childNode->vertex.hsv = child.brushHsv;
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
            r.node->setChunk(r.isChunk);

            if(r.resultType == SpaceType::Surface) {
                if(!r.node->isLeaf()) {
                    OctreeNode *childNodes[8] = { NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL };
                    r.node->getChildren(*allocator, childNodes);
                    r.selectedLod = 0;
                    r.selectedChunkLod = 0;
                    // Propagate the most common brushIndex among children
                    // (excluding DISCARD_BRUSH_INDEX) so every node from all LoD
                    // levels carries the dominant material of its subtree; coarse
                    // cells then texture as the majority brush instead of
                    // inheriting an arbitrary single child's brush. The winning
                    // child's hsv travels with the brush so the painted tint
                    // reaches the root; if no child carries a brush, keep this
                    // node's own brush/hsv instead of resetting to DISCARD.
                    std::unordered_map<int, int> brushCounts;
                    int bestCount = 0;
                    for(OctreeNode * childNode : childNodes) {
                        if(childNode != NULL && childNode->getType() == SpaceType::Surface) {
                            const uint8_t childLod = childNode->getLod();
                            const uint8_t childChunkLod = childNode->getChunkLod();

                            r.selectedLod = (r.selectedLod == 0 ? childLod : glm::min(r.selectedLod, childLod));
                            r.selectedChunkLod = (r.selectedChunkLod == 0 ? childChunkLod : glm::min(r.selectedChunkLod, childChunkLod));
        
                            const int childBrush = childNode->getBrush();
                            if(childBrush > DISCARD_BRUSH_INDEX) {
                                const int count = ++brushCounts[childBrush];
                                if(count > bestCount) {
                                    bestCount = count;
                                    r.brushIndex = childBrush;
                                    r.brushHsv = childNode->vertex.hsv;
                                }
                            }
                        }
                    }

                }
            }
            else if(process) {
                // Compression: a fully Solid/Empty subtree collapses to a leaf
                // (children are released) before the lod rule runs, so the
                // collapsed node follows the leaf rule, not max(child)+1.
                // Only when the cell descended (process=true) do the children
                // represent this subtree's real state; a process=false cell's
                // children were never re-evaluated and must be preserved.
                r.node->clear(*allocator, NULL);
            }


            r.node->setBrush(r.brushIndex);
            r.node->vertex.hsv = r.brushHsv;
          
            if(r.isChunk) {
                r.node->setChunkLod(1);
                r.selectedChunkLod = 1u;
            }
            else {
                r.node->setChunkLod(r.selectedChunkLod == 0 ? 0 : r.selectedChunkLod + 1);
            }
            
            if(r.node->isLeaf()) {
                // A leaf's stored lod is its TRUE ladder level, derived from
                // its size (lodForCellSize): frontier leaves are 1, coarse
                // leaves left by coarser passes (e.g. the minSize=120 demo
                // box) carry their own level (2, 3, …) instead of claiming
                // the frontier. This propagates the true interpolated lod so
                // the walk emits each cell at exactly its ladder level.
                const uint8_t sizeLod = lodForCellSize(nodeLength, chunkSize);
                r.node->setLod(sizeLod);
                r.selectedLod = sizeLod;
            }
            else {
                r.node->setLod(r.selectedLod == 0 ? 0 : r.selectedLod + 1);
            }
            // Dispatch a mesh event for every cell with a chunkLod (stored
            // 1..5). Each cell publishes exactly ONE mesh tagged with its own
            // level; the GPU cull keeps only the cell level matching the
            // camera distance band (fine cells near, coarse cells far). Cells
            // whose Surface-Nets mesh comes out empty (no zero crossing at
            // that resolution) simply publish nothing — the walk above emits
            // only cells marked Surface, and the renderer skips empty
            // geometry.
            if(r.node->getChunkLod() > 0) {
                ++r.node->version;
                OctreeNodeData data = OctreeNodeData(frame.level, r.node, frame.cube, nullptr);
                r.resultType == SpaceType::Surface ? updateHandler(data) : deleteHandler(data);
            }
        }
    }

    // Coarse leaves skipped by the gate above (process=false with an existing
    // node — the shape does not reach this cell) never re-enter the lod rule,
    // so they keep whatever lod an earlier pass left behind: a 60^3/120^3 leaf
    // created by a coarser pass (e.g. the minSize=120 demo box) would keep
    // claiming the frontier (lod 1). Propagate its true interpolated lod here
    // so the walk emits it at its own ladder level and not at every level.
    if(r.node != NULL && r.node->isLeaf() && !r.isLeaf) {
        const uint8_t sizeLod = lodForCellSize(nodeLength, chunkSize);
        if(r.node->getLod() != sizeLod) {
            r.node->setLod(sizeLod);
        }
    }
}

void Octree::iterate(OctreeNodeData &data, const IterateHandler &iterateHandler, const IterateOrderHandler &getOrderHandler) {
    OctreeSharedLock lock(treeMutex);
	IteratorHandler handler;
	handler.iterate(*this, data, iterateHandler, getOrderHandler);
}

void Octree::iterate(const IterateHandler &iterateHandler, const IterateOrderHandler &getOrderHandler) {
    OctreeSharedLock lock(treeMutex);
    OctreeNodeData data(0, root, *this, nullptr);
	IteratorHandler handler;
	handler.iterate(*this, data, iterateHandler, getOrderHandler);
}

void Octree::iterateFlat(OctreeNodeData &data, const IterateHandler &iterateHandler, const IterateOrderHandler &getOrderHandler) {
    OctreeSharedLock lock(treeMutex);
    IteratorHandler handler;
    handler.iterateFlatIn(*this, data, iterateHandler, getOrderHandler);
}

void Octree::iterateMultiThreaded(const IterateHandler &iterateHandler, const IterateOrderHandler &getOrderHandler, const IterateThreadedHandler &iterateThreadedHandler) {
    OctreeSharedLock lock(treeMutex);
    OctreeNodeData data(0, root, *this, nullptr);
    IteratorHandler handler;
    handler.iterateMultiThreaded(*this, data, threadPool, iterateHandler, getOrderHandler, iterateThreadedHandler);
}

void Octree::iterateFlat(const IterateHandler &iterateHandler, const IterateOrderHandler &getOrderHandler) {
    OctreeSharedLock lock(treeMutex);
    OctreeNodeData data(0, root, *this, nullptr);
    IteratorHandler handler;
    handler.iterateFlatIn(*this, data, iterateHandler, getOrderHandler);
}

void Octree::iterateParallel(OctreeNodeData &data, const IterateHandler &iterateHandler, const IterateOrderHandler &getOrderHandler) {
    OctreeSharedLock lock(treeMutex);
    IteratorHandler handler;
    handler.iterateBFS(*this, data, iterateHandler, getOrderHandler);
}

void Octree::iterateParallel(const IterateHandler &iterateHandler, const IterateOrderHandler &getOrderHandler) {
    OctreeSharedLock lock(treeMutex);
    OctreeNodeData data(0, root, *this, nullptr);
    IteratorHandler handler;
    handler.iterateBFS(*this, data, iterateHandler, getOrderHandler);
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
    threadPool.stop();
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
        file << "\"hsv\":[" << v.hsv.x << "," << v.hsv.y << "," << v.hsv.z << "],";
        file << "\"bits\":" << (int)node->bits << ",";
        glm::vec3 cubeMin = cube.getMin();
        glm::vec3 cubeLen = cube.getLength();
        file << "\"min\":[" << cubeMin.x << "," << cubeMin.y << "," << cubeMin.z << "],";
        file << "\"length\":[" << cubeLen.x << "," << cubeLen.y << "," << cubeLen.z << "],";

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
        // hsv
        std::vector<double> hsvArr = { node->vertex.hsv.x, node->vertex.hsv.y, node->vertex.hsv.z };
        std::vector<uint8_t> hsvDoc = makeDoubleArrayDoc(hsvArr);
        doc.push_back(0x04); appendCString(doc, "hsv"); doc.insert(doc.end(), hsvDoc.begin(), hsvDoc.end());

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
