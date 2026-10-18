#include "OctreeTriangleTessellator.hpp"
#include "Octree.hpp"
#include "OctreeNode.hpp"
#include "OctreeSharedLock.hpp"
#include "../sdf/SDF.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

// Moved verbatim from Octree::iterateTriangles (space/Octree.cpp): the walk
// itself is unchanged, only the owning object became a `const Octree&`
// parameter (member accesses are tree.*) and getNodeIndex(pos, cube) — a
// forwarder — became cube.getChildIndex(pos), which it is value-identical to.

namespace octess {

void iterateTriangles(const Octree &tree,
                      OctreeNode * from,
                      const BoundingCube &fromCube,
                      int fromLevel,
                      OctreeNodeTriangleHandler &func,
                      ThreadContext * context,
                      int targetLod,
                      const std::function<Vertex(OctreeNode*, const BoundingCube&)>& vertexFn) {
    OctreeSharedLock lock(tree.treeMutex);

    struct EdgeCell {
        OctreeNode *node = NULL;
        BoundingCube cube;
        int level = 0;

        // A cell is "surface at the walk's resolution": a ladder cell exactly
        // at targetLod. With targetLod >= 1 the descent already stops at cells
        // with lod == targetLod, so requiring lod equality here keeps coarse
        // ladder levels (internal, non-simplified nodes) emitting their own
        // cells while neighbors one level finer/coarser stay out. lod and
        // targetLod are both in the +1-shifted STORED space.
        bool isSurface(int targetLod) const {
            return node != NULL && node->getType() == SpaceType::Surface &&
                node->getLod() == targetLod;
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

    // `&tree` replaces the implicit `this` of the original member function.
    auto findCellAt = [&tree, context, &hint, &fromCube, &cachedChainNode, &cachedChainNodes,
            &cachedChainIndices, &cachedChainLen, &cachedChainOk, &cellCache, targetLod](const glm::vec3 &pos) {
        auto cacheHit = cellCache.find(pos);
        if(cacheHit != cellCache.end()) {
            return cacheHit->second;
        }
        EdgeCell result;
        if(tree.root == NULL || !tree.contains(pos)) {
            cellCache[pos] = result;
            return result;
        }

        // Default to the original root descent; the cache below only shortens it.
        OctreeNode *startNode = tree.root;
        BoundingCube startCube = static_cast<const BoundingCube&>(tree);

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
            // contains pos. (Depth is bounded; MAX_CHAIN is a generous upper
            // bound. If exceeded we simply fall through to the default root
            // descent below — correct, just less optimal.)
            // The chain is cached in cachedChain* and only rebuilt when the
            // anchor (hint.node) changes; it is independent of pos.
            if(hint.node != cachedChainNode) {
                cachedChainNode = hint.node;
                cachedChainLen = 0;
                cachedChainOk = true;
                OctreeNode *n = hint.node;
                while(n != NULL && n != tree.root) {
                    auto it = context->parentOf.find(n);
                    if(it == context->parentOf.end()) { cachedChainOk = false; break; }
                    if(cachedChainLen >= MAX_CHAIN) { cachedChainOk = false; break; }
                    cachedChainNodes[cachedChainLen] = n;
                    cachedChainIndices[cachedChainLen] = it->second.second;
                    ++cachedChainLen;
                    n = it->second.first;
                }
                if(!(cachedChainOk && n == tree.root && cachedChainLen > 0)) {
                    cachedChainOk = false;
                } else {
                    for(int i = 0; i < cachedChainLen / 2; ++i) {
                        std::swap(cachedChainNodes[i], cachedChainNodes[cachedChainLen - 1 - i]);
                        std::swap(cachedChainIndices[i], cachedChainIndices[cachedChainLen - 1 - i]);
                    }
                }
            }
            if(cachedChainOk) {
                BoundingCube c = static_cast<const BoundingCube&>(tree);
                int chosen = -1;
                for(int i = 0; i < cachedChainLen; ++i) {
                    c = c.getChild(cachedChainIndices[i]);
                    if(c.contains(pos)) chosen = i;
                    else break;
                }
                if(chosen >= 0) {
                    BoundingCube cc = static_cast<const BoundingCube&>(tree);
                    for(int i = 0; i <= chosen; ++i) cc = cc.getChild(cachedChainIndices[i]);
                    startNode = cachedChainNodes[chosen];
                    startCube = cc;
                }
            }
        }

        OctreeNode *node = startNode;
        BoundingCube cube = startCube;

        while(node != NULL && !node->isLeaf() &&
              (node->getLod() > targetLod)) {
            ChildBlock *block = node->getBlock(*tree.allocator);
            if(block == NULL) {
                node = NULL;
                break;
            }

            int childIndex = cube.getChildIndex(pos);
            OctreeNode *child = block->get(childIndex, *tree.allocator);
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
        if(end - start <= edge.eps * 2.0f) {
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
        edge.eps = glm::max(tree.getLengthX(), cellCube.getLengthX()) * 1e-6f;
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

    auto nodeVertex = [&](OctreeNode *n, const BoundingCube &c) -> Vertex {
        if (vertexFn) return vertexFn(n, c);
        glm::vec3 pos = SDF::getPosition(n->sdf, c);
        glm::vec3 norm = SDF::getNormalFromPosition(n->sdf, c, pos);
        Vertex v(pos, norm, glm::vec2(0.0f), n->brushIndex);
        v.hsv = n->hsv;
        return v;
    };

    auto emitTriangle = [&](const Vertex &a, const Vertex &b, const Vertex &c, float eps) {
        // a/b/c are vertices (not pointers) now; no null check needed.
        if(samePosition(a.position, b.position, eps)) return;
        if(samePosition(b.position, c.position, eps)) return;
        if(samePosition(c.position, a.position, eps)) return;
        // Brush fallback for unattributed vertices: rung-parent chunks
        // aggregate finer non-chunk cells whose vertices were solved
        // (positions valid) but never painted (brush DISCARD), because paint
        // runs on chunk assignment, not on subdivision. Such triangles are
        // real surface; attribute them with the anchor chunk's brush (`from`
        // carries the rung being tessellated) instead of dropping them, which
        // left holes in chunk wireframes/meshes. Copies only — shared octree
        // vertex state is never mutated. When the anchor itself is
        // unattributed the old drop behavior is kept (Tesselator::handle
        // below enforces the same rule).
        Vertex ca = a, cb = b, cc = c;
        const int fallbackBrush = (from != NULL) ? from->brushIndex : DISCARD_BRUSH_INDEX;
        if(ca.brushIndex <= DISCARD_BRUSH_INDEX) ca.brushIndex = fallbackBrush;
        if(cb.brushIndex <= DISCARD_BRUSH_INDEX) cb.brushIndex = fallbackBrush;
        if(cc.brushIndex <= DISCARD_BRUSH_INDEX) cc.brushIndex = fallbackBrush;
        if(ca.brushIndex <= DISCARD_BRUSH_INDEX || cb.brushIndex <= DISCARD_BRUSH_INDEX
            || cc.brushIndex <= DISCARD_BRUSH_INDEX) return;

        func.handle(ca, cb, cc);
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

        // Emit the segment for every cell that owns the edge. scanCell already
        // only calls emitSegment for edges whose own corners differ in sign, so
        // a shared edge between two adjacent cells is produced by both cells,
        // giving a watertight seam. We intentionally do NOT skip the segment
        // based on whether the owner cell lies inside `from`'s cube: gating on
        // the owner's position dropped one side of a chunk-boundary edge, leaving
        // the neighbouring chunk's triangle missing and a crack in the mesh.
        // Each chunk tessellates its own cells, so both sides of a shared edge
        // are always emitted. This duplicates boundary geometry but keeps the
        // mesh watertight across chunk seams.

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
                    || samePosition(nodeVertex(plist[pcount - 1].node, plist[pcount - 1].cube).position, nodeVertex(cells[q].node, cells[q].cube).position, edge.eps));
            if(!duplicate && pcount < 4) {
                plist[pcount++] = cells[q];
            }
        }

        if(pcount > 1) {
            const EdgeCell &first = plist[0];
            const EdgeCell &last = plist[pcount - 1];
            if(first.node == last.node || samePosition(nodeVertex(first.node, first.cube).position, nodeVertex(last.node, last.cube).position, edge.eps)) {
                --pcount;
            }
        }

        for(int i = 0; i < pcount; ++i) {
            for(int j = i + 1; j < pcount; ++j) {
                if(plist[i].node == plist[j].node
                    || samePosition(nodeVertex(plist[i].node, plist[i].cube).position, nodeVertex(plist[j].node, plist[j].cube).position, edge.eps)) {
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
                emitTriangle(nodeVertex(plist[0].node, plist[0].cube), nodeVertex(plist[1].node, plist[1].cube), nodeVertex(plist[2].node, plist[2].cube), edge.eps);
            else
                emitTriangle(nodeVertex(plist[0].node, plist[0].cube), nodeVertex(plist[2].node, plist[2].cube), nodeVertex(plist[1].node, plist[1].cube), edge.eps);
        } else if(pcount == 4) {
            if(solidAtStart) {
                emitTriangle(nodeVertex(plist[0].node, plist[0].cube), nodeVertex(plist[1].node, plist[1].cube), nodeVertex(plist[2].node, plist[2].cube), edge.eps);
                emitTriangle(nodeVertex(plist[0].node, plist[0].cube), nodeVertex(plist[2].node, plist[2].cube), nodeVertex(plist[3].node, plist[3].cube), edge.eps);
            } else {
                emitTriangle(nodeVertex(plist[0].node, plist[0].cube), nodeVertex(plist[3].node, plist[3].cube), nodeVertex(plist[2].node, plist[2].cube), edge.eps);
                emitTriangle(nodeVertex(plist[0].node, plist[0].cube), nodeVertex(plist[2].node, plist[2].cube), nodeVertex(plist[1].node, plist[1].cube), edge.eps);
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

    // Ladder mode: the level-k mesh is the AGGREGATE of the cells at lod k
    // inside the anchor (each cell's own stored corner samples, which are
    // correct). A single Surface-Nets cell over the whole anchor would miss
    // interior surface detail — the anchor's own corners have no zero
    // crossing when the surface is inside it — so every level emits its own
    // cell resolution (cell size frontierCell*2^k), which is what the
    // distance bands consume.
    std::function<void(OctreeNode*, const BoundingCube&)> walkLadder = [&](OctreeNode *node, const BoundingCube &cube) {
        if(node == NULL) return;
        // Stored (+1-shifted → uint8) lod: 0 = unset, 1 = frontier, k+1 = parent.
        const uint8_t lod = node->getLod();
        if(lod == targetLod) {
            scanCell(node, cube);
        }
        else if(lod > targetLod) {
            ChildBlock *block = node->getBlock(*tree.allocator);
            if(block == NULL) {
                return;
            }
            for(uint i = 0; i < 8; ++i) {
                OctreeNode *child = block->get(i, *tree.allocator);
                if(child != NULL) {
                    walkLadder(child, cube.getChild(i));
                }
            }
        }

    };
    walkLadder(from, fromCube);
}

} // namespace octess
