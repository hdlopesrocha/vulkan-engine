#include "OctreeTriangleTessellator.hpp"
#include "EdgeCell.hpp"
#include "EdgeSpan.hpp"
#include "Octree.hpp"
#include "OctreeNode.hpp"
#include "OctreeSharedLock.hpp"
#include "../sdf/SDF.hpp"
#include <algorithm>
#include <cmath>
#include <vector>

// Moved verbatim from Octree::iterateTriangles (space/Octree.cpp): the walk
// itself is unchanged, only the owning object became OctreeTriangleTessellator's
// `tree` member (member accesses are tree.*) and getNodeIndex(pos, cube) — a
// forwarder — became cube.getChildIndex(pos), which it is value-identical to.
// The walk's steps are members of Walk below (declared in the header, defined
// here); they used to be function-local lambdas.

// One walk: the fixed inputs, the per-call state (anchor and caches) and the
// steps of the walk. iterateTriangles() builds one per call.
//
// The steps used to be function-local lambdas. As members they are not
// constructed on every call, and the two recursive steps (collectBreaks,
// walkLadder) recurse directly instead of through a std::function.
struct OctreeTriangleTessellator::Walk {
    // Inputs, fixed for the duration of the walk.
    const Octree &tree;
    OctreeNode *from = NULL;
    const BoundingCube &fromCube;
    OctreeNodeTriangleHandler &func;
    OctreeThreadContext *context = NULL;
    int targetLod = 0;
    const std::function<Vertex(OctreeNode*, const BoundingCube&)> &vertexFn;

    // Per-call state.
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

    Walk(const Octree &tree_, OctreeNode *from_, const BoundingCube &fromCube_,
         OctreeNodeTriangleHandler &func_, OctreeThreadContext *context_, int targetLod_,
         const std::function<Vertex(OctreeNode*, const BoundingCube&)>& vertexFn_)
        : tree(tree_), from(from_), fromCube(fromCube_), func(func_),
          context(context_), targetLod(targetLod_), vertexFn(vertexFn_) {
        // Anchor the cache on `from`: its ancestor chain (seeded by the
        // traversal, rooted at the world root) is already in context->parentOf,
        // so neighbor lookups climb from a cached parent instead of the root.
        hint.node = from;
        hint.cube = fromCube;
    }

    static bool samePosition(const glm::vec3 &a, const glm::vec3 &b, float eps) {
        glm::vec3 d = a - b;
        return glm::dot(d, d) <= eps * eps;
    }

    // Descends to the cell containing pos, starting from the cached anchor
    // when context->parentOf can rebuild the parent chain, and memoizes it.
    EdgeCell findCellAt(const glm::vec3 &pos) {
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
    }

    static bool addBreak(std::vector<float> &breaks, float value, float start, float end, float eps) {
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
    }

    static void sortUnique(std::vector<float> &values, float eps) {
        std::sort(values.begin(), values.end());
        values.erase(std::unique(values.begin(), values.end(), [eps](float a, float b) {
            return std::fabs(a - b) <= eps;
        }), values.end());
    }

    // Fixed-buffer variant of addBreak/sortUnique to avoid a heap allocation in
    // the per-recursion localBreaks (its size is bounded; see collectBreaks).
    static bool addToBuffer(float *buf, int &count, int cap, float value, float start, float end, float eps) {
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
    }

    static void sortUniqueBuf(float *buf, int &count, float eps) {
        std::sort(buf, buf + count);
        float *newEnd = std::unique(buf, buf + count, [eps](float a, float b) {
            return std::fabs(a - b) <= eps;
        });
        count = int(newEnd - buf);
    }

    void collectBreaks(const EdgeSpan &edge, float start, float end, std::vector<float> &breaks, int depth) {
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
            EdgeCell cell = findCellAt(edge.samplePoint(mid, q));
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
    }

    Vertex nodeVertex(OctreeNode *n, const BoundingCube &c) const {
        if (vertexFn) return vertexFn(n, c);
        glm::vec3 pos = SDF::getPosition(n->sdf, c);
        glm::vec3 norm = SDF::getNormalFromPosition(n->sdf, c, pos);
        Vertex v(pos, norm, n->data, n->brushIndex);
        v.hsv = n->hsv;
        v.textureType = n->textureType;
        return v;
    }

    void emitTriangle(const Vertex &a, const Vertex &b, const Vertex &c, float eps) {
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
        const int fallbackTextureType = (from != NULL) ? from->textureType : 0;
        const glm::vec2 fallbackUv = (from != NULL) ? from->data : glm::vec2(0.0f);
        if(ca.brushIndex <= DISCARD_BRUSH_INDEX) { ca.brushIndex = fallbackBrush; ca.textureType = fallbackTextureType; ca.texCoord = fallbackUv; }
        if(cb.brushIndex <= DISCARD_BRUSH_INDEX) { cb.brushIndex = fallbackBrush; cb.textureType = fallbackTextureType; cb.texCoord = fallbackUv; }
        if(cc.brushIndex <= DISCARD_BRUSH_INDEX) { cc.brushIndex = fallbackBrush; cc.textureType = fallbackTextureType; cc.texCoord = fallbackUv; }
        if(ca.brushIndex <= DISCARD_BRUSH_INDEX || cb.brushIndex <= DISCARD_BRUSH_INDEX
            || cc.brushIndex <= DISCARD_BRUSH_INDEX) return;

        func.handle(ca, cb, cc);
    }

    void emitSegment(const EdgeSpan &edge, float start, float end, bool sign0, bool sign1) {
        float mid = start + (end - start) * 0.5f;
        EdgeCell cells[4];
        for(int q = 0; q < 4; ++q) {
            cells[q] = findCellAt(edge.samplePoint(mid, q));
        }

        // Owner = positionally-smallest SURFACE ring cell. Non-surface
        // quadrants (Empty/Solid space) never own the segment, but they do NOT
        // abort it either: the segment still closes the mesh at the surface
        // boundary (shape faces, chunk ladder seams, world edge) with the
        // polygon of the surface quadrants alone. Aborting there left open edge
        // chains along every face aligned with the cell grid. Quadrants of a
        // coarser ladder level are surface cells too and own the segment when
        // they are the only crossing, which is how coarse cells bridge against
        // finer LOD cells instead of leaving a gap at the band boundary.
        EdgeCell owner = cells[0];
        bool ownerValid = false;
        for(int q = 0; q < 4; ++q) {
            if(cells[q].isSurface() && (!ownerValid || EdgeCell::ownerLess(cells[q], owner))) {
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

        glm::vec3 p0 = edge.point(start);
        glm::vec3 p1 = edge.point(end);
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
            // The owner's interpolation can land exactly on a cell corner/face
            // where the field is zero. Zero is classified as positive, so both
            // endpoint signs collapse and the whole segment was dropped — the
            // missing surface along the x/y/z axis planes (and any boundary
            // where a contour vertex sits exactly on a cell corner). The
            // producing cell only emits edges whose own corners differ in sign,
            // so those signs are authoritative for existence; keep the owner's
            // ordering/winding whenever it does resolve the crossing.
            s0 = sign0;
            s1 = sign1;
        }
        if(s0 == s1) {
            return;
        }

        // Bounded stack buffer: built from the surface quadrant cells
        // (deduped), so it never exceeds 4 elements — avoids a per-segment
        // heap allocation. Non-surface quadrants are skipped: their nodes
        // have no usable vertex and their cells never tessellate.
        // Collect the distinct surface quadrants. A coarse ancestor can appear
        // in two OPPOSITE quadrants of the ring (common on the world-axis cell
        // boundaries, where the sample offsets stay inside one large cell). The
        // old code only removed adjacent duplicates, then aborted the whole
        // segment when it found any remaining pair — exactly the missing
        // triangles along x/y/z = 0. Dedup against every already-collected
        // quadrant instead; the remaining 3/4 corners still form the surface
        // polygon, and the order is preserved for triangulation.
        EdgeCell plist[4];
        Vertex pverts[4];
        int pcount = 0;
        for(int q = 0; q < 4; ++q) {
            if(!cells[q].isSurface()) {
                continue;
            }
            Vertex vq = nodeVertex(cells[q].node, cells[q].cube);
            bool duplicate = false;
            for(int k = 0; k < pcount; ++k) {
                if(plist[k].node == cells[q].node
                    || samePosition(pverts[k].position, vq.position, edge.eps)) {
                    duplicate = true;
                    break;
                }
            }
            if(!duplicate && pcount < 4) {
                plist[pcount] = cells[q];
                pverts[pcount] = vq;
                ++pcount;
            }
        }

        if(pcount > 1) {
            const EdgeCell &first = plist[0];
            const EdgeCell &last = plist[pcount - 1];
            if(first.node == last.node || samePosition(pverts[0].position, pverts[pcount - 1].position, edge.eps)) {
                --pcount;
            }
        }
        if(pcount < 3) {
            return;
        }

        // Rotate the polygon so the OWNER (the finest shared quadrant cell) is
        // at index 0. Both sides of a chunk boundary resolve the same owner,
        // so their duplicated boundary quads get the same vertex order and the
        // same triplanar UV plane. Rotating around `from` instead made each
        // chunk rotate the same quad to a different cell: the two copies
        // z-fought with different textures, leaving bright seams along every
        // chunk border.
        {
            int it = -1;
            for(int k = 0; k < pcount; ++k) {
                if(plist[k].node == owner.node
                    || samePosition(nodeVertex(plist[k].node, plist[k].cube).position,
                                    nodeVertex(owner.node, owner.cube).position, edge.eps)) {
                    it = k;
                    break;
                }
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
    }

    // Scan ONE Surface-Nets cell: walk the 12 edges of cellCube with the
    // cell's OWN stored corner samples and emit the segments whose owner
    // (finest of the four quadrant cells at the walk's resolution) lies
    // inside the outer from-cube. Quadrant cells of a coarser ladder level
    // stay in the segment (isSurface() accepts any surface node) so that a
    // chunk and its coarser neighbour emit the SAME polygon on a shared edge,
    // which is what closes the surface across LoD band boundaries; the finer
    // side's bridging triangles cover the level step. Non-surface cells
    // (Empty/Solid space) are still dropped and the polygon closes with the
    // remaining surface quadrants.
    void scanCell(OctreeNode *cellNode, const BoundingCube &cellCube) {
        for(int edgeIndex = 0; edgeIndex < 12; ++edgeIndex) {
            glm::ivec2 edgeCorners = SDF_EDGES[edgeIndex];
            bool sign0 = cellNode->sdf[edgeCorners.x] < 0.0f;
            bool sign1 = cellNode->sdf[edgeCorners.y] < 0.0f;

            if(sign0 == sign1) {
                continue;
            }

            EdgeSpan edge = EdgeSpan::fromCellEdge(edgeIndex, cellCube, tree.getLengthX());
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
    }

    // Ladder mode: the level-k mesh is the AGGREGATE of the cells at lod k
    // inside the anchor (each cell's own stored corner samples, which are
    // correct). A single Surface-Nets cell over the whole anchor would miss
    // interior surface detail — the anchor's own corners have no zero
    // crossing when the surface is inside it — so every level emits its own
    // cell resolution (cell size frontierCell*2^k), which is what the
    // distance bands consume.
    void walkLadder(OctreeNode *node, const BoundingCube &cube) {
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

    }

    // Runs the walk: rejects non-ladder anchors, then walks the ladder.
    void run() {
        // Accept any ladder cell as the `from` anchor: a frontier simplified leaf
        // (simplification 1, developed raw lod == 1 in the stored space) or an
        // internal ancestor (simplification 0 but developed > 0). Reject only
        // non-ladder leaves (stored lod 0).
        if(from == NULL || from->getType() != SpaceType::Surface ||
           (from->getLod() == 0u && from->getLod() == 0)) {
            return;
        }

        walkLadder(from, fromCube);
    }
};

void OctreeTriangleTessellator::iterateTriangles(
        OctreeNode * from,
        const BoundingCube &fromCube,
        int fromLevel,
        OctreeNodeTriangleHandler &func,
        OctreeThreadContext * context,
        int targetLod,
        const std::function<Vertex(OctreeNode*, const BoundingCube&)>& vertexFn) {
    OctreeSharedLock lock(tree.treeMutex);

    Walk walk(tree, from, fromCube, func, context, targetLod, vertexFn);
    walk.run();
}
