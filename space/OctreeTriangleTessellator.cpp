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
    int targetLod = 0;
    const std::function<CellVertex(OctreeNode*, const BoundingCube&)> &vertexFn;

    // Per-call memo for findCellAt. The 12 edges and their segments repeatedly
    // query identical quadrant/face sample positions (the same edge midpoint is
    // sampled by both collectBreaks and emitSegment, and adjacent edges share
    // corners), so memoizing avoids repeated O(depth) descents. findCellAt is a
    // pure function of pos during the walk, so cached results are bit-identical
    // to a fresh lookup.
    tsl::robin_map<glm::vec3, EdgeCell> cellCache;

    Walk(const Octree &tree_, OctreeNode *from_, const BoundingCube &fromCube_,
         OctreeNodeTriangleHandler &func_, int targetLod_,
         const std::function<CellVertex(OctreeNode*, const BoundingCube&)>& vertexFn_)
        : tree(tree_), from(from_), fromCube(fromCube_), func(func_),
          targetLod(targetLod_), vertexFn(vertexFn_) {}

    static bool samePosition(const glm::vec3 &a, const glm::vec3 &b, float eps) {
        glm::vec3 d = a - b;
        return glm::dot(d, d) <= eps * eps;
    }

    // Surface point + normal of a cell, supplied by the caller's vertexFn (the
    // Blender exporter's SDF solver). The walk never solves a Cartesian vertex
    // itself: runtime chunk vertices are GPU-only.
    CellVertex cellVertex(OctreeNode *n, const BoundingCube &c) const {
        return vertexFn(n, c);
    }

    // Descends from the world root to the cell containing pos and memoizes
    // the result. Cubes are replayed root-consistently, so the result is
    // bit-identical to the GPU's root descent.
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

        OctreeNode *node = tree.root;
        BoundingCube cube = static_cast<const BoundingCube&>(tree);

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

    // Emit one triangle from its three corner CELLS: each corner's node plus
    // that cell's root-consistent cube. The corner point/normal is resolved by
    // the caller's vertexFn, so no Vertex crosses the walk and the material is
    // taken off the node by the handler. CPU mirror of the GPU's
    // emitTriangle(Cell, Cell, Cell, eps) with cellPosCached/cellNormal.
    void emitTriangle(OctreeNode &a, OctreeNode &b, OctreeNode &c,
                      const BoundingCube &ca, const BoundingCube &cb, const BoundingCube &cc,
                      float eps) {
        const CellVertex va = cellVertex(&a, ca);
        const CellVertex vb = cellVertex(&b, cb);
        const CellVertex vc = cellVertex(&c, cc);
        // Degenerate-triangle guard on the resolved surface points (the same test
        // the shader applies); the polygon is already deduped in emitSegment.
        if(samePosition(va.position, vb.position, eps)) return;
        if(samePosition(vb.position, vc.position, eps)) return;
        if(samePosition(vc.position, va.position, eps)) return;
        // A discard-brush vertex was never painted: do not tessellate it and do
        // not borrow the anchor chunk's brush (that resurrected unpainted cells
        // as geometry with a borrowed material). Same rule as the GPU
        // emitTriangle and the ring gate in EdgeCell::isSurface.
        if(a.brushIndex <= DISCARD_BRUSH_INDEX || b.brushIndex <= DISCARD_BRUSH_INDEX
            || c.brushIndex <= DISCARD_BRUSH_INDEX) return;

        func.handle(a, b, c, va, vb, vc, eps);
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
        // Corner point/normal of each kept cell, resolved once by cellVertex
        // and reused for the dedup and the owner rotation, so the walk never
        // builds a Vertex. The emit below re-resolves per triangle corner via
        // emitTriangle; this array is not rotated with plist.
        CellVertex pverts[4];
        int pcount = 0;
        for(int q = 0; q < 4; ++q) {
            if(!cells[q].isSurface()) {
                continue;
            }
            const CellVertex vq = cellVertex(cells[q].node, cells[q].cube);
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
            const CellVertex ownerV = cellVertex(owner.node, owner.cube);
            int it = -1;
            for(int k = 0; k < pcount; ++k) {
                if(plist[k].node == owner.node
                    || samePosition(pverts[k].position, ownerV.position, edge.eps)) {
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
                emitTriangle(*plist[0].node, *plist[1].node, *plist[2].node, plist[0].cube, plist[1].cube, plist[2].cube, edge.eps);
            else
                emitTriangle(*plist[0].node, *plist[2].node, *plist[1].node, plist[0].cube, plist[2].cube, plist[1].cube, edge.eps);
        } else if(pcount == 4) {
            if(solidAtStart) {
                emitTriangle(*plist[0].node, *plist[1].node, *plist[2].node, plist[0].cube, plist[1].cube, plist[2].cube, edge.eps);
                emitTriangle(*plist[0].node, *plist[2].node, *plist[3].node, plist[0].cube, plist[2].cube, plist[3].cube, edge.eps);
            } else {
                emitTriangle(*plist[0].node, *plist[3].node, *plist[2].node, plist[0].cube, plist[3].cube, plist[2].cube, edge.eps);
                emitTriangle(*plist[0].node, *plist[2].node, *plist[1].node, plist[0].cube, plist[2].cube, plist[1].cube, edge.eps);
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
            // A discard-brush cell was never painted: do not even scan it.
            // Mirrors the GPU scan-list filter in GpuTopologyBuilder::collectCells.
            if(node->brushIndex > DISCARD_BRUSH_INDEX) {
                scanCell(node, cube);
            }
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
           from->getLod() == 0u) {
            return;
        }

        walkLadder(from, fromCube);
    }
};

void OctreeTriangleTessellator::iterateTriangles(
        OctreeNode * from,
        const BoundingCube &fromCube,
        OctreeNodeTriangleHandler &func,
        int targetLod,
        const std::function<CellVertex(OctreeNode*, const BoundingCube&)>& vertexFn) {
    if (!vertexFn) return; // no solver -> no Cartesian output (Blender-only path)
    OctreeSharedLock lock(tree.treeMutex);

    Walk walk(tree, from, fromCube, func, targetLod, vertexFn);
    walk.run();
}
