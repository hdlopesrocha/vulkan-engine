#include "Processor.hpp"
#include "Octree.hpp"
#include "OctreeNode.hpp"
#include "ThreadPool.hpp"
#include "Tesselator.hpp"


Processor::Processor(long * count_, ThreadPool &threadPool_, ThreadContext * context_, const BoundingCube &targetCube_, float * cellSizeOut_): threadPool(threadPool_), context(context_), count(count_), targetCube(targetCube_) {

}

bool Processor::iterate(const Octree &tree, OctreeNodeData &params) {
    // Only Surface nodes can contain surface geometry. Node type is the
    // conservative union of its children (see childToParent), so an Empty/Solid
    // node has no Surface descendants — skip the entire subtree instead of
    // walking empty space. This is the dominant traversal cost cut: the octree
    // is mostly void, and we now iterate only the surface shell.
    if(params.node->getType() != SpaceType::Surface) {
        return false;
    }
    // Prune everything off this chunk's root path: only the nodes whose cube
    // contains the chunk (the chunk and its ancestors) contribute to this
    // request's ladder — sibling chunks' subtrees are skipped entirely. The
    // ancestor tessellations still cover their whole regions internally.
    // Without this prune every chunk event re-tessellates the WHOLE tree and
    // the walk emits every chunk's level-0 mesh — which the caller publishes
    // under the EVENT chunk's NodeID, so slots end up holding another chunk's
    // finest geometry ("wrong chunkLod geometry").
    if(!params.cube.contains(targetCube.getCenter())) {
        return false;
    }
    // ONE Tesselator per node, from the WORLD ROOT down: every node on the
    // chunk's root path has a chunkLod and tessellates at targetLod = its own
    // chunkLod — the chunk (stored chunkLod 1) emits the frontier level-0
    // mesh, each ancestor (stored chunkLod k) emits the lod-k cell mesh. One
    // walk therefore returns the whole ladder (all lods, no decimation), and
    // the node's own root-descended cube (handed to the handler) is the
    // tessellation bounds — no reconstructed cubes, no drift.
    // lod/chunkLod are stored +1 shifted (0 = unset), so the chunk's stored
    // chunkLod is 1 and its ancestors are 1,2,3... The walker compares stored
    // values against this same stored target, keeping the whole ladder walk
    // self-consistent. Only the level handed to onGeometry (the renderer's
    // 0-based LADDER LEVEL: 0 = frontier mesh) is decoded back by (-1).
    const uint8_t chunkLodStored = params.node->getChunkLod();
    if(chunkLodStored > 0) {
        long trianglesCount = 0;
        Tesselator nodeTesselator(&trianglesCount);
        tree.iterateTriangles(params.node, params.cube, params.level, nodeTesselator, context, chunkLodStored);
        if(!nodeTesselator.geometry.indices.empty()) {
            onGeometry(params.level, params.node, params.cube, nodeTesselator.geometry, chunkLodStored);
        }
    }
    // Keep descending along the root path: the children hold the finer ladder
    // levels. Cells without a chunkLod (stored 0) never tessellate and end the
    // walk — their parent links are already propagated for neighbor lookups.
    return chunkLodStored > 0;
}

void Processor::getOrder(const Octree &tree, OctreeNodeData &params, uint8_t * order){
    for(int i = 0 ; i < 8 ; ++i) {
        order[i] = i;
    }
}
