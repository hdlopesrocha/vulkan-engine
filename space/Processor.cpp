#include "Processor.hpp"
#include "Octree.hpp"
#include "OctreeNode.hpp"
#include "ThreadPool.hpp"
#include "Tesselator.hpp"


Processor::Processor(long * count_, ThreadPool &threadPool_, ThreadContext * context_, const BoundingCube &targetCube_, float * cellSizeOut_): threadPool(threadPool_), context(context_), count(count_), targetCube(targetCube_), firstCellSize(0.0f), cellSizeOut(cellSizeOut_) {

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
    if(!params.cube.contains(targetCube.getCenter())) {
        return false;
    }
    // ONE Tesselator per node, from the WORLD ROOT down: every node on the
    // chunk's root path has chunkLod >= 0 and tessellates at targetLod = its
    // own chunkLod — the chunk (chunkLod 0) emits the frontier level-0 mesh,
    // each ancestor (chunkLod k) emits the lod-k cell mesh. One walk therefore
    // returns the whole ladder (all lods, no decimation), and the node's own
    // root-descended cube (handed to the handler) is the tessellation bounds —
    // no reconstructed cubes, no drift.
    const int nodeChunkLod = params.node->getChunkLod();
    if(nodeChunkLod >= 0) {
        // Capture the frontier (level-0) cell size for the caller; coarse
        // ancestor cells do not count (they are log-distance banded).
        if(cellSizeOut && firstCellSize == 0.0f && params.node->getLod() == 0) {
            firstCellSize = params.cube.getLengthX();
        }
        long trianglesCount = 0;
        Tesselator nodeTesselator(&trianglesCount);
        tree.iterateTriangles(params.node, params.cube, params.level, nodeTesselator, context, nodeChunkLod);
#ifdef DEBUG
        std::cout << "[proc] chunkLod=" << nodeChunkLod << " lod=" << (int)params.node->getLod()
                  << " cube=" << params.cube.getLengthX() << " tris=" << trianglesCount
                  << " verts=" << nodeTesselator.geometry.vertices.size() << std::endl;
#endif
        if(onGeometry && !nodeTesselator.geometry.indices.empty()) {
            onGeometry(nodeChunkLod, params.node, params.cube, std::move(nodeTesselator.geometry), nodeChunkLod);
        }
    }
    // Keep descending along the root path: the children hold the finer ladder
    // levels. Cells below chunks (chunkLod -1) never tessellate and end the
    // walk — their parent links are already propagated for neighbor lookups.
    return nodeChunkLod != -1;
}

void Processor::getOrder(const Octree &tree, OctreeNodeData &params, uint8_t * order){
    for(int i = 0 ; i < 8 ; ++i) {
        order[i] = i;
    }
}
