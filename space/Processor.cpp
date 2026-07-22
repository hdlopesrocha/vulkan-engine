#include "Processor.hpp"
#include "Octree.hpp"
#include "OctreeNode.hpp"
#include "ThreadPool.hpp"
#include "ForwardingHandler.hpp"


Processor::Processor(long * count_, ThreadPool &threadPool_, ThreadContext * context_, std::vector<OctreeNodeTriangleHandler*> * handlers): threadPool(threadPool_), context(context_), count(count_), fh(count_, handlers) {

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
    if(params.node->isSimplified()) {
        tree.iterateTriangles(params.node, params.cube, params.level, fh, context);
        return false;
    }
    return !params.node->isLeaf();
}

void Processor::getOrder(const Octree &tree, OctreeNodeData &params, uint8_t * order){
    for(int i = 0 ; i < 8 ; ++i) {
        order[i] = i;
    }
}

