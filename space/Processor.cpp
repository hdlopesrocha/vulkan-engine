#include "Processor.hpp"
#include "Octree.hpp"
#include "OctreeNode.hpp"
#include "ThreadPool.hpp"
#include "ForwardingHandler.hpp"


Processor::Processor(long * count, ThreadPool &threadPool, ThreadContext * context, std::vector<OctreeNodeTriangleHandler*> * handlers): threadPool(threadPool), context(context), count(count), fh(count, handlers) {

}

bool Processor::test(const Octree &tree, OctreeNodeData &params) {
    if(params.context != NULL) {
        return false;
    }
    else if(params.node->getType() == SpaceType::Surface) {    
        // Set context for leaf nodes OR chunk nodes (chunks are tessellation boundaries)
        if(params.node->isSimplified()) {
            // Forwarding handler: forwards triangle calls to the provided handlers vector
            // Iterate triangles and forward them to the registered handlers
            tree.iterateTriangles(params.node, params.cube, fh, context);
            return false;
        }
        return true;
    }
    return false;
}

void Processor::before(const Octree &tree, OctreeNodeData &params) {        

}

void Processor::after(const Octree &tree, OctreeNodeData &params) {

}

void Processor::getOrder(const Octree &tree, OctreeNodeData &params, uint8_t * order){
    for(int i = 0 ; i < 8 ; ++i) {
        order[i] = i;
    }
}

