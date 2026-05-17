#include "Processor.hpp"
#include "Octree.hpp"
#include "OctreeNode.hpp"
#include "ThreadPool.hpp"


Processor::Processor(long * count, ThreadPool &threadPool, ThreadContext * context, std::vector<OctreeNodeTriangleHandler*> * handlers): threadPool(threadPool), context(context), handlers(handlers), count(count) {

}

bool Processor::test(const Octree &tree, OctreeNodeData &params) {
    if(params.context != NULL) {
        return false;
    }
    else if(params.node->getType() == SpaceType::Surface) {    
        // Set context for leaf nodes OR chunk nodes (chunks are tessellation boundaries)
        if(params.node->isSimplified()) {
            params.context = params.node;
        }
        return true;
    }
    return false;
}

void Processor::before(const Octree &tree, OctreeNodeData &params) {        

}
//std::mutex processorMutex;
void Processor::after(const Octree &tree, OctreeNodeData &params) {
    if(params.context != NULL) {
        // Forwarding handler: forwards triangle calls to the provided handlers vector
        class ForwardingHandler : public OctreeNodeTriangleHandler {
        public:
            std::vector<OctreeNodeTriangleHandler*> * handlersVec;
            ForwardingHandler(long *count, std::vector<OctreeNodeTriangleHandler*> *h) : OctreeNodeTriangleHandler(count), handlersVec(h) {}
            void handle(Vertex &v0, Vertex &v1, Vertex &v2) override {
                for (auto h : *handlersVec) h->handle(v0, v1, v2);
            }
        } fh(this->count, handlers);

        // Iterate triangles and forward them to the registered handlers
        tree.iterateTriangles(params.node, params.cube, fh, context);
    }
}

void Processor::getOrder(const Octree &tree, OctreeNodeData &params, uint8_t * order){
    for(int i = 0 ; i < 8 ; ++i) {
        order[i] = i;
    }
}

