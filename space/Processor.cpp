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
    else {    
        // Set context for leaf nodes OR chunk nodes (chunks are tessellation boundaries)
        if(params.node->isSimplified()) {
            params.context = params.node;
        }
        return params.node->getType() == SpaceType::Surface;
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
        tree.iterateTriangles(params.node, params.cube, params.level, fh, context);

        // Also ensure the source cube is invoked once (dedup)
        glm::vec4 invokedKey = glm::vec4(params.cube.getMin(), params.level);
        auto res = context->invokedCubeCalls.emplace(invokedKey);
        bool shouldInvoke = res.second; // true if inserted (wasn't present)
        if (shouldInvoke) {
            tree.handleQuadNodes(params.cube, params.level, params.node->sdf, handlers, true, context);
        }
    }
}

void Processor::getOrder(const Octree &tree, OctreeNodeData &params, uint8_t * order){
    for(int i = 0 ; i < 8 ; ++i) {
        order[i] = i;
    }
}

