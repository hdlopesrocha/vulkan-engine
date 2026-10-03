#include "IteratorHandler.hpp"
#include "ThreadPool.hpp"
#include "Octree.hpp"
#include "OctreeNode.hpp"

#include <functional>

void IteratorHandler::iterateMultiThreaded(
    const Octree &tree, 
    OctreeNodeData &params, 
    ThreadPool& pool,
    const Octree::IterateHandler& iterateHandler, 
    const Octree::IterateOrderHandler& getOrderHandler,
    const Octree::IterateThreadedHandler& iterateThreadedHandler
) {    
    bool isThreaded = false;
    if(params.node != NULL) {
        
        if(params.node != NULL && iterateHandler(tree, params)) {
            uint8_t internalOrder[8];
            getOrderHandler(tree, params, internalOrder);

            OctreeNode* children[8] = { NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL };
            params.node->getChildren(*tree.allocator, children);
            std::vector<std::future<void>> futures;
            futures.reserve(8);
            for(int i=0; i <8 ; ++i) {
                uint8_t j = internalOrder[i];
                OctreeNode * child = children[j];
                BoundingCube childCube = params.cube.getChild(j);
                if (child == params.node) {
                    throw std::runtime_error("Wrong pointer @ iter!");
                }                
                if(child != NULL && params.node != child) {
                    isThreaded = iterateThreadedHandler(tree, params);
                    if(isThreaded) {
                        futures.push_back(pool.enqueue([this, &tree, &pool,child,childCube, params, &iterateHandler, &getOrderHandler, &iterateThreadedHandler]() mutable {
                            OctreeNodeData data = OctreeNodeData( params.level+1, child, childCube, params.context);
                            this->iterateMultiThreaded(tree, data, pool, iterateHandler, getOrderHandler, iterateThreadedHandler);
                        }));
                    } else {
                        OctreeNodeData data = OctreeNodeData( params.level+1, child, childCube, params.context);
                        this->iterateMultiThreaded(tree, data, pool, iterateHandler, getOrderHandler, iterateThreadedHandler);
                    }
                }
            }
            for(auto &fut : futures) {
                // Cooperative wait: help drain the pool while waiting. A plain
                // fut.get() here starves the pool whenever the number of
                // nested blocked tasks reaches the worker count — every worker
                // ends up waiting on a future whose task is still queued, and
                // the walk never finishes (the sceneProcessThread then hangs
                // forever and clean() deadlocks joining it at window close).
                pool.getCooperative(fut);
            }
        }
    }
}

