#include "IteratorHandler.hpp"
#include "ConcurrentQueue.hpp"
#include "ThreadPool.hpp"
#include "Octree.hpp"
#include "OctreeNode.hpp"
#include <queue>

#include <atomic>
#include <functional>

void IteratorHandler::iterateParallelBFS(const Octree &tree, OctreeNodeData &rootParams, ThreadPool& pool,
        const Octree::IterateHandler &iterateHandler, const Octree::IterateOrderHandler &getOrderHandler)
{
    if (rootParams.node == NULL)
        return;

    ConcurrentQueue<OctreeNodeData> queue;
    std::atomic<int> activeTasks {0};

    // Push root node
    queue.push(rootParams);

    std::mutex doneMutex;
    std::condition_variable doneCV;

    auto worker = [&]() {
        OctreeNodeData params = OctreeNodeData();

        while (true) {
            // Try get work
            if (!queue.tryPop(params)) {
                // Exit condition: no work and no active tasks
                if (activeTasks.load() == 0 && queue.empty())
                    break;
                std::this_thread::yield();
                continue;
            }

            activeTasks.fetch_add(1);

            if (params.node != NULL) {
                if (params.node != NULL && iterateHandler(tree, params)) {
                    uint8_t internalOrder[8];
                    getOrderHandler(tree, params, internalOrder);

                    OctreeNode* children[8] = {
                        NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL
                    };

                    params.node->getChildren(*tree.allocator, children);

                    for (int i = 0; i < 8; ++i) {
                        uint8_t j = internalOrder[i];
                        OctreeNode* child = children[j];

                        if (child == params.node) {
                            throw std::runtime_error("Wrong pointer @ iter!");
                        }

                        if (child != NULL && child != params.node) {
                                            OctreeNodeData childData(
                                                params.level + 1,
                                                child,
                                                params.cube.getChild(j), 
                                                params.context
                                            );

                                            queue.push(childData);
                        }
                    }
                }
            }

            activeTasks.fetch_sub(1);
        }
    };

    // Launch worker threads (fire-and-forget — futures are unused)
    size_t threads = pool.threadCount();
    for (size_t i = 0; i < threads; ++i) {
        pool.enqueueDetached(worker);
    }

    // Wait until all work is complete
    {
        std::unique_lock<std::mutex> lock(doneMutex);
        doneCV.wait(lock, [&]() {
            return queue.empty() && activeTasks.load() == 0;
        });
    }
}


void IteratorHandler::iterateBFS(const Octree &tree, OctreeNodeData &rootParams,
        const Octree::IterateHandler &iterateHandler, const Octree::IterateOrderHandler &getOrderHandler)
{
    if (rootParams.node == NULL)
        return;


    std::vector<std::future<bool>> futures;
	futures.reserve(8);

    std::queue<OctreeNodeData> q;
    q.push(rootParams);

    while (!q.empty()) {
        OctreeNodeData params = q.front();
        q.pop();

        if (params.node == NULL)
            continue;

        // Same as recursive version

        if (params.node != NULL && iterateHandler(tree, params)) {
            uint8_t internalOrder[8];
            getOrderHandler(tree, params, internalOrder);

            OctreeNode* children[8] = {
                NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL
            };

            params.node->getChildren(*tree.allocator, children);

            for (int i = 0; i < 8; ++i) {
                uint8_t j = internalOrder[i];
                OctreeNode* child = children[j];

                if (child == params.node) {
                    throw std::runtime_error("Wrong pointer @ iter!");
                }

                if (child != NULL && child != params.node) {
                    OctreeNodeData childData(
                        params.level + 1,
                        child,
                        params.cube.getChild(j),
                        params.context
                    );

                    // BFS: push instead of recursive call
                    q.push(childData);
                }
            }

        }
    }
}



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
                fut.get();
            }
        }
    }
}
      

void IteratorHandler::iterate(
    const Octree &tree, 
    OctreeNodeData &params,
    const Octree::IterateHandler &iterateHandler, 
    const Octree::IterateOrderHandler &getOrderHandler
) {
    if(params.node != NULL && iterateHandler(tree, params)) {
        uint8_t internalOrder[8];
        getOrderHandler(tree, params, internalOrder);

        OctreeNode* children[8] = { NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL };
        params.node->getChildren(*tree.allocator, children);
        for(int i=0; i <8 ; ++i) {
            uint8_t j = internalOrder[i];
            OctreeNode * child = children[j];
            if (child == params.node) {
                throw std::runtime_error("Wrong pointer @ iter!");
            }                
            if(child != NULL && params.node != child) {
                OctreeNodeData data = OctreeNodeData( params.level+1, child, params.cube.getChild(j), params.context);
                this->iterate(tree, data, iterateHandler, getOrderHandler);
            }
        }
    }
    
}

void IteratorHandler::iterateFlatIn(
    const Octree &tree, 
    OctreeNodeData &params,
    const Octree::IterateHandler &iterateHandler, 
    const Octree::IterateOrderHandler &getOrderHandler
) {
    if (params.context != NULL) {
        ThreadContext *tc = reinterpret_cast<ThreadContext*>(params.context);
        tc->parentOf.clear();

        // Seed parentOf with the world-root path down to params.node (the chunk
        // root) so findCellAt can rebuild root-consistent cubes. Without this,
        // the cached cubes drift from the root-descended cubes and getNodeIndex
        // flips at chunk-boundary faces, returning the wrong (cross-chunk) cell
        // and leaving holes in the mesh.
        OctreeNode *n = tree.root;
        BoundingCube c = static_cast<const BoundingCube&>(tree);
        for (uint l = 0; l < params.level && n != NULL; ++l) {
            ChildBlock *block = n->getBlock(*tree.allocator);
            if (block == NULL) break;
            const glm::vec3 center = params.cube.getCenter();
            const glm::vec3 cc = c.getCenter();
            int idx = (center.x >= cc.x ? 4 : 0)
                    + (center.y >= cc.y ? 2 : 0)
                    + (center.z >= cc.z ? 1 : 0);
            OctreeNode *child = block->get(idx, *tree.allocator);
            if (child == NULL) break;
            tc->parentOf.try_emplace(child, n, idx);
            c = c.getChild(idx);
            n = child;
        }
        // If the descent didn't land exactly on params.node, the seed is
        // unusable; disable the cache (findCellAt falls back to root descent,
        // which is correct, just not optimized).
        if (n != params.node) {
            tc->parentOf.clear();
        }
    }

    uint8_t internalOrder[8];

    flatData.push(params);
    while (!flatData.empty()) {
        OctreeNodeData data = flatData.top();
        flatData.pop();

        OctreeNode* node = data.node;

        if (node != NULL && iterateHandler(tree, data)) {
            getOrderHandler(tree, data, internalOrder);
            OctreeNode* children[8] = { NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL };
            node->getChildren(*tree.allocator, children);
            for (int i = 7; i >= 0; --i) {
                uint8_t j = internalOrder[i];
                OctreeNode* child = children[j];

                if (child == node) {
                    throw std::runtime_error("Wrong pointer!");
                }

                if (child != NULL) {
                    // Cache the parent link (with child index) for upper
                    // traversal during triangle iteration.
                    if (data.context != NULL) {
                        reinterpret_cast<ThreadContext*>(data.context)
                            ->parentOf.try_emplace(child, node, j);
                    }
                    flatData.push(OctreeNodeData(
                        data.level + 1,
                        child,
                        data.cube.getChild(j),
                        data.context
                    ));
                }
            }

        }
    }
}

void IteratorHandler::iterateFlatOut(
    const Octree &tree, 
    OctreeNodeData &params,
    const Octree::IterateHandler &iterateHandler, 
    const Octree::IterateOrderHandler &getOrderHandler
) {
    if (!params.node) return;
    params.context = NULL;

    stackOut.push(StackFrameOut(params, false));

    // A single shared array to hold the child processing order.
    uint8_t internalOrder[8];

    while (!stackOut.empty()) {
        StackFrameOut &frame = stackOut.top();

        if (!frame.visited) {
            
            // First visit: execute before() and update context.
            frame.visited = true;

            // Only process children if the test passes.
            if (!iterateHandler(tree, frame)) {
                stackOut.pop();
                continue;
            }

            // Compute the child order for this node.
            getOrderHandler(tree, frame, internalOrder);

            // Push all valid children in reverse order so that they are processed
            // in the original (correct) order when popped.
            for (int i = 7; i >= 0; --i) {
                uint8_t j = internalOrder[i];
                OctreeNode * node = frame.node;
                ChildBlock * block = node->getBlock(*tree.allocator);
                OctreeNode* child = block->get(j, *tree.allocator);
                if (child) {
                    stackOut.push(StackFrameOut(OctreeNodeData(frame.level + 1, child, frame.cube.getChild(j), frame.context), false));
                }
            }
        } else {
            // Second visit: all children have been processed; now call after().
            stackOut.pop();
        }
    }
}
