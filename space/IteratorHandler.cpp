#include "IteratorHandler.hpp"
#include "ThreadPool.hpp"
#include "Octree.hpp"
#include "OctreeNode.hpp"

#include <array>
#include <exception>

void IteratorHandler::iterateMultiThreaded(
    const Octree &tree, 
    OctreeNodeData &params, 
    ThreadPool& pool,
    const Octree::IterateHandler& iterateHandler, 
    const Octree::IterateOrderHandler& getOrderHandler,
    const Octree::IterateThreadedHandler& iterateThreadedHandler
) {
    if (params.node == NULL) return;
    if (!iterateHandler(tree, params)) return;

    uint8_t internalOrder[8];
    getOrderHandler(tree, params, internalOrder);

    OctreeNode* children[8] = { NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL };
    params.node->getChildren(*tree.allocator, children);

    // One fork decision per node, not per child: the predicate depends only on
    // params, so the old loop overwrote the same value up to eight times.
    const bool isThreaded = iterateThreadedHandler(tree, params);

    // One stable record per threaded child. The records live in this frame,
    // which stays alive until every future below has been joined, so the task
    // pointers cannot dangle. The task lambda captures only this pointer, which
    // keeps SmallFunction inline (no per-task heap allocation).
    struct ChildTask {
        OctreeNodeData data;
        IteratorHandler* self;
        const Octree* tree;
        ThreadPool* pool;
        const Octree::IterateHandler* iterate;
        const Octree::IterateOrderHandler* order;
        const Octree::IterateThreadedHandler* threaded;
    };
    std::array<ChildTask, 8> taskData;
    std::array<std::future<void>, 8> futures;
    int futureCount = 0;
    std::exception_ptr error;

    try {
        for (int i = 0; i < 8; ++i) {
            const uint8_t j = internalOrder[i];
            OctreeNode* child = children[j];
            const BoundingCube childCube = params.cube.getChild(j);
            if (child == params.node) {
                throw std::runtime_error("Wrong pointer @ iter!");
            }
            if (child == NULL) continue;

            if (isThreaded) {
                const int idx = futureCount;
                taskData[idx] = ChildTask{
                    OctreeNodeData(params.level + 1, child, childCube, params.context),
                    this, &tree, &pool, &iterateHandler, &getOrderHandler, &iterateThreadedHandler};
                try {
                    futures[futureCount++] = pool.enqueue([task = &taskData[idx]] {
                        task->self->iterateMultiThreaded(*task->tree, task->data, *task->pool,
                                                         *task->iterate, *task->order, *task->threaded);
                    });
                } catch (...) {
                    if (!error) error = std::current_exception();
                    break;
                }
            } else {
                OctreeNodeData data(params.level + 1, child, childCube, params.context);
                this->iterateMultiThreaded(tree, data, pool, iterateHandler, getOrderHandler,
                                           iterateThreadedHandler);
            }
        }
    } catch (...) {
        if (!error) error = std::current_exception();
    }

    // Join every enqueued task even after a failure, so no task can outlive the
    // taskData/frame it points at. Collect the first error, then rethrow once
    // all tasks have completed.
    for (int i = 0; i < futureCount; ++i) {
        try {
            pool.getCooperative(futures[i]);
        } catch (...) {
            if (!error) error = std::current_exception();
        }
    }
    if (error) std::rethrow_exception(error);
}
