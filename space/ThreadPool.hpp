#pragma once
#include <vector>
#include <queue>
#include <memory>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <future>
#include <functional>
#include <stdexcept>
#include <type_traits>
#include "SmallFunction.hpp"

class ThreadPool {
public:
    explicit ThreadPool(size_t threads);
    ~ThreadPool();

    // Enqueue a task, returns a future
    template<class F, class... Args>
    auto enqueue(F&& f, Args&&... args) 
        -> std::future<std::invoke_result_t<F, Args...>>;

    // Enqueue a fire-and-forget task (no future, no heap-allocated packaged_task)
    template<class F, class... Args>
    void enqueueDetached(F&& f, Args&&... args);

    // Wait on a future from THIS pool while helping run queued tasks. A plain
    // fut.get() inside a pool task can starve the pool: if every worker is
    // blocked waiting on its own subtasks, the queued subtasks never run and
    // the waits never resolve (fork-join starvation deadlock). Running queued
    // tasks while waiting guarantees progress because the task graph is a
    // finite tree. Safe to call from any thread; task lifetimes are sound
    // because a task's captured references live in a frame that is itself
    // blocked in this wait and cannot return before the task completes.
    template<typename T>
    T getCooperative(std::future<T>& fut);

    // Explicitly stop workers and drain the queue.  Safe to call multiple times
    // and from any thread.  Must be called before the objects referenced by
    // enqueued tasks are destroyed to avoid use-after-free during shutdown.
    void stop();

    // Pop and run one queued task if any; returns false when the queue is
    // empty. Used by getCooperative(); exposed so other blocking joins on this
    // pool can help drain work too. Also safe after stop(): workers drain the
    // remaining queue, and so may a cooperative waiter.
    bool runOneTask();

    size_t threadCount() const;

private:
    // Worker threads
    std::vector<std::thread> workers;

    // Task queue (small-buffer-optimized, avoids heap allocation for small callables)
    std::queue<SmallFunction> tasks;

    // Synchronization
    std::mutex queue_mutex;
    std::condition_variable condition;
    bool stopping;
};

// Constructor: launch worker threads
// (definition moved to ThreadPool.cpp)

// Enqueue a new task
#include "ThreadPool.tpp"

 
