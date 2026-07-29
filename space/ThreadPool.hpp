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

    // Explicitly stop workers and drain the queue.  Safe to call multiple times
    // and from any thread.  Must be called before the objects referenced by
    // enqueued tasks are destroyed to avoid use-after-free during shutdown.
    void stop();

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

 
