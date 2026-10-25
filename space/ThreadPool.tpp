#pragma once

#include "ThreadPool.hpp"

template<class F, class... Args>
auto ThreadPool::enqueue(F&& f, Args&&... args)
    -> std::future<std::invoke_result_t<F, Args...>>
{
    using return_type = std::invoke_result_t<F, Args...>;

    auto task = std::make_shared<std::packaged_task<return_type()>>(
        [func = std::forward<F>(f), ... capturedArgs = std::forward<Args>(args)]() mutable {
            return func(capturedArgs...);
        }
    );

    std::future<return_type> res = task->get_future();
    {
        std::unique_lock<std::mutex> lock(queue_mutex);
        if(stopping) throw std::runtime_error("enqueue on stopped ThreadPool");
        // The wrapper lambda captures only a shared_ptr (16 B) — fits in SmallFunction inline buffer
        tasks.emplace([task]{ (*task)(); });
    }
    condition.notify_one();
    return res;
}

template<class F, class... Args>
void ThreadPool::enqueueDetachedNoNotify(F&& f, Args&&... args)
{
    SmallFunction sf(
        [func = std::forward<F>(f), ... capturedArgs = std::forward<Args>(args)]() mutable {
            func(capturedArgs...);
        }
    );
    {
        std::unique_lock<std::mutex> lock(queue_mutex);
        if(stopping) throw std::runtime_error("enqueueDetachedNoNotify on stopped ThreadPool");
        tasks.emplace(std::move(sf));
    }
    // Deliberately no notify: the caller batches the enqueues and wakes the
    // pool once with notifyAll().
}

// Wait for a pool future, running queued tasks in the meantime so nested
// fork-join waits can never starve the pool (see the header comment). While
// the queue has work the waiter drains it back-to-back. Once the queue is
// empty the future's task is running on another worker (a queued task would
// have been popped by runOneTask), so fut.wait() is an event-driven wait that
// cannot deadlock; the running task completes without this waiter and any
// nested fork it performs uses the same drain-then-wait discipline.
template<typename T>
T ThreadPool::getCooperative(std::future<T>& fut)
{
    while (fut.wait_for(std::chrono::milliseconds(0)) != std::future_status::ready) {
        if (!runOneTask())
            fut.wait();
    }
    return fut.get();
}

