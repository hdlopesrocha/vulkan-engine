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
void ThreadPool::enqueueDetached(F&& f, Args&&... args)
{
    SmallFunction sf(
        [func = std::forward<F>(f), ... capturedArgs = std::forward<Args>(args)]() mutable {
            func(capturedArgs...);
        }
    );
    {
        std::unique_lock<std::mutex> lock(queue_mutex);
        if(stopping) throw std::runtime_error("enqueueDetached on stopped ThreadPool");
        tasks.emplace(std::move(sf));
    }
    condition.notify_one();
}

// Wait for a pool future, running queued tasks in the meantime so nested
// fork-join waits can never starve the pool (see the header comment). While
// the queue has work, the waiter drains it back-to-back; the brief 1 ms nap
// only happens when the queue is momentarily empty (the future's task is
// executing on another worker and about to resolve the wait anyway).
template<typename T>
T ThreadPool::getCooperative(std::future<T>& fut)
{
    while (fut.wait_for(std::chrono::milliseconds(0)) != std::future_status::ready) {
        if (!runOneTask())
            fut.wait_for(std::chrono::milliseconds(1));
    }
    return fut.get();
}

