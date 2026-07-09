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
        if(stop) throw std::runtime_error("enqueue on stopped ThreadPool");
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
        if(stop) throw std::runtime_error("enqueueDetached on stopped ThreadPool");
        tasks.emplace(std::move(sf));
    }
    condition.notify_one();
}

