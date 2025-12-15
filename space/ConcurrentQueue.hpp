#ifndef SPACE_CONCURRENT_QUEUE_HPP
#define SPACE_CONCURRENT_QUEUE_HPP
#include <queue>
#include <mutex>
#include <condition_variable>

template<typename T>
class ConcurrentQueue {
public:
    void push(const T& value);

    bool tryPop(T& out);

    bool empty() const;

private:
    mutable std::mutex mutex_;
    std::queue<T> queue_;
    std::condition_variable cv_;
};

#include "ConcurrentQueue.tpp"

#endif
