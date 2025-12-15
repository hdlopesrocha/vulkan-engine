// ConcurrentQueue.tpp - template implementations

#ifndef SPACE_CONCURRENT_QUEUE_TPP
#define SPACE_CONCURRENT_QUEUE_TPP

// header already includes this tpp; avoid recursive include

template<typename T>
void ConcurrentQueue<T>::push(const T& value) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        queue_.push(value);
    }
    cv_.notify_one();
}

template<typename T>
bool ConcurrentQueue<T>::tryPop(T& out) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (queue_.empty())
        return false;
    out = std::move(queue_.front());
    queue_.pop();
    return true;
}

template<typename T>
bool ConcurrentQueue<T>::empty() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return queue_.empty();
}

#endif // SPACE_CONCURRENT_QUEUE_TPP
