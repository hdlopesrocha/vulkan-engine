#pragma once

#include <atomic>
#include <cstddef>
#include <memory>
#include <utility>

namespace streaming {

// Lock-free Multiple-Producer / Single-Consumer queue (Dmitry Vyukov's MPSC).
//
// Why MPSC and not a generic MPMC: terrain streaming has MANY worker threads
// generating meshes (producers) but exactly ONE consumer — UploadManager::
// processUploads() runs once per frame on the main thread. A SPSC/MPMC queue
// would add unnecessary CAS contention on the consumer side for no benefit.
//
// - Producers: lock-free (a single atomic exchange + store). Heap allocation of
//   the node happens per push, which is acceptable because the mesh payload
//   itself is already heap-allocated by the generator.
// - Consumer: wait-free (a single atomic load + store), runs only on the
//   render thread, so it never contends with producers beyond one atomic.
//
// T must be move-constructible. UploadJob is, so this is safe.
template <typename T>
class MPSCQueue {
    struct Node {
        T                  value;
        std::atomic<Node*> next{nullptr};
    };

public:
    MPSCQueue() {
        // Stub node so the consumer always has a non-null `head` to advance from.
        stub_ = new Node();
        head_ = stub_;
        tail_ = stub_;
    }

    ~MPSCQueue() {
        // Drain anything left, then free the stub.
        T v;
        while (tryPop(v)) {}
        delete stub_;
    }

    // Producer side. Thread-safe, lock-free.
    void push(T&& value) {
        Node* n = new Node();
        n->value = std::move(value);
        // Publish: make this node the new tail, getting the previous tail.
        Node* prev = tail_.exchange(n, std::memory_order_relaxed);
        // Link the previous tail to us. Release so the consumer sees `value`.
        prev->next.store(n, std::memory_order_release);
    }

    // Consumer side. Single-threaded only. Returns false if empty.
    bool tryPop(T& out) {
        Node* head = head_.load(std::memory_order_relaxed);
        Node* next = head->next.load(std::memory_order_acquire);
        if (next == nullptr) {
            return false; // empty
        }
        // Advance head; the old head node is now free.
        head_.store(next, std::memory_order_relaxed);
        out = std::move(next->value);
        delete head;
        return true;
    }

    bool empty() const {
        Node* head = head_.load(std::memory_order_relaxed);
        return head->next.load(std::memory_order_acquire) == nullptr;
    }

private:
    MPSCQueue(const MPSCQueue&) = delete;
    MPSCQueue& operator=(const MPSCQueue&) = delete;

    std::atomic<Node*> head_;
    std::atomic<Node*> tail_;
    Node* stub_;
};

} // namespace streaming
