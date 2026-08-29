#pragma once

#include <vulkan/vulkan.h>

#include <atomic>
#include <list>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

// Per-submit busy interval used by the queue-usage slotted (timeline) view.
// Every submit to any queue records a segment {queue, startNs, endNs, frame}
// so the UI can draw exactly when each queue is processing. `endNs` is 0
// while the submission is in flight and is resolved to the completion time
// when its fence signals.
struct QueueSegment {
    VkQueue  queue    = VK_NULL_HANDLE;
    VkFence  fence    = VK_NULL_HANDLE;
    uint64_t frame    = 0;
    uint64_t submitId = 0;
    uint64_t startNs  = 0;
    uint64_t endNs    = 0; // 0 while in flight; resolved on completion
};

// Encapsulates one Vulkan queue together with all queue-local state that used
// to be scattered across VulkanApp (a handle keyed into global maps). A single
// Queue owns:
//   - its VkQueue handle, family index and a human-readable name,
//   - a per-queue submit mutex so submissions to *different* queues never
//     serialize against each other (only aliased queues share a mutex because
//     they are the same Queue object),
//   - activity counters (in-flight / cumulative submitted / completed),
//   - the queue-timeline segment list used by the slotted view.
//
// Global command-buffer lifecycle tracking (fences, deferred destroys,
// extra-wait semaphores, layout pre-apply) intentionally stays in VulkanApp;
// this class only owns the queue-local concerns so the repeated per-queue
// bookkeeping is no longer duplicated across every queue.
class Queue {
public:
    Queue() = default;
    Queue(VkQueue handle, uint32_t family, std::string name)
        : handle_(handle), family_(family), name_(std::move(name)) {}

    // Implicit conversion to VkQueue so existing call sites that pass a raw
    // queue handle keep compiling unchanged.
    operator VkQueue() const { return handle_; }
    VkQueue handle() const { return handle_; }
    uint32_t family() const { return family_; }
    const std::string& name() const { return name_; }

    // Per-queue submit mutex. Distinct (non-aliased) queues each lock their own
    // mutex, so concurrent submissions to separate queues don't contend.
    std::mutex& submitMutex() { return submitMutex_; }

    // ---- Activity counters ----
    int      pending()   const { return pending_.load(std::memory_order_relaxed); }
    uint64_t submitted() const { return submitted_.load(std::memory_order_relaxed); }
    uint64_t completed() const { return completed_.load(std::memory_order_relaxed); }
    void onSubmitted() {
        submitted_.fetch_add(1, std::memory_order_relaxed);
        pending_.fetch_add(1, std::memory_order_relaxed);
    }
    // Undo a prior onSubmitted() when a submission fails before reaching the GPU
    // (e.g. vkQueueSubmit2 returns an error). Only drops the in-flight count; the
    // cumulative submitted counter is left as-is (the attempt was still counted).
    void undoSubmit() {
        int expected = pending_.load(std::memory_order_relaxed);
        while (expected > 0 && !pending_.compare_exchange_weak(expected, expected - 1,
                 std::memory_order_relaxed, std::memory_order_relaxed)) {}
    }
    void onCompleted() {
        // pending may already be 0 if accounting was undone after a failed submit
        int expected = pending_.load(std::memory_order_relaxed);
        while (expected > 0 && !pending_.compare_exchange_weak(expected, expected - 1,
                 std::memory_order_relaxed, std::memory_order_relaxed)) {}
        completed_.fetch_add(1, std::memory_order_relaxed);
    }

    // ---- Timeline (queue-usage slotted view) ----
    void recordSegment(VkFence fence, uint64_t submitId, uint64_t frame);
    void markSegmentDone(VkFence fence, uint64_t endNs);
    void getTimeline(std::vector<QueueSegment>& out) const;

private:
    VkQueue     handle_   = VK_NULL_HANDLE;
    uint32_t    family_   = 0;
    std::string name_;
    std::mutex  submitMutex_;

    std::atomic<int>      pending_{0};
    std::atomic<uint64_t> submitted_{0};
    std::atomic<uint64_t> completed_{0};

    mutable std::mutex                          timelineMtx_;
    std::list<QueueSegment>                     segments_;
    std::unordered_map<VkFence, std::list<QueueSegment>::iterator> live_;
};
