#include "Queue.hpp"

#include <chrono>

static uint64_t nowNs() {
    using namespace std::chrono;
    return static_cast<uint64_t>(
        duration_cast<nanoseconds>(steady_clock::now().time_since_epoch()).count());
}

void Queue::recordSegment(VkFence fence, uint64_t submitId, uint64_t frame) {
    if (handle_ == VK_NULL_HANDLE) return;
    std::lock_guard<std::mutex> lk(timelineMtx_);
    QueueSegment seg;
    seg.queue    = handle_;
    seg.fence    = fence;
    seg.frame    = frame;
    seg.submitId = submitId;
    seg.startNs  = nowNs();
    // A null fence (e.g. present) is recorded as an instantaneous event: mark it
    // already-done so the UI shows a single-slot marker instead of an open-ended
    // "ongoing" segment.
    seg.endNs    = (fence == VK_NULL_HANDLE) ? seg.startNs : 0;
    segments_.push_back(seg);
    if (fence != VK_NULL_HANDLE)
        live_[fence] = std::prev(segments_.end());
    // Cap history to keep the snapshot cheap for the UI thread.
    if (segments_.size() > 8192) {
        auto& front = segments_.front();
        auto live = live_.find(front.fence);
        if (live != live_.end()) live_.erase(live);
        segments_.pop_front();
    }
}

void Queue::markSegmentDone(VkFence fence, uint64_t endNs) {
    if (fence == VK_NULL_HANDLE) return;
    std::lock_guard<std::mutex> lk(timelineMtx_);
    auto it = live_.find(fence);
    if (it != live_.end()) {
        it->second->endNs = endNs;
        live_.erase(it);
    }
}

void Queue::getTimeline(std::vector<QueueSegment>& out) const {
    std::lock_guard<std::mutex> lk(timelineMtx_);
    out.reserve(out.size() + segments_.size());
    for (const auto& s : segments_) out.push_back(s);
}
