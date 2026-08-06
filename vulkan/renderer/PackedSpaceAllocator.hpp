#pragma once

#include <cstdint>
#include <vector>
#include <mutex>
#include <algorithm>

// Free-space allocator for the packed vertex/index pools of the slotted
// IndirectRenderer. The pools are pre-reserved (fixed GPU buffers, no global
// rebuilds) and this allocator hands out variable-size spans of elements to
// individual (chunk, level) geometry. Multiple chunks therefore share the
// pool's space ("packed slots"): a chunk reserves only what its mesh actually
// needs instead of a fixed worst-case budget, so the active chunk count is
// bounded by total geometry bytes rather than by a slot count.
//
// Allocation is best-fit over the free-span lists (reduces fragmentation and
// lets small chunks fill the holes left by removed ones). Freed spans merge
// with their neighbours. There is NO compaction: a span, once allocated, is
// only released by an explicit free (the GPU data is referenced by absolute
// draw commands, so no repacking of other chunks ever happens).
//
// Thread safety: all public methods are thread-safe via internal mutex.
class PackedSpaceAllocator {
public:
    struct Span {
        uint32_t offset = 0;
        uint32_t size   = 0;
    };

    explicit PackedSpaceAllocator() = default;

    // Pre-allocate the element pools. `vertexElements`/`indexElements` are the
    // TOTAL capacities of the merged vertex/index buffers. Can be called again
    // to grow the pool (existing free spans are preserved, new space is
    // appended as one span).
    void reserve(uint32_t vertexElements, uint32_t indexElements) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (vertexElements > totalVertex_) {
            freeVertex_.push_back(Span{totalVertex_, vertexElements - totalVertex_});
            totalVertex_ = vertexElements;
        }
        if (indexElements > totalIndex_) {
            freeIndex_.push_back(Span{totalIndex_, indexElements - totalIndex_});
            totalIndex_ = indexElements;
        }
        std::sort(freeVertex_.begin(), freeVertex_.end(),
                  [](const Span& a, const Span& b) { return a.offset < b.offset; });
        std::sort(freeIndex_.begin(), freeIndex_.end(),
                  [](const Span& a, const Span& b) { return a.offset < b.offset; });
        mergeAdjacent(freeVertex_);
        mergeAdjacent(freeIndex_);
    }

    // Allocate `n` consecutive vertex elements. Returns the element offset or
    // UINT32_MAX when no free span is large enough.
    uint32_t allocateVertex(uint32_t n) {
        std::lock_guard<std::mutex> lock(mutex_);
        uint32_t off = takeBestFit(freeVertex_, n);
        if (off != UINT32_MAX) usedVertex_ += n;
        return off;
    }

    void freeVertex(uint32_t offset, uint32_t n) {
        if (n == 0 || offset == UINT32_MAX) return;
        std::lock_guard<std::mutex> lock(mutex_);
        insertSpan(freeVertex_, offset, n);
        usedVertex_ -= n;
    }

    // Allocate `n` consecutive index elements. Returns the element offset or
    // UINT32_MAX when no free span is large enough.
    uint32_t allocateIndex(uint32_t n) {
        std::lock_guard<std::mutex> lock(mutex_);
        uint32_t off = takeBestFit(freeIndex_, n);
        if (off != UINT32_MAX) usedIndex_ += n;
        return off;
    }

    void freeIndex(uint32_t offset, uint32_t n) {
        if (n == 0 || offset == UINT32_MAX) return;
        std::lock_guard<std::mutex> lock(mutex_);
        insertSpan(freeIndex_, offset, n);
        usedIndex_ -= n;
    }

    uint32_t totalVertex() const { return totalVertex_; }
    uint32_t totalIndex()  const { return totalIndex_; }
    uint64_t usedVertex()  const { return usedVertex_; }
    uint64_t usedIndex()   const { return usedIndex_; }

private:
    // Remove the smallest free span that fits `n` (best fit), return its
    // offset and keep the remainder as a new span. UINT32_MAX when no fit.
    uint32_t takeBestFit(std::vector<Span>& spans, uint32_t n) {
        if (n == 0) return 0; // zero-size allocation consumes nothing
        size_t best = spans.size();
        for (size_t i = 0; i < spans.size(); ++i) {
            if (spans[i].size >= n &&
                (best == spans.size() || spans[i].size < spans[best].size)) {
                best = i;
            }
        }
        if (best == spans.size()) return UINT32_MAX;

        Span& s = spans[best];
        uint32_t off = s.offset;
        if (s.size == n) {
            // Exact fit: remove the span entirely
            spans.erase(spans.begin() + static_cast<ptrdiff_t>(best));
        } else {
            s.offset += n;
            s.size   -= n;
        }
        return off;
    }

    // Insert a span and merge it with any adjacent free span.
    void insertSpan(std::vector<Span>& spans, uint32_t offset, uint32_t size) {
        if (size == 0) return;
        spans.push_back(Span{offset, size});
        std::sort(spans.begin(), spans.end(),
                  [](const Span& a, const Span& b) { return a.offset < b.offset; });
        mergeAdjacent(spans);
    }

    static void mergeAdjacent(std::vector<Span>& spans) {
        if (spans.size() < 2) return;
        size_t w = 0;
        for (size_t r = 1; r < spans.size(); ++r) {
            Span& cur = spans[w];
            const Span& next = spans[r];
            if (cur.offset + cur.size == next.offset) {
                cur.size += next.size;
            } else {
                ++w;
                spans[w] = next;
            }
        }
        spans.resize(w + 1);
    }

    mutable std::mutex mutex_;
    std::vector<Span> freeVertex_;
    std::vector<Span> freeIndex_;
    uint32_t totalVertex_ = 0;
    uint32_t totalIndex_  = 0;
    uint64_t usedVertex_  = 0;
    uint64_t usedIndex_   = 0;
};
