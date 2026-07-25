#pragma once

#include <cstdint>
#include <vector>
#include <mutex>
#include <algorithm>
#include <cassert>

// Manages a pre-allocated pool of "slots" within a combined vertex/index
// buffer. Each slot has a fixed maximum capacity (reserved at allocation
// time). The allocator never compacts: once a slot index is assigned, it
// remains valid until explicitly freed. Freed slots are recycled to a
// free list.
//
// This eliminates the need for global rebuilds when chunks change:
// an update only touches the specific slot belonging to that chunk.
//
// Thread safety: all public methods are thread-safe via internal mutex.
class SlotAllocator {
public:
    struct Slot {
        bool     active      = false;
        uint32_t vertexOffset = 0; // in elements
        uint32_t vertexCapacity = 0;
        uint32_t vertexCount = 0;
        uint32_t indexOffset  = 0; // in elements
        uint32_t indexCapacity = 0;
        uint32_t indexCount   = 0;
    };

    explicit SlotAllocator() = default;

    // Pre-allocate the slot array. `count` is the number of slots to
    // reserve; the vertex and index capacities for each slot are
    // specified separately (typically the worst-case chunk size).
    // Can be called again to grow the pool (existing slots are preserved).
    void reserve(uint32_t slotCount,
                 uint32_t vertexCapacityPerSlot,
                 uint32_t indexCapacityPerSlot)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (slotCount <= slots_.size()) return;

        size_t oldSize = slots_.size();
        slots_.resize(slotCount);

        // Initialize new slots with contiguous vertex/index offsets
        for (size_t i = oldSize; i < slotCount; ++i) {
            auto& s = slots_[i];
            s.vertexOffset   = static_cast<uint32_t>(i) * vertexCapacityPerSlot;
            s.vertexCapacity = vertexCapacityPerSlot;
            s.indexOffset    = static_cast<uint32_t>(i) * indexCapacityPerSlot;
            s.indexCapacity  = indexCapacityPerSlot;
            s.vertexCount    = 0;
            s.indexCount     = 0;
            s.active         = false;
            freeSlots_.push_back(static_cast<uint32_t>(i));
        }

        vertexCapacityPerSlot_ = vertexCapacityPerSlot;
        indexCapacityPerSlot_  = indexCapacityPerSlot;
    }

    // Allocate a slot for a mesh with the given vertex/index counts.
    // Returns the slot index, or UINT32_MAX if no slot is available
    // AND the pool cannot be grown (caller must handle fallback).
    uint32_t allocate(uint32_t neededVertices, uint32_t neededIndices) {
        std::lock_guard<std::mutex> lock(mutex_);

        if (freeSlots_.empty()) {
            return UINT32_MAX;
        }

        uint32_t slotIdx = freeSlots_.back();
        freeSlots_.pop_back();

        auto& s = slots_[slotIdx];
        assert(!s.active);
        assert(neededVertices <= s.vertexCapacity);
        assert(neededIndices  <= s.indexCapacity);

        s.active      = true;
        s.vertexCount = neededVertices;
        s.indexCount  = neededIndices;

        return slotIdx;
    }

    // Free a previously allocated slot. The slot index becomes available
    // for reuse. The slot's vertex/index data in the buffer is NOT
    // zeroed (the caller may zero the indirect command separately).
    void free(uint32_t slotIdx) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (slotIdx >= slots_.size()) return;

        auto& s = slots_[slotIdx];
        if (!s.active) return;

        s.active      = false;
        s.vertexCount = 0;
        s.indexCount  = 0;
        freeSlots_.push_back(slotIdx);
    }

    // Update the vertex/index counts for an active slot.
    void updateCounts(uint32_t slotIdx, uint32_t vertexCount_, uint32_t indexCount_) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (slotIdx >= slots_.size()) return;

        auto& s = slots_[slotIdx];
        assert(s.active);
        assert(vertexCount_ <= s.vertexCapacity);
        assert(indexCount_  <= s.indexCapacity);
        s.vertexCount = vertexCount_;
        s.indexCount  = indexCount_;
    }

    // Access a slot (thread-safe copy).
    Slot getSlot(uint32_t slotIdx) const {
        std::lock_guard<std::mutex> lock(mutex_);
        if (slotIdx < slots_.size()) {
            return slots_[slotIdx];
        }
        return Slot{};
    }

    // Number of active (allocated) slots.
    uint32_t activeCount() const {
        std::lock_guard<std::mutex> lock(mutex_);
        uint32_t count = 0;
        for (const auto& s : slots_) {
            if (s.active) ++count;
        }
        return count;
    }

    // Total capacity (allocated + free) in slots.
    uint32_t capacity() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return static_cast<uint32_t>(slots_.size());
    }

    // Total vertex elements across allocated + free slots.
    uint32_t totalVertexCapacity() const {
        return capacity() * vertexCapacityPerSlot_;
    }

    // Total index elements across allocated + free slots.
    uint32_t totalIndexCapacity() const {
        return capacity() * indexCapacityPerSlot_;
    }

    // Visit every active slot with a visitor function.
    // The visitor is called with (slotIndex, const Slot&).
    template<typename F>
    void visitActive(F&& visitor) const {
        std::lock_guard<std::mutex> lock(mutex_);
        for (uint32_t i = 0; i < slots_.size(); ++i) {
            if (slots_[i].active) {
                std::forward<F>(visitor)(i, slots_[i]);
            }
        }
    }

private:
    mutable std::mutex mutex_;
    std::vector<Slot> slots_;
    std::vector<uint32_t> freeSlots_;
    uint32_t vertexCapacityPerSlot_ = 0;
    uint32_t indexCapacityPerSlot_  = 0;
};
