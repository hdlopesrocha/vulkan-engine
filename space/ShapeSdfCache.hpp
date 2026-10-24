#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>
#include <glm/glm.hpp>
#if __has_include(<tsl/robin_map.h>)
#include <tsl/robin_map.h>
#else
#include <unordered_map>
namespace tsl {
    template <typename K, typename V, typename H = std::hash<K>>
    using robin_map = std::unordered_map<K, V, H>;
}
#endif

// Per-thread cache for shape SDF values (perf report 29, C1).
//
// Shape queries are corners or centers of the cubes the recursion visits.
// For this apply's own leaf cubes those points lie on its lattice:
// cacheOrigin + n * spacing, with spacing = L_leaf / 2 (corners are whole
// L_leaf steps, centers half steps). The stored tree can hold finer nodes than
// the current leaf (an earlier apply ran with a smaller minSize) and the
// recursion then visits cubes off this lattice; those queries use the
// fallback map, exactly like any point whose float bits do not reconstruct
// from a rounded index. The lattice index packs into 21 bits per axis, so the
// fast path is a flat open-addressed table keyed by a uint64: no three-float
// hash, no node allocation per entry.
//
// Exactness. The table is only probed for a position that reconstructs from
// its rounded lattice index bit-for-bit (Octree::evaluateSDF checks this), so a
// hit can only return the value that was computed for exactly those float
// coordinates in this apply. Everything else -- non-lattice positions, or a
// float representation that is not the canonical lattice point -- goes to the
// float-keyed fallback map, which is what this cache used to be, so those
// queries behave exactly as before.
//
// Generations. A context can outlive an apply (thread_local worker contexts),
// so entries are tagged with the apply's generation: a different tag reads as
// empty, which makes starting a new apply free. Stale entries are dropped when
// the table rehashes, and the table keeps its capacity, so it is already sized
// to the previous apply's peak.
class ShapeSdfCache {
public:
    static constexpr uint32_t kAxisBits = 21;
    static constexpr uint64_t kAxisLimit = uint64_t(1) << kAxisBits;
    static constexpr uint64_t kAxisMask = kAxisLimit - 1;

    static uint64_t pack(int64_t x, int64_t y, int64_t z) {
        return (uint64_t(x) & kAxisMask)
             | ((uint64_t(y) & kAxisMask) << kAxisBits)
             | ((uint64_t(z) & kAxisMask) << (2 * kAxisBits));
    }

    // Switch the cache to a new apply generation. Generation 0 means "not in an
    // apply": the table stays disabled and the fallback map is kept as-is.
    void beginGeneration(uint32_t generation);

    // Probe the table. False means "not computed in this generation" (or the
    // table is still empty).
    bool find(uint64_t key, float &out) const;

    void insert(uint64_t key, float value);

    // Slow path for positions that are not canonical lattice points.
    tsl::robin_map<glm::vec3, float> &fallback() { return fallback_; }

    size_t capacity() const { return entries_.size(); }

private:
    struct Entry {
        uint64_t key;
        uint32_t generation;
        float value;
    };

    void place(uint64_t key, float value);
    void grow();

    std::vector<Entry> entries_;
    size_t live_ = 0;
    uint32_t generation_ = 0;
    tsl::robin_map<glm::vec3, float> fallback_;
};

inline void ShapeSdfCache::beginGeneration(uint32_t generation) {
    if (generation == generation_) return;
    generation_ = generation;
    live_ = 0;
    fallback_.clear();
    // The table is deliberately not cleared: every existing entry now has a
    // stale generation and reads as empty. Capacity is retained, so the table
    // is already reserved to the previous apply's peak.
}

inline bool ShapeSdfCache::find(uint64_t key, float &out) const {
    if (entries_.empty() || generation_ == 0) return false;
    const size_t mask = entries_.size() - 1;
    size_t i = size_t((key * 0x9E3779B97F4A7C15ull) >> 32) & mask;
    for (;;) {
        const Entry &e = entries_[i];
        if (e.generation != generation_) return false;
        if (e.key == key) {
            out = e.value;
            return true;
        }
        i = (i + 1) & mask;
    }
}

inline void ShapeSdfCache::insert(uint64_t key, float value) {
    if (generation_ == 0) return;
    if (entries_.empty()) entries_.resize(1024);
    // Grow at 70% load so a probe always finds an empty (stale) slot.
    if ((live_ + 1) * 10 > entries_.size() * 7) grow();
    place(key, value);
}

inline void ShapeSdfCache::place(uint64_t key, float value) {
    const size_t mask = entries_.size() - 1;
    size_t i = size_t((key * 0x9E3779B97F4A7C15ull) >> 32) & mask;
    for (;;) {
        Entry &e = entries_[i];
        if (e.generation != generation_) {
            e = Entry{key, generation_, value};
            ++live_;
            return;
        }
        if (e.key == key) {
            e.value = value;
            return;
        }
        i = (i + 1) & mask;
    }
}

inline void ShapeSdfCache::grow() {
    std::vector<Entry> old;
    old.swap(entries_);
    entries_.resize(old.size() * 2);
    live_ = 0;
    for (const Entry &e : old) {
        if (e.generation != generation_) continue;
        place(e.key, e.value);
    }
}
