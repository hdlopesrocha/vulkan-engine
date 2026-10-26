#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>

// Per-chunk mesh-build profile. Written by the worker/render threads after each
// GPU topology build and read by the ImGui overlay / log reporting. It is
// deliberately outside OctreeNode: the octree stores scene topology, not
// transient renderer capacity telemetry.
enum class ChunkMeshProfileStatus : uint8_t {
    Built = 0,  // geometry published (indexCount >= 3)
    Empty = 1,  // valid rebuilt revision with no triangles
    Failed = 2, // span/overflow/invalid-slot failure; chunk not published
};

struct ChunkMeshProfile {
    uint32_t chunkId = 0;
    int32_t layer = 0;
    uint8_t lod = 0;
    ChunkMeshProfileStatus status = ChunkMeshProfileStatus::Failed;
    uint32_t crossEdgeCount = 0;
    uint32_t maxSegmentsPerCrossEdge = 1;
    uint32_t vertexCapacity = 0; // dense surface-node slot span reserved
    uint32_t indexCapacity = 0;  // cross-edge topology bound reserved
    uint32_t vertexCount = 0;    // distinct vertex payloads actually written
    uint32_t vertexExtent = 0;   // highest referenced slot + 1 (published span)
    uint32_t indexCount = 0;     // indices actually written
    uint32_t indexOverflow = 0;  // triangles rejected by the index span
    uint32_t vertexOverflow = 0; // triangles rejected by the vertex span
    uint32_t invalidSlots = 0;   // corners with no host-assigned slot
    uint32_t attempts = 1;       // geometry dispatches submitted
    bool grew = false;           // span released and re-reserved larger

    static constexpr uint32_t kVertexStrideBytes = 64; // math/geometry/Vertex
    static constexpr uint32_t kIndexStrideBytes = 4;

    bool hasGeometry() const {
        return status == ChunkMeshProfileStatus::Built && vertexCount > 0 && indexCount > 0;
    }
    bool overflowed() const {
        return grew || indexOverflow != 0 || vertexOverflow != 0 || invalidSlots != 0;
    }

    static double percent(uint64_t used, uint64_t capacity) {
        return capacity != 0 ? 100.0 * static_cast<double>(used) /
                                   static_cast<double>(capacity)
                             : 0.0;
    }

    // Independent vertex/index utilization.
    double vertexUtilization() const { return percent(vertexCount, vertexCapacity); }
    double indexUtilization() const { return percent(indexCount, indexCapacity); }

    // Byte-weighted combined utilization: one reserved vertex slot is 64 B,
    // one reserved index is 4 B. Using the byte ratio (not the average of the
    // two percentages) keeps the metric comparable to the pool footprint.
    double combinedUtilization() const {
        const uint64_t used = uint64_t(vertexCount) * kVertexStrideBytes +
                              uint64_t(indexCount) * kIndexStrideBytes;
        const uint64_t capacity = uint64_t(vertexCapacity) * kVertexStrideBytes +
                                  uint64_t(indexCapacity) * kIndexStrideBytes;
        return percent(used, capacity);
    }
};

struct ChunkMeshUtilizationStats {
    double min = 0.0;
    double max = 0.0;
    double mean = 0.0;
    double median = 0.0;
};

struct ChunkMeshProfileAggregate {
    size_t chunks = 0;
    size_t built = 0;
    size_t empty = 0;
    size_t failed = 0;
    size_t overflowChunks = 0;
    size_t lowUtilizationChunks = 0; // built, combined utilization < 25 %

    uint64_t crossEdges = 0;
    uint64_t vertexCapacity = 0;
    uint64_t indexCapacity = 0;
    uint64_t vertexCount = 0;
    uint64_t indexCount = 0;

    // Aggregate utilization is computed from the totals, not averaged from
    // per-chunk percentages.
    double vertexUtilization = 0.0;
    double indexUtilization = 0.0;
    double combinedUtilization = 0.0;

    ChunkMeshUtilizationStats vertexStats;
    ChunkMeshUtilizationStats indexStats;
    ChunkMeshUtilizationStats combinedStats;
};

struct ChunkMeshProfileSnapshot {
    std::vector<ChunkMeshProfile> chunks; // sorted by (layer, chunkId)
    ChunkMeshProfileAggregate aggregate;
};

// Thread-safe registry with a versioned snapshot cache: recording a build
// invalidates the cache, while the UI/log path reuses the last sorted vector
// and aggregate until something changes (no per-frame traversal/sort).
class ChunkMeshProfileRegistry {
public:
    // Project convention for pool-utilization warnings (see
    // IndirectRenderer::logUtilization).
    static constexpr double kLowUtilizationRatio = 0.25;

    void record(const ChunkMeshProfile& profile) {
        std::lock_guard<std::mutex> lock(mutex_);
        chunks_[key(profile.layer, profile.chunkId)] = profile;
        ++generation_;
    }

    void erase(int32_t layer, uint32_t chunkId) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (chunks_.erase(key(layer, chunkId)) != 0) ++generation_;
    }

    void clearLayer(int32_t layer) {
        std::lock_guard<std::mutex> lock(mutex_);
        const uint64_t prefix = uint64_t(static_cast<uint32_t>(layer)) << 32;
        bool removed = false;
        for (auto it = chunks_.begin(); it != chunks_.end();) {
            if ((it->first & 0xFFFFFFFF00000000ull) == prefix) {
                it = chunks_.erase(it);
                removed = true;
            } else {
                ++it;
            }
        }
        if (removed) ++generation_;
    }

    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (chunks_.empty()) return;
        chunks_.clear();
        ++generation_;
    }

    std::shared_ptr<const ChunkMeshProfileSnapshot> snapshot() const {
        std::lock_guard<std::mutex> lock(mutex_);
        if (cache_ && cachedGeneration_ == generation_) return cache_;
        auto snap = std::make_shared<ChunkMeshProfileSnapshot>();
        snap->chunks.reserve(chunks_.size());
        for (const auto& kv : chunks_) snap->chunks.push_back(kv.second);
        std::sort(snap->chunks.begin(), snap->chunks.end(),
                  [](const ChunkMeshProfile& a, const ChunkMeshProfile& b) {
                      if (a.layer != b.layer) return a.layer < b.layer;
                      return a.chunkId < b.chunkId;
                  });
        snap->aggregate = aggregate(snap->chunks);
        cache_ = snap;
        cachedGeneration_ = generation_;
        return cache_;
    }

private:
    static uint64_t key(int32_t layer, uint32_t chunkId) {
        return (uint64_t(static_cast<uint32_t>(layer)) << 32) | chunkId;
    }

    static ChunkMeshUtilizationStats stats(std::vector<double> values) {
        ChunkMeshUtilizationStats out;
        if (values.empty()) return out;
        std::sort(values.begin(), values.end());
        out.min = values.front();
        out.max = values.back();
        double sum = 0.0;
        for (double v : values) sum += v;
        out.mean = sum / static_cast<double>(values.size());
        const size_t n = values.size();
        out.median = (n & 1u) ? values[n / 2]
                              : 0.5 * (values[n / 2 - 1] + values[n / 2]);
        return out;
    }

    static ChunkMeshProfileAggregate aggregate(const std::vector<ChunkMeshProfile>& chunks) {
        ChunkMeshProfileAggregate a;
        std::vector<double> vertexUtils, indexUtils, combinedUtils;
        vertexUtils.reserve(chunks.size());
        indexUtils.reserve(chunks.size());
        combinedUtils.reserve(chunks.size());

        for (const ChunkMeshProfile& p : chunks) {
            ++a.chunks;
            switch (p.status) {
                case ChunkMeshProfileStatus::Built: ++a.built; break;
                case ChunkMeshProfileStatus::Empty: ++a.empty; break;
                case ChunkMeshProfileStatus::Failed: ++a.failed; break;
            }
            if (p.overflowed()) ++a.overflowChunks;
            if (!p.hasGeometry()) continue;

            a.crossEdges += p.crossEdgeCount;
            a.vertexCapacity += p.vertexCapacity;
            a.indexCapacity += p.indexCapacity;
            a.vertexCount += p.vertexCount;
            a.indexCount += p.indexCount;

            const double vu = p.vertexUtilization();
            const double iu = p.indexUtilization();
            const double cu = p.combinedUtilization();
            vertexUtils.push_back(vu);
            indexUtils.push_back(iu);
            combinedUtils.push_back(cu);
            if (cu < kLowUtilizationRatio * 100.0) ++a.lowUtilizationChunks;
        }

        a.vertexUtilization = ChunkMeshProfile::percent(a.vertexCount, a.vertexCapacity);
        a.indexUtilization = ChunkMeshProfile::percent(a.indexCount, a.indexCapacity);
        const uint64_t usedBytes = a.vertexCount * ChunkMeshProfile::kVertexStrideBytes +
                                   a.indexCount * ChunkMeshProfile::kIndexStrideBytes;
        const uint64_t capBytes = a.vertexCapacity * ChunkMeshProfile::kVertexStrideBytes +
                                  a.indexCapacity * ChunkMeshProfile::kIndexStrideBytes;
        a.combinedUtilization = ChunkMeshProfile::percent(usedBytes, capBytes);

        a.vertexStats = stats(std::move(vertexUtils));
        a.indexStats = stats(std::move(indexUtils));
        a.combinedStats = stats(std::move(combinedUtils));
        return a;
    }

    mutable std::mutex mutex_;
    std::unordered_map<uint64_t, ChunkMeshProfile> chunks_;
    uint64_t generation_ = 1;
    mutable uint64_t cachedGeneration_ = 0;
    mutable std::shared_ptr<const ChunkMeshProfileSnapshot> cache_;
};
