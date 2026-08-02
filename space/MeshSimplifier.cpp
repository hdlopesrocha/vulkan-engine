#include "MeshSimplifier.hpp"
#include "../math/BoundingCube.hpp"
#include <tsl/robin_map.h>
#include <cstdint>
#include <functional>

namespace {

// Grid cell key, shared by the interior weld grid (world-space cell indices at
// clusterSize scale) and the border dedup grid (positions quantized to 1e-4).
struct CellKey {
    int64_t ix, iy, iz;
};

bool operator==(const CellKey& a, const CellKey& b) {
    return a.ix == b.ix && a.iy == b.iy && a.iz == b.iz;
}

} // namespace

namespace std {

template <>
struct hash<CellKey> {
    size_t operator()(const CellKey& k) const noexcept {
        uint64_t h = 0x9e3779b97f4a7c15ULL;
        auto mix = [](uint64_t v) {
            v ^= v >> 33;
            v *= 0xff51afd7ed558ccdULL;
            v ^= v >> 33;
            return v;
        };
        h ^= mix(static_cast<uint64_t>(k.ix) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2));
        h ^= mix(static_cast<uint64_t>(k.iy) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2));
        h ^= mix(static_cast<uint64_t>(k.iz) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2));
        return h;
    }
};

} // namespace std

bool decimateVertexCluster(const Geometry& in,
                           float clusterSize,
                           float borderStrip,
                           const BoundingCube& chunkBounds,
                           Geometry& out) {
    if (in.vertices.empty() || in.indices.size() < 3 || clusterSize <= 0.0f) {
        return false;
    }

    const glm::vec3 minP = chunkBounds.getMin();
    const glm::vec3 maxP = chunkBounds.getMax();
    // Strip width is the frontier cell size regardless of cluster size: only
    // the boundary cells' vertices must stay exact (the boundary-edge vertices
    // on the shared plane are the seam). Clusters may touch the strip — their
    // welded points simply exclude the exact strip vertices.
    const float strip = std::max(borderStrip, 1e-3f);

    // clusterIndex -> welded vertex slot. Border vertices use a separate grid
    // quantized to 1e-4 so identical boundary positions dedupe to one vertex.
    struct WeldSlot {
        glm::vec3 sum;
        uint32_t count;
        glm::vec3 exact;
        uint32_t srcVertex; // first source vertex, carries the attributes
        bool border;
    };
    tsl::robin_map<CellKey, uint32_t> clusterMap;
    std::vector<WeldSlot> slots;
    std::vector<uint32_t> remap(in.vertices.size());

    const int64_t borderQuant = 10000; // 1e-4 unit quantization for exact positions
    const float invCluster = 1.0f / clusterSize;

    for (size_t vi = 0; vi < in.vertices.size(); ++vi) {
        const glm::vec3& p = in.vertices[vi].position;
        const bool nearBorder =
            (p.x - minP.x < strip) || (maxP.x - p.x < strip) ||
            (p.y - minP.y < strip) || (maxP.y - p.y < strip) ||
            (p.z - minP.z < strip) || (maxP.z - p.z < strip);

        CellKey key;
        if (nearBorder) {
            key = CellKey{
                static_cast<int64_t>(std::llround(static_cast<double>(p.x) * borderQuant)),
                static_cast<int64_t>(std::llround(static_cast<double>(p.y) * borderQuant)),
                static_cast<int64_t>(std::llround(static_cast<double>(p.z) * borderQuant))};
        } else {
            key = CellKey{
                static_cast<int64_t>(std::floor(p.x * invCluster)),
                static_cast<int64_t>(std::floor(p.y * invCluster)),
                static_cast<int64_t>(std::floor(p.z * invCluster))};
        }

        auto it = clusterMap.find(key);
        uint32_t idx;
        if (it == clusterMap.end()) {
            idx = static_cast<uint32_t>(slots.size());
            clusterMap.emplace(key, idx);
            WeldSlot slot{};
            slot.sum = p;
            slot.count = 1;
            slot.exact = p;
            slot.srcVertex = static_cast<uint32_t>(vi);
            slot.border = nearBorder;
            slots.push_back(slot);
        } else {
            idx = it->second;
            WeldSlot& slot = slots[idx];
            slot.sum += p;
            ++slot.count;
        }
        remap[vi] = idx;
    }

    if (slots.size() < 3) {
        return false; // everything welded to a point: nothing left to draw
    }

    // Resolve welded positions: border slots stay exact, interior slots use
    // the cluster average. Attributes are carried over from the slot's first
    // source vertex (color, texture, normal, brush, hsv).
    std::vector<Vertex> verts;
    verts.reserve(slots.size());
    for (size_t si = 0; si < slots.size(); ++si) {
        const WeldSlot& slot = slots[si];
        glm::vec3 pos = slot.border ? slot.exact : slot.sum / static_cast<float>(slot.count);
        Vertex v = in.vertices[slot.srcVertex];
        v.position = pos;
        verts.push_back(v);
    }

    // Rebuild triangles through the weld map, dropping degenerates.
    out.vertices.clear();
    out.indices.clear();
    out.vertices = std::move(verts);
    out.indices.reserve(in.indices.size());
    for (size_t ti = 0; ti + 2 < in.indices.size(); ti += 3) {
        const uint32_t a = remap[in.indices[ti]];
        const uint32_t b = remap[in.indices[ti + 1]];
        const uint32_t c = remap[in.indices[ti + 2]];
        if (a == b || b == c || a == c) {
            continue; // collapsed sliver
        }
        out.indices.push_back(a);
        out.indices.push_back(b);
        out.indices.push_back(c);
    }

    out.setCenter();
    return !out.indices.empty();
}
