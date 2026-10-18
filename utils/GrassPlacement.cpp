#include "GrassPlacement.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>

namespace grass {

uint32_t xorshift32(uint32_t& state) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

float randFloat(uint32_t& state) {
    return float(xorshift32(state) & 0x00FFFFFFu) / float(0x01000000u);
}

uint32_t posHash(const glm::vec3& p) {
    glm::ivec3 qi = glm::ivec3(glm::round(p * 8.0f));
    uint32_t h = uint32_t(qi.x) * 1640531513u;
    h ^= uint32_t(qi.y) * 2246822519u;
    h ^= uint32_t(qi.z) * 3266489917u;
    return h;
}

uint32_t cellHash(glm::ivec2 c) {
    uint32_t h = uint32_t(c.x) * 1640531513u ^ uint32_t(c.y) * 2246822519u;
    h ^= h >> 13;
    h *= 0x45d9f3bu;
    h ^= h >> 16;
    return h;
}

float biomeNoise(const glm::vec2& xz) {
    const float kBiomeScale = 50.0f;
    glm::vec2 p = xz / kBiomeScale;
    glm::ivec2 i = glm::ivec2(glm::floor(p));
    glm::vec2 f = p - glm::vec2(i);
    glm::vec2 u = f * f * (3.0f - 2.0f * f);

    float a = float(cellHash(i + glm::ivec2(0, 0))) / 4294967295.0f;
    float b = float(cellHash(i + glm::ivec2(1, 0))) / 4294967295.0f;
    float c = float(cellHash(i + glm::ivec2(0, 1))) / 4294967295.0f;
    float d = float(cellHash(i + glm::ivec2(1, 1))) / 4294967295.0f;

    return glm::mix(glm::mix(a, b, u.x), glm::mix(c, d, u.x), u.y);
}

namespace {

inline float fade(float t) {
    return t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f);
}

inline float hash12(const glm::vec2& p) {
    glm::vec3 p3 = glm::fract(glm::vec3(p.x, p.y, p.x) * 0.1031f);
    p3 += glm::vec3(glm::dot(p3, glm::vec3(p3.y, p3.z, p3.x) + 33.33f));
    return glm::fract((p3.x + p3.y) * p3.z);
}

inline glm::vec2 grad2(const glm::vec2& ip) {
    static const glm::vec2 kGrads[8] = {
        { 1.0f, 0.0f },
        { 0.70710678f, 0.70710678f },
        { 0.0f, 1.0f },
        { -0.70710678f, 0.70710678f },
        { -1.0f, 0.0f },
        { -0.70710678f, -0.70710678f },
        { 0.0f, -1.0f },
        { 0.70710678f, -0.70710678f },
    };
    const int idx = int(hash12(ip) * 8.0f) & 7;
    return kGrads[idx];
}

} // namespace

float perlin2(const glm::vec2& p) {
    glm::vec2 i = glm::floor(p);
    glm::vec2 f = glm::fract(p);
    glm::vec2 u = glm::vec2(fade(f.x), fade(f.y));

    const float n00 = glm::dot(grad2(i + glm::vec2(0.0f, 0.0f)), f - glm::vec2(0.0f, 0.0f));
    const float n10 = glm::dot(grad2(i + glm::vec2(1.0f, 0.0f)), f - glm::vec2(1.0f, 0.0f));
    const float n01 = glm::dot(grad2(i + glm::vec2(0.0f, 1.0f)), f - glm::vec2(0.0f, 1.0f));
    const float n11 = glm::dot(grad2(i + glm::vec2(1.0f, 1.0f)), f - glm::vec2(1.0f, 1.0f));

    const float nx0 = glm::mix(n00, n10, u.x);
    const float nx1 = glm::mix(n01, n11, u.x);
    return glm::mix(nx0, nx1, u.y);
}

float heightScaleAt(const glm::vec2& worldXZ) {
    const float n = perlin2(worldXZ * 0.008f);
    return 0.6f + 0.8f * (n * 0.5f + 0.5f);
}

void collectSlots(const Geometry& geom, uintptr_t nodeId, std::vector<uint32_t>& out) {
    out.clear();
    if (geom.vertices.empty() || geom.indices.size() < 3) return;

    const uint32_t chunkSeed = static_cast<uint32_t>(nodeId ^ (nodeId >> 32)) ^ 0x9e3779b9u;
    std::mt19937 samplingRng(chunkSeed);
    std::uniform_real_distribution<float> unitDist(0.0f, 1.0f);

    const size_t vertexCount = geom.vertices.size();
    for (size_t i = 0; i + 2 < geom.indices.size(); i += 3) {
        const uint32_t i0 = geom.indices[i + 0];
        const uint32_t i1 = geom.indices[i + 1];
        const uint32_t i2 = geom.indices[i + 2];
        if (i0 >= vertexCount || i1 >= vertexCount || i2 >= vertexCount) continue;

        // Permissive: any grass-touching triangle contributes slots; the
        // per-instance dominant-vertex test trims the spill later.
        const bool hasBrush =
            geom.vertices[i0].brushIndex == kGrassBrush ||
            geom.vertices[i1].brushIndex == kGrassBrush ||
            geom.vertices[i2].brushIndex == kGrassBrush;
        if (!hasBrush) continue;

        const glm::vec3& v0 = geom.vertices[i0].position;
        const glm::vec3& v1 = geom.vertices[i1].position;
        const glm::vec3& v2 = geom.vertices[i2].position;
        const glm::vec3 faceNormal = glm::cross(v1 - v0, v2 - v0);
        const float area = 0.5f * glm::length(faceNormal);
        const float expectedInstances = std::max(0.0f, area * kDensity);
        uint32_t slotCount = static_cast<uint32_t>(std::floor(expectedInstances));
        const float fractional = expectedInstances - static_cast<float>(slotCount);
        if (unitDist(samplingRng) < fractional) {
            ++slotCount;
        }
        for (uint32_t s = 0; s < slotCount; ++s) {
            out.push_back(i0);
            out.push_back(i1);
            out.push_back(i2);
        }
    }

    // Shuffle virtual triangle slots per chunk so reducing the instance count
    // keeps a random spatial subset instead of always dropping the tail.
    if (out.size() >= 6) {
        std::mt19937 shuffleRng(chunkSeed ^ 0x85ebca6bu);
        const size_t triangleCount = out.size() / 3;
        for (size_t slot = triangleCount - 1; slot > 0; --slot) {
            std::uniform_int_distribution<size_t> dist(0, slot);
            const size_t other = dist(shuffleRng);
            if (other == slot) continue;
            for (size_t component = 0; component < 3; ++component) {
                std::swap(out[slot * 3 + component], out[other * 3 + component]);
            }
        }
    }
}

void placeInstances(const std::vector<glm::vec3>& positions,
                    const std::vector<glm::vec3>& normals,
                    const std::vector<int>& brushes,
                    const std::vector<uint32_t>& slots,
                    uint32_t billboardCount,
                    uint32_t seed,
                    std::vector<Instance>& out) {
    const uint32_t billboardCnt = (billboardCount > 0) ? billboardCount : 1u;
    const uint32_t triCount = static_cast<uint32_t>(slots.size()) / 3;

    const bool hasNormals = (normals.size() == positions.size());
    const bool hasBrushes = (brushes.size() == positions.size());

    for (uint32_t tri = 0; tri < triCount; ++tri) {
        const uint32_t tb = tri * 3;
        const uint32_t i0 = slots[tb + 0];
        const uint32_t i1 = slots[tb + 1];
        const uint32_t i2 = slots[tb + 2];
        if (i0 >= positions.size() || i1 >= positions.size() || i2 >= positions.size()) continue;

        const glm::vec3 v0 = positions[i0];
        const glm::vec3 v1 = positions[i1];
        const glm::vec3 v2 = positions[i2];

        // Steep triangles carry rock/mix-brushed verts, so the dominant-vertex
        // test below rejects their samples.
        const glm::vec3 tc = (v0 + v1 + v2) / 3.0f;
        const uint32_t tch = posHash(tc);

        uint32_t rng = seed ^ tch ^ (tri * 2654435761u) ^ (0u * 19349663u);
        float u = randFloat(rng);
        float v = randFloat(rng);
        if (u + v > 1.0f) { u = 1.0f - u; v = 1.0f - v; }
        const float w = 1.0f - u - v;
        const glm::vec3 pos = u * v0 + v * v1 + w * v2;

        // Dominant-vertex brush test: keep the sample only if the closest
        // vertex (largest barycentric weight) is grass-brushed.
        if (hasBrushes && i0 < brushes.size() && i1 < brushes.size() && i2 < brushes.size()) {
            const int dominantBrush = (u >= v && u >= w) ? brushes[i0]
                                    : (v >= w)           ? brushes[i1]
                                                         : brushes[i2];
            if (dominantBrush != kGrassBrush) continue;
        }

        // Biome from spatially-coherent noise.
        const float noise = biomeNoise(glm::vec2(pos.x, pos.z));
        if (noise < 0.40f) continue; // empty biome — skip entirely

        const float remapped = (noise - 0.40f) / 0.60f;
        const uint32_t bi = std::min(uint32_t(remapped * float(billboardCnt)), billboardCnt - 1u);

        uint32_t rs = seed ^ posHash(pos);
        const float rf = randFloat(rs);

        glm::vec3 nrm(0.0f, 1.0f, 0.0f);
        if (hasNormals) {
            nrm = u * normals[i0] + v * normals[i1] + w * normals[i2];
            const float n2 = glm::dot(nrm, nrm);
            nrm = (n2 > 1e-8f) ? nrm * (1.0f / std::sqrt(n2)) : glm::vec3(0.0f, 1.0f, 0.0f);
        }

        Instance inst;
        inst.position = pos;
        inst.normal = nrm;
        inst.variant = static_cast<int>(bi);
        inst.rotation = rf;
        out.push_back(inst);
    }
}

void placeInstances(const Geometry& geom, const std::vector<uint32_t>& slots,
                    uintptr_t nodeId, uint32_t billboardCount,
                    std::vector<Instance>& out) {
    std::vector<glm::vec3> positions;
    std::vector<glm::vec3> normals;
    std::vector<int> brushes;
    positions.reserve(geom.vertices.size());
    normals.reserve(geom.vertices.size());
    brushes.reserve(geom.vertices.size());
    for (const Vertex& v : geom.vertices) {
        positions.push_back(v.position);
        normals.push_back(v.normal);
        brushes.push_back(v.brushIndex);
    }
    const uint32_t seed = static_cast<uint32_t>(nodeId & 0xffffffffull);
    placeInstances(positions, normals, brushes, slots, billboardCount, seed, out);
}

} // namespace grass
