// Generic CPU-side SDF scene. Keeps no dependency on space/Octree or on
// sdf/*DistanceFunction; only standard C++23 + GLM + SdfUBO.hpp.
#include "sdf/gpu/SdfScene.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>

namespace sdf_gpu {
namespace {

// Deform flag bits (must match SdfDeformFlags in SdfUBO.hpp).
constexpr uint32_t kDeformNoise = 1u << 0;
constexpr uint32_t kDeformTwist = 1u << 1;
constexpr uint32_t kDeformBend = 1u << 2;
constexpr uint32_t kDeformTaper = 1u << 3;
constexpr uint32_t kDeformRepeat = 1u << 4;

glm::mat3 rotationFromEuler(const glm::vec3& e) {
    // MUST match sdfEulerMat() in shaders/includes/sdf_ops.glsl exactly:
    // R = Rx * Ry * Rz (same matrix entries and product order). The GPU
    // inverts this to reach primitive-local space, and the bounds below must
    // enclose the same oriented shape, so any divergence here breaks both
    // culling and the SDF itself once euler is non-zero.
    const glm::mat4 rx = glm::rotate(glm::mat4(1.0f), e.x, glm::vec3(1.0f, 0.0f, 0.0f));
    const glm::mat4 ry = glm::rotate(glm::mat4(1.0f), e.y, glm::vec3(0.0f, 1.0f, 0.0f));
    const glm::mat4 rz = glm::rotate(glm::mat4(1.0f), e.z, glm::vec3(0.0f, 0.0f, 1.0f));
    return glm::mat3(rx * ry * rz);
}

// Local (pre-rotation, pre-translation) half extents for a definition.
// radial = scale*radiusScale, height = scale*heightScale, uni = scale.
glm::vec3 localHalfExtents(SdfPrimitiveType prim, const glm::vec4& p0,
                           float radial, float height, float uni) {
    const float r = std::max(p0.x, 0.0f);
    switch (prim) {
        case SdfPrimitiveType::Sphere: {
            const float rr = std::max(r * radial, 0.001f);
            return glm::vec3(rr);
        }
        case SdfPrimitiveType::Box:
        case SdfPrimitiveType::RoundedBox: {
            // params0.xyz = half extents.
            glm::vec3 h(std::max(p0.x, 0.0f) * radial,
                        std::max(p0.y, 0.0f) * height,
                        std::max(p0.z, 0.0f) * radial);
            return glm::vec3(std::max(h.x, 0.001f), std::max(h.y, 0.001f), std::max(h.z, 0.001f));
        }
        case SdfPrimitiveType::Capsule: {
            // params0.x = radius, params0.y = half height of cylinder section.
            const float rad = std::max(p0.x, 0.0f) * radial;
            const float hh = std::max(p0.y, 0.0f) * height;
            return glm::vec3(std::max(rad, 0.001f), std::max(hh + rad, 0.001f), std::max(rad, 0.001f));
        }
        case SdfPrimitiveType::Cylinder: {
            const float rad = std::max(p0.x, 0.0f) * radial;
            const float hh = std::max(p0.y, 0.0f) * height;
            return glm::vec3(std::max(rad, 0.001f), std::max(hh, 0.001f), std::max(rad, 0.001f));
        }
        case SdfPrimitiveType::Cone: {
            // params0.x = base radius, params0.y = full height (centered).
            const float rad = std::max(p0.x, 0.0f) * radial;
            const float h = std::max(p0.y, 0.0f) * height;
            return glm::vec3(std::max(rad, 0.001f), std::max(h * 0.5f, 0.001f), std::max(rad, 0.001f));
        }
        case SdfPrimitiveType::Torus: {
            // params0.x = major R, params0.y = minor r, ring in XZ plane.
            const float R = std::max(p0.x, 0.0f) * radial;
            const float mr = std::max(p0.y, 0.0f) * std::max(radial, height);
            const float ext = R + mr;
            return glm::vec3(std::max(ext, 0.001f), std::max(mr, 0.001f), std::max(ext, 0.001f));
        }
        case SdfPrimitiveType::Plane: {
            // Infinite plane -> large thin box.
            const float s = std::max(uni, 0.001f);
            return glm::vec3(500.0f * s, 0.5f * s, 500.0f * s);
        }
        case SdfPrimitiveType::Flame: {
            // Base-anchored tapered capsule: local box y in [-R, H+R] with
            // R = base radius, H = height. Bounds are computed as a CENTERED
            // box of double height, a conservative superset under any
            // rotation about the anchor (slightly loose below the anchor;
            // buried part is occluded by the composite depth test anyway).
            // Spike displacement (<= spikiness) is covered by the deformation
            // padding below.
            const float R = std::max(p0.x, 0.0f) * radial;
            const float H = std::max(p0.y, 0.0f) * height;
            return glm::vec3(std::max(R, 0.001f), std::max(H + R, 0.001f),
                             std::max(R, 0.001f));
        }
    }
    return glm::vec3(0.5f * uni);
}

float deformationPadding(const SdfScene::Definition& def, float turbulence) {
    // Generous padding: base formula turbulence*0.5+0.5, plus smoothK for
    // smooth ops and small extras for analytic deformations.
    float pad = 0.5f + turbulence * 0.5f;
    if (def.op == SdfOpType::SmoothUnion || def.op == SdfOpType::SmoothIntersection ||
        def.op == SdfOpType::SmoothSubtraction) {
        pad += std::max(def.smoothK, 0.0f);
    }
    if (def.deformFlags & kDeformTwist) pad += 0.5f;
    if (def.deformFlags & kDeformBend) pad += 0.5f;
    if (def.deformFlags & kDeformTaper) pad += 0.5f;
    if (def.deformFlags & kDeformRepeat) pad += 2.0f;
    // Noise flag is already covered by the turbulence term; keep a minimum.
    (void)kDeformNoise;
    return std::max(pad, 0.5f);
}

} // namespace

uint32_t SdfScene::addDefinition(const Definition& d) {
    definitions_.push_back(d);
    return static_cast<uint32_t>(definitions_.size() - 1);
}

uint32_t SdfScene::addMaterial(const Material& m) {
    materials_.push_back(m);
    return static_cast<uint32_t>(materials_.size() - 1);
}

uint32_t SdfScene::addContainer(const glm::vec3& minp, const glm::vec3& maxp, glm::uvec3 res) {
    Container c;
    c.minp = minp;
    c.maxp = maxp;
    c.resolution = glm::uvec3(std::max(res.x, 1u), std::max(res.y, 1u), std::max(res.z, 1u));
    containers_.push_back(std::move(c));
    return static_cast<uint32_t>(containers_.size() - 1);
}

uint32_t SdfScene::addInstance(const Instance& in) {
    instances_.push_back(in);
    const uint32_t idx = static_cast<uint32_t>(instances_.size() - 1);
    // Keep the container's index list in sync when the target exists.
    if (in.containerIdx < containers_.size()) {
        containers_[in.containerIdx].instanceIndices.push_back(idx);
    }
    return idx;
}

void SdfScene::clear() {
    definitions_.clear();
    materials_.clear();
    instances_.clear();
    containers_.clear();
}

glm::vec4 SdfScene::computeInstanceBoundsMin(const Instance& in) const {
    glm::vec3 half(0.5f);
    float pad = 0.5f;
    if (in.defIdx < definitions_.size()) {
        const Definition& def = definitions_[in.defIdx];
        float turbulence = 0.6f;
        if (in.matIdx < materials_.size()) turbulence = materials_[in.matIdx].turbulence;
        const float uni = std::max(in.scale, 0.0001f);
        const float radial = uni * std::max(in.radiusScale, 0.0001f);
        const float height = uni * std::max(in.heightScale, 0.0001f);
        half = localHalfExtents(def.prim, def.params0, radial, height, uni);
        pad = deformationPadding(def, turbulence);
        if (def.prim == SdfPrimitiveType::Flame) {
            // Spike displacement is spikiness (local) * instance scale in
            // world units; without this, giant spiky flames escape their
            // bounds and get wrongly culled by the bound test / grid.
            pad += std::max(def.params1.y, 0.0f) * uni;
        }
    } else {
        half = glm::vec3(0.5f * std::max(in.scale, 0.0001f));
    }
    half += glm::vec3(pad);
    const glm::mat3 r = rotationFromEuler(in.euler);
    const glm::mat3 ar(glm::vec3(std::abs(r[0].x), std::abs(r[0].y), std::abs(r[0].z)),
                       glm::vec3(std::abs(r[1].x), std::abs(r[1].y), std::abs(r[1].z)),
                       glm::vec3(std::abs(r[2].x), std::abs(r[2].y), std::abs(r[2].z)));
    const glm::vec3 worldHalf = ar * half;
    return glm::vec4(in.pos - worldHalf, 0.0f);
}

glm::vec4 SdfScene::computeInstanceBoundsMax(const Instance& in) const {
    glm::vec3 half(0.5f);
    float pad = 0.5f;
    if (in.defIdx < definitions_.size()) {
        const Definition& def = definitions_[in.defIdx];
        float turbulence = 0.6f;
        if (in.matIdx < materials_.size()) turbulence = materials_[in.matIdx].turbulence;
        const float uni = std::max(in.scale, 0.0001f);
        const float radial = uni * std::max(in.radiusScale, 0.0001f);
        const float height = uni * std::max(in.heightScale, 0.0001f);
        half = localHalfExtents(def.prim, def.params0, radial, height, uni);
        pad = deformationPadding(def, turbulence);
        if (def.prim == SdfPrimitiveType::Flame) {
            // Spike displacement is spikiness (local) * instance scale in
            // world units; without this, giant spiky flames escape their
            // bounds and get wrongly culled by the bound test / grid.
            pad += std::max(def.params1.y, 0.0f) * uni;
        }
    } else {
        half = glm::vec3(0.5f * std::max(in.scale, 0.0001f));
    }
    half += glm::vec3(pad);
    const glm::mat3 r = rotationFromEuler(in.euler);
    const glm::mat3 ar(glm::vec3(std::abs(r[0].x), std::abs(r[0].y), std::abs(r[0].z)),
                       glm::vec3(std::abs(r[1].x), std::abs(r[1].y), std::abs(r[1].z)),
                       glm::vec3(std::abs(r[2].x), std::abs(r[2].y), std::abs(r[2].z)));
    const glm::vec3 worldHalf = ar * half;
    return glm::vec4(in.pos + worldHalf, 0.0f);
}

SdfScene::BuiltGrid SdfScene::buildContainerGrid(uint32_t containerIdx) const {
    BuiltGrid out;
    if (containerIdx >= containers_.size()) return out;
    const Container& c = containers_[containerIdx];
    const uint32_t nx = std::max(c.resolution.x, 1u);
    const uint32_t ny = std::max(c.resolution.y, 1u);
    const uint32_t nz = std::max(c.resolution.z, 1u);
    const size_t cellCount = static_cast<size_t>(nx) * ny * nz;
    out.cells.resize(cellCount);
    for (auto& cell : out.cells) {
        cell.offset = 0;
        cell.count = 0;
        cell._pad0 = 0;
        cell._pad1 = 0;
    }
    if (cellCount == 0) return out;

    glm::vec3 extent = c.maxp - c.minp;
    glm::vec3 cellSize(1.0f);
    if (extent.x > 1e-6f) cellSize.x = extent.x / static_cast<float>(nx);
    if (extent.y > 1e-6f) cellSize.y = extent.y / static_cast<float>(ny);
    if (extent.z > 1e-6f) cellSize.z = extent.z / static_cast<float>(nz);

    // Candidate instances: union of the container's index list and any
    // instance whose containerIdx matches (deduped).
    std::vector<uint32_t> candidates;
    candidates.reserve(c.instanceIndices.size() + 8);
    std::vector<char> seen(instances_.size(), 0);
    for (uint32_t gi : c.instanceIndices) {
        if (gi < instances_.size() && !seen[gi]) {
            seen[gi] = 1;
            candidates.push_back(gi);
        }
    }
    for (uint32_t gi = 0; gi < static_cast<uint32_t>(instances_.size()); ++gi) {
        if (instances_[gi].containerIdx == containerIdx && !seen[gi]) {
            seen[gi] = 1;
            candidates.push_back(gi);
        }
    }

    std::vector<std::vector<uint32_t>> tmp(cellCount);
    for (uint32_t gi : candidates) {
        const Instance& inst = instances_[gi];
        const glm::vec4 bmin4 = computeInstanceBoundsMin(inst);
        const glm::vec4 bmax4 = computeInstanceBoundsMax(inst);
        const glm::vec3 bmin(bmin4.x, bmin4.y, bmin4.z);
        const glm::vec3 bmax(bmax4.x, bmax4.y, bmax4.z);
        // Skip instances fully outside the container.
        if (bmax.x < c.minp.x || bmax.y < c.minp.y || bmax.z < c.minp.z) continue;
        if (bmin.x > c.maxp.x || bmin.y > c.maxp.y || bmin.z > c.maxp.z) continue;
        glm::vec3 cbmin(std::max(bmin.x, c.minp.x), std::max(bmin.y, c.minp.y), std::max(bmin.z, c.minp.z));
        glm::vec3 cbmax(std::min(bmax.x, c.maxp.x), std::min(bmax.y, c.maxp.y), std::min(bmax.z, c.maxp.z));
        auto axisRange = [&](float lo, float hi, float origin, float cs, uint32_t n) {
            int i0 = 0, i1 = 0;
            if (cs > 1e-9f) {
                i0 = static_cast<int>(std::floor((lo - origin) / cs));
                i1 = static_cast<int>(std::floor((hi - origin) / cs));
            }
            i0 = std::clamp(i0, 0, static_cast<int>(n) - 1);
            i1 = std::clamp(i1, 0, static_cast<int>(n) - 1);
            if (i1 < i0) std::swap(i0, i1);
            return std::pair<int, int>(i0, i1);
        };
        const auto rx = axisRange(cbmin.x, cbmax.x, c.minp.x, cellSize.x, nx);
        const auto ry = axisRange(cbmin.y, cbmax.y, c.minp.y, cellSize.y, ny);
        const auto rz = axisRange(cbmin.z, cbmax.z, c.minp.z, cellSize.z, nz);
        for (int z = rz.first; z <= rz.second; ++z) {
            for (int y = ry.first; y <= ry.second; ++y) {
                for (int x = rx.first; x <= rx.second; ++x) {
                    const size_t ci =
                        static_cast<size_t>(x) + static_cast<size_t>(nx) * (static_cast<size_t>(y) + static_cast<size_t>(ny) * static_cast<size_t>(z));
                    tmp[ci].push_back(gi);
                }
            }
        }
    }

    uint32_t offset = 0;
    out.indices.reserve(candidates.size() > 0 ? candidates.size() * 2 : 0);
    for (size_t i = 0; i < cellCount; ++i) {
        out.cells[i].offset = offset;
        out.cells[i].count = static_cast<uint32_t>(tmp[i].size());
        out.cells[i]._pad0 = 0;
        out.cells[i]._pad1 = 0;
        for (uint32_t gi : tmp[i]) out.indices.push_back(gi);
        offset += static_cast<uint32_t>(tmp[i].size());
    }
    return out;
}

void SdfScene::flattenDefinitions(std::vector<SdfDefinitionGPU>& out) const {
    out.resize(definitions_.size());
    for (size_t i = 0; i < definitions_.size(); ++i) {
        const Definition& d = definitions_[i];
        out[i].params0 = d.params0;
        out[i].params1 = d.params1;
        out[i].meta = glm::uvec4(static_cast<uint32_t>(d.prim), static_cast<uint32_t>(d.op), d.deformFlags,
                                 std::bit_cast<uint32_t>(d.smoothK));
    }
}

void SdfScene::flattenMaterials(std::vector<SdfMaterialGPU>& out) const {
    out.resize(materials_.size());
    for (size_t i = 0; i < materials_.size(); ++i) {
        const Material& m = materials_[i];
        out[i].baseColor = m.baseColor;
        out[i].surfaceParams = glm::vec4(m.roughness, m.metallic, m.opacity, static_cast<float>(static_cast<uint32_t>(m.mode)));
        out[i].emission = m.emission;
        out[i].volumeParams = glm::vec4(m.density, m.absorption, m.scattering, m.tempScale);
        // No per-material smoothK field exists on the CPU Material; store the
        // framework default (matches Definition::smoothK default).
        out[i].extra = glm::vec4(0.5f, m.noiseScale, m.turbulence, m.riseSpeed);
    }
}

void SdfScene::flattenInstances(std::vector<SdfInstanceGPU>& out) const {
    out.resize(instances_.size());
    for (size_t i = 0; i < instances_.size(); ++i) {
        const Instance& in = instances_[i];
        out[i].posScale = glm::vec4(in.pos, in.scale);
        out[i].rotSeed = glm::vec4(in.euler, in.seed);
        out[i].sizeParams = glm::vec4(in.heightScale, in.radiusScale, in.intensity, 0.0f);
        out[i].indices = glm::uvec4(in.defIdx, in.matIdx, in.containerIdx, 0u);
        out[i].boundsMin = computeInstanceBoundsMin(in);
        out[i].boundsMax = computeInstanceBoundsMax(in);
    }
}

void SdfScene::flattenContainers(std::vector<SdfContainerGPU>& out) const {
    out.resize(containers_.size());
    for (size_t i = 0; i < containers_.size(); ++i) {
        const Container& c = containers_[i];
        // Count instances belonging to this container (union of the stored
        // list and matching containerIdx, as in buildContainerGrid).
        uint32_t count = 0;
        {
            std::vector<char> seen(instances_.size(), 0);
            for (uint32_t gi : c.instanceIndices) {
                if (gi < instances_.size() && !seen[gi]) {
                    seen[gi] = 1;
                    ++count;
                }
            }
            for (uint32_t gi = 0; gi < static_cast<uint32_t>(instances_.size()); ++gi) {
                if (instances_[gi].containerIdx == static_cast<uint32_t>(i) && !seen[gi]) {
                    seen[gi] = 1;
                    ++count;
                }
            }
        }
        out[i].boundsMin = glm::vec4(c.minp, 0.0f);
        out[i].boundsMax = glm::vec4(c.maxp, 0.0f);
        out[i].gridInfo = glm::uvec4(c.resolution.x, c.resolution.y, c.resolution.z, 0u);
        out[i].gridOffset = glm::uvec4(0u, 0u, count, 0u);
    }
}

void SdfScene::flattenGrid(std::vector<SdfGridCellGPU>& outCells, std::vector<uint32_t>& outIndices) const {
    outCells.clear();
    outIndices.clear();
    for (uint32_t ci = 0; ci < static_cast<uint32_t>(containers_.size()); ++ci) {
        BuiltGrid g = buildContainerGrid(ci);
        const uint32_t cellStart = static_cast<uint32_t>(outCells.size());
        const uint32_t indexStart = static_cast<uint32_t>(outIndices.size());
        (void)cellStart;
        for (auto& cell : g.cells) cell.offset += indexStart;
        outCells.insert(outCells.end(), g.cells.begin(), g.cells.end());
        outIndices.insert(outIndices.end(), g.indices.begin(), g.indices.end());
    }
}

SdfScene::FlattenedScene SdfScene::flatten() const {
    FlattenedScene f;
    flattenDefinitions(f.definitions);
    flattenMaterials(f.materials);
    flattenInstances(f.instances);
    f.containers.resize(containers_.size());
    f.cells.clear();
    f.indices.clear();
    for (uint32_t ci = 0; ci < static_cast<uint32_t>(containers_.size()); ++ci) {
        const Container& c = containers_[ci];
        BuiltGrid g = buildContainerGrid(ci);
        const uint32_t cellStart = static_cast<uint32_t>(f.cells.size());
        const uint32_t indexStart = static_cast<uint32_t>(f.indices.size());
        // Count members for gridOffset.z (same union rule as the grid).
        uint32_t count = 0;
        {
            std::vector<char> seen(instances_.size(), 0);
            for (uint32_t gi : c.instanceIndices) {
                if (gi < instances_.size() && !seen[gi]) {
                    seen[gi] = 1;
                    ++count;
                }
            }
            for (uint32_t gi = 0; gi < static_cast<uint32_t>(instances_.size()); ++gi) {
                if (instances_[gi].containerIdx == ci && !seen[gi]) {
                    seen[gi] = 1;
                    ++count;
                }
            }
        }
        f.containers[ci].boundsMin = glm::vec4(c.minp, 0.0f);
        f.containers[ci].boundsMax = glm::vec4(c.maxp, 0.0f);
        f.containers[ci].gridInfo = glm::uvec4(c.resolution.x, c.resolution.y, c.resolution.z, indexStart);
        f.containers[ci].gridOffset = glm::uvec4(cellStart, indexStart, count, 0u);
        for (auto& cell : g.cells) cell.offset += indexStart;
        f.cells.insert(f.cells.end(), g.cells.begin(), g.cells.end());
        f.indices.insert(f.indices.end(), g.indices.begin(), g.indices.end());
    }
    return f;
}

SdfScene SdfScene::createFireDemo(uint32_t flameCount, glm::vec3 center, float areaSize) {    SdfScene scene;
    // 1 capsule-based flame definition with noise deformation.
    Definition flame;
    flame.prim = SdfPrimitiveType::Capsule;
    flame.op = SdfOpType::Union;
    flame.params0 = glm::vec4(0.5f, 1.0f, 0.0f, 0.0f); // radius, half height
    flame.params1 = glm::vec4(0.0f);
    flame.deformFlags = kDeformNoise;
    flame.smoothK = 0.5f;
    scene.addDefinition(flame);

    // 1 volumetric fire material (defaults are already fire-like).
    Material fire;
    fire.mode = SdfMaterialType::Volume;
    fire.baseColor = glm::vec4(1.0f, 0.5f, 0.1f, 1.0f);
    fire.roughness = 0.5f;
    fire.metallic = 0.0f;
    fire.opacity = 1.0f;
    fire.emission = glm::vec4(1.0f, 0.92f, 0.78f, 2.5f);
    fire.density = 1.0f;
    fire.absorption = 0.5f;
    fire.scattering = 0.5f;
    fire.tempScale = 1.0f;
    fire.noiseScale = 2.5f;
    fire.turbulence = 0.6f;
    fire.riseSpeed = 1.5f;
    scene.addMaterial(fire);

    // 1 container AABB around the fire area.
    const glm::vec3 minp = center + glm::vec3(-areaSize * 0.5f, 0.0f, -areaSize * 0.5f);
    const glm::vec3 maxp = center + glm::vec3(areaSize * 0.5f, 8.0f, areaSize * 0.5f);
    scene.addContainer(minp, maxp, glm::uvec3(8u, 8u, 8u));

    // N instances with random pos/scale/seed.
    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> dxzX(minp.x, maxp.x);
    std::uniform_real_distribution<float> dxzZ(minp.z, maxp.z);
    std::uniform_real_distribution<float> dy(minp.y, minp.y + 2.0f);
    std::uniform_real_distribution<float> dscale(0.5f, 1.5f);
    std::uniform_real_distribution<float> dseed(0.0f, 100.0f);
    for (uint32_t i = 0; i < flameCount; ++i) {
        Instance in;
        in.defIdx = 0;
        in.matIdx = 0;
        in.pos = glm::vec3(dxzX(rng), dy(rng), dxzZ(rng));
        in.euler = glm::vec3(0.0f);
        in.scale = dscale(rng);
        in.heightScale = 1.0f;
        in.radiusScale = 1.0f;
        in.intensity = 1.0f;
        in.seed = dseed(rng);
        in.containerIdx = 0;
        scene.addInstance(in);
    }
    return scene;
}

namespace {
// Shared flame definition + fire material for all fire-scene builders.
// Keeps createFireDemo and createFireFromAnchors on the same look.
void addFlameDefinitionAndMaterial(SdfScene& scene,
                                   const SdfScene::FlameShape& shape) {
    SdfScene::Definition flame;
    flame.prim = SdfPrimitiveType::Flame;
    flame.op = SdfOpType::Union;
    flame.params0 = glm::vec4(shape.baseRadius, shape.height, 0.0f, 0.0f);
    flame.params1 = glm::vec4(shape.tipRadius, shape.spikiness,
                               shape.spikeFreq, 0.0f);
    flame.deformFlags = kDeformNoise; // waviness (rise scroll + sway)
    flame.smoothK = 0.5f;
    scene.addDefinition(flame);

    SdfScene::Material fire;
    fire.mode = SdfMaterialType::Volume;
    fire.baseColor = glm::vec4(1.0f, 0.5f, 0.1f, 1.0f);
    fire.roughness = 0.5f;
    fire.metallic = 0.0f;
    fire.opacity = 1.0f;
    // Near-white tint: the temperature gradient (white-hot base through
    // yellow/orange to dark-red tip) carries the hue. An orange tint here
    // would multiply the whites/yellows down and paint every flame flat.
    fire.emission = glm::vec4(1.0f, 0.92f, 0.78f, 2.5f);
    fire.density = shape.density;
    fire.absorption = 0.5f;
    fire.scattering = 0.5f;
    fire.tempScale = 1.0f;
    fire.noiseScale = 2.5f;
    fire.turbulence = 0.6f;
    fire.riseSpeed = 1.5f;
    scene.addMaterial(fire);
}
} // namespace

SdfScene SdfScene::createFireFromAnchors(const std::vector<FlameAnchor>& anchors,
                                          const FlameShape& shape) {
    SdfScene scene;
    addFlameDefinitionAndMaterial(scene, shape);
    if (anchors.empty()) return scene; // def+mat, no containers -> renders nothing

    glm::vec3 mn(anchors[0].pos), mx(anchors[0].pos);
    float maxScale = 1.0f;
    for (const auto& a : anchors) {
        mn = glm::min(mn, a.pos);
        mx = glm::max(mx, a.pos);
        maxScale = std::max(maxScale, a.scale);
    }
    // Pad for flame extent: flames rise ~(hh+radius)*scale above the anchor
    // and sway sideways; the per-instance bounds already cover deformation,
    // this only sizes the container + grid so big flames are never cut at
    // the container walls (the march interval is container-clipped).
    const float padXZ = std::max(3.0f, maxScale * 2.5f);
    mn -= glm::vec3(padXZ, std::max(1.0f, maxScale), padXZ);
    mx += glm::vec3(padXZ, std::max(6.0f, maxScale * 3.0f), padXZ);
    const glm::vec3 extent = mx - mn;
    // Adaptive grid targeting ~6 m cells (clamped 1..24 per axis).
    auto axisRes = [](float e) {
        return std::clamp(static_cast<uint32_t>(std::ceil(e / 6.0f)), 1u, 24u);
    };
    scene.addContainer(mn, mx,
        glm::uvec3(axisRes(extent.x), axisRes(extent.y), axisRes(extent.z)));

    for (const auto& a : anchors) {
        Instance in;
        in.defIdx = 0;
        in.matIdx = 0;
        in.pos = a.pos;
        in.euler = a.euler;
        in.scale = std::max(a.scale, 0.05f);
        in.heightScale = 1.0f;
        in.radiusScale = 1.0f;
        in.intensity = a.intensity;
        in.seed = a.seed;
        in.containerIdx = 0;
        scene.addInstance(in);
    }
    return scene;
}

} // namespace sdf_gpu

glm::vec3 sdf_gpu::SdfScene::eulerAlignYToNormal(const glm::vec3& n) {
    const float l2 = glm::dot(n, n);
    if (l2 < 1e-12f) return glm::vec3(0.0f); // degenerate: keep Y-up
    const glm::vec3 nn = n * (1.0f / std::sqrt(l2));
    const glm::vec3 up(0.0f, 1.0f, 0.0f);
    const float d = std::clamp(glm::dot(up, nn), -1.0f, 1.0f);
    glm::quat q;
    if (d > 1.0f - 1e-6f) {
        q = glm::quat(1.0f, 0.0f, 0.0f, 0.0f); // already Y-up
    } else if (d < -1.0f + 1e-6f) {
        q = glm::angleAxis(3.14159265358979f, glm::vec3(1.0f, 0.0f, 0.0f));
    } else {
        q = glm::angleAxis(std::acos(d), glm::normalize(glm::cross(up, nn)));
    }
    // Extract XYZ euler for R = Rx(x) * Ry(y) * Rz(z) (sdfEulerMat order).
    // With row notation: R02 = sin(y); R12 = -sin(x)cos(y); R22 = cos(x)cos(y);
    // R00 = cos(y)cos(z); R01 = -cos(y)sin(z).
    const glm::mat3 r = glm::mat3_cast(q);
    const float sy = std::clamp(r[2][0], -1.0f, 1.0f);
    glm::vec3 e(0.0f);
    if (std::abs(sy) < 1.0f - 1e-6f) {
        e.y = std::asin(sy);
        e.x = std::atan2(-r[2][1], r[2][2]);
        e.z = std::atan2(-r[1][0], r[0][0]);
    } else {
        // Gimbal lock (normal straight up/down): keep z = 0.
        e.y = (sy > 0.0f) ? 1.57079632679490f : -1.57079632679490f;
        e.x = std::atan2(sy * r[0][1], r[1][1]);
        e.z = 0.0f;
    }
    return e;
}
