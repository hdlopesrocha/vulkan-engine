// CPU-side SDF scene builder. Stores the canonical GPU-layout structs
// directly (sdf/types/*GPU.hpp) so the renderer uploads them verbatim.
// Keeps no dependency on space/Octree, on sdf/*DistanceFunction, or on
// vulkan/; only standard C++23 + GLM + sdf/types + math/BoundingBox.
#include "sdf/types/SdfScene.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>

#include "sdf/types/SdfDeformFlags.hpp"

namespace sdf_gpu {
namespace {

constexpr uint32_t kDeformNoise = static_cast<uint32_t>(SdfDeformFlags::Noise);
constexpr uint32_t kDeformTwist = static_cast<uint32_t>(SdfDeformFlags::Twist);
constexpr uint32_t kDeformBend = static_cast<uint32_t>(SdfDeformFlags::Bend);
constexpr uint32_t kDeformTaper = static_cast<uint32_t>(SdfDeformFlags::Taper);
constexpr uint32_t kDeformRepeat = static_cast<uint32_t>(SdfDeformFlags::Repeat);

glm::mat3 rotationFromEuler(const glm::vec3& e) {
    // MUST match sdfEulerMat() in shaders/includes/sdf/SdfOps.glsl exactly:
    // R = Rx * Ry * Rz (same matrix entries and product order). The GPU
    // inverts this to reach primitive-local space, and the bounds below must
    // enclose the same oriented shape, so any divergence here breaks both
    // culling and the SDF itself once rotation is non-zero.
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
        case SdfPrimitiveType::Sphere:
        case SdfPrimitiveType::Smoke: {
            // Smoke uses the sphere domain at its maximum radius (growth is
            // density-side, so bounds stay static while the look loops).
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

float deformationPadding(const SdfDefinition& def, float turbulence) {
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

uint32_t SdfScene::addDefinition(const SdfDefinition& d) {
    definitions_.push_back(d);
    return static_cast<uint32_t>(definitions_.size() - 1);
}

uint32_t SdfScene::addMaterial(const SdfMaterial& m) {
    materials_.push_back(m);
    return static_cast<uint32_t>(materials_.size() - 1);
}

uint32_t SdfScene::addContainer(const glm::vec3& minp, const glm::vec3& maxp, glm::uvec3 res) {
    SdfContainer c;
    c.boundsMin = minp;
    c.boundsMax = maxp;
    c.resX = std::max(res.x, 1u);
    c.resY = std::max(res.y, 1u);
    c.resZ = std::max(res.z, 1u);
    c.cellStart = 0u;
    containers_.push_back(c);
    return static_cast<uint32_t>(containers_.size() - 1);
}

uint32_t SdfScene::addInstance(const SdfInstance& in) {
    instances_.push_back(in);
    // Fill the world AABB immediately; rebuild() refreshes it after edits.
    const BoundingBox b = computeInstanceBounds(instances_.back());
    instances_.back().boundsMin = b.getMin();
    instances_.back().boundsMax = b.getMax();
    return static_cast<uint32_t>(instances_.size() - 1);
}

void SdfScene::clear() {
    definitions_.clear();
    materials_.clear();
    instances_.clear();
    containers_.clear();
    cells_.clear();
    indices_.clear();
}

BoundingBox SdfScene::computeInstanceBounds(const SdfInstance& in) const {
    glm::vec3 half(0.5f);
    float pad = 0.5f;
    if (in.defIdx < definitions_.size()) {
        const SdfDefinition& def = definitions_[in.defIdx];
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
    const glm::mat3 r = rotationFromEuler(in.rotation);
    const glm::mat3 ar(glm::vec3(std::abs(r[0].x), std::abs(r[0].y), std::abs(r[0].z)),
                       glm::vec3(std::abs(r[1].x), std::abs(r[1].y), std::abs(r[1].z)),
                       glm::vec3(std::abs(r[2].x), std::abs(r[2].y), std::abs(r[2].z)));
    const glm::vec3 worldHalf = ar * half;
    return BoundingBox(in.position - worldHalf, in.position + worldHalf);
}

void SdfScene::rebuild() {
    // 1. Instance world AABBs (rotation/scale edits included).
    for (SdfInstance& in : instances_) {
        const BoundingBox b = computeInstanceBounds(in);
        in.boundsMin = b.getMin();
        in.boundsMax = b.getMax();
    }

    // 2. Uniform grid per container, rebased into the global cell/index
    //    arrays. Membership derives from Instance::containerIdx; an instance
    //    is inserted into every cell its AABB overlaps (clamped to the
    //    container). Cell index = x + resX * (y + resY * z).
    cells_.clear();
    indices_.clear();
    for (uint32_t ci = 0; ci < static_cast<uint32_t>(containers_.size()); ++ci) {
        SdfContainer& c = containers_[ci];
        const uint32_t nx = std::max(c.resX, 1u);
        const uint32_t ny = std::max(c.resY, 1u);
        const uint32_t nz = std::max(c.resZ, 1u);
        const size_t cellCount = static_cast<size_t>(nx) * ny * nz;
        c.cellStart = static_cast<uint32_t>(cells_.size());
        if (cellCount == 0) continue;

        const glm::vec3 length = c.boundsMax - c.boundsMin;
        glm::vec3 cellSize(1.0f);
        if (length.x > 1e-6f) cellSize.x = length.x / static_cast<float>(nx);
        if (length.y > 1e-6f) cellSize.y = length.y / static_cast<float>(ny);
        if (length.z > 1e-6f) cellSize.z = length.z / static_cast<float>(nz);

        std::vector<std::vector<uint32_t>> tmp(cellCount);
        for (uint32_t gi = 0; gi < static_cast<uint32_t>(instances_.size()); ++gi) {
            const SdfInstance& inst = instances_[gi];
            if (inst.containerIdx != ci) continue;
            const glm::vec3 bmin = inst.boundsMin;
            const glm::vec3 bmax = inst.boundsMax;
            // Skip instances fully outside the container.
            if (bmax.x < c.boundsMin.x || bmax.y < c.boundsMin.y || bmax.z < c.boundsMin.z) continue;
            if (bmin.x > c.boundsMax.x || bmin.y > c.boundsMax.y || bmin.z > c.boundsMax.z) continue;
            const glm::vec3 cbmin = glm::max(bmin, c.boundsMin);
            const glm::vec3 cbmax = glm::min(bmax, c.boundsMax);
            auto axisRange = [](float lo, float hi, float origin, float cs, uint32_t n) {
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
            const auto rx = axisRange(cbmin.x, cbmax.x, c.boundsMin.x, cellSize.x, nx);
            const auto ry = axisRange(cbmin.y, cbmax.y, c.boundsMin.y, cellSize.y, ny);
            const auto rz = axisRange(cbmin.z, cbmax.z, c.boundsMin.z, cellSize.z, nz);
            for (int z = rz.first; z <= rz.second; ++z) {
                for (int y = ry.first; y <= ry.second; ++y) {
                    for (int x = rx.first; x <= rx.second; ++x) {
                        const size_t ki = static_cast<size_t>(x)
                            + static_cast<size_t>(nx) * (static_cast<size_t>(y)
                            + static_cast<size_t>(ny) * static_cast<size_t>(z));
                        tmp[ki].push_back(gi);
                    }
                }
            }
        }

        uint32_t offset = static_cast<uint32_t>(indices_.size());
        if (indices_.capacity() < indices_.size() + cellCount * 2) {
            indices_.reserve(indices_.size() + cellCount * 2);
        }
        for (size_t i = 0; i < cellCount; ++i) {
            SdfGridCell cell;
            cell.offset = offset;
            cell.count = static_cast<uint32_t>(tmp[i].size());
            cells_.push_back(cell);
            for (uint32_t gi : tmp[i]) indices_.push_back(gi);
            offset += cell.count;
        }
    }
}

SdfScene SdfScene::merge(const SdfScene& a, const SdfScene& b) {
    SdfScene out;
    out.definitions_ = a.definitions_;
    out.definitions_.insert(out.definitions_.end(), b.definitions_.begin(), b.definitions_.end());
    out.materials_ = a.materials_;
    out.materials_.insert(out.materials_.end(), b.materials_.begin(), b.materials_.end());
    const uint32_t defBase = static_cast<uint32_t>(a.definitions_.size());
    const uint32_t matBase = static_cast<uint32_t>(a.materials_.size());
    const uint32_t contBase = static_cast<uint32_t>(a.containers_.size());
    out.containers_ = a.containers_;
    for (const SdfContainer& c : b.containers_) {
        SdfContainer nc = c;
        nc.cellStart = 0u; // rebuilt by rebuild()
        out.containers_.push_back(nc);
    }
    out.instances_ = a.instances_;
    for (const SdfInstance& in : b.instances_) {
        SdfInstance ni = in;
        ni.defIdx += defBase;
        ni.matIdx += matBase;
        ni.containerIdx += contBase;
        out.instances_.push_back(ni);
    }
    return out;
}

SdfScene SdfScene::createSmokeBomb(const glm::vec3& center, float scale, float seed) {
    SdfScene scene;
    SdfDefinition d;
    d.prim = SdfPrimitiveType::Smoke;
    d.op = SdfOpType::Union;
    d.params0 = glm::vec4(std::max(scale, 1.0f), seed, 0.0f, 0.0f);
    d.params1 = glm::vec4(0.0f);
    d.deformFlags = 0u; // smoke warps its own noise domain; no flame deform
    d.smoothK = 0.5f;
    scene.addDefinition(d);

    SdfMaterial m;
    m.mode = SdfMaterialType::Volume;
    m.baseColor = glm::vec4(0.62f, 0.60f, 0.58f, 1.0f); // neutral gray body
    m.roughness = 1.0f;
    m.metallic = 0.0f;
    m.opacity = 1.0f;
    m.emission = glm::vec3(0.0f); // scattering-shaded, no blackbody emission
    m.emissionIntensity = 0.0f;
    m.density = 0.5f;
    m.absorption = 0.6f;
    m.scattering = 0.7f;
    m.tempScale = 0.0f; // temperature path disabled for smoke
    m.noiseScale = 2.5f;
    m.turbulence = 0.6f;
    m.riseSpeed = 1.5f;
    scene.addMaterial(m);

    const float pad = std::max(scale * 0.25f, 4.0f);
    scene.addContainer(center - glm::vec3(pad + scale),
                       center + glm::vec3(pad + scale),
                       glm::uvec3(4u, 4u, 4u));
    SdfInstance in;
    in.defIdx = 0;
    in.matIdx = 0;
    in.position = center;
    in.rotation = glm::vec3(0.0f);
    in.scale = 1.0f;
    in.heightScale = 1.0f;
    in.radiusScale = 1.0f;
    in.intensity = 1.0f;
    in.seed = seed;
    in.containerIdx = 0;
    scene.addInstance(in);
    return scene;
}

SdfScene SdfScene::createFireDemo(uint32_t flameCount, glm::vec3 center, float areaSize) {
    SdfScene scene;
    // 1 capsule-based flame definition with noise deformation.
    SdfDefinition flame;
    flame.prim = SdfPrimitiveType::Capsule;
    flame.op = SdfOpType::Union;
    flame.params0 = glm::vec4(0.5f, 1.0f, 0.0f, 0.0f); // radius, half height
    flame.params1 = glm::vec4(0.0f);
    flame.deformFlags = kDeformNoise;
    flame.smoothK = 0.5f;
    scene.addDefinition(flame);

    // 1 volumetric fire material (defaults are already fire-like).
    SdfMaterial fire;
    fire.mode = SdfMaterialType::Volume;
    fire.baseColor = glm::vec4(1.0f, 0.5f, 0.1f, 1.0f);
    fire.roughness = 0.5f;
    fire.metallic = 0.0f;
    fire.opacity = 1.0f;
    fire.emission = glm::vec3(1.0f, 0.92f, 0.78f);
    fire.emissionIntensity = 2.5f;
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
        SdfInstance in;
        in.defIdx = 0;
        in.matIdx = 0;
        in.position = glm::vec3(dxzX(rng), dy(rng), dxzZ(rng));
        in.rotation = glm::vec3(0.0f);
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
    SdfDefinition flame;
    flame.prim = SdfPrimitiveType::Flame;
    flame.op = SdfOpType::Union;
    flame.params0 = glm::vec4(shape.baseRadius, shape.height, 0.0f, 0.0f);
    flame.params1 = glm::vec4(shape.tipRadius, shape.spikiness,
                               shape.spikeFreq, 0.0f);
    flame.deformFlags = kDeformNoise; // waviness (rise scroll + sway)
    flame.smoothK = 0.5f;
    scene.addDefinition(flame);

    SdfMaterial fire;
    fire.mode = SdfMaterialType::Volume;
    fire.baseColor = glm::vec4(1.0f, 0.5f, 0.1f, 1.0f);
    fire.roughness = 0.5f;
    fire.metallic = 0.0f;
    fire.opacity = 1.0f;
    // Near-white tint: the temperature gradient (white-hot base through
    // yellow/orange to dark-red tip) carries the hue. An orange tint here
    // would multiply the whites/yellows down and paint every flame flat.
    fire.emission = glm::vec3(1.0f, 0.92f, 0.78f);
    fire.emissionIntensity = 2.5f;
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
        SdfInstance in;
        in.defIdx = 0;
        in.matIdx = 0;
        in.position = a.pos;
        in.rotation = a.euler;
        in.scale = std::max(a.scale, 0.05f);
        in.heightScale = a.heightScale;
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
