#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>

#include <glm/glm.hpp>

// ── CPU mirror of the shared wind field's ambient fBm ─────────────────────
// Reproduces shaders/includes/noise/Perlin.glsl (pcgHash / hash33 /
// perlinNoise3D) and shaders/includes/vegetation/WindField.glsl (windFbm3 +
// windAmbient) exactly, so the host can evaluate the SAME wind velocity the
// GPU consumers sample.
//
// Used for the cloud drift: the cloud pattern's bulk offset is a path integral
// of the wind velocity, which the GPU cannot accumulate statelessly, so the
// host integrates this CPU sample once per frame and forwards the displacement
// through SkyUniform::cloudWindDrift.
//
// Keep every constant/offset in sync with WindField.glsl.
namespace WindFieldCpu {

// FBM parameters (the subset of VegetationRenderer::WindSettings the ambient
// field reads; mirrors WindField.ambientA.w / ambientB / ambientC).
struct FbmParams {
    float frequency = 0.002f;   // ambientB.x  first-octave spatial frequency (1/m)
    float lacunarity = 2.0f;    // ambientB.z  frequency gain per octave
    float gain = 0.5f;          // ambientB.w  amplitude gain per octave
    float amplitude = 16.0f;    // ambientC.x  peak wind speed (m/s)
    float vertical = 1.0f;      // ambientC.y  vertical component weight
    float seed = 0.0f;          // ambientC.z  domain offset
    float speed = 0.75f;        // ambientA.w  temporal evolution speed
    int octaves = 3;            // ambientB.y  1..6
};

inline uint32_t pcgHash(uint32_t v) {
    const uint32_t state = v * 747796405u + 2891336453u;
    const uint32_t word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

inline float uintToUnitFloat(uint32_t x) {
    const uint32_t bits = 0x3f800000u | (x >> 9u);
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f - 1.0f;
}

inline uint32_t floatBits(float f) {
    uint32_t u;
    std::memcpy(&u, &f, sizeof(u));
    return u;
}

// GLSL hash33: three decorrelated [-1,1] gradient components per lattice cell.
inline glm::vec3 hash33(const glm::vec3& p) {
    const uint32_t ux = floatBits(p.x);
    const uint32_t uy = floatBits(p.y);
    const uint32_t uz = floatBits(p.z);
    uint32_t s = pcgHash(ux);
    s = pcgHash(s ^ uy);
    s = pcgHash(s ^ uz);
    return glm::vec3(uintToUnitFloat(pcgHash(s + 0x9E3779B9u)),
                     uintToUnitFloat(pcgHash(s + 0x85EBCA6Bu)),
                     uintToUnitFloat(pcgHash(s + 0xC2B2AE35u))) * 2.0f - 1.0f;
}

// GLSL perlinNoise3D: gradient noise, quintic fade, trilinear blend.
inline float perlinNoise3D(const glm::vec3& p) {
    const glm::vec3 i(std::floor(p.x), std::floor(p.y), std::floor(p.z));
    const glm::vec3 f(p.x - i.x, p.y - i.y, p.z - i.z);
    const glm::vec3 u = f * f * f * (f * (f * 6.0f - 15.0f) + 10.0f);

    auto corner = [&](float cx, float cy, float cz) {
        const glm::vec3 c(cx, cy, cz);
        return glm::dot(hash33(i + c), f - c);
    };
    const float n000 = corner(0.0f, 0.0f, 0.0f);
    const float n001 = corner(0.0f, 0.0f, 1.0f);
    const float n010 = corner(0.0f, 1.0f, 0.0f);
    const float n011 = corner(0.0f, 1.0f, 1.0f);
    const float n100 = corner(1.0f, 0.0f, 0.0f);
    const float n101 = corner(1.0f, 0.0f, 1.0f);
    const float n110 = corner(1.0f, 1.0f, 0.0f);
    const float n111 = corner(1.0f, 1.0f, 1.0f);

    const float nx00 = n000 + u.x * (n100 - n000);
    const float nx01 = n001 + u.x * (n101 - n001);
    const float nx10 = n010 + u.x * (n110 - n010);
    const float nx11 = n011 + u.x * (n111 - n011);
    const float nxy0 = nx00 + u.y * (nx10 - nx00);
    const float nxy1 = nx01 + u.y * (nx11 - nx01);
    return nxy0 + u.z * (nxy1 - nxy0);
}

// Vector fBm: each component is its own octave stack at a decorrelated domain
// offset (same offsets as WindField.glsl).
inline glm::vec3 windFbm3(const glm::vec3& p, float time, const FbmParams& q) {
    const glm::vec3 kComponentOffsetY(31.416f, 17.234f, 11.719f);
    const glm::vec3 kComponentOffsetZ(-13.137f, 41.309f, 7.913f);

    const int octaves = static_cast<int>(
        glm::clamp(static_cast<float>(q.octaves), 1.0f, 6.0f) + 0.5f);
    const float frequency = glm::clamp(q.frequency, 1e-6f, 1.0f);
    const float lacunarity = glm::clamp(q.lacunarity, 1.0f, 4.0f);
    const float gain = glm::clamp(q.gain, 0.0f, 0.95f);
    const float seed = glm::clamp(q.seed, -1000.0f, 1000.0f);
    const float speed = std::max(q.speed, 0.0f);

    const glm::vec3 worldOffset =
        glm::vec3(0.7f, 0.3f, 1.0f) * (time * 3.0f * speed) +
        glm::vec3(seed * 137.13f, seed * 73.31f, seed * 39.77f);

    glm::vec3 sum(0.0f);
    float amp = 1.0f;
    float ampSum = 0.0f;
    float f = frequency;
    for (int i = 0; i < 6; ++i) {
        if (i >= octaves) break;
        const glm::vec3 base = p * f + worldOffset * f;
        sum.x += perlinNoise3D(base) * amp;
        sum.y += perlinNoise3D(base + kComponentOffsetY) * amp;
        sum.z += perlinNoise3D(base + kComponentOffsetZ) * amp;
        ampSum += amp;
        f *= lacunarity;
        amp *= gain;
    }
    return sum / std::max(ampSum, 1e-6f);
}

// Ambient wind velocity (m/s): the 3D fBm field scaled by the amplitude, with
// the vertical component weighted. No mean flow (the local vector is the
// direction), mirroring windAmbient().
inline glm::vec3 windAmbient(const glm::vec3& p, float time, const FbmParams& q) {
    const float gustAmplitude = glm::clamp(q.amplitude, 0.0f, 128.0f);
    if (gustAmplitude <= 0.0f) return glm::vec3(0.0f);

    const float t = glm::clamp(time, 0.0f, 86400.0f);
    const glm::vec3 pc(glm::clamp(p.x, -1e5f, 1e5f),
                       glm::clamp(p.y, -1e4f, 1e4f),
                       glm::clamp(p.z, -1e5f, 1e5f));
    glm::vec3 gust = windFbm3(pc, t, q) * gustAmplitude;
    gust.y *= glm::clamp(q.vertical, 0.0f, 1.0f);
    return gust;
}

} // namespace WindFieldCpu
