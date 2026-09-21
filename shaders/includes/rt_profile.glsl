// Hybrid RT per-operation GPU profiling (opt-in, RT_PROFILE shader variants).
//
// The inline RT paths are embedded in raster shaders (solid fragment mirror
// rays, water reflection/refraction/depth, bounce chains, contact shadows), so
// per-op GPU time cannot be bracketed with timestamp queries. This include
// instruments every ray-query site with:
//   * a ray counter (query initializations),
//   * a hit counter (committed intersections),
//   * accumulated device-clock nanoseconds (VK_KHR_shader_clock /
//     GL_EXT_shader_realtime_clock), summed over invocations. The sum is
//     "thread-time": it attributes total shader execution cost across ops
//     (ops run concurrently, so the sum exceeds wall time — compare shares,
//     not absolute frame time).
//
// A 1/RT_PROFILE_SAMPLE_STRIDE checkerboard sample keeps atomic traffic
// negligible; the CPU scales counts/time by the stride for display.
//
// Op ids mirror RTProfileOp in vulkan/renderer/RayTracingResources.hpp; the
// counter layout mirrors RTProfileCounters there (std430, 4-byte uints).
// Profile variants are only created when the device supports shaderDeviceClock
// (feature-detected in VulkanApp); production variants never include this
// file's RT_PROFILE section, so they carry no clock capability.

#ifndef RT_PROFILE_GLSL
#define RT_PROFILE_GLSL

// Op ids are ALWAYS defined: call sites use the macros below, which compile to
// no-ops without RT_PROFILE, so the production and profile sources stay
// textually identical at the instrumented sites.
const uint RT_PROFILE_OP_SOLID_REFLECTION = 0u;
const uint RT_PROFILE_OP_WATER_REFLECTION = 1u;
const uint RT_PROFILE_OP_WATER_REFRACTION = 2u;
const uint RT_PROFILE_OP_WATER_DEPTH = 3u;
const uint RT_PROFILE_OP_BOUNCE = 4u;
const uint RT_PROFILE_OP_CONTACT_SHADOW = 5u;
const uint RT_PROFILE_OP_WATER_HIT_REFRACT = 6u;
const uint RT_PROFILE_OP_COUNT = 7u;

#ifdef RT_PROFILE
// GL_EXT_shader_realtime_clock is enabled at the TOP of the including shader
// (main.frag / main.tese): a mid-file #extension is illegal in GLSL.

layout(set = 0, binding = 26, std430) buffer RTProfileBlock {
    uint rtProfRays[RT_PROFILE_OP_COUNT];
    uint rtProfHits[RT_PROFILE_OP_COUNT];
    uint rtProfTime[RT_PROFILE_OP_COUNT]; // device-clock ns >> 6 (64 ns units)
} rtProfile;

const int RT_PROFILE_SAMPLE_STRIDE = 4;

// Fragment stages sample a checkerboard; the TES (water-region depth) samples
// by input primitive (gl_FragCoord is unavailable there).
bool rtProfSample() {
#ifdef RT_PROFILE_TES
    return (gl_PrimitiveID & (RT_PROFILE_SAMPLE_STRIDE - 1)) == 0;
#else
    return ((int(gl_FragCoord.x) + int(gl_FragCoord.y))
        & (RT_PROFILE_SAMPLE_STRIDE - 1)) == 0;
#endif
}

struct RTProfileScope {
    uint op;
    uint t0;
    bool sampled;
};

// Device-scope realtime clock in nanoseconds; only the low 32 bits are kept
// (a single ray operation is far below the 4.29 s 32-bit wrap). The clock is
// read only for sampled invocations.
RTProfileScope rtProfBegin(uint op) {
    RTProfileScope s;
    s.op = op;
    s.sampled = rtProfSample();
    s.t0 = s.sampled ? clockRealtime2x32EXT().x : 0u;
    if (s.sampled) atomicAdd(rtProfile.rtProfRays[op], 1u);
    return s;
}

void rtProfEnd(RTProfileScope s) {
    if (!s.sampled) return;
    // 64 ns units: a u32 holds ~275 s of summed thread-time (no overflow).
    atomicAdd(rtProfile.rtProfTime[s.op],
        (clockRealtime2x32EXT().x - s.t0) >> 6);
}

void rtProfHit(uint op) {
    if (rtProfSample()) atomicAdd(rtProfile.rtProfHits[op], 1u);
}

#define RT_PROF_BEGIN(name, op) RTProfileScope name = rtProfBegin(op)
#define RT_PROF_END(name) rtProfEnd(name)
#define RT_PROF_HIT(op) rtProfHit(op)
#else
// Empty compound statements: valid GLSL, and call sites keep their trailing
// semicolon (`RT_PROF_BEGIN(x, op);` -> `{};`).
#define RT_PROF_BEGIN(name, op) {}
#define RT_PROF_END(name) {}
#define RT_PROF_HIT(op) {}
#endif // RT_PROFILE

#endif // RT_PROFILE_GLSL
