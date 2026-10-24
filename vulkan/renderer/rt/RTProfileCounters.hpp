#pragma once

#include <cstdint>

// Per-op RT profiling counters. Layout mirrors the std430 block in
// shaders/includes/rt/RtProfile.glsl; op ids are the RT_PROFILE_OP_* constants
// there. The profile shader variants sample 1/4 of invocations
// (rt_profile.glsl: RT_PROFILE_SAMPLE_STRIDE), so counts and time are scaled
// by 4 on display. `time` is device-clock nanoseconds >> 6 (64 ns units),
// summed over invocations = "thread-time" (ops run concurrently; compare
// shares, not absolute frame time).
struct RTProfileCounters {
    static constexpr uint32_t kOpCount = 7;
    uint32_t rays[kOpCount] = {};  // ray queries issued
    uint32_t hits[kOpCount] = {};  // committed intersections
    uint32_t time[kOpCount] = {};  // 64 ns units, summed thread-time
};
static_assert(sizeof(RTProfileCounters) == RTProfileCounters::kOpCount * 3 * sizeof(uint32_t));
