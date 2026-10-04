#pragma once

// Tapered base-anchored flame definition parameters.
namespace sdf_gpu {

struct FlameShape {
    float baseRadius = 1.0f;  // local units, scaled by instance scale
    float height = 3.2f;     // base disk (y=0) to tip, local units
    float tipRadius = 0.25f; // 0 = sharp cone tip, = base = capsule
    float spikiness = 0.35f; // spike amplitude, 0 = smooth rounded capsule
    float spikeFreq = 2.0f;  // tongue count around the axis
    float density = 0.35f;   // volumetric density multiplier (lower = glassier)
};

} // namespace sdf_gpu
