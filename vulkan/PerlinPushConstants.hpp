#pragma once
#include <cstdint>

// Push constants for compute shader
// Must match shader layout in perlin_noise.comp
//
// Perf report 23 C2: the layer indices (primary/secondary/target) were removed
// with the array-view bindings; the descriptor set now carries the layer
// selection as single-layer views. debugOutput is unused padding kept so the
// 40-byte layout matches perlin_noise.comp.
struct PerlinPushConstants {
    float scale;              // offset 0
    float octaves;            // offset 4
    float persistence;        // offset 8
    float lacunarity;         // offset 12
    uint32_t seed;            // offset 16
    float brightness;         // offset 20
    float contrast;           // offset 24
    uint32_t textureSize;     // offset 28
    float time;               // offset 32
    uint32_t debugOutput;     // offset 36
    // Total: 40 bytes (padded to 4-byte boundary)
};
