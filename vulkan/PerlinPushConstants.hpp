#pragma once
#include <cstdint>

// Push constants for compute shader
// Must match shader layout in perlin_noise.comp
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
    uint32_t primaryLayer;    // offset 36
    uint32_t secondaryLayer;  // offset 40
    uint32_t targetLayer;     // offset 44
    // Total: 48 bytes
};
