#pragma once
#include <cstdint>

// Push constants for compute shader
struct PerlinPushConstants {
    float scale;
    float octaves;
    float persistence;
    float lacunarity;
    uint32_t seed;
    float brightness;
    float contrast;
    uint32_t textureSize;
    float time;  // Time parameter for animated noise
    float padding[3];  // Padding for alignment
};
