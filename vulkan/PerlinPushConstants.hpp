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
    uint32_t primaryLayer;
    uint32_t secondaryLayer;
    uint32_t targetLayer;
    float padding[1];  // Padding for alignment
};
