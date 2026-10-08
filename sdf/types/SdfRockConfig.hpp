#pragma once

// Rock-boulder tuning (SDF boulder instances collected from brush-7
// terrain chunks). CPU-only (no direct GPU twin): density/scale shape the
// ingested candidates, the shape fields pack into the rock Definition and
// the rock Material (texture layer / tiling) at scene rebuild.
//
// Density works in two stages so the widget stays real-time: chunks are
// ingested once at `minSpacing` (the densest candidate set retained), and
// every rebuild deterministically decimates that set down to `spacing`
// (1 rock per spacing x spacing metres). Moving the spacing slider never
// needs the chunk geometry again.
#include <cstdint>

#include <glm/glm.hpp>

struct SdfRockConfig {
    bool enabled = true;
    // Placement: one rock per `spacing` x `spacing` m of rock surface.
    float spacing = 512.0f;    // m between rocks (default 1 per 512 x 512 m)
    float minSpacing = 64.0f;  // densest candidate set retained at ingest
    int maxPerChunk = 64;      // per-chunk candidate cap
    uint32_t maxAnchors = 4096;// global candidate cap
    // Size: base radius (m) with a symmetric random variation.
    float scale = 64.0f;
    float scaleVariation = 0.4f; // +/- fraction of scale
    float embed = 0.35f;         // fraction of the radius sunk below the surface
    // Shape: static Perlin displacement of the sphere (frequency is per
    // local unit, so features scale with the boulder; ~2.5 puts a handful
    // of lumps across a unit sphere).
    float noiseScale = 2.5f;      // noise frequency per local unit
    float noiseAmplitude = 0.35f; // displacement as a fraction of the radius
    // Surface: scene texture-array layer (7 = Rough_rock_021) and world
    // metres per texture repeat.
    float textureLayer = 7.0f;
    float textureTiling = 64.0f;
    float roughness = 0.85f;
    float metallic = 0.0f;
    glm::vec3 tint = glm::vec3(1.0f); // multiplies the sampled albedo
};
