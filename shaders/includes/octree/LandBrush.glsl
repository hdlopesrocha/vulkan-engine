#ifndef LAND_BRUSH_GLSL
#define LAND_BRUSH_GLSL

// GPU port of utils/LandBrush.cpp: classify a terrain surface point into a
// material index from its world height, slope and — in the flat grass band — a
// Perlin biome field. Runs per vertex in OctreeTriangles.comp (where the
// surface position and normal are already computed) so the octree build never
// has to call SDF::getPosition for terrain.
//
// Layer layout: the indices below mirror utils/SceneMaterialTable.cpp. Source
// triples occupy [0, LAND_SOURCE_COUNT); generated mixer layers follow in
// sceneMixerDescs() order. Keep both files in sync when adding textures or
// mixers.
#include "../noise/Perlin2D.glsl"

const int LAND_BRUSH_INDEX = 255;
// NormalBrush sentinel (math/BrushMode.hpp NORMAL_BRUSH_INDEX). Resolved per
// node: keep only nodes facing +Y, the rest are discarded.
const int NORMAL_BRUSH = 254;

const int LAND_SOURCE_COUNT   = 24;
const int LAND_MIXER_FIRST    = LAND_SOURCE_COUNT;

const int LAND_UNDERGROUND    = -1;
const int LAND_ROCK           = 7;   // textures/Rough_rock_021
const int LAND_SAND           = 8;   // textures/Sand_007
const int LAND_SNOW           = 9;   // textures/Snow_001
const int LAND_SOFT_SAND      = 10;  // textures/Sand_002

// Generated land-transition layers (sceneMixerDescs order).
const int LAND_GRASS_MIX_SAND = LAND_MIXER_FIRST + 0;
const int LAND_GRASS_MIX_SNOW = LAND_MIXER_FIRST + 1;
const int LAND_ROCK_MIX_GRASS = LAND_MIXER_FIRST + 2;
const int LAND_ROCK_MIX_SNOW  = LAND_MIXER_FIRST + 3;
const int LAND_ROCK_MIX_SAND  = LAND_MIXER_FIRST + 4;

// ── Grass-band biomes ───────────────────────────────────────────────────────
// The flat band that used to be a single LAND_GRASS material is a linear
// Perlin palette. Mixer LAND_BIOME_MIX_FIRST + i blends palette entry i into
// entry i+1, so every band boundary uses a pre-generated TextureMixer layer.
// Palette order (lush -> dry): Grass_001, Ground_Grass_001, Dead_leaves_001,
// Ground_Dirt_007, Ground_Dirt_008.
const int LAND_BIOME_GRASS        = 3;   // textures/Grass_001
const int LAND_BIOME_GROUND_GRASS = 22;  // textures/Ground_Grass_001
const int LAND_BIOME_LEAVES       = 2;   // textures/Dead_leaves_001
const int LAND_BIOME_DIRT         = 1;   // textures/Ground_Dirt_007
const int LAND_BIOME_BARE_DIRT    = 21;  // textures/Ground_Dirt_008
const int LAND_BIOME_MIX_FIRST    = LAND_MIXER_FIRST + 5;

int landBiomeBase(int band) {
    if (band <= 0) return LAND_BIOME_GRASS;
    if (band == 1) return LAND_BIOME_GROUND_GRASS;
    if (band == 2) return LAND_BIOME_LEAVES;
    if (band == 3) return LAND_BIOME_DIRT;
    return LAND_BIOME_BARE_DIRT;
}

// Smooth [0,1] biome field. Two Perlin octaves give the broad patches and a
// higher-frequency third octave warps the band edges so the seams are organic
// instead of clean contour lines around the world origin. The tanh transfer
// flattens Perlin's bell-shaped value distribution so every palette band
// covers a similar share of the map instead of only the middle bands
// appearing.
float landBiomeField(vec2 xz) {
    vec2 p = xz * 0.01;
    float base = perlin2(p) + 0.5 * perlin2(p * 2.07 + vec2(19.3, -7.1));
    float warp = perlin2(xz * 0.075 + vec2(4.7, 23.9));
    float n = clamp((base * 0.6667 + warp * 0.12) * 1.9, -1.0, 1.0);
    return 0.5 + 0.5 * tanh(2.5 * n);
}

int landBiomeMaterial(vec2 xz) {
    const int kBiomeCount = 5;
    const float kBlend = 0.15; // fraction of a band that uses the mix layer

    float t = landBiomeField(xz) * float(kBiomeCount);
    int band = clamp(int(floor(t)), 0, kBiomeCount - 1);
    float f = t - float(band);

    if (band + 1 < kBiomeCount && f > 1.0 - kBlend) return LAND_BIOME_MIX_FIRST + band;
    if (band > 0 && f < kBlend) return LAND_BIOME_MIX_FIRST + (band - 1);
    return landBiomeBase(band);
}

int landMaterialIndex(in vec3 pos, in vec3 normal) {
    vec3 n = normalize(normal);
    float steepness = dot(vec3(0.0, 1.0, 0.0), n);
    const float grassLevel = 256.0;
    const float sandLevel = 16.0;
    const float softSandLevel = 2.0;
    int brushIndex;

    if (steepness <= 0.01) {
        brushIndex = LAND_UNDERGROUND;
    } else if (steepness < 0.980) {
        brushIndex = LAND_ROCK;
    } else if (steepness < 0.985) {
        if (pos.y < softSandLevel) {
            brushIndex = LAND_ROCK;
        } else if (pos.y < sandLevel) {
            brushIndex = LAND_ROCK_MIX_SAND;
        } else if (pos.y < grassLevel) {
            brushIndex = LAND_ROCK_MIX_GRASS;
        } else {
            brushIndex = LAND_ROCK_MIX_SNOW;
        }
    } else if (pos.y < softSandLevel) {
        brushIndex = LAND_SOFT_SAND;
    } else if (pos.y < sandLevel) {
        brushIndex = LAND_SAND;
    } else if (pos.y < sandLevel + 4.0) {
        brushIndex = LAND_GRASS_MIX_SAND;
    } else if (pos.y < grassLevel) {
        // Same height/slope range the single grass material covered; the
        // Perlin field picks the biome (and its transition mixers).
        brushIndex = landBiomeMaterial(pos.xz);
    } else if (pos.y < grassLevel + 32.0) {
        brushIndex = LAND_GRASS_MIX_SNOW;
    } else {
        brushIndex = LAND_SNOW;
    }
    return brushIndex;
}

// Replace the LandBrush sentinel with the classified terrain material.
int resolveLandBrush(in int brush, in vec3 pos, in vec3 normal) {
    return (brush == LAND_BRUSH_INDEX) ? landMaterialIndex(pos, normal) : brush;
}

#endif // LAND_BRUSH_GLSL
