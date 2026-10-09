#pragma once

// Grass-clump tuning (SDF grass derived from the existing vegetation
// instances). CPU-only (no direct GPU twin): caps/region fields shape the
// collected anchor set, the shape fields pack into the Grass SdfDefinitions
// and SdfMaterials at scene rebuild.
//
// Anchors are streamed 1:1 with the vegetation chunks by VegetationRenderer
// (position + vegetation type/biome + smooth normal); each retained anchor
// becomes ONE Grass SDF instance whose shader evaluates a small procedural
// group of blades. The density controls here only cap the SDF-side set so
// the SSBO/grid stay bounded — they never re-place grass.
#include <cstdint>

#include <glm/glm.hpp>

struct SdfGrassConfig {
    bool enabled = true;
    // Caps (mirror the lava/rock collectors): per-chunk retained clumps and
    // the global retained set. The vegetation density is already applied by
    // the vegetation generator; these only bound the SDF scene.
    int maxPerChunk = 256;       // clumps retained per vegetation chunk
    uint32_t maxAnchors = 32768; // global clump cap
    // Region partition: anchors are grouped into regionSize x regionSize m
    // containers (the generic SDF container grid is the coarse cull level);
    // each region targets cellSize m grid cells so a cell holds few clumps.
    float regionSize = 512.0f;
    float cellSize = 24.0f;
    // Shape (local units; one unit = the per-instance world scale).
    float clumpRadius = 0.5f;
    float bladeHeight = 1.0f;
    float bladeWidth = 0.05f;
    int bladeCount = 24;
    float curvature = 0.35f;
    float maxLean = 0.35f;   // radians
    float windGain = 0.12f;  // lean radians per m/s
    float tipWidth = 0.35f;
    float roughness = 0.85f;
    glm::vec3 tint = glm::vec3(1.0f);
    // Shadow-caster LOD: the EVSM grass march uses this fixed camera-scale
    // value instead of the real camera distance, so shadow texels evaluate
    // the reduced blade set rather than the full clump. Shadows are blurred
    // and cover far more area per texel than the main view, so a coarse LOD
    // is both sufficient and required for performance. At >= impostorFull it
    // evaluates the impostor only (zero blades).
    float shadowLodScale = 32.0f;
    // SDF impostor fade band (camera distance in clump scales): below
    // impostorStart the blade LOD only, impostorStart..impostorFull the
    // blades unioned with the fading impostor, at/above impostorFull the
    // impostor only (cheapest). Editable in the Impostors widget; setting
    // both to 0 forces the impostor everywhere ("impostors only" preview).
    float impostorStart = 48.0f;
    float impostorFull = 80.0f;
};
