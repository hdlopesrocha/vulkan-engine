#pragma once

// Grass clump definition/material parameters. CPU-only build input: packed
// into the three per-vegetation-type Grass SdfDefinitions (params0/params1)
// and SdfMaterials (color/roughness) by createGrassFromAnchors. GLSL twin of
// the shape: shaders/includes/sdf/SdfGrass.glsl (SDF_PRIM_GRASS).
//
// The clump stays in local units (one unit = one instance scale, the existing
// vegetation size): clumpRadius/bladeHeight/bladeWidth are fractions of the
// per-instance world scale so the SDF grass matches the physical dimensions
// of the vegetation instances it is derived from.
#include <glm/glm.hpp>

namespace sdf_gpu {

struct GrassShape {
    float clumpRadius = 0.5f;    // local XZ radius of the blade roots
    float bladeHeight = 1.0f;    // local blade height
    float bladeWidth = 0.02f;    // local blade half-width at the base
    int bladeCount = 8;          // blades evaluated per clump (upper bound)
    float curvature = 0.35f;     // sideways tip offset as a fraction of height
    float maxLean = 0.35f;       // maximum wind lean (radians)
    float windGain = 0.12f;      // lean radians per m/s of shared wind
    float tipWidth = 0.35f;      // tip radius as a fraction of the base width
    float regionSize = 512.0f;   // metres per grass region container (grid tile)
    float cellSize = 24.0f;      // target grid cell size (candidate budget)
    float roughness = 0.85f;
    glm::vec3 tint = glm::vec3(1.0f); // multiplies the per-type blade colors
};

} // namespace sdf_gpu
