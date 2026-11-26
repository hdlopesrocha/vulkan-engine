#pragma once

#include <cstdint>

#include <glm/glm.hpp>

#include "../math/TextureType.hpp"
#include "MaterialKind.hpp"
#include "MaterialBlendMode.hpp"

// Material properties per texture (extracted from TextureManager.hpp)
struct MaterialProperties {
    // Mapping enabled: false = disabled, true = enable tessellation + bump mapping
    bool mappingMode = false;
    // Mirror height map V coordinate vertically when true
    bool invertHeight = false;
    // Tessellation-specific height scale (used when mappingMode == 2)
    float tessHeightScale = 0.2f;
    // Tessellation level for hardware displacement (used when mappingMode == 2)
    float tessLevel = 16.0f;
    // Per-material distance-based tessellation range
    float tessMinLevel = 4.0f;
    float tessMaxLevel = 32.0f;

    float ambientFactor = 0.4f;

    float specularStrength = 0.5f;
    float shininess = 32.0f;
    // Mirror amount [0..1]: 1 = full environment mirror at every viewing
    // angle, 0 = no reflection. Terrain materials keep the 0 default so they
    // stay matte; reflective objects opt in with explicit values (e.g. the
    // scene's mirror box and polished spheres).
    float reflectionStrength = 0.0f;

    // Flip height map U coordinate horizontally
    bool invertWidth = false;
    // Normal map conventions (defaults keep existing behavior)
    bool normalFlipY = false;   // flip normal map Y (green) channel if true
    bool normalSwapXZ = false;  // swap R/B channels (X/Z) if true
        // default: legacy invert behavior for backward compatibility

    // Multiplier on the sampled roughness texture [0..1]. Neutral default is
    // 1.0 (full texture range): lower values compress every material toward a
    // mirror finish (at 0.5 even a fully rough texture can only reach 50%
    // matte), for both the RT mirror mix and the specular exponent in
    // SolidRenderer.frag. Tune per material in the texture viewer.
    float roughnessFactor = 1.0f;
    float aoFactor = 1.0f;
    bool useAO = true;

    // ── Material kind / texture selection ────────────────────────────────
    // Textures: samples the material's own layer in the scene arrays.
    // Video: samples slot `textureIndex` of the video texture array.
    // Water: water params palette (surface shaders).
    MaterialKind kind = MaterialKind::Texture;
    // Layer the material samples. -1 = the material's own index (the classic
    // one-material-per-layer behaviour); >= 0 lets several materials share a
    // texture layer or point a Video material at a video slot.
    int textureIndex = -1;

    // ── Texture coordinates ─────────────────────────────────────────────
    // The technique is a per-vertex value (Vertex::textureType, painted by the
    // scene's BrushPainter). The material only carries the Angular
    // center/radius: xyz = bounding-volume center, w = radius (0 keeps the
    // default world-space sphere, useful for objects whose SDF bounds are
    // unknown).
    glm::vec4 mappingOrigin = glm::vec4(0.0f);

    // ── Layer blending ──────────────────────────────────────────────────
    MaterialBlendMode blendMode = MaterialBlendMode::Normal;

    // ── True raymarched displacement ─────────────────────────────────────
    // Independent of tessellation (which stays as the legacy height-bump path
    // for compatibility). The octree raymarch folds this offset into the SDF:
    //   F(p) = d_original(p) - displacement(p)
    // and intersects/refines the actual displaced zero set. The raster
    // tessellation path is not involved.
    // Amplitude is signed world units at the normalized height extremes;
    // 0 keeps the new path disabled (legacy behaviour unchanged).
    float displacementAmplitude = 0.0f;
    // UV multiplier applied on top of the per-vertex texture scale. The
    // inverse is the world-space feature wavelength used by the conservative
    // march.
    float displacementFrequency = 1.0f;
    // Normalized height offset added before the amplitude: +0.5 raises the
    // neutral surface by half the amplitude (negative values recess it).
    float displacementOffset = 0.0f;
    // 0 disables the new path; 1 applies the full amplitude. Intermediate
    // values scale the geometric offset.
    float displacementStrength = 0.0f;
    // Optional clamp on |displacement| in world units (0 = |amplitude|*strength).
    float displacementBounds = 0.0f;
    // Height-map layer used for displacement. -1 = this material's own layer;
    // >= 0 selects another material index's height texture through the brush
    // map (letting several materials share one displacement source).
    int displacementTexture = -1;

    // ── Secondary SDF displacement ────────────────────────────────────────
    // Alternative to the height texture: a GPU-evaluated scale pattern oriented
    // by the brush segment velocity. 0 = texture path, 1 = triangular scales,
    // 2 = hexagonal scales, 3 = animated Perlin, 4 = animated Voronoi.
    // The pattern is never stored in the octree.
    int displacementPattern = 0;
    // Peak world-unit displacement of the pattern and its world-unit repeat
    // cell (a larger repeat makes larger scales).
    float displacementPatternDisplacement = 8.0f;
    float displacementPatternRepeat = 64.0f;
    // Last dragged brush segment. `time` is the application clock, velocity is
    // world-space m/s, and intensity [0..1] scales the pattern displacement.
    float displacementSegmentTime = 0.0f;
    glm::vec3 displacementSegmentVelocity{0.0f, 1.0f, 0.0f};
    float displacementSegmentIntensity = 1.0f;
};
