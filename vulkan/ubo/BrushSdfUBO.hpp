#pragma once

#include <glm/glm.hpp>
#include <cstdint>

// Per-frame brush SDF description (std140 block, set=1 binding=0), the single
// source of truth shared by the CPU, the brush preview raymarcher
// (BrushSdf.frag) and the solid shader's PAINT/REMOVE intersection test
// (SolidSurface.glsl). One UBO per frame-in-flight is streamed by
// BrushSdfRenderer::updateParams().
//
// Layout contract (all members 16-byte aligned, matching the GLSL twin in
// shaders/includes/brush/BrushSdf.glsl):
//   xTranslate : xyz = model.translate (world)
//   xScale     : xyz = model.scale (Behavior per primitive, see BrushSdf.glsl)
//   xRotation  : quaternion (x, y, z, w) of model.quaternion
//   xParams0   : primitive params: capsule a.xyz / r1 in w, torus R,r in xy,
//                tapered radii r1,r2 in xy (per-sdfType convention)
//   xParams1   : capsule b.xyz / r2 in w
//   xEffect0   : amplitude, frequency, threshold, cellSize
//   xEffect1   : brightness, contrast, unused, unused
//   xSweepStart: xyz = previous world translate (sweep trail start)
//   xBounds    : xyz = world bounding sphere center, w = radius
//   xFlags     : x = sdfType (0..9), y = effectType + 1 (0 = no effect),
//                z = material index (texture array layer), w = sweepMode
//   xHsv       : xyz = paint tint HSV
//   xViewport  : xy = target size in pixels (ray reconstruction)
struct BrushSdfUBO {
    glm::vec4  translate   = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
    glm::vec4  scale       = glm::vec4(1.0f);
    glm::vec4  rotation    = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
    glm::vec4  params0     = glm::vec4(0.0f);
    glm::vec4  params1     = glm::vec4(0.0f);
    glm::vec4  effect0     = glm::vec4(0.0f);
    glm::vec4  effect1     = glm::vec4(0.0f);
    glm::vec4  sweepStart  = glm::vec4(0.0f);
    glm::vec4  bounds      = glm::vec4(0.0f);   // radius 0 = no brush
    glm::uvec4 flags       = glm::uvec4(0u);
    glm::vec4  hsv         = glm::vec4(0.0f, 0.5f, 0.5f, 1.0f);
    glm::vec4  viewport    = glm::vec4(1.0f, 1.0f, 0.0f, 0.0f);
};

static_assert(sizeof(BrushSdfUBO) == 12 * 16,
              "BrushSdfUBO must match the std140 GLSL block layout (12 x vec4)");
