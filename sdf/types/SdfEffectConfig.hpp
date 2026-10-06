#pragma once

// Whole-module tuning: one instance owned by SdfRenderer, shared with the
// ray marching widget. Previously both sides kept
// mirror copies that could drift; now the renderer owns this single copy
// and widgets bind its fields directly, while renderer setters keep
// clamping + dirty-flag routing.
//
// This is plain tuning data, NOT GPU state: the SSBO mirrors (SmokeFragBulletGPU
// etc.) and scene rebuilds are derived from it by the renderer.

#include "SdfLavaConfig.hpp"
#include "SdfSmokeConfig.hpp"
#include "SdfBulletConfig.hpp"

struct SdfEffectConfig {
    SdfLavaConfig lava;
    SdfSmokeConfig smoke;
    SdfBulletConfig bullet;
};
