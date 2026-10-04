#pragma once

// Whole-module tuning: one instance owned by SdfRenderer, shared with the
// effect widgets (SdfWidget, SmokeBulletWidget). Previously both sides kept
// mirror copies that could drift; now the renderer owns this single copy
// and widgets bind its fields directly, while renderer setters keep
// clamping + dirty-flag routing.
//
// This is plain tuning data, NOT GPU state: the SSBO mirrors (SmokeState
// etc.) and scene rebuilds are derived from it by the renderer.

#include "types/SdfLavaConfig.hpp"
#include "types/SdfSmokeConfig.hpp"
#include "types/SdfBulletConfig.hpp"

struct SdfEffectConfig {
    SdfLavaConfig lava;
    SdfSmokeConfig smoke;
    SdfBulletConfig bullet;
};
